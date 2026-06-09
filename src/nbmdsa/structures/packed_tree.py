"""Packed (subset) steepest-descent tree.

Like Tree but covers an explicit subset of grid nodes. Internal arrays are
indexed 0..m-1 (local indices); all public API takes and returns grid indices.

Three modes: 'implicit', 'par', 'full' — same semantics as Tree.

PackedTree fields
-----------------
  nodes           int64[m]   — grid indices of active nodes
  grid_to_local   int64[n]   — grid_idx → local_idx, -1 for non-members
  parents         int64[m]   — local indices (_EMPTY for implicit)
  child_ptr       int64[m+1] — CSR ptr (_EMPTY unless full)
  child_data      int64[m]   — CSR data (_EMPTY unless full)
  parent          closure    — parent(grid_idx) → grid_idx
  children        closure    — children(grid_idx, buf) or (grid_idx, z, buf)
  neighbours_fn   closure
  mask            uint8[n]   — copy of creation mask

Callers must pass only valid grid indices in nodes (mask != 0).
Mode detection: same as Tree (len(parents)==0 → implicit, etc.).
"""

from collections import namedtuple

import numpy as np
import numba as nb


PackedTree = namedtuple('PackedTree', [
    'nodes', 'grid_to_local',
    'parents', 'child_ptr', 'child_data',
    'parent', 'children',
    'neighbours_fn', 'mask',
])

_EMPTY = np.empty(0, np.int64)


# ── njit helpers ──────────────────────────────────────────────────────────────

@nb.njit
def _packed_steepest_local(local_idx, nodes, grid_to_local, z, mask, nbuf, neighbours_fn):
    grid_idx = nodes[local_idx]
    neighbours_fn(grid_idx, nbuf)
    best   = local_idx
    best_z = z[grid_idx]
    for k in range(nb.int64(8)):
        j_grid  = nbuf[k]
        if j_grid == nb.int64(-1):
            continue
        j_local = grid_to_local[j_grid]
        if j_local == nb.int64(-1) or mask[j_grid] == nb.uint8(0):
            continue
        if z[j_grid] < best_z:
            best_z = z[j_grid]
            best   = j_local
    return best


@nb.njit
def _build_packed_parents(nodes, grid_to_local, z, mask, parents, neighbours_fn):
    nbuf = np.empty(nb.int64(8), nb.int64)
    for li in range(nb.int64(len(nodes))):
        gi = nodes[li]
        if mask[gi] == nb.uint8(3):
            parents[li] = li
        else:
            parents[li] = _packed_steepest_local(
                li, nodes, grid_to_local, z, mask, nbuf, neighbours_fn)


@nb.njit
def _build_packed_csr(parents, child_ptr, child_data):
    m = nb.int64(len(parents))
    child_ptr[:] = nb.int64(0)
    for li in range(m):
        p = parents[li]
        if p != li:
            child_ptr[p] += nb.int64(1)
    total = nb.int64(0)
    for li in range(m):
        c             = child_ptr[li]
        child_ptr[li] = total
        total        += c
    child_ptr[m] = total
    tmp = child_ptr[:m].copy()
    for li in range(m):
        p = parents[li]
        if p != li:
            child_data[tmp[p]] = li
            tmp[p]            += nb.int64(1)


@nb.njit
def _apply_external_parents(nodes, grid_to_local, new_parents_grid, parents):
    for li in range(nb.int64(len(nodes))):
        gi            = nodes[li]
        parents[li]   = grid_to_local[new_parents_grid[gi]]


# ── factory ───────────────────────────────────────────────────────────────────

def make_packed_tree(nodes, z, neighbours_fn, mask, mode='par'):
    """Build a packed steepest-descent tree over an explicit node subset.

    Parameters
    ----------
    nodes         : int64 array of grid indices (size m) — must all have mask != 0
    z             : float64 array of size n (full grid)
    neighbours_fn : closure from make_neighbours (1D)
    mask          : uint8 array of size n  — 0=nodata, 1=node, 3=root
    mode          : 'implicit' | 'par' | 'full'
    """
    n             = len(z)
    m             = len(nodes)
    _nodes        = np.asarray(nodes, dtype=np.int64)
    _mask         = mask.copy()

    grid_to_local = np.full(n, -1, dtype=np.int64)
    for li in range(m):
        grid_to_local[_nodes[li]] = li

    if mode == 'implicit':
        @nb.njit
        def parent(grid_idx, z):
            nbuf     = np.empty(nb.int64(8), nb.int64)
            local_idx = grid_to_local[grid_idx]
            p_local  = _packed_steepest_local(
                local_idx, _nodes, grid_to_local, z, _mask, nbuf, neighbours_fn)
            return _nodes[p_local]

        @nb.njit
        def children(grid_idx, z, buf):
            nbuf      = np.empty(nb.int64(8), nb.int64)
            nbuf2     = np.empty(nb.int64(8), nb.int64)
            local_idx = grid_to_local[grid_idx]
            neighbours_fn(grid_idx, nbuf)
            for k in range(nb.int64(8)):
                j_grid  = nbuf[k]
                if j_grid == nb.int64(-1):
                    buf[k] = nb.int64(-1)
                    continue
                j_local = grid_to_local[j_grid]
                if j_local == nb.int64(-1) or _mask[j_grid] == nb.uint8(0):
                    buf[k] = nb.int64(-1)
                    continue
                p_local = _packed_steepest_local(
                    j_local, _nodes, grid_to_local, z, _mask, nbuf2, neighbours_fn)
                buf[k] = j_grid if p_local == local_idx else nb.int64(-1)

        return PackedTree(_nodes, grid_to_local,
                          _EMPTY, _EMPTY, _EMPTY,
                          parent, children, neighbours_fn, _mask)

    elif mode == 'par':
        parents = np.empty(m, np.int64)
        _build_packed_parents(_nodes, grid_to_local, z, _mask, parents, neighbours_fn)

        @nb.njit
        def parent(grid_idx):
            return _nodes[parents[grid_to_local[grid_idx]]]

        @nb.njit
        def children(grid_idx, buf):
            nbuf      = np.empty(nb.int64(8), nb.int64)
            local_idx = grid_to_local[grid_idx]
            neighbours_fn(grid_idx, nbuf)
            for k in range(nb.int64(8)):
                j_grid  = nbuf[k]
                if j_grid == nb.int64(-1):
                    buf[k] = nb.int64(-1)
                    continue
                j_local = grid_to_local[j_grid]
                buf[k]  = j_grid \
                    if (j_local != nb.int64(-1) and parents[j_local] == local_idx) \
                    else nb.int64(-1)

        return PackedTree(_nodes, grid_to_local,
                          parents, _EMPTY, _EMPTY,
                          parent, children, neighbours_fn, _mask)

    elif mode == 'full':
        parents    = np.empty(m, np.int64)
        child_ptr  = np.empty(m + 1, np.int64)
        child_data = np.empty(m, np.int64)
        _build_packed_parents(_nodes, grid_to_local, z, _mask, parents, neighbours_fn)
        _build_packed_csr(parents, child_ptr, child_data)

        @nb.njit
        def parent(grid_idx):
            return _nodes[parents[grid_to_local[grid_idx]]]

        @nb.njit
        def children(grid_idx, buf):
            nbuf      = np.empty(nb.int64(8), nb.int64)
            local_idx = grid_to_local[grid_idx]
            neighbours_fn(grid_idx, nbuf)
            for k in range(nb.int64(8)):
                j_grid  = nbuf[k]
                if j_grid == nb.int64(-1):
                    buf[k] = nb.int64(-1)
                    continue
                j_local = grid_to_local[j_grid]
                buf[k]  = j_grid \
                    if (j_local != nb.int64(-1) and parents[j_local] == local_idx) \
                    else nb.int64(-1)

        return PackedTree(_nodes, grid_to_local,
                          parents, child_ptr, child_data,
                          parent, children, neighbours_fn, _mask)

    else:
        raise ValueError(f"mode must be 'implicit', 'par', or 'full', got {mode!r}")
