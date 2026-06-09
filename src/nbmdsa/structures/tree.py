"""Steepest-descent tree structures.

Three modes
-----------
'implicit'  nothing stored; parent/children recomputed from z at every call.
            API: parent(idx, z)   children(idx, z, buf)
'par'       parents array stored; children derived from parents.
            API: parent(idx)      children(idx, buf)
'full'      parents + CSR children stored; same query API as 'par'.
            Raw arrays (parents, child_ptr, child_data) accessible for
            algorithm-level traversal.

All modes
---------
- Built with make_tree(z, neighbours_fn, mask, mode)
- mask: 0=nodata (excluded), 1=normal node, 3=forced root
- Indices are original 1D grid indices — no remapping.
- z is NOT stored. Updates live in nbmdsa.algorithms.tree_update.

Tree fields
-----------
  parents, child_ptr, child_data  — arrays (_EMPTY when not applicable)
  parent, children                — query closures
  neighbours_fn                   — the neighbourer passed at construction
  mask                            — copy of mask passed at construction

Detection of mode from tree fields:
  implicit : len(tree.parents) == 0
  par      : len(tree.parents) > 0 and len(tree.child_ptr) == 0
  full     : len(tree.parents) > 0 and len(tree.child_ptr) > 0

children buf: same 8-slot positional layout as the neighbourer.
  slot k = same spatial direction, value = child index or -1.
"""

from collections import namedtuple

import numpy as np
import numba as nb


Tree = namedtuple('Tree', ['parents', 'child_ptr', 'child_data',
                           'parent', 'children',
                           'neighbours_fn', 'mask'])

_EMPTY = np.empty(0, np.int64)


# ── module-level njit helpers ─────────────────────────────────────────────────

@nb.njit
def _steepest_parent(idx, z, mask, nbuf, neighbours_fn):
    if mask[idx] == nb.uint8(3):   # forced root — never routes elsewhere
        return idx
    neighbours_fn(idx, nbuf)
    best   = idx
    best_z = z[idx]
    for k in range(nb.int64(8)):
        j = nbuf[k]
        if j == nb.int64(-1) or mask[j] == nb.uint8(0):
            continue
        if z[j] < best_z:
            best_z = z[j]
            best   = j
    return best


@nb.njit
def _build_parents(z, mask, parents, neighbours_fn):
    nbuf = np.empty(nb.int64(8), nb.int64)
    for i in range(nb.int64(len(z))):
        i64 = nb.int64(i)
        if mask[i] == nb.uint8(0):
            parents[i] = nb.int64(-1)
        elif mask[i] == nb.uint8(3):
            parents[i] = i64
        else:
            parents[i] = _steepest_parent(i64, z, mask, nbuf, neighbours_fn)


@nb.njit
def _build_csr(parents, mask, child_ptr, child_data):
    n = nb.int64(len(parents))
    child_ptr[:] = nb.int64(0)
    for i in range(n):
        if mask[i] == nb.uint8(0):
            continue
        p = parents[i]
        if p != nb.int64(i):
            child_ptr[p] += nb.int64(1)
    total = nb.int64(0)
    for i in range(n):
        c            = child_ptr[i]
        child_ptr[i] = total
        total       += c
    child_ptr[n] = total
    tmp = child_ptr[:n].copy()
    for i in range(n):
        if mask[i] == nb.uint8(0):
            continue
        p = parents[i]
        if p != nb.int64(i):
            child_data[tmp[p]] = nb.int64(i)
            tmp[p]            += nb.int64(1)


# ── factory ───────────────────────────────────────────────────────────────────

def make_tree(z, neighbours_fn, mask, mode='par'):
    """Build a steepest-descent tree.

    Parameters
    ----------
    z             : float64 array, shape (n,)
    neighbours_fn : closure from make_neighbours (1D)
    mask          : uint8 array, shape (n,)  — 0=nodata, 1=node, 3=root
    mode          : 'implicit' | 'par' | 'full'
    """
    n     = len(z)
    _mask = mask.copy()

    if mode == 'implicit':
        @nb.njit
        def parent(idx, z):
            nbuf = np.empty(nb.int64(8), nb.int64)
            return _steepest_parent(nb.int64(idx), z, _mask, nbuf, neighbours_fn)

        @nb.njit
        def children(idx, z, buf):
            nbuf  = np.empty(nb.int64(8), nb.int64)
            nbuf2 = np.empty(nb.int64(8), nb.int64)
            neighbours_fn(nb.int64(idx), nbuf)
            for k in range(nb.int64(8)):
                j = nbuf[k]
                if j == nb.int64(-1) or _mask[j] == nb.uint8(0):
                    buf[k] = nb.int64(-1)
                    continue
                p      = _steepest_parent(j, z, _mask, nbuf2, neighbours_fn)
                buf[k] = j if p == nb.int64(idx) else nb.int64(-1)

        return Tree(_EMPTY, _EMPTY, _EMPTY, parent, children, neighbours_fn, _mask)

    elif mode == 'par':
        parents = np.empty(n, np.int64)
        _build_parents(z, mask, parents, neighbours_fn)

        @nb.njit
        def parent(idx):
            return parents[idx]

        @nb.njit
        def children(idx, buf):
            nbuf = np.empty(nb.int64(8), nb.int64)
            neighbours_fn(nb.int64(idx), nbuf)
            for k in range(nb.int64(8)):
                j      = nbuf[k]
                buf[k] = j if (j != nb.int64(-1) and parents[j] == nb.int64(idx)) \
                           else nb.int64(-1)

        return Tree(parents, _EMPTY, _EMPTY, parent, children, neighbours_fn, _mask)

    elif mode == 'full':
        parents    = np.empty(n, np.int64)
        child_ptr  = np.empty(n + 1, np.int64)
        child_data = np.empty(n, np.int64)
        _build_parents(z, mask, parents, neighbours_fn)
        _build_csr(parents, mask, child_ptr, child_data)

        @nb.njit
        def parent(idx):
            return parents[idx]

        @nb.njit
        def children(idx, buf):
            nbuf = np.empty(nb.int64(8), nb.int64)
            neighbours_fn(nb.int64(idx), nbuf)
            for k in range(nb.int64(8)):
                j      = nbuf[k]
                buf[k] = j if (j != nb.int64(-1) and parents[j] == nb.int64(idx)) \
                           else nb.int64(-1)

        return Tree(parents, child_ptr, child_data, parent, children, neighbours_fn, _mask)

    else:
        raise ValueError(f"mode must be 'implicit', 'par', or 'full', got {mode!r}")
