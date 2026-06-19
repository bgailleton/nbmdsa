"""Multiple-flow direction (MFD) topological order and traversal.

No tree structure is stored. Receiver/donor relationships are computed on the
fly from z and the neighbourer at every step.

  Receiver of i : neighbour j where z[j] < z[i]  (strictly lower)
  Donor    of i : neighbour j where z[j] > z[i]  (strictly higher)

Topological order: Kahn's on the MFD DAG.
  in-degree of i = number of strictly-higher valid neighbours (donors).
  Default output: donors first (local maxima → sinks), analogous to
  leaves-first in tree mode.  reverse=True gives sinks first.

Kernel signature (all functions):
    kernel(idx, z, mask, neighbours_fn, *extra_args)

direction='up'   — toward donors   (higher cells / headwaters)
direction='down' — toward receivers (lower cells / outlets)
direction='none' — no DAG direction; kernel applied to all valid cells in
                   linear order (full) or all reachable neighbours (partial)
"""

import numpy as np
import numba as nb

from nbmdsa.algorithms.tree_traversal import _hpush, _hpop

_UP   = 1
_DOWN = 2
_NONE = 0


def _parse_direction(direction):
    if direction == 'up':   return _UP
    if direction == 'down': return _DOWN
    if direction == 'none': return _NONE
    raise ValueError(f"direction must be 'up', 'down', or 'none', got {direction!r}")


# ── Kahn's topological sort on MFD DAG ───────────────────────────────────────

@nb.njit
def _mfd_kahn(z, mask, neighbours_fn, order):
    n      = nb.int64(len(z))
    in_deg = np.zeros(n, nb.int64)
    nbuf   = np.empty(nb.int64(8), nb.int64)

    for i in range(n):
        if mask[i] == nb.uint8(0):
            continue
        neighbours_fn(nb.int64(i), nbuf)
        for k in range(nb.int64(8)):
            j = nbuf[k]
            if j == nb.int64(-1) or mask[j] == nb.uint8(0):
                continue
            if z[j] > z[i]:                   # j is a donor of i
                in_deg[i] += nb.int64(1)

    q  = np.empty(n, nb.int64)
    qh = qt = nb.int64(0)
    for i in range(n):
        if mask[i] != nb.uint8(0) and in_deg[i] == nb.int64(0):
            q[qt] = nb.int64(i); qt += nb.int64(1)

    count = nb.int64(0)
    while qh < qt:
        i            = q[qh]; qh += nb.int64(1)
        order[count] = i;     count += nb.int64(1)
        neighbours_fn(nb.int64(i), nbuf)
        for k in range(nb.int64(8)):
            j = nbuf[k]
            if j == nb.int64(-1) or mask[j] == nb.uint8(0):
                continue
            if z[j] < z[i]:                   # j is a receiver of i
                in_deg[j] -= nb.int64(1)
                if in_deg[j] == nb.int64(0):
                    q[qt] = j; qt += nb.int64(1)
    return count


def mfd_topo_order(z, mask, neighbours_fn, reverse=False):
    """Topological order of the MFD DAG.

    reverse=False — donors first  (local maxima → sinks; use for accumulation)
    reverse=True  — receivers first (sinks → local maxima; use for distribution)
    """
    order = np.empty(len(z), np.int64)
    count = _mfd_kahn(z, mask, neighbours_fn, order)
    result = order[:count]
    return result[::-1].copy() if reverse else result


# ── full traversal ────────────────────────────────────────────────────────────

@nb.njit
def _run_mfd_full(topo_order, z, mask, neighbours_fn, kernel, extra_args, direction):
    n = nb.int64(len(topo_order))
    if direction == 1:          # up — forward (donors first)
        for i in range(n):
            idx = topo_order[i]
            if mask[idx] != nb.uint8(0):
                kernel(idx, z, mask, neighbours_fn, *extra_args)
    elif direction == 2:        # down — reversed (receivers first)
        for i in range(n - nb.int64(1), nb.int64(-1), nb.int64(-1)):
            idx = topo_order[i]
            if mask[idx] != nb.uint8(0):
                kernel(idx, z, mask, neighbours_fn, *extra_args)
    else:                       # none — linear scan, ignore DAG order
        for idx in range(nb.int64(len(mask))):
            if mask[idx] != nb.uint8(0):
                kernel(idx, z, mask, neighbours_fn, *extra_args)


def mfd_traversal_full(topo_order, z, mask, neighbours_fn, kernel, extra_args,
                        direction='up'):
    """Apply kernel to every node following a pre-computed MFD topo order.

    topo_order : int64[:] from mfd_topo_order() — donors-first by default.
    direction  : 'up'   → iterate as given (donors first)
                 'down' → iterate reversed (receivers first)
                 'none' → linear scan over all valid cells; topo_order ignored
    extra_args : tuple forwarded to kernel after standard arguments.
    """
    _run_mfd_full(topo_order, z, mask, neighbours_fn, kernel, extra_args,
                  _parse_direction(direction))


# ── partial traversal helpers ─────────────────────────────────────────────────

@nb.njit
def _partial_mfd_bfs(start_nodes, z, mask, neighbours_fn,
                      kernel, extra_args, direction, multi_enabled):
    n    = nb.int64(len(z))
    qcap = n * nb.int64(9) if multi_enabled else n
    q    = np.empty(qcap, nb.int64)
    qh   = qt = nb.int64(0)
    vis  = np.zeros(n, nb.uint8)
    nbuf = np.empty(nb.int64(8), nb.int64)

    for s in range(nb.int64(len(start_nodes))):
        idx = start_nodes[s]
        if multi_enabled:
            q[qt] = idx; qt += nb.int64(1)
        elif vis[idx] == nb.uint8(0):
            vis[idx] = nb.uint8(1)
            q[qt] = idx; qt += nb.int64(1)

    while qh < qt:
        idx = q[qh]; qh += nb.int64(1)
        kernel(idx, z, mask, neighbours_fn, *extra_args)
        neighbours_fn(nb.int64(idx), nbuf)
        for k in range(nb.int64(8)):
            j = nbuf[k]
            if j == nb.int64(-1) or mask[j] == nb.uint8(0):
                continue
            if direction == 1:              # up — donors only (strictly higher)
                if z[j] <= z[idx]: continue
            elif direction == 2:            # down — receivers only (strictly lower)
                if z[j] >= z[idx]: continue
            # none — all valid neighbours pass through
            if multi_enabled:
                q[qt] = j; qt += nb.int64(1)
            elif vis[j] == nb.uint8(0):
                vis[j] = nb.uint8(1)
                q[qt] = j; qt += nb.int64(1)


@nb.njit
def _partial_mfd_dfs(start_nodes, z, mask, neighbours_fn,
                      kernel, extra_args, direction, multi_enabled):
    n     = nb.int64(len(z))
    scap  = n * nb.int64(9) if multi_enabled else n
    stack = np.empty(scap, nb.int64)
    top   = nb.int64(0)
    vis   = np.zeros(n, nb.uint8)
    nbuf  = np.empty(nb.int64(8), nb.int64)

    for s in range(nb.int64(len(start_nodes)) - nb.int64(1), nb.int64(-1), nb.int64(-1)):
        idx = start_nodes[s]
        if multi_enabled:
            stack[top] = idx; top += nb.int64(1)
        elif vis[idx] == nb.uint8(0):
            vis[idx] = nb.uint8(1)
            stack[top] = idx; top += nb.int64(1)

    while top > nb.int64(0):
        top -= nb.int64(1)
        idx = stack[top]
        kernel(idx, z, mask, neighbours_fn, *extra_args)
        neighbours_fn(nb.int64(idx), nbuf)
        for k in range(nb.int64(8) - nb.int64(1), nb.int64(-1), nb.int64(-1)):
            j = nbuf[k]
            if j == nb.int64(-1) or mask[j] == nb.uint8(0):
                continue
            if direction == 1:
                if z[j] <= z[idx]: continue
            elif direction == 2:
                if z[j] >= z[idx]: continue
            if multi_enabled:
                stack[top] = j; top += nb.int64(1)
            elif vis[j] == nb.uint8(0):
                vis[j] = nb.uint8(1)
                stack[top] = j; top += nb.int64(1)


@nb.njit
def _partial_mfd_pq(start_nodes, z, mask, neighbours_fn,
                     kernel, extra_args, direction, multi_enabled, negate):
    n        = nb.int64(len(z))
    hcap     = n * nb.int64(9) if multi_enabled else n
    hscores  = np.empty(hcap, nb.float64)
    hindices = np.empty(hcap, nb.int64)
    hsize    = nb.int64(0)
    vis      = np.zeros(n, nb.uint8)
    nbuf     = np.empty(nb.int64(8), nb.int64)

    for s in range(nb.int64(len(start_nodes))):
        idx = start_nodes[s]
        sc  = -z[idx] if negate else z[idx]
        if multi_enabled:
            hsize = _hpush(hscores, hindices, hsize, sc, idx)
        elif vis[idx] == nb.uint8(0):
            vis[idx] = nb.uint8(1)
            hsize = _hpush(hscores, hindices, hsize, sc, idx)

    while hsize > nb.int64(0):
        idx, hsize = _hpop(hscores, hindices, hsize)
        kernel(idx, z, mask, neighbours_fn, *extra_args)
        neighbours_fn(nb.int64(idx), nbuf)
        for k in range(nb.int64(8)):
            j = nbuf[k]
            if j == nb.int64(-1) or mask[j] == nb.uint8(0):
                continue
            if direction == 1:
                if z[j] <= z[idx]: continue
            elif direction == 2:
                if z[j] >= z[idx]: continue
            sc = -z[j] if negate else z[j]
            if multi_enabled:
                hsize = _hpush(hscores, hindices, hsize, sc, j)
            elif vis[j] == nb.uint8(0):
                vis[j] = nb.uint8(1)
                hsize = _hpush(hscores, hindices, hsize, sc, j)


# ── public API ────────────────────────────────────────────────────────────────

def mfd_traversal_partial(start_nodes, z, mask, neighbours_fn, kernel, extra_args,
                           direction='up', multi_enabled=False,
                           mode='bfs', min_heap=True):
    """Traverse the MFD DAG expanding from start_nodes.

    start_nodes  : int64[:] starting grid indices
    z            : float64[:] elevation array
    mask         : uint8[:] — 0=nodata, 1=normal, 3=outlet
    neighbours_fn: closure from make_neighbours (1D)
    kernel       : @nb.njit  kernel(idx, z, mask, neighbours_fn, *extra_args)
    extra_args   : tuple of additional arguments forwarded to kernel
    direction    : 'up'   → expand toward donors (higher cells)
                   'down' → expand toward receivers (lower cells)
                   'none' → expand to all valid neighbours (no DAG direction)
    multi_enabled: True → a node may be processed more than once
    mode         : 'bfs' | 'dfs' | 'pq'
    min_heap     : True → ascending z / False → descending z  (mode='pq' only)
    """
    d = _parse_direction(direction)
    if mode == 'bfs':
        _partial_mfd_bfs(start_nodes, z, mask, neighbours_fn,
                          kernel, extra_args, d, multi_enabled)
    elif mode == 'dfs':
        _partial_mfd_dfs(start_nodes, z, mask, neighbours_fn,
                          kernel, extra_args, d, multi_enabled)
    elif mode == 'pq':
        _partial_mfd_pq(start_nodes, z, mask, neighbours_fn,
                         kernel, extra_args, d, multi_enabled, not min_heap)
    else:
        raise ValueError(f"mode must be 'bfs', 'dfs', or 'pq', got {mode!r}")
