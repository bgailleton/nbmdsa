"""Tree traversal algorithms.

Full traversal iterates a pre-calculated topological order.
Partial traversal expands from a set of start nodes.

Kernel signature — par/full trees:
    kernel(idx, parents, child_ptr, child_data, mask, neighbours_fn, *extra_args)
    child_ptr / child_data are empty arrays for par-mode trees.

Kernel signature — implicit trees (tree_traversal_partial_implicit):
    kernel(idx, z, mask, neighbours_fn, *extra_args)

upstream=True  — toward children (headwaters / leaves direction)
upstream=False — toward parents  (outlet / root direction)
"""

import numpy as np
import numba as nb

from nbmdsa.structures.tree import _steepest_parent


# ── inline min-heap ops ───────────────────────────────────────────────────────
# Kept module-level and @njit so they inline into traversal functions.

@nb.njit
def _hpush(scores, indices, size, score, idx):
    scores[size]  = score
    indices[size] = idx
    i = size
    while i > nb.int64(0):
        p = (i - nb.int64(1)) >> nb.int64(1)
        if scores[p] > scores[i]:
            scores[p],  scores[i]  = scores[i],  scores[p]
            indices[p], indices[i] = indices[i], indices[p]
            i = p
        else:
            break
    return size + nb.int64(1)


@nb.njit
def _hpop(scores, indices, size):
    idx  = indices[nb.int64(0)]
    size -= nb.int64(1)
    indices[nb.int64(0)] = indices[size]
    scores[nb.int64(0)]  = scores[size]
    i = nb.int64(0)
    while True:
        l = nb.int64(2) * i + nb.int64(1)
        r = l + nb.int64(1)
        s = i
        if l < size and scores[l] < scores[s]: s = l
        if r < size and scores[r] < scores[s]: s = r
        if s == i: break
        scores[i],  scores[s]  = scores[s],  scores[i]
        indices[i], indices[s] = indices[s], indices[i]
        i = s
    return idx, size


# ── child lookup (par: scan neighbours; full: CSR) ────────────────────────────

@nb.njit
def _get_children(idx, parents, child_ptr, child_data, mask, neighbours_fn, nbuf, cbuf):
    count = nb.int64(0)
    if len(child_ptr) > nb.int64(0):                        # full mode — CSR
        for ci in range(child_ptr[idx], child_ptr[idx + nb.int64(1)]):
            j = child_data[ci]
            if mask[j] != nb.uint8(0):
                cbuf[count] = j; count += nb.int64(1)
    else:                                                    # par mode — scan neighbours
        neighbours_fn(nb.int64(idx), nbuf)
        for k in range(nb.int64(8)):
            j = nbuf[k]
            if j != nb.int64(-1) and mask[j] != nb.uint8(0) \
                    and parents[j] == nb.int64(idx):
                cbuf[count] = j; count += nb.int64(1)
    return count


# ── full traversal ────────────────────────────────────────────────────────────

@nb.njit
def _run_full(topo_order, parents, child_ptr, child_data, mask,
              neighbours_fn, kernel, extra_args, upstream):
    n = nb.int64(len(topo_order))
    if upstream:
        for i in range(n):
            idx = topo_order[i]
            if mask[idx] != nb.uint8(0):
                kernel(idx, parents, child_ptr, child_data, mask,
                       neighbours_fn, *extra_args)
    else:
        for i in range(n - nb.int64(1), nb.int64(-1), nb.int64(-1)):
            idx = topo_order[i]
            if mask[idx] != nb.uint8(0):
                kernel(idx, parents, child_ptr, child_data, mask,
                       neighbours_fn, *extra_args)


def tree_traversal_full(topo_order, tree, kernel, extra_args, upstream=True):
    """Apply kernel to every node following a pre-calculated topological order.

    topo_order : int64[:] — output of topo_order() (leaves-first by default)
    upstream   : True  → iterate as given / False → iterate reversed
    extra_args : tuple forwarded to kernel after standard arguments
    """
    _run_full(topo_order, tree.parents, tree.child_ptr, tree.child_data,
              tree.mask, tree.neighbours_fn, kernel, extra_args, upstream)


# ── partial traversal — BFS ───────────────────────────────────────────────────

@nb.njit
def _partial_bfs(start_nodes, parents, child_ptr, child_data, mask, neighbours_fn,
                 kernel, extra_args, upstream, multi_enabled):
    n    = nb.int64(len(parents))
    q    = np.empty(n, nb.int64)
    qh   = qt = nb.int64(0)
    vis  = np.zeros(n, nb.uint8)
    nbuf = np.empty(nb.int64(8), nb.int64)
    cbuf = np.empty(nb.int64(8), nb.int64)

    for s in range(nb.int64(len(start_nodes))):
        idx = start_nodes[s]
        if multi_enabled:
            q[qt] = idx; qt += nb.int64(1)
        elif vis[idx] == nb.uint8(0):
            vis[idx] = nb.uint8(1)
            q[qt] = idx; qt += nb.int64(1)

    while qh < qt:
        idx = q[qh]; qh += nb.int64(1)
        kernel(idx, parents, child_ptr, child_data, mask, neighbours_fn, *extra_args)
        if upstream:
            cc = _get_children(idx, parents, child_ptr, child_data, mask,
                                neighbours_fn, nbuf, cbuf)
            for i in range(cc):
                j = cbuf[i]
                if multi_enabled:
                    q[qt] = j; qt += nb.int64(1)
                elif vis[j] == nb.uint8(0):
                    vis[j] = nb.uint8(1)
                    q[qt] = j; qt += nb.int64(1)
        else:
            p = parents[idx]
            if p != nb.int64(idx):
                if multi_enabled:
                    q[qt] = p; qt += nb.int64(1)
                elif vis[p] == nb.uint8(0):
                    vis[p] = nb.uint8(1)
                    q[qt] = p; qt += nb.int64(1)


# ── partial traversal — DFS ───────────────────────────────────────────────────

@nb.njit
def _partial_dfs(start_nodes, parents, child_ptr, child_data, mask, neighbours_fn,
                 kernel, extra_args, upstream, multi_enabled):
    n     = nb.int64(len(parents))
    stack = np.empty(n, nb.int64)
    top   = nb.int64(0)
    vis   = np.zeros(n, nb.uint8)
    nbuf  = np.empty(nb.int64(8), nb.int64)
    cbuf  = np.empty(nb.int64(8), nb.int64)

    # push in reverse so first start_node is processed first
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
        kernel(idx, parents, child_ptr, child_data, mask, neighbours_fn, *extra_args)
        if upstream:
            cc = _get_children(idx, parents, child_ptr, child_data, mask,
                                neighbours_fn, nbuf, cbuf)
            for i in range(cc - nb.int64(1), nb.int64(-1), nb.int64(-1)):
                j = cbuf[i]
                if multi_enabled:
                    stack[top] = j; top += nb.int64(1)
                elif vis[j] == nb.uint8(0):
                    vis[j] = nb.uint8(1)
                    stack[top] = j; top += nb.int64(1)
        else:
            p = parents[idx]
            if p != nb.int64(idx):
                if multi_enabled:
                    stack[top] = p; top += nb.int64(1)
                elif vis[p] == nb.uint8(0):
                    vis[p] = nb.uint8(1)
                    stack[top] = p; top += nb.int64(1)


# ── partial traversal — PQ ────────────────────────────────────────────────────
# Pass negate=True for max-heap behaviour (negates z scores internally).

@nb.njit
def _partial_pq(start_nodes, z, parents, child_ptr, child_data, mask, neighbours_fn,
                kernel, extra_args, upstream, multi_enabled, negate):
    n        = nb.int64(len(parents))
    hscores  = np.empty(n, nb.float64)
    hindices = np.empty(n, nb.int64)
    hsize    = nb.int64(0)
    vis      = np.zeros(n, nb.uint8)
    nbuf     = np.empty(nb.int64(8), nb.int64)
    cbuf     = np.empty(nb.int64(8), nb.int64)

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
        kernel(idx, parents, child_ptr, child_data, mask, neighbours_fn, *extra_args)
        if upstream:
            cc = _get_children(idx, parents, child_ptr, child_data, mask,
                                neighbours_fn, nbuf, cbuf)
            for i in range(cc):
                j  = cbuf[i]
                sc = -z[j] if negate else z[j]
                if multi_enabled:
                    hsize = _hpush(hscores, hindices, hsize, sc, j)
                elif vis[j] == nb.uint8(0):
                    vis[j] = nb.uint8(1)
                    hsize = _hpush(hscores, hindices, hsize, sc, j)
        else:
            p = parents[idx]
            if p != nb.int64(idx):
                sc = -z[p] if negate else z[p]
                if multi_enabled:
                    hsize = _hpush(hscores, hindices, hsize, sc, p)
                elif vis[p] == nb.uint8(0):
                    vis[p] = nb.uint8(1)
                    hsize = _hpush(hscores, hindices, hsize, sc, p)


# ── public API ────────────────────────────────────────────────────────────────

def tree_traversal_partial(start_nodes, tree, kernel, extra_args,
                            upstream=True, multi_enabled=False,
                            mode='bfs', z=None, min_heap=True):
    """Traverse a subtree expanding from start_nodes.

    start_nodes  : int64[:] starting grid indices
    tree         : Tree in par or full mode (not implicit)
    kernel       : @nb.njit function with signature
                   kernel(idx, parents, child_ptr, child_data, mask, neighbours_fn,
                          *extra_args)
    extra_args   : tuple of additional arguments forwarded to kernel
    upstream     : True → expand toward children / False → expand toward parents
    multi_enabled: True → a node may be processed more than once
    mode         : 'bfs' | 'dfs' | 'pq'
    z            : float64[:] score array — required for mode='pq'
    min_heap     : True → ascending z (min first) / False → descending z (max first)
    """
    p, cp, cd, m, nb_fn = (tree.parents, tree.child_ptr, tree.child_data,
                            tree.mask, tree.neighbours_fn)
    if mode == 'bfs':
        _partial_bfs(start_nodes, p, cp, cd, m, nb_fn,
                     kernel, extra_args, upstream, multi_enabled)
    elif mode == 'dfs':
        _partial_dfs(start_nodes, p, cp, cd, m, nb_fn,
                     kernel, extra_args, upstream, multi_enabled)
    elif mode == 'pq':
        if z is None:
            raise ValueError("mode='pq' requires z")
        _partial_pq(start_nodes, z, p, cp, cd, m, nb_fn,
                    kernel, extra_args, upstream, multi_enabled, not min_heap)
    else:
        raise ValueError(f"mode must be 'bfs', 'dfs', or 'pq', got {mode!r}")


# ── implicit-tree partial traversal ──────────────────────────────────────────
# Parent/child lookups recompute _steepest_parent from z at every step.
# Kernel: kernel(idx, z, mask, neighbours_fn, *extra_args)

@nb.njit
def _get_children_implicit(idx, z, mask, neighbours_fn, nbuf, nbuf2, cbuf):
    count = nb.int64(0)
    neighbours_fn(nb.int64(idx), nbuf)
    for k in range(nb.int64(8)):
        j = nbuf[k]
        if j == nb.int64(-1) or mask[j] == nb.uint8(0):
            continue
        if _steepest_parent(j, z, mask, nbuf2, neighbours_fn) == nb.int64(idx):
            cbuf[count] = j; count += nb.int64(1)
    return count


@nb.njit
def _partial_bfs_implicit(start_nodes, z, mask, neighbours_fn,
                           kernel, extra_args, upstream, multi_enabled):
    n     = nb.int64(len(z))
    q     = np.empty(n, nb.int64)
    qh    = qt = nb.int64(0)
    vis   = np.zeros(n, nb.uint8)
    nbuf  = np.empty(nb.int64(8), nb.int64)
    nbuf2 = np.empty(nb.int64(8), nb.int64)
    cbuf  = np.empty(nb.int64(8), nb.int64)

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
        if upstream:
            cc = _get_children_implicit(idx, z, mask, neighbours_fn, nbuf, nbuf2, cbuf)
            for i in range(cc):
                j = cbuf[i]
                if multi_enabled:
                    q[qt] = j; qt += nb.int64(1)
                elif vis[j] == nb.uint8(0):
                    vis[j] = nb.uint8(1)
                    q[qt] = j; qt += nb.int64(1)
        else:
            p = _steepest_parent(nb.int64(idx), z, mask, nbuf, neighbours_fn)
            if p != nb.int64(idx):
                if multi_enabled:
                    q[qt] = p; qt += nb.int64(1)
                elif vis[p] == nb.uint8(0):
                    vis[p] = nb.uint8(1)
                    q[qt] = p; qt += nb.int64(1)


@nb.njit
def _partial_dfs_implicit(start_nodes, z, mask, neighbours_fn,
                           kernel, extra_args, upstream, multi_enabled):
    n     = nb.int64(len(z))
    stack = np.empty(n, nb.int64)
    top   = nb.int64(0)
    vis   = np.zeros(n, nb.uint8)
    nbuf  = np.empty(nb.int64(8), nb.int64)
    nbuf2 = np.empty(nb.int64(8), nb.int64)
    cbuf  = np.empty(nb.int64(8), nb.int64)

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
        if upstream:
            cc = _get_children_implicit(idx, z, mask, neighbours_fn, nbuf, nbuf2, cbuf)
            for i in range(cc - nb.int64(1), nb.int64(-1), nb.int64(-1)):
                j = cbuf[i]
                if multi_enabled:
                    stack[top] = j; top += nb.int64(1)
                elif vis[j] == nb.uint8(0):
                    vis[j] = nb.uint8(1)
                    stack[top] = j; top += nb.int64(1)
        else:
            p = _steepest_parent(nb.int64(idx), z, mask, nbuf, neighbours_fn)
            if p != nb.int64(idx):
                if multi_enabled:
                    stack[top] = p; top += nb.int64(1)
                elif vis[p] == nb.uint8(0):
                    vis[p] = nb.uint8(1)
                    stack[top] = p; top += nb.int64(1)


@nb.njit
def _partial_pq_implicit(start_nodes, z, mask, neighbours_fn,
                          kernel, extra_args, upstream, multi_enabled, negate):
    n        = nb.int64(len(z))
    hscores  = np.empty(n, nb.float64)
    hindices = np.empty(n, nb.int64)
    hsize    = nb.int64(0)
    vis      = np.zeros(n, nb.uint8)
    nbuf     = np.empty(nb.int64(8), nb.int64)
    nbuf2    = np.empty(nb.int64(8), nb.int64)
    cbuf     = np.empty(nb.int64(8), nb.int64)

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
        if upstream:
            cc = _get_children_implicit(idx, z, mask, neighbours_fn, nbuf, nbuf2, cbuf)
            for i in range(cc):
                j  = cbuf[i]
                sc = -z[j] if negate else z[j]
                if multi_enabled:
                    hsize = _hpush(hscores, hindices, hsize, sc, j)
                elif vis[j] == nb.uint8(0):
                    vis[j] = nb.uint8(1)
                    hsize = _hpush(hscores, hindices, hsize, sc, j)
        else:
            p = _steepest_parent(nb.int64(idx), z, mask, nbuf, neighbours_fn)
            if p != nb.int64(idx):
                sc = -z[p] if negate else z[p]
                if multi_enabled:
                    hsize = _hpush(hscores, hindices, hsize, sc, p)
                elif vis[p] == nb.uint8(0):
                    vis[p] = nb.uint8(1)
                    hsize = _hpush(hscores, hindices, hsize, sc, p)


def tree_traversal_partial_implicit(start_nodes, tree, z, kernel, extra_args,
                                     upstream=True, multi_enabled=False,
                                     mode='bfs', min_heap=True):
    """Partial traversal for implicit-mode trees.

    Parent and child lookups recompute steepest descent from z at every step.

    start_nodes  : int64[:] starting grid indices
    tree         : Tree in implicit mode
    z            : float64[:] elevation array (used for parent/child lookup and PQ score)
    kernel       : @nb.njit  kernel(idx, z, mask, neighbours_fn, *extra_args)
    extra_args   : tuple of additional arguments forwarded to kernel
    upstream     : True → expand toward children / False → expand toward parents
    multi_enabled: True → a node may be processed more than once
    mode         : 'bfs' | 'dfs' | 'pq'
    min_heap     : True → ascending z (min first) / False → descending z (max first)
    """
    m, nb_fn = tree.mask, tree.neighbours_fn
    if mode == 'bfs':
        _partial_bfs_implicit(start_nodes, z, m, nb_fn,
                               kernel, extra_args, upstream, multi_enabled)
    elif mode == 'dfs':
        _partial_dfs_implicit(start_nodes, z, m, nb_fn,
                               kernel, extra_args, upstream, multi_enabled)
    elif mode == 'pq':
        _partial_pq_implicit(start_nodes, z, m, nb_fn,
                              kernel, extra_args, upstream, multi_enabled, not min_heap)
    else:
        raise ValueError(f"mode must be 'bfs', 'dfs', or 'pq', got {mode!r}")
