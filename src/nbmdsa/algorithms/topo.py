"""Topological ordering of trees.

Works on par and full modes only — not implicit (no stored parents).

Full mode dispatches to a CSR-based BFS (one pass, no child counting).
Par  mode uses Kahn's algorithm (two passes over parents array).

topo_order(tree, reverse=False)
    Tree (grid-indexed). Returns int64 array of grid indices.
    Default: leaves first (upstream). reverse=True: roots first (downstream).

topo_order_packed(tree, reverse=False)
    PackedTree. Returns int64 array of grid indices.
"""

import numpy as np
import numba as nb


# ── par mode: Kahn's on parents array ────────────────────────────────────────
# Natural output: leaves first.

@nb.njit
def _topo_grid_kahn(parents, mask, order):
    n           = nb.int64(len(parents))
    child_count = np.zeros(n, nb.int64)
    for i in range(n):
        if mask[i] == nb.uint8(0):
            continue
        p = parents[i]
        if p != nb.int64(i):
            child_count[p] += nb.int64(1)
    q  = np.empty(n, nb.int64)
    qh = qt = nb.int64(0)
    for i in range(n):
        if mask[i] != nb.uint8(0) and child_count[i] == nb.int64(0):
            q[qt] = nb.int64(i)
            qt   += nb.int64(1)
    k = nb.int64(0)
    while qh < qt:
        c        = q[qh]; qh += nb.int64(1)
        order[k] = c;     k  += nb.int64(1)
        p = parents[c]
        if p != c:
            child_count[p] -= nb.int64(1)
            if child_count[p] == nb.int64(0):
                q[qt] = p; qt += nb.int64(1)
    return k


@nb.njit
def _topo_local_kahn(parents, order):
    m           = nb.int64(len(parents))
    child_count = np.zeros(m, nb.int64)
    for li in range(m):
        p = parents[li]
        if p != li:
            child_count[p] += nb.int64(1)
    q  = np.empty(m, nb.int64)
    qh = qt = nb.int64(0)
    for li in range(m):
        if child_count[li] == nb.int64(0):
            q[qt] = nb.int64(li)
            qt   += nb.int64(1)
    k = nb.int64(0)
    while qh < qt:
        c        = q[qh]; qh += nb.int64(1)
        order[k] = c;     k  += nb.int64(1)
        p = parents[c]
        if p != c:
            child_count[p] -= nb.int64(1)
            if child_count[p] == nb.int64(0):
                q[qt] = p; qt += nb.int64(1)
    return k


# ── full mode: BFS from roots via CSR ────────────────────────────────────────
# Natural output: roots first (one pass, no counting).

@nb.njit
def _topo_grid_csr(parents, mask, child_ptr, child_data, order):
    n  = nb.int64(len(parents))
    q  = np.empty(n, nb.int64)
    qh = qt = nb.int64(0)
    for i in range(n):
        if mask[i] != nb.uint8(0) and parents[i] == nb.int64(i):
            q[qt] = nb.int64(i)
            qt   += nb.int64(1)
    k = nb.int64(0)
    while qh < qt:
        c        = q[qh]; qh += nb.int64(1)
        order[k] = c;     k  += nb.int64(1)
        for ci in range(child_ptr[c], child_ptr[c + nb.int64(1)]):
            q[qt] = child_data[ci]; qt += nb.int64(1)
    return k


@nb.njit
def _topo_local_csr(parents, child_ptr, child_data, order):
    m  = nb.int64(len(parents))
    q  = np.empty(m, nb.int64)
    qh = qt = nb.int64(0)
    for li in range(m):
        if parents[li] == li:
            q[qt] = nb.int64(li)
            qt   += nb.int64(1)
    k = nb.int64(0)
    while qh < qt:
        c        = q[qh]; qh += nb.int64(1)
        order[k] = c;     k  += nb.int64(1)
        for ci in range(child_ptr[c], child_ptr[c + nb.int64(1)]):
            q[qt] = child_data[ci]; qt += nb.int64(1)
    return k


# ── public API ────────────────────────────────────────────────────────────────

def topo_order(tree, reverse=False):
    """Topological order for Tree (par or full). Returns grid indices.

    reverse=False — leaves first (upstream, default)
    reverse=True  — roots first (downstream)
    """
    order = np.empty(len(tree.parents), np.int64)
    if len(tree.child_ptr) > 0:                         # full: CSR BFS, roots-first natural
        count  = _topo_grid_csr(tree.parents, tree.mask,
                                tree.child_ptr, tree.child_data, order)
        result = order[:count]
        return result if reverse else result[::-1].copy()
    else:                                                # par: Kahn's, leaves-first natural
        count  = _topo_grid_kahn(tree.parents, tree.mask, order)
        result = order[:count]
        return result[::-1].copy() if reverse else result


def topo_order_packed(tree, reverse=False):
    """Topological order for PackedTree (par or full). Returns grid indices.

    reverse=False — leaves first (upstream, default)
    reverse=True  — roots first (downstream)
    """
    local_order = np.empty(len(tree.parents), np.int64)
    if len(tree.child_ptr) > 0:                         # full: CSR BFS, roots-first natural
        count  = _topo_local_csr(tree.parents,
                                 tree.child_ptr, tree.child_data, local_order)
        result = tree.nodes[local_order[:count]]
        return result if reverse else result[::-1].copy()
    else:                                                # par: Kahn's, leaves-first natural
        count  = _topo_local_kahn(tree.parents, local_order)
        result = tree.nodes[local_order[:count]]
        return result[::-1].copy() if reverse else result
