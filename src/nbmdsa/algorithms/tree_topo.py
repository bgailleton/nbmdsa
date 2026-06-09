"""Topological ordering of steepest-descent trees.

Produces an array of node indices ordered from roots to farthest leaves
(root-first topological order).  Two traversal strategies:

  BFS  — level-order; sequential writes, best cache behaviour
  DFS  — pre-order depth-first; groups subtrees contiguously

Both strategies build a temporary CSR children representation from the
parents array and then traverse in O(N).

Low-level (parents array, usable in @njit):
    topo_order_from_parents_bfs(parents: int64[::1]) -> int64[::1]
    topo_order_from_parents_dfs(parents: int64[::1]) -> int64[::1]

High-level (tree object, Python only; g ignored for Par/Full):
    topo_order_bfs(tree, g=None) -> int64[::1]
    topo_order_dfs(tree, g=None) -> int64[::1]
"""

import numpy as np
import numba as nb

from nbmdsa.structures.tree_sd import (
    SteepTreeImpl, SteepTreePar, SteepTreeFull,
    _build_parents,
)

# ── Internal helpers to extract fields via @njit ──────────────────────────────

@nb.njit
def _get_z(t):       return t.z
@nb.njit
def _get_parents(t): return t.parents

# ── CSR build helper shared by BFS and DFS ────────────────────────────────────

@nb.njit
def _build_csr(parents):
    n = len(parents)
    counts = np.zeros(n, nb.int64)
    for i in range(n):
        p = parents[i]
        if p != i:
            counts[p] += nb.int64(1)
    ptr = np.zeros(n + nb.int64(1), nb.int64)
    for i in range(n):
        ptr[i + 1] = ptr[i] + counts[i]
    flat = np.empty(ptr[n], nb.int64)
    pos = ptr[:n].copy()
    for i in range(n):
        p = parents[i]
        if p != i:
            flat[pos[p]] = nb.int64(i)
            pos[p] += nb.int64(1)
    return ptr, flat

# ── Core @njit functions ──────────────────────────────────────────────────────

@nb.njit
def topo_order_from_parents_bfs(parents):
    """Root-to-leaf topological order via BFS (level-order)."""
    n = len(parents)
    ptr, flat = _build_csr(parents)
    order = np.empty(n, nb.int64)
    queue = np.empty(n, nb.int64)
    head = nb.int64(0)
    tail = nb.int64(0)
    for i in range(n):
        if parents[i] == i:
            queue[tail] = nb.int64(i)
            tail += nb.int64(1)
    k = nb.int64(0)
    while head < tail:
        node = queue[head]
        head += nb.int64(1)
        order[k] = node
        k += nb.int64(1)
        for j in range(ptr[node], ptr[node + nb.int64(1)]):
            queue[tail] = flat[j]
            tail += nb.int64(1)
    return order


@nb.njit
def topo_order_from_parents_dfs(parents):
    """Root-to-leaf topological order via pre-order DFS (subtrees contiguous)."""
    n = len(parents)
    ptr, flat = _build_csr(parents)
    order = np.empty(n, nb.int64)
    stack = np.empty(n, nb.int64)
    top = nb.int64(0)
    k = nb.int64(0)
    # collect roots, push in reverse so smallest-index root is processed first
    roots = np.empty(n, nb.int64)
    nroots = nb.int64(0)
    for i in range(n):
        if parents[i] == i:
            roots[nroots] = nb.int64(i)
            nroots += nb.int64(1)
    for i in range(nroots - nb.int64(1), -nb.int64(1), -nb.int64(1)):
        stack[top] = roots[i]
        top += nb.int64(1)
    while top > nb.int64(0):
        top -= nb.int64(1)
        node = stack[top]
        order[k] = node
        k += nb.int64(1)
        # push children in reverse order so left child is processed first
        for j in range(ptr[node + nb.int64(1)] - nb.int64(1),
                       ptr[node] - nb.int64(1), -nb.int64(1)):
            stack[top] = flat[j]
            top += nb.int64(1)
    return order

# ── Source and root detection ─────────────────────────────────────────────────
# Sources: nodes with no children (child count == 0 from the parents array).
# Roots:   nodes where parents[i] == i.  An ignore_mask (bool[N]) can mark
#          expected/boundary roots so only unexpected internal roots are returned.

@nb.njit
def find_sources_from_parents(parents):
    """Return indices of all source nodes (no children) from a parents array."""
    n = len(parents)
    counts = np.zeros(n, nb.int64)
    for i in range(n):
        p = parents[i]
        if p != i:
            counts[p] += nb.int64(1)
    nsrc = nb.int64(0)
    for i in range(n):
        if counts[i] == nb.int64(0):
            nsrc += nb.int64(1)
    out = np.empty(nsrc, nb.int64)
    k = nb.int64(0)
    for i in range(n):
        if counts[i] == nb.int64(0):
            out[k] = nb.int64(i)
            k += nb.int64(1)
    return out


@nb.njit
def find_roots_from_parents(parents):
    """Return indices of all root nodes (parents[i] == i) from a parents array."""
    n = len(parents)
    nroots = nb.int64(0)
    for i in range(n):
        if parents[i] == i:
            nroots += nb.int64(1)
    out = np.empty(nroots, nb.int64)
    k = nb.int64(0)
    for i in range(n):
        if parents[i] == i:
            out[k] = nb.int64(i)
            k += nb.int64(1)
    return out


@nb.njit
def find_roots_from_parents_masked(parents, ignore_mask):
    """Return root indices excluding nodes where ignore_mask[i] is True.

    ignore_mask marks nodes that are expected to be roots (e.g. boundary outlets).
    Only roots not covered by the mask — internal/unexpected ones — are returned.
    """
    n = len(parents)
    nroots = nb.int64(0)
    for i in range(n):
        if parents[i] == i and not ignore_mask[i]:
            nroots += nb.int64(1)
    out = np.empty(nroots, nb.int64)
    k = nb.int64(0)
    for i in range(n):
        if parents[i] == i and not ignore_mask[i]:
            out[k] = nb.int64(i)
            k += nb.int64(1)
    return out


# ── Python-level wrappers (g is optional, ignored for Par/Full) ───────────────

def topo_order_bfs(tree, g=None) -> np.ndarray:
    """Root-to-leaf BFS topological order.  g required only for SteepTreeImpl."""
    if isinstance(tree, SteepTreeImpl):
        parents = _build_parents(_get_z(tree), g)
    else:
        parents = _get_parents(tree)
    return topo_order_from_parents_bfs(parents)


def topo_order_dfs(tree, g=None) -> np.ndarray:
    """Root-to-leaf DFS (pre-order) topological order.  g required only for SteepTreeImpl."""
    if isinstance(tree, SteepTreeImpl):
        parents = _build_parents(_get_z(tree), g)
    else:
        parents = _get_parents(tree)
    return topo_order_from_parents_dfs(parents)


def find_sources(tree, g=None) -> np.ndarray:
    """Return indices of source nodes (no children).  g required only for SteepTreeImpl."""
    if isinstance(tree, SteepTreeImpl):
        parents = _build_parents(_get_z(tree), g)
    else:
        parents = _get_parents(tree)
    return find_sources_from_parents(parents)


# ── Subtree labelling ─────────────────────────────────────────────────────────
# Propagates the label stored at each root node down to all its descendants.
# With a pre-computed topo order the inner loop is a single linear scan.

@nb.njit
def label_from_roots_from_parents_ordered(topo_order, parents, labels):
    """Propagate root labels using a pre-computed topological order.  Returns a copy."""
    out = labels.copy()
    for i in range(len(topo_order)):
        node = topo_order[i]
        if parents[node] != node:
            out[node] = out[parents[node]]
    return out


@nb.njit
def label_from_roots_from_parents(parents, labels):
    """Propagate root labels to all descendants.  Computes topo order internally."""
    return label_from_roots_from_parents_ordered(
        topo_order_from_parents_bfs(parents), parents, labels
    )


def label_from_roots(tree, labels, g=None, topo_order=None):
    """Propagate root labels to all descendants.  g required only for SteepTreeImpl.

    Pass topo_order (int64[N]) to skip recomputing it.
    """
    if isinstance(tree, SteepTreeImpl):
        parents = _build_parents(_get_z(tree), g)
    else:
        parents = _get_parents(tree)
    lbl = np.asarray(labels)
    if topo_order is not None:
        return label_from_roots_from_parents_ordered(topo_order, parents, lbl)
    return label_from_roots_from_parents(parents, lbl)


def find_roots(tree, g=None, ignore_mask=None) -> np.ndarray:
    """Return indices of root nodes (parents[i] == i).

    ignore_mask: optional bool array of length N.  Nodes where ignore_mask[i]
    is True are excluded from the result — use it to mark known boundary roots
    so only unexpected internal roots are returned.
    g is required only for SteepTreeImpl.
    """
    if isinstance(tree, SteepTreeImpl):
        parents = _build_parents(_get_z(tree), g)
    else:
        parents = _get_parents(tree)
    if ignore_mask is None:
        return find_roots_from_parents(parents)
    return find_roots_from_parents_masked(
        parents, np.asarray(ignore_mask, dtype=np.bool_)
    )
