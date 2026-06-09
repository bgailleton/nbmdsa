"""Subtree extraction from a steepest-descent tree.

Given a set of starting nodes and a direction, extracts the sub-forest of all
nodes reachable in that direction and returns a new parents array plus a mask.

Direction:
    upstream   (upstream=True)  — from starting nodes toward leaves (their descendants)
    downstream (upstream=False) — from starting nodes toward root   (their ancestors)

Returns: (new_parents: int64[N], mask: bool[N])
    new_parents — valid parents array of the same size N; nodes outside the
                  subtree satisfy new_parents[i] == i (isolated roots).
                  Subtree roots (whose original parent was outside the subtree)
                  also satisfy new_parents[i] == i.
    mask        — True for nodes that are part of the subtree.

The mask is needed because new_parents[i]==i is ambiguous between "subtree root"
and "not in subtree".

If a topo_order is supplied the extraction uses a single linear scan (O(N))
instead of a BFS from starts (also O(N) but with higher constant and CSR build).

──────────────────────────────────────────────────────────────────────────────
@nb.njit low-level (take raw arrays, composable inside other @njit functions):
    subtree_upstream_bfs    (parents, ptr, flat, starts)
    subtree_upstream_ordered(topo_order, parents, starts)
    subtree_downstream_bfs  (parents, starts)
    subtree_downstream_ordered(topo_order, parents, starts)

Python wrappers (take tree object):
    subtree(starts, tree, g=None, upstream=True, topo_order=None)
"""

import numpy as np
import numba as nb

from nbmdsa.structures.tree_sd import SteepTreeImpl, _build_parents
from nbmdsa.algorithms.tree_topo import _build_csr

# ── Internal helpers ──────────────────────────────────────────────────────────

@nb.njit
def _get_parents(t): return t.parents
@nb.njit
def _get_z(t):       return t.z

def _extract_parents(tree, g):
    if isinstance(tree, SteepTreeImpl):
        return _build_parents(_get_z(tree), g)
    return _get_parents(tree)


@nb.njit
def _build_new_parents(parents, mask):
    """Build subtree parents: in-mask nodes keep parent if also in mask, else self."""
    n = len(parents)
    out = np.empty(n, nb.int64)
    for i in range(n):
        if mask[i]:
            p = parents[i]
            out[i] = p if mask[p] else nb.int64(i)
        else:
            out[i] = nb.int64(i)
    return out

# ── @njit low-level functions ─────────────────────────────────────────────────

@nb.njit
def subtree_upstream_bfs(parents, ptr, flat, starts):
    """Upstream subtree via BFS: collect all descendants of starts."""
    n = len(parents)
    mask = np.zeros(n, nb.bool_)
    queue = np.empty(n, nb.int64)
    head = nb.int64(0)
    tail = nb.int64(0)
    for i in range(len(starts)):
        s = starts[i]
        if not mask[s]:
            mask[s] = True
            queue[tail] = s
            tail += nb.int64(1)
    while head < tail:
        node = queue[head]
        head += nb.int64(1)
        for j in range(ptr[node], ptr[node + nb.int64(1)]):
            child = flat[j]
            if not mask[child]:
                mask[child] = True
                queue[tail] = child
                tail += nb.int64(1)
    return _build_new_parents(parents, mask), mask


@nb.njit
def subtree_upstream_ordered(topo_order, parents, starts):
    """Upstream subtree via topo scan: a node is included if its parent is included."""
    n = len(parents)
    mask = np.zeros(n, nb.bool_)
    for i in range(len(starts)):
        mask[starts[i]] = True
    for i in range(len(topo_order)):
        node = topo_order[i]
        if not mask[node] and mask[parents[node]]:
            mask[node] = True
    return _build_new_parents(parents, mask), mask


@nb.njit
def subtree_downstream_bfs(parents, starts):
    """Downstream subtree: follow parent chains from each start to the root."""
    n = len(parents)
    mask = np.zeros(n, nb.bool_)
    for i in range(len(starts)):
        node = starts[i]
        while not mask[node]:
            mask[node] = True
            if parents[node] == node:
                break
            node = parents[node]
    return _build_new_parents(parents, mask), mask


@nb.njit
def subtree_downstream_ordered(topo_order, parents, starts):
    """Downstream subtree via reverse topo scan: propagate mask from starts toward root."""
    n = len(parents)
    mask = np.zeros(n, nb.bool_)
    for i in range(len(starts)):
        mask[starts[i]] = True
    for i in range(len(topo_order) - nb.int64(1), -nb.int64(1), -nb.int64(1)):
        node = topo_order[i]
        if mask[node] and parents[node] != node:
            mask[parents[node]] = True
    return _build_new_parents(parents, mask), mask

# ── Python wrapper ────────────────────────────────────────────────────────────

def subtree(starts, tree, g=None, upstream=True, topo_order=None):
    """Extract subtree from starts in the given direction.

    upstream=True  — descendants of starts (toward leaves)
    upstream=False — ancestors of starts (toward root)
    topo_order     — pre-computed int64[N] topo order; skips BFS/CSR build if supplied
    g              — required only for SteepTreeImpl

    Returns (new_parents: int64[N], mask: bool[N]).
    """
    parents = _extract_parents(tree, g)
    starts_arr = np.asarray(starts, dtype=np.int64)

    if topo_order is not None:
        if upstream:
            return subtree_upstream_ordered(topo_order, parents, starts_arr)
        else:
            return subtree_downstream_ordered(topo_order, parents, starts_arr)

    if upstream:
        ptr, flat = _build_csr(parents)
        return subtree_upstream_bfs(parents, ptr, flat, starts_arr)
    else:
        return subtree_downstream_bfs(parents, starts_arr)
