"""Tree traversal with user-supplied kernels.

Kernel signature (must be @nb.njit):
    kernel(tree, g, node: int64, extra: tuple) -> None
    tree  — steepest-descent tree structref (read neighbours, z, …)
    g     — grid neighbourer structref
    node  — current node index
    extra — user tuple of extra parameters, unpack inside kernel

Direction:
    upstream   (upstream=True)  — from roots toward leaves (children direction)
    downstream (upstream=False) — from leaves toward root  (parent direction)

──────────────────────────────────────────────────────────────────────────────
Full traversal  (all N nodes)
──────────────────────────────────────────────────────────────────────────────
@nb.njit-able low-level (take parents + CSR, tree, g):
    traverse_full_bfs(parents, tree, g, kernel, extra, upstream)
    traverse_full_dfs(parents, tree, g, kernel, extra, upstream)
    traverse_full_ordered(topo_order, tree, g, kernel, extra, upstream)

Python wrappers (take tree object, g optional for SteepTreeImpl):
    full_bfs(tree, g, kernel, extra, upstream=True)
    full_dfs(tree, g, kernel, extra, upstream=True)
    full_ordered(topo_order, tree, g, kernel, extra, upstream=True)

──────────────────────────────────────────────────────────────────────────────
Partial traversal  (from a set of starting nodes)
──────────────────────────────────────────────────────────────────────────────
revisit=False — kernel runs at most once per node; converging paths stop at
                already-visited nodes.
revisit=True  — kernel runs every time a node is reached, even if visited
                before.  Meaningful for downstream traversal where multiple
                paths converge; for upstream, subtrees of overlapping starts
                are re-processed.

@nb.njit-able low-level (take parents + CSR):
    traverse_partial_bfs(parents, ptr, flat, starts, tree, g, kernel, extra, upstream, revisit)
    traverse_partial_dfs(parents, ptr, flat, starts, tree, g, kernel, extra, upstream, revisit)
    traverse_partial_pq (parents, ptr, flat, starts, z, tree, g, kernel, extra, upstream, revisit)
        z: float64[N] priority scores — nodes processed lowest score first.

Python wrappers:
    partial_bfs(starts, tree, g, kernel, extra, upstream=True, revisit=False)
    partial_dfs(starts, tree, g, kernel, extra, upstream=True, revisit=False)
    partial_pq (starts, z, tree, g, kernel, extra, upstream=True, revisit=False)
"""

import numpy as np
import numba as nb
from numba.experimental import structref

from nbmdsa.structures.tree_sd import SteepTreeImpl, _build_parents
from nbmdsa.structures.heap_st import MinHeapStF64
from nbmdsa.algorithms.tree_topo import (
    topo_order_from_parents_bfs,
    topo_order_from_parents_dfs,
    _build_csr,
)

# ── Internal helpers ──────────────────────────────────────────────────────────

@nb.njit
def _get_parents(t): return t.parents
@nb.njit
def _get_z(t):       return t.z

def _extract_parents(tree, g):
    if isinstance(tree, SteepTreeImpl):
        return _build_parents(_get_z(tree), g)
    return _get_parents(tree)

# ── Full traversal — @njit cores ──────────────────────────────────────────────

@nb.njit
def traverse_full_ordered(topo_order, tree, g, kernel, extra, upstream):
    """Apply kernel to all nodes in the given topological order.

    upstream=True  — iterate topo_order as-is (roots → leaves).
    upstream=False — iterate in reverse (leaves → root).
    """
    n = len(topo_order)
    if upstream:
        for i in range(n):
            kernel(tree, g, topo_order[i], extra)
    else:
        for i in range(n - 1, -1, -1):
            kernel(tree, g, topo_order[i], extra)


@nb.njit
def traverse_full_bfs(parents, tree, g, kernel, extra, upstream):
    """Full traversal; computes BFS topo order internally."""
    traverse_full_ordered(
        topo_order_from_parents_bfs(parents), tree, g, kernel, extra, upstream
    )


@nb.njit
def traverse_full_dfs(parents, tree, g, kernel, extra, upstream):
    """Full traversal; computes DFS topo order internally."""
    traverse_full_ordered(
        topo_order_from_parents_dfs(parents), tree, g, kernel, extra, upstream
    )

# ── Partial traversal — @njit cores ──────────────────────────────────────────

@nb.njit
def traverse_partial_bfs(parents, ptr, flat, starts, tree, g, kernel, extra,
                          upstream, revisit):
    """Partial BFS from starts.  ptr/flat: CSR children structure."""
    n = len(parents)
    visited = np.zeros(n, nb.bool_)
    queue = np.empty(n, nb.int64)
    head = nb.int64(0)
    tail = nb.int64(0)

    for i in range(len(starts)):
        s = starts[i]
        if revisit or not visited[s]:
            visited[s] = True
            queue[tail] = s
            tail += nb.int64(1)

    while head < tail:
        node = queue[head]
        head += nb.int64(1)
        kernel(tree, g, node, extra)

        if upstream:
            for j in range(ptr[node], ptr[node + nb.int64(1)]):
                child = flat[j]
                if revisit or not visited[child]:
                    visited[child] = True
                    queue[tail] = child
                    tail += nb.int64(1)
        else:
            p = parents[node]
            if p != node:
                if revisit or not visited[p]:
                    visited[p] = True
                    queue[tail] = p
                    tail += nb.int64(1)


@nb.njit
def traverse_partial_dfs(parents, ptr, flat, starts, tree, g, kernel, extra,
                          upstream, revisit):
    """Partial DFS from starts.  ptr/flat: CSR children structure."""
    n = len(parents)
    visited = np.zeros(n, nb.bool_)
    stack = np.empty(n, nb.int64)
    top = nb.int64(0)

    for i in range(len(starts) - 1, -1, -1):
        s = starts[i]
        if revisit or not visited[s]:
            visited[s] = True
            stack[top] = s
            top += nb.int64(1)

    while top > nb.int64(0):
        top -= nb.int64(1)
        node = stack[top]
        kernel(tree, g, node, extra)

        if upstream:
            for j in range(ptr[node + nb.int64(1)] - nb.int64(1),
                           ptr[node] - nb.int64(1), -nb.int64(1)):
                child = flat[j]
                if revisit or not visited[child]:
                    visited[child] = True
                    stack[top] = child
                    top += nb.int64(1)
        else:
            p = parents[node]
            if p != node:
                if revisit or not visited[p]:
                    visited[p] = True
                    stack[top] = p
                    top += nb.int64(1)


@nb.njit
def traverse_partial_pq(parents, ptr, flat, starts, z, tree, g, kernel, extra,
                         upstream, revisit):
    """Partial traversal from starts, nodes processed in ascending z order.

    Uses an internal min-heap seeded with (start, z[start]).
    Nodes are expanded upstream (children) or downstream (parent) after
    kernel is applied.
    """
    n = len(parents)
    visited = np.zeros(n, nb.bool_)

    heap = structref.new(MinHeapStF64)
    heap.indices = np.empty(n, nb.int64)
    heap.scores  = np.empty(n, nb.float64)
    heap.size     = nb.int64(0)
    heap.max_size = nb.int64(n)

    for i in range(len(starts)):
        s = starts[i]
        if revisit or not visited[s]:
            visited[s] = True
            heap.emplace(s, z[s])

    while not heap.is_empty():
        node, _ = heap.top()
        heap.pop()
        kernel(tree, g, node, extra)

        if upstream:
            for j in range(ptr[node], ptr[node + nb.int64(1)]):
                child = flat[j]
                if revisit or not visited[child]:
                    visited[child] = True
                    heap.emplace(child, z[child])
        else:
            p = parents[node]
            if p != node:
                if revisit or not visited[p]:
                    visited[p] = True
                    heap.emplace(p, z[p])

# ── Python wrappers ───────────────────────────────────────────────────────────

def full_bfs(tree, g, kernel, extra, upstream=True):
    """Full BFS traversal.  g required only for SteepTreeImpl."""
    parents = _extract_parents(tree, g)
    traverse_full_bfs(parents, tree, g, kernel, extra, upstream)


def full_dfs(tree, g, kernel, extra, upstream=True):
    """Full DFS traversal.  g required only for SteepTreeImpl."""
    parents = _extract_parents(tree, g)
    traverse_full_dfs(parents, tree, g, kernel, extra, upstream)


def full_ordered(topo_order, tree, g, kernel, extra, upstream=True):
    """Full traversal using a pre-computed topo order.  g required only for SteepTreeImpl."""
    traverse_full_ordered(topo_order, tree, g, kernel, extra, upstream)


def _prep_partial(tree, g):
    parents = _extract_parents(tree, g)
    ptr, flat = _build_csr(parents)
    return parents, ptr, flat


def partial_bfs(starts, tree, g, kernel, extra, upstream=True, revisit=False):
    """Partial BFS from starts.  g required only for SteepTreeImpl."""
    parents, ptr, flat = _prep_partial(tree, g)
    traverse_partial_bfs(
        parents, ptr, flat,
        np.asarray(starts, dtype=np.int64),
        tree, g, kernel, extra, upstream, revisit,
    )


def partial_dfs(starts, tree, g, kernel, extra, upstream=True, revisit=False):
    """Partial DFS from starts.  g required only for SteepTreeImpl."""
    parents, ptr, flat = _prep_partial(tree, g)
    traverse_partial_dfs(
        parents, ptr, flat,
        np.asarray(starts, dtype=np.int64),
        tree, g, kernel, extra, upstream, revisit,
    )


def partial_pq(starts, z, tree, g, kernel, extra, upstream=True, revisit=False):
    """Partial PQ traversal from starts, ascending z priority.  g required only for SteepTreeImpl."""
    parents, ptr, flat = _prep_partial(tree, g)
    traverse_partial_pq(
        parents, ptr, flat,
        np.asarray(starts, dtype=np.int64),
        np.asarray(z, dtype=np.float64),
        tree, g, kernel, extra, upstream, revisit,
    )
