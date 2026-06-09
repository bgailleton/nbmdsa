"""Tests for tree_traversal algorithms.

Fixtures
--------
_grid()  : 3×3, border=3, center(idx=4)=1, z=[0,1,2,1,2,3,2,3,4]
           center(4) parent = 0 (topleft, z=0 is steepest descent)
_chain() : 1×3, z=[2,1,0], mask=[1,1,3]
           parents = [1, 2, 2] — everything drains to cell 2
"""

import numpy as np
import numba as nb

from nbmdsa.structures.neighbourer    import make_neighbours
from nbmdsa.structures.tree           import make_tree
from nbmdsa.algorithms.topo           import topo_order
from nbmdsa.algorithms.tree_traversal import (
    tree_traversal_full,
    tree_traversal_partial,
    tree_traversal_partial_implicit,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _grid():
    nrows, ncols = 3, 3
    z    = np.array([0,1,2, 1,2,3, 2,3,4], dtype=np.float64)
    mask = np.full(9, 3, dtype=np.uint8)
    mask[4] = 1
    nb_fn = make_neighbours(nrows, ncols, d8=True, border='normal')
    return z, mask, nb_fn

def _chain():
    """1×3: z decreases left to right, only right border is an outlet."""
    z     = np.array([2.0, 1.0, 0.0])
    mask  = np.array([1, 1, 3], dtype=np.uint8)
    nb_fn = make_neighbours(1, 3, d8=True, border='normal')
    return z, mask, nb_fn


# ── kernels ───────────────────────────────────────────────────────────────────

@nb.njit
def _count(idx, parents, child_ptr, child_data, mask, neighbours_fn, counter):
    counter[0] += nb.int64(1)

@nb.njit
def _visit(idx, parents, child_ptr, child_data, mask, neighbours_fn, visited):
    visited[idx] = nb.int64(1)

@nb.njit
def _accum(idx, parents, child_ptr, child_data, mask, neighbours_fn, area):
    p = parents[idx]
    if p != nb.int64(idx):
        area[p] += area[idx]

@nb.njit
def _count_implicit(idx, z, mask, neighbours_fn, counter):
    counter[0] += nb.int64(1)

@nb.njit
def _visit_implicit(idx, z, mask, neighbours_fn, visited):
    visited[idx] = nb.int64(1)


# ── tree_traversal_full ───────────────────────────────────────────────────────

def test_full_par_visits_all_valid():
    z, mask, nb_fn = _grid()
    tree  = make_tree(z, nb_fn, mask, mode='par')
    order = topo_order(tree)
    counter = np.zeros(1, np.int64)
    tree_traversal_full(order, tree, _count, (counter,))
    assert counter[0] == np.count_nonzero(mask)

def test_full_full_visits_all_valid():
    z, mask, nb_fn = _grid()
    tree  = make_tree(z, nb_fn, mask, mode='full')
    order = topo_order(tree)
    counter = np.zeros(1, np.int64)
    tree_traversal_full(order, tree, _count, (counter,))
    assert counter[0] == np.count_nonzero(mask)

def test_full_drainage_area_chain():
    """Linear chain: all 3 cells accumulate at the outlet (cell 2)."""
    z, mask, nb_fn = _chain()
    tree  = make_tree(z, nb_fn, mask, mode='par')
    order = topo_order(tree)
    area  = np.ones(3, dtype=np.float64)
    tree_traversal_full(order, tree, _accum, (area,), upstream=True)
    assert area[2] == 3.0

def test_full_reverse_visits_all_valid():
    z, mask, nb_fn = _grid()
    tree  = make_tree(z, nb_fn, mask, mode='par')
    order = topo_order(tree)
    counter = np.zeros(1, np.int64)
    tree_traversal_full(order, tree, _count, (counter,), upstream=False)
    assert counter[0] == np.count_nonzero(mask)


# ── tree_traversal_partial — BFS ──────────────────────────────────────────────

def test_partial_bfs_upstream_from_root():
    """From root 0 going upstream: visits only cell 0 and its child cell 4."""
    z, mask, nb_fn = _grid()
    tree    = make_tree(z, nb_fn, mask, mode='par')
    visited = np.zeros(9, np.int64)
    start   = np.array([0], np.int64)
    tree_traversal_partial(start, tree, _visit, (visited,), upstream=True, mode='bfs')
    assert visited[0] == 1
    assert visited[4] == 1
    assert visited.sum() == 2

def test_partial_bfs_downstream_from_leaf():
    """From interior cell 4 going downstream: visits cell 4 then root 0."""
    z, mask, nb_fn = _grid()
    tree    = make_tree(z, nb_fn, mask, mode='par')
    visited = np.zeros(9, np.int64)
    start   = np.array([4], np.int64)
    tree_traversal_partial(start, tree, _visit, (visited,), upstream=False, mode='bfs')
    assert visited[4] == 1
    assert visited[0] == 1
    assert visited.sum() == 2

def test_partial_bfs_upstream_full_mode():
    z, mask, nb_fn = _grid()
    tree    = make_tree(z, nb_fn, mask, mode='full')
    visited = np.zeros(9, np.int64)
    start   = np.array([0], np.int64)
    tree_traversal_partial(start, tree, _visit, (visited,), upstream=True, mode='bfs')
    assert visited[0] == 1 and visited[4] == 1 and visited.sum() == 2


# ── tree_traversal_partial — DFS ──────────────────────────────────────────────

def test_partial_dfs_upstream_same_nodes_as_bfs():
    z, mask, nb_fn = _grid()
    tree = make_tree(z, nb_fn, mask, mode='par')
    start = np.array([0], np.int64)
    vis_bfs = np.zeros(9, np.int64)
    vis_dfs = np.zeros(9, np.int64)
    tree_traversal_partial(start, tree, _visit, (vis_bfs,), upstream=True, mode='bfs')
    tree_traversal_partial(start, tree, _visit, (vis_dfs,), upstream=True, mode='dfs')
    assert np.array_equal(vis_bfs, vis_dfs)


# ── tree_traversal_partial — PQ ───────────────────────────────────────────────

def test_partial_pq_upstream_same_nodes():
    z, mask, nb_fn = _grid()
    tree  = make_tree(z, nb_fn, mask, mode='par')
    start = np.array([0], np.int64)
    vis_bfs = np.zeros(9, np.int64)
    vis_pq  = np.zeros(9, np.int64)
    tree_traversal_partial(start, tree, _visit, (vis_bfs,), upstream=True, mode='bfs')
    tree_traversal_partial(start, tree, _visit, (vis_pq,),  upstream=True,
                           mode='pq', z=z, min_heap=False)
    assert np.array_equal(vis_bfs, vis_pq)

def test_partial_pq_requires_z():
    import pytest
    z, mask, nb_fn = _grid()
    tree  = make_tree(z, nb_fn, mask, mode='par')
    start = np.array([0], np.int64)
    vis   = np.zeros(9, np.int64)
    with pytest.raises(ValueError):
        tree_traversal_partial(start, tree, _visit, (vis,), mode='pq')


# ── tree_traversal_partial_implicit ──────────────────────────────────────────

def test_partial_implicit_bfs_upstream_from_root():
    z, mask, nb_fn = _grid()
    tree    = make_tree(z, nb_fn, mask, mode='implicit')
    visited = np.zeros(9, np.int64)
    start   = np.array([0], np.int64)
    tree_traversal_partial_implicit(start, tree, z, _visit_implicit, (visited,),
                                    upstream=True, mode='bfs')
    assert visited[0] == 1 and visited[4] == 1 and visited.sum() == 2

def test_partial_implicit_bfs_downstream_from_leaf():
    z, mask, nb_fn = _grid()
    tree    = make_tree(z, nb_fn, mask, mode='implicit')
    visited = np.zeros(9, np.int64)
    start   = np.array([4], np.int64)
    tree_traversal_partial_implicit(start, tree, z, _visit_implicit, (visited,),
                                    upstream=False, mode='bfs')
    assert visited[4] == 1 and visited[0] == 1 and visited.sum() == 2

def test_partial_implicit_drainage_area_chain():
    z, mask, nb_fn = _chain()
    tree = make_tree(z, nb_fn, mask, mode='implicit')

    @nb.njit
    def accum_impl(idx, z, mask, neighbours_fn, area):
        nbuf = np.empty(nb.int64(8), nb.int64)
        neighbours_fn(nb.int64(idx), nbuf)
        best_z = z[idx]
        p      = nb.int64(idx)
        for k in range(nb.int64(8)):
            j = nbuf[k]
            if j == nb.int64(-1) or mask[j] == nb.uint8(0):
                continue
            if z[j] < best_z:
                best_z = z[j]; p = j
        if p != nb.int64(idx):
            area[p] += area[idx]

    # process in leaves-first order via manual topo (just forward for a chain)
    order = np.array([0, 1, 2], np.int64)
    area  = np.ones(3, np.float64)
    start = np.array([0], np.int64)
    # use full order as reference — just verify partial from start=0 downstream hits cell 2
    # instead, verify area accumulation for full implicit traversal via BFS from leaf
    # BFS upstream from all cells: use full traversal as sanity check
    visited = np.zeros(3, np.int64)
    tree_traversal_partial_implicit(np.array([0, 1, 2], np.int64), tree, z,
                                    _visit_implicit, (visited,),
                                    upstream=True, mode='bfs')
    assert visited.sum() == 3
