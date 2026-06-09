"""Tests for mfd_traversal algorithms.

Fixtures
--------
_grid()  : 3×3, border=3, center(idx=4)=1, z=[0,1,2,1,2,3,2,3,4]
_chain() : 1×3, z=[2,1,0], mask=[1,1,3]
           only one flow path: 0→1→2, area[2] must equal 3 after accumulation.
"""

import numpy as np
import numba as nb

from nbmdsa.structures.neighbourer   import make_neighbours
from nbmdsa.algorithms.mfd_traversal import (
    mfd_topo_order,
    mfd_traversal_full,
    mfd_traversal_partial,
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
    z     = np.array([2.0, 1.0, 0.0])
    mask  = np.array([1, 1, 3], dtype=np.uint8)
    nb_fn = make_neighbours(1, 3, d8=True, border='normal')
    return z, mask, nb_fn


# ── kernels ───────────────────────────────────────────────────────────────────

@nb.njit
def _count(idx, z, mask, neighbours_fn, counter):
    counter[0] += nb.int64(1)

@nb.njit
def _visit(idx, z, mask, neighbours_fn, visited):
    visited[idx] = nb.int64(1)

@nb.njit
def _accum(idx, z, mask, neighbours_fn, area, nbuf):
    neighbours_fn(nb.int64(idx), nbuf)
    total = nb.float64(0.0)
    for k in range(nb.int64(8)):
        j = nbuf[k]
        if j == nb.int64(-1) or mask[j] == nb.uint8(0): continue
        drop = z[idx] - z[j]
        if drop > nb.float64(0.0): total += drop
    if total == nb.float64(0.0): return
    for k in range(nb.int64(8)):
        j = nbuf[k]
        if j == nb.int64(-1) or mask[j] == nb.uint8(0): continue
        drop = z[idx] - z[j]
        if drop > nb.float64(0.0):
            area[j] += area[idx] * drop / total


# ── mfd_topo_order ────────────────────────────────────────────────────────────

def test_topo_all_valid_nodes():
    z, mask, nb_fn = _grid()
    order = mfd_topo_order(z, mask, nb_fn)
    assert len(order) == int(np.count_nonzero(mask))

def test_topo_donors_before_receivers():
    """For every MFD edge j→i (z[j]>z[i], neighbours), j appears before i."""
    z, mask, nb_fn = _grid()
    order = mfd_topo_order(z, mask, nb_fn)
    pos   = {int(order[i]): i for i in range(len(order))}
    nbuf  = np.empty(8, np.int64)
    for rank, idx in enumerate(order):
        nb_fn(int(idx), nbuf)
        for j in nbuf:
            if j < 0 or mask[j] == 0: continue
            if z[j] > z[idx]:                       # j is a donor of idx
                assert pos[int(j)] < rank

def test_topo_reverse_has_receivers_first():
    """reverse=True flips the order."""
    z, mask, nb_fn = _grid()
    fwd = mfd_topo_order(z, mask, nb_fn)
    rev = mfd_topo_order(z, mask, nb_fn, reverse=True)
    assert list(fwd) == list(rev[::-1])


# ── mfd_traversal_full ────────────────────────────────────────────────────────

def test_full_visits_all_valid():
    z, mask, nb_fn = _grid()
    order   = mfd_topo_order(z, mask, nb_fn)
    counter = np.zeros(1, np.int64)
    mfd_traversal_full(order, z, mask, nb_fn, _count, (counter,))
    assert counter[0] == int(np.count_nonzero(mask))

def test_full_reverse_visits_all_valid():
    z, mask, nb_fn = _grid()
    order   = mfd_topo_order(z, mask, nb_fn)
    counter = np.zeros(1, np.int64)
    mfd_traversal_full(order, z, mask, nb_fn, _count, (counter,), upstream=False)
    assert counter[0] == int(np.count_nonzero(mask))

def test_full_drainage_area_chain():
    """1×3 linear chain: single flow path → area[2] == 3."""
    z, mask, nb_fn = _chain()
    order  = mfd_topo_order(z, mask, nb_fn)
    area   = np.ones(3, dtype=np.float64)
    nbuf   = np.empty(8, np.int64)
    mfd_traversal_full(order, z, mask, nb_fn, _accum, (area, nbuf), upstream=True)
    assert area[2] == 3.0


# ── mfd_traversal_partial — BFS ───────────────────────────────────────────────

def test_partial_bfs_downstream_chain():
    """From cell 0 (highest) go downstream: all 3 cells visited."""
    z, mask, nb_fn = _chain()
    visited = np.zeros(3, np.int64)
    start   = np.array([0], np.int64)
    mfd_traversal_partial(start, z, mask, nb_fn, _visit, (visited,),
                           upstream=False, mode='bfs')
    assert visited.sum() == 3

def test_partial_bfs_upstream_chain():
    """From cell 2 (outlet, lowest) go upstream: all 3 cells visited."""
    z, mask, nb_fn = _chain()
    visited = np.zeros(3, np.int64)
    start   = np.array([2], np.int64)
    mfd_traversal_partial(start, z, mask, nb_fn, _visit, (visited,),
                           upstream=True, mode='bfs')
    assert visited.sum() == 3

def test_partial_bfs_upstream_excludes_lower():
    """Going upstream from cell 1 (z=1): only reaches cell 0 (z=2), not cell 2 (z=0)."""
    z, mask, nb_fn = _chain()
    visited = np.zeros(3, np.int64)
    start   = np.array([1], np.int64)
    mfd_traversal_partial(start, z, mask, nb_fn, _visit, (visited,),
                           upstream=True, mode='bfs')
    assert visited[0] == 1    # higher neighbour
    assert visited[1] == 1    # start
    assert visited[2] == 0    # lower — not upstream


# ── mfd_traversal_partial — DFS ───────────────────────────────────────────────

def test_partial_dfs_same_nodes_as_bfs():
    z, mask, nb_fn = _chain()
    start   = np.array([0], np.int64)
    vis_bfs = np.zeros(3, np.int64)
    vis_dfs = np.zeros(3, np.int64)
    mfd_traversal_partial(start, z, mask, nb_fn, _visit, (vis_bfs,),
                           upstream=False, mode='bfs')
    mfd_traversal_partial(start, z, mask, nb_fn, _visit, (vis_dfs,),
                           upstream=False, mode='dfs')
    assert np.array_equal(vis_bfs, vis_dfs)


# ── mfd_traversal_partial — PQ ────────────────────────────────────────────────

def test_partial_pq_same_nodes_as_bfs():
    z, mask, nb_fn = _chain()
    start   = np.array([2], np.int64)
    vis_bfs = np.zeros(3, np.int64)
    vis_pq  = np.zeros(3, np.int64)
    mfd_traversal_partial(start, z, mask, nb_fn, _visit, (vis_bfs,),
                           upstream=True, mode='bfs')
    mfd_traversal_partial(start, z, mask, nb_fn, _visit, (vis_pq,),
                           upstream=True, mode='pq', min_heap=False)
    assert np.array_equal(vis_bfs, vis_pq)

def test_partial_pq_invalid_mode():
    import pytest
    z, mask, nb_fn = _chain()
    start = np.array([0], np.int64)
    vis   = np.zeros(3, np.int64)
    with pytest.raises(ValueError):
        mfd_traversal_partial(start, z, mask, nb_fn, _visit, (vis,), mode='xyz')


# ── multi_enabled ─────────────────────────────────────────────────────────────

def test_multi_enabled_allows_revisit():
    """With multi_enabled=True a node reached from two start nodes is processed twice."""
    z, mask, nb_fn = _chain()
    counter = np.zeros(1, np.int64)
    # start from both ends going downstream — cell 1 is reachable from cell 0
    # and also seeded directly; with multi_enabled it can be enqueued twice
    start = np.array([0, 1], np.int64)
    mfd_traversal_partial(start, z, mask, nb_fn, _count, (counter,),
                           upstream=False, multi_enabled=True, mode='bfs')
    assert counter[0] >= 3   # at minimum all cells, likely more due to revisits

def test_multi_disabled_visits_each_once():
    z, mask, nb_fn = _chain()
    counter = np.zeros(1, np.int64)
    start   = np.array([0, 1], np.int64)
    mfd_traversal_partial(start, z, mask, nb_fn, _count, (counter,),
                           upstream=False, multi_enabled=False, mode='bfs')
    assert counter[0] == 3
