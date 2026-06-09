"""Tests for topological ordering.

Two code paths:
  par  mode → Kahn's algorithm   (leaves-first natural)
  full mode → CSR BFS            (roots-first natural, reversed for default)

Property checked: in leaves-first order every node appears before its parent.
Argsort check:    monotonic 1×5 chain → topo order == argsort(z)[::-1].
"""

import numpy as np
import numba as nb

from nbmdsa.structures.neighbourer import make_neighbours
from nbmdsa.structures.tree        import make_tree
from nbmdsa.structures.packed_tree import make_packed_tree
from nbmdsa.algorithms.topo        import topo_order, topo_order_packed


# ── helpers ───────────────────────────────────────────────────────────────────

def _check_leaves_first(order, tree):
    """Every node must appear before its parent (roots excluded)."""
    pos = {int(n): k for k, n in enumerate(order)}
    for node in order:
        p = int(tree.parent(int(node)))
        if p != int(node):
            assert pos[p] > pos[int(node)], \
                f"node {node} at pos {pos[int(node)]} but parent {p} at pos {pos[p]}"

def _check_roots_first(order, tree):
    """Every node must appear after its parent (roots excluded)."""
    pos = {int(n): k for k, n in enumerate(order)}
    for node in order:
        p = int(tree.parent(int(node)))
        if p != int(node):
            assert pos[p] < pos[int(node)], \
                f"node {node} at pos {pos[int(node)]} but parent {p} at pos {pos[p]}"

def _linear_chain():
    """1×5, z monotonically increasing, single root at cell 0."""
    nrows, ncols = 1, 5
    z    = np.arange(5, dtype=np.float64)
    mask = np.ones(5, dtype=np.uint8)
    mask[0] = 3
    nb_fn = make_neighbours(nrows, ncols, d8=True, border='normal')
    return z, mask, nb_fn

def _grid_3x3():
    nrows, ncols = 3, 3
    z    = np.array([0,1,2, 1,2,3, 2,3,4], dtype=np.float64)
    mask = np.full(9, 3, dtype=np.uint8)
    mask[4] = 1
    nb_fn = make_neighbours(nrows, ncols, d8=True, border='normal')
    return z, mask, nb_fn


# ── par mode (Kahn's) ─────────────────────────────────────────────────────────

def test_par_leaves_first_property():
    z, mask, nb_fn = _grid_3x3()
    tree  = make_tree(z, nb_fn, mask, mode='par')
    order = topo_order(tree)
    assert len(order) == 9
    _check_leaves_first(order, tree)

def test_par_roots_first_property():
    z, mask, nb_fn = _grid_3x3()
    tree  = make_tree(z, nb_fn, mask, mode='par')
    order = topo_order(tree, reverse=True)
    _check_roots_first(order, tree)


# ── full mode (CSR BFS) ───────────────────────────────────────────────────────

def test_full_leaves_first_property():
    z, mask, nb_fn = _grid_3x3()
    tree  = make_tree(z, nb_fn, mask, mode='full')
    order = topo_order(tree)
    assert len(order) == 9
    _check_leaves_first(order, tree)

def test_full_roots_first_property():
    z, mask, nb_fn = _grid_3x3()
    tree  = make_tree(z, nb_fn, mask, mode='full')
    order = topo_order(tree, reverse=True)
    _check_roots_first(order, tree)


# ── par vs full — same node set ───────────────────────────────────────────────

def test_par_full_same_nodes():
    z, mask, nb_fn = _grid_3x3()
    par_order  = topo_order(make_tree(z, nb_fn, mask, mode='par'))
    full_order = topo_order(make_tree(z, nb_fn, mask, mode='full'))
    assert set(par_order) == set(full_order)


# ── argsort equivalence on linear chain ──────────────────────────────────────

def test_vs_argsort_par():
    z, mask, nb_fn = _linear_chain()
    tree  = make_tree(z, nb_fn, mask, mode='par')
    order = topo_order(tree)
    assert list(order) == list(np.argsort(z)[::-1])

def test_vs_argsort_full():
    z, mask, nb_fn = _linear_chain()
    tree  = make_tree(z, nb_fn, mask, mode='full')
    order = topo_order(tree)
    assert list(order) == list(np.argsort(z)[::-1])

def test_reverse_vs_argsort():
    z, mask, nb_fn = _linear_chain()
    tree  = make_tree(z, nb_fn, mask, mode='par')
    order = topo_order(tree, reverse=True)
    assert list(order) == list(np.argsort(z))


# ── PackedTree ────────────────────────────────────────────────────────────────

def test_packed_par_leaves_first():
    z, mask, nb_fn = _grid_3x3()
    nodes = np.array([0, 4], dtype=np.int64)
    tree  = make_packed_tree(nodes, z, nb_fn, mask, mode='par')
    order = topo_order_packed(tree)
    assert set(order) == {0, 4}
    _check_leaves_first(order, tree)

def test_packed_full_leaves_first():
    z, mask, nb_fn = _grid_3x3()
    nodes = np.array([0, 4], dtype=np.int64)
    tree  = make_packed_tree(nodes, z, nb_fn, mask, mode='full')
    order = topo_order_packed(tree)
    assert set(order) == {0, 4}
    _check_leaves_first(order, tree)

def test_packed_par_vs_full_same_nodes():
    z, mask, nb_fn = _grid_3x3()
    nodes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    par_order  = topo_order_packed(make_packed_tree(nodes, z, nb_fn, mask, mode='par'))
    full_order = topo_order_packed(make_packed_tree(nodes, z, nb_fn, mask, mode='full'))
    assert set(par_order) == set(full_order)
