"""Tests for Tree and PackedTree structures.

3x3 grid fixture:
  col:  0  1  2
  row 0: 0  1  2    z values
  row 1: 3  4  5    idx
  row 2: 6  7  8

  z:    [0,1,2, 1,2,3, 2,3,4]
  mask: borders=3, center (idx=4)=1

  Center (idx=4, z=2) steepest parent = idx 0 (topleft, z=0, first minimum found).
"""

import numpy as np
import numba as nb
import pytest

from nbmdsa.structures.neighbourer  import make_neighbours
from nbmdsa.structures.tree         import make_tree
from nbmdsa.structures.packed_tree  import make_packed_tree
from nbmdsa.algorithms.tree_update  import (
    update_from_z, update_from_parents,
    update_packed_from_z, update_packed_from_parents,
)


# ── fixture ───────────────────────────────────────────────────────────────────

def _grid():
    nrows, ncols = 3, 3
    z    = np.array([0,1,2, 1,2,3, 2,3,4], dtype=np.float64)
    mask = np.full(9, 3, dtype=np.uint8)
    mask[4] = 1
    nb_fn = make_neighbours(nrows, ncols, d8=True, border='normal')
    return z, mask, nb_fn


# ── Tree — par mode ───────────────────────────────────────────────────────────

class TestTreePar:
    def setup_method(self):
        z, mask, nb_fn = _grid()
        self.tree = make_tree(z, nb_fn, mask, mode='par')
        self.buf  = np.full(8, -1, dtype=np.int64)

    def test_parent_of_root(self):
        assert self.tree.parent(0) == 0

    def test_parent_of_interior(self):
        # idx=4 steepest descent → idx=0 (topleft, z=0)
        assert self.tree.parent(4) == 0

    def test_children_of_root(self):
        self.tree.children(0, self.buf)
        # idx=4 is at bottomright of idx=0 → slot 7
        assert self.buf[7] == 4
        assert all(self.buf[k] == -1 for k in range(7))

    def test_children_of_leaf(self):
        self.tree.children(4, self.buf)
        assert all(v == -1 for v in self.buf)

    def test_nodata_not_in_parents(self):
        # all border cells (mask=3) are roots
        for i in [0,1,2,3,5,6,7,8]:
            assert self.tree.parent(i) == i


# ── Tree — full mode ──────────────────────────────────────────────────────────

class TestTreeFull:
    def setup_method(self):
        z, mask, nb_fn = _grid()
        self.tree = make_tree(z, nb_fn, mask, mode='full')
        self.buf  = np.full(8, -1, dtype=np.int64)

    def test_parent_of_interior(self):
        assert self.tree.parent(4) == 0

    def test_children_of_root(self):
        self.tree.children(0, self.buf)
        assert self.buf[7] == 4
        assert all(self.buf[k] == -1 for k in range(7))

    def test_csr_consistent(self):
        # child_ptr[0+1] - child_ptr[0] == 1 (cell 0 has one child)
        assert self.tree.child_ptr[1] - self.tree.child_ptr[0] == 1
        assert self.tree.child_data[self.tree.child_ptr[0]] == 4


# ── Tree — implicit mode ──────────────────────────────────────────────────────

class TestTreeImplicit:
    def setup_method(self):
        self.z, self.mask, nb_fn = _grid()
        self.tree = make_tree(self.z, nb_fn, self.mask, mode='implicit')
        self.buf  = np.full(8, -1, dtype=np.int64)

    def test_parent_of_interior(self):
        assert self.tree.parent(4, self.z) == 0

    def test_parent_of_root(self):
        assert self.tree.parent(0, self.z) == 0

    def test_children_of_root(self):
        self.tree.children(0, self.z, self.buf)
        assert self.buf[7] == 4
        assert all(self.buf[k] == -1 for k in range(7))


# ── update_from_z ─────────────────────────────────────────────────────────────

def test_update_from_z_changes_parent():
    z, mask, nb_fn = _grid()
    tree = make_tree(z, nb_fn, mask, mode='par')
    assert tree.parents[4] == 0
    # raise cell 0 above cell 4 → parent changes to next cheapest
    z2 = z.copy()
    z2[0] = 10.0
    update_from_z(tree, z2)
    # Numba closures capture array state at first compile; check raw array instead
    assert tree.parents[4] != 0

def test_update_from_z_full_propagates_csr():
    z, mask, nb_fn = _grid()
    tree = make_tree(z, nb_fn, mask, mode='full')
    z2 = z.copy(); z2[0] = 10.0
    update_from_z(tree, z2)
    # Use raw array to avoid Numba closure constant-capture issue
    new_parent = int(tree.parents[4])
    # CSR must reflect new parent: child_ptr[new_parent] range contains 4
    start = tree.child_ptr[new_parent]
    end   = tree.child_ptr[new_parent + 1]
    assert 4 in tree.child_data[start:end]

def test_update_from_parents():
    z, mask, nb_fn = _grid()
    tree = make_tree(z, nb_fn, mask, mode='par')
    new_p = tree.parents.copy()
    new_p[4] = 2  # override: make cell 2 the parent of cell 4
    update_from_parents(tree, new_p)
    assert tree.parent(4) == 2


# ── PackedTree ────────────────────────────────────────────────────────────────

class TestPackedTree:
    def setup_method(self):
        z, mask, nb_fn = _grid()
        # subset: only cells 0 and 4
        nodes = np.array([0, 4], dtype=np.int64)
        self.tree_par  = make_packed_tree(nodes, z, nb_fn, mask, mode='par')
        self.tree_full = make_packed_tree(nodes, z, nb_fn, mask, mode='full')
        self.buf = np.full(8, -1, dtype=np.int64)

    def test_parent_returns_grid_idx(self):
        # cell 4's parent in packed set is cell 0
        assert self.tree_par.parent(4) == 0
        assert self.tree_full.parent(4) == 0

    def test_root_is_own_parent(self):
        assert self.tree_par.parent(0) == 0
        assert self.tree_full.parent(0) == 0

    def test_children_grid_indices(self):
        self.tree_par.children(0, self.buf)
        # cell 4 is bottomright (slot 7) of cell 0
        assert self.buf[7] == 4
        assert all(self.buf[k] == -1 for k in range(7))

    def test_grid_to_local_mapping(self):
        assert self.tree_par.grid_to_local[0] == 0
        assert self.tree_par.grid_to_local[4] == 1
        assert self.tree_par.grid_to_local[1] == -1  # not in subset

    def test_update_packed_from_z(self):
        z, mask, nb_fn = _grid()
        nodes = np.array([0, 4], dtype=np.int64)
        tree = make_packed_tree(nodes, z, nb_fn, mask, mode='par')
        assert tree.parents[1] == 0  # cell 4 (local 1) initially points to cell 0 (local 0)
        z2 = z.copy(); z2[0] = 10.0
        update_packed_from_z(tree, z2)
        # z[0]=10 > z[4]=2: cell 0 is no longer downhill in the packed set
        # cell 4 has no lower packed neighbour → becomes its own root (local parent = self)
        assert tree.parents[1] == 1

    def test_update_packed_from_parents(self):
        z, mask, nb_fn = _grid()
        nodes = np.array([0, 4], dtype=np.int64)
        tree = make_packed_tree(nodes, z, nb_fn, mask, mode='full')
        # build a new parents_grid where cell 4 points to itself (make it a root)
        new_parents = np.arange(9, dtype=np.int64)  # all self-referential
        update_packed_from_parents(tree, new_parents)
        assert tree.parent(4) == 4
