"""Tree update algorithms.

Grid-indexed trees (Tree)
  update_from_z(tree, z, mask=None)
  update_from_parents(tree, new_parents_grid, mask=None)
      new_parents_grid: grid-indexed array, values are grid indices

Packed trees (PackedTree)
  update_packed_from_z(tree, z, mask=None)
  update_packed_from_parents(tree, new_parents_grid, mask=None)
      new_parents_grid: grid-indexed array, values are grid indices
"""

import numpy as np

from nbmdsa.structures.tree import _build_parents, _build_csr
from nbmdsa.structures.packed_tree import (
    _build_packed_parents, _build_packed_csr, _apply_external_parents,
)


def update_from_z(tree, z, mask=None):
    m = mask if mask is not None else tree.mask
    if len(tree.parents) == 0:          # implicit — nothing to precompute
        if mask is not None:
            tree.mask[:] = mask
        return
    _build_parents(z, m, tree.parents, tree.neighbours_fn)
    if len(tree.child_ptr) > 0:         # full mode
        _build_csr(tree.parents, m, tree.child_ptr, tree.child_data)


def update_from_parents(tree, new_parents_grid, mask=None):
    """new_parents_grid: grid-indexed array whose values are grid indices."""
    if len(tree.parents) == 0:          # implicit — nothing stored
        return
    tree.parents[:] = new_parents_grid
    if len(tree.child_ptr) > 0:         # full mode — propagate to CSR
        m = mask if mask is not None else tree.mask
        _build_csr(tree.parents, m, tree.child_ptr, tree.child_data)


# ── PackedTree updates ────────────────────────────────────────────────────────

def update_packed_from_z(tree, z, mask=None):
    m = mask if mask is not None else tree.mask
    if len(tree.parents) == 0:          # implicit
        if mask is not None:
            tree.mask[:] = mask
        return
    _build_packed_parents(tree.nodes, tree.grid_to_local, z, m,
                          tree.parents, tree.neighbours_fn)
    if len(tree.child_ptr) > 0:         # full mode
        _build_packed_csr(tree.parents, tree.child_ptr, tree.child_data)


def update_packed_from_parents(tree, new_parents_grid, mask=None):
    """new_parents_grid: grid-indexed array whose values are grid indices."""
    if len(tree.parents) == 0:          # implicit — nothing stored
        return
    _apply_external_parents(tree.nodes, tree.grid_to_local,
                             new_parents_grid, tree.parents)
    if len(tree.child_ptr) > 0:         # full mode — propagate to CSR
        _build_packed_csr(tree.parents, tree.child_ptr, tree.child_data)
