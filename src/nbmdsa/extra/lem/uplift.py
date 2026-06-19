"""
Explicit uplift step for LEM solvers.

U_arr  : flat float64 array, uplift rate [m/yr] per cell (spatially variable)
dt     : float64, timestep [yr]

Uplift is NOT applied to outlet cells (mask == 3).
"""

import numba as nb
import numpy as np

from nbmdsa.algorithms.tree_traversal import tree_traversal_full


@nb.njit
def _uplift_kernel(idx, parents, child_ptr, child_data, mask, neighbours_fn,
                   z, U_arr, dt):
    if mask[idx] != nb.uint8(3):
        z[idx] += U_arr[idx] * dt


def apply_uplift(order, tree, z, U_arr, dt):
    """Apply one explicit uplift step in-place on z."""
    tree_traversal_full(order, tree, _uplift_kernel,
                        (z, U_arr, np.float64(dt)),
                        direction='none')
