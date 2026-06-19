"""Drainage area from a DEM using steepest-descent tree + kernel traversal."""

import numpy as np
import numba as nb
import topotoolbox as ttb
import matplotlib.pyplot as plt

from nbmdsa.structures import make_grid_nb, make_steep_tree_par, make_steep_tree_full, make_steep_tree_impl
from nbmdsa.algorithms import full_bfs, priority_flood_epsilon
from nbmdsa.structures.tree_sd import _ctor_par
from priority_flood_inline import priority_flood_epsilon_inline
import richdem as rd

import time

# ── Load DEM ──────────────────────────────────────────────────────────────────

dem   = ttb.load_dem('greenriver')
nrows, ncols = dem.shape
z     = dem.z.astype(np.float64).ravel()
cs    = float(dem.cellsize)


st = time.perf_counter()
for i in range(10):
    # richdem needs a 2D rdarray with a nodata value
    dem_rd = rd.rdarray(z.reshape(nrows, ncols), no_data=-9999.0)
    z_filled_rd = rd.FillDepressions(dem_rd, epsilon=True, in_place=False)
     # = np.array(dem_rd).ravel()
print('priority flood OG took', time.perf_counter() - st)

# ── Grid neighbourer ──────────────────────────────────────────────────────────
# make_grid_nb(nrows, ncols,
#   border   = 'normal'  # 'normal' | 'ew' (east-west periodic) | 'ns' (north-south periodic)
#   indexing = '1dr'     # '1dr' (row-major 1D) | '1dc' (col-major 1D) | '2d'
#   d8       = False     # False = D4 (4-connectivity) | True = D8 (8-connectivity)
#   mask     = None      # bool[nrows*ncols] — False cells are inactive
#   dx       = 1.0       # cell width  (east-west distance)
#   dy       = 1.0       # cell height (north-south distance)
# )

g = make_grid_nb(nrows, ncols, dx=cs, dy=cs, d8=True, border='normal', indexing='1dr')

# ── Depression filling ────────────────────────────────────────────────────────

border        = np.zeros(nrows * ncols, dtype=bool)
border[:ncols] = border[-ncols:] = True
border[::ncols] = border[ncols-1::ncols] = True
z = priority_flood_epsilon(z, g, border)

st = time.perf_counter()
for i in range(10):
    z = priority_flood_epsilon(z, g, border)
print('priority flood nbmdsa  took', time.perf_counter() - st)

z_inline = priority_flood_epsilon_inline(z, nb.int64(nrows), nb.int64(ncols))
st = time.perf_counter()
for i in range(10):
    z_inline = priority_flood_epsilon_inline(z, nb.int64(nrows), nb.int64(ncols))
print('priority flood inline  took', time.perf_counter() - st)
# ── Steepest-descent tree ─────────────────────────────────────────────────────
# make_steep_tree_par(z, g)
#   Stores z + parents array. parent(idx) and is_root(idx) are O(1).
#   children(idx, g) still requires the neighbourer.
#   → best default for kernel traversals.
#
# make_steep_tree_full(z, g)
#   Stores z + parents + CSR children. All queries O(1), no g at query time.
#   → use when children are queried heavily without g available.
#
# make_steep_tree_impl(z)
#   Stores only z. Everything computed on the fly; requires g at every query.
#   → use when memory is tight and queries are rare.
#
# Rebuild from a pre-computed parents array (e.g. after priority_flood):
#   tree = _ctor_par(z_filled, parents)
st = time.perf_counter()
for i in range(10):
    # tree = make_steep_tree_impl(z_filled)
    tree = make_steep_tree_full(z, g)
print('tree creation tool', time.perf_counter() - st)

# ── Drainage area kernel ──────────────────────────────────────────────────────
# upstream=False → reverse topo order (leaves first): each node's total is
# complete before it propagates to its parent.

@nb.njit
def accumulate(tree, g, node, extra):
    area, = extra
    p = tree.parent_of(node, g)
    if p != node:
        area[p] += area[node]

area = np.full(nrows * ncols, cs * cs, dtype=np.float64)
full_bfs(tree, g, accumulate, (area,), upstream=False)
st = time.perf_counter()
for i in range(10):
    area.fill(cs*cs)
    full_bfs(tree, g, accumulate, (area,), upstream=False)
print('tree creation tool', time.perf_counter() - st)


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots()
ax.imshow(dem.hillshade(), cmap='gray')
im = ax.imshow(np.log1p(area).reshape(nrows, ncols), cmap='Blues', alpha=0.8)
plt.colorbar(im, label='log(drainage area)')
plt.tight_layout()
plt.show()
