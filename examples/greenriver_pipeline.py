"""Greenriver DEM pipeline: priority flood → par-mode tree → topo order → drainage area.

Steps
-----
1. Load greenriver DEM from topotoolbox.
2. Build mask (nodata=0, border=3, interior=1) and priority-flood fill depressions.
3. Build a par-mode tree (parents array only — no CSR; "semi-explicit").
4. Compute leaves-first topological order via Kahn's algorithm.
5. Accumulate drainage area with a custom @nb.njit kernel via tree_traversal_full.
6. Plot log-drainage area overlaid on hillshade.
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import topotoolbox as ttb

from nbmdsa.structures.neighbourer   import make_neighbours
from nbmdsa.structures.primitives    import make_heap, make_queue
from nbmdsa.algorithms.priority_flood import make_priority_flood
from nbmdsa.structures.tree          import make_tree
from nbmdsa.algorithms.topo          import topo_order
from nbmdsa.algorithms.tree_traversal import tree_traversal_full


# ── 1. load DEM ───────────────────────────────────────────────────────────────

dem          = ttb.load_dem('greenriver')
nrows, ncols = dem.shape
z            = dem.z.astype(np.float64).ravel().copy()
cs           = float(dem.cellsize)
n            = nrows * ncols

# ── 2. mask + priority flood ──────────────────────────────────────────────────

nodata = float(dem.nodata) if hasattr(dem, 'nodata') else -9999.0
mask   = np.ones(n, dtype=np.uint8)
mask[z == nodata] = 0

# grid border cells are outlets
border          = np.zeros(n, dtype=bool)
border[:ncols]  = True
border[-ncols:] = True
border[::ncols] = True
border[ncols - 1::ncols] = True
mask[border & (mask != 0)] = 3

nb_fn      = make_neighbours(nrows, ncols, d8=True, border='normal')
heap_ops   = make_heap()
queue_ops  = make_queue()
pf         = make_priority_flood()

z_filled = pf(z.copy(), mask, nb_fn, heap_ops, queue_ops)

# ── 3. par-mode tree (parents array, no CSR) ──────────────────────────────────

tree = make_tree(z_filled, nb_fn, mask, mode='par')

# ── 4. topological order (leaves first) ──────────────────────────────────────

order = topo_order(tree)          # int64[:], leaves-first

# ── 5. drainage area via custom kernel ────────────────────────────────────────

@nb.njit
def accumulate_area(idx, parents, child_ptr, child_data, mask, neighbours_fn, area):
    p = parents[idx]
    if p != nb.int64(idx):
        area[p] += area[idx]

area           = np.zeros(n, dtype=np.float64)
area[mask > 0] = cs * cs   # each valid cell starts with its own area

tree_traversal_full(order, tree, accumulate_area, (area,), direction='up')

# ── 6. plot ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(dem.hillshade(), cmap='gray', interpolation='bilinear')
im = ax.imshow(
    np.log1p(area).reshape(nrows, ncols),
    cmap='Blues', alpha=0.7,
)
plt.colorbar(im, ax=ax, label='log(drainage area  m²)')
ax.set_title('Greenriver — drainage area (nbmdsa)')
ax.axis('off')
plt.tight_layout()
plt.show()
