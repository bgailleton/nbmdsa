"""Greenriver MFD drainage area — flow partitioned proportionally to slope.

Steps
-----
1. Load greenriver DEM, build mask, priority-flood fill depressions.
2. Compute MFD topological order (Kahn's on the donor/receiver DAG).
3. Accumulate drainage area with a custom @nb.njit kernel:
   each cell distributes its area to ALL strictly lower neighbours,
   weighted by (z[i] - z[j]) / sum_of_drops.
4. Plot log-drainage area on hillshade.
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import topotoolbox as ttb

from nbmdsa.structures.neighbourer    import make_neighbours
from nbmdsa.structures.primitives     import make_heap, make_queue
from nbmdsa.algorithms.priority_flood import make_priority_flood
from nbmdsa.algorithms.mfd_traversal  import mfd_topo_order, mfd_traversal_full, mfd_traversal_partial


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

border          = np.zeros(n, dtype=bool)
border[:ncols]  = True
border[-ncols:] = True
border[::ncols] = True
border[ncols - 1::ncols] = True
mask[border & (mask != 0)] = 3

nb_fn     = make_neighbours(nrows, ncols, d8=True, border='normal')
heap_ops  = make_heap()
queue_ops = make_queue()
pf        = make_priority_flood()

z_filled = pf(z.copy(), mask, nb_fn, heap_ops, queue_ops)

# ── 3. MFD topological order ──────────────────────────────────────────────────
# donors-first (local maxima → sinks): guarantees every cell's area is fully
# accumulated before it distributes to its receivers.

order = mfd_topo_order(z_filled, mask, nb_fn)

# ── 4. MFD drainage area kernel ───────────────────────────────────────────────
# Weights: w(i→j) = (z[i] - z[j]) / Σ(z[i] - z[k]) for all receivers k.
# nbuf passed via extra_args to avoid per-call heap allocation.

@nb.njit
def mfd_accumulate(idx, z, mask, neighbours_fn, area, nbuf):
    neighbours_fn(nb.int64(idx), nbuf)

    total = nb.float64(0.0)
    for k in range(nb.int64(8)):
        j = nbuf[k]
        if j == nb.int64(-1) or mask[j] == nb.uint8(0):
            continue
        drop = z[idx] - z[j]
        if drop > nb.float64(0.0):
            total += drop

    if total == nb.float64(0.0):
        return   # sink or flat — no receivers to distribute to

    for k in range(nb.int64(8)):
        j = nbuf[k]
        if j == nb.int64(-1) or mask[j] == nb.uint8(0):
            continue
        drop = z[idx] - z[j]
        if drop > nb.float64(0.0):
            area[j] += area[idx] * drop / total


area           = np.zeros(n, dtype=np.float64)
# area[mask > 0] = cs * cs
start_nodes = [512 * dem.columns + 512]
area[start_nodes] = 50

nbuf_scratch = np.empty(8, dtype=np.int64)   # reused scratch buffer

# mfd_traversal_full(order, z_filled, mask, nb_fn,
#                    mfd_accumulate, (area, nbuf_scratch),
#                    upstream=True)

mfd_traversal_partial(start_nodes, z_filled, mask, nb_fn,
                        mfd_accumulate, (area, nbuf_scratch),
                        direction='down', multi_enabled=False,
                        mode='pq', min_heap=False)

# ── 5. plot ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(dem.hillshade(), cmap='gray', interpolation='bilinear')
im = ax.imshow(
    np.log1p(area).reshape(nrows, ncols),
    cmap='Blues', alpha=0.7,
)
plt.colorbar(im, ax=ax, label='log(drainage area  m²)')
ax.set_title('Greenriver — MFD drainage area (slope-proportional)')
ax.axis('off')
plt.tight_layout()
plt.show()


# cmake -S . -B build -DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9 -DCMAKE_CUDA_HOST_COMPILER=g++-9