"""Benchmark: priority-flood + epsilon, three implementations.

  richdem   — C++ reference (FillDepressions with epsilon=True)
  inline    — raw @nb.njit, no framework (priority_flood_inline.py)
  nbmdsa    — this framework (closure neighbourer + heap/queue + algorithm)
"""

import time

import numpy as np
import numba as nb
import topotoolbox as ttb
import richdem as rd

from priority_flood_inline import priority_flood_epsilon_inline
from nbmdsa.structures.neighbourer import make_neighbours
from nbmdsa.structures.primitives  import make_heap, make_queue
from nbmdsa.algorithms.priority_flood import make_priority_flood

RUNS = 10

# ── data ──────────────────────────────────────────────────────────────────────

dem         = ttb.load_dem('greenriver')
nrows, ncols = dem.shape
z0          = dem.z.astype(np.float32).ravel()

# mask: 3 = outlet (border cells), 1 = normal, 0 = nodata (none here)
mask = np.ones(nrows * ncols, dtype=np.uint8)
mask[:ncols] = mask[-ncols:] = 3
mask[::ncols] = mask[ncols - 1::ncols] = 3

# ── warm-up JIT ───────────────────────────────────────────────────────────────

nb_fn  = make_neighbours(nrows, ncols, d8=True, border='normal')
heap   = make_heap(nb.float32)
queue  = make_queue(nb.int32)
pf     = make_priority_flood(nb.float32)

priority_flood_epsilon_inline(z0.copy(), nb.int32(nrows), nb.int32(ncols))
pf(z0.copy(), mask, nb_fn, heap, queue)

# ── richdem ───────────────────────────────────────────────────────────────────

dem_rd = rd.rdarray(z0.reshape(nrows, ncols), no_data=-9999.0)
st = time.perf_counter()
for _ in range(RUNS):
    rd.FillDepressions(dem_rd, epsilon=True, in_place=False)
print(f'richdem  {time.perf_counter() - st:.3f}s')

# ── inline numba ──────────────────────────────────────────────────────────────

st = time.perf_counter()
for _ in range(RUNS):
    priority_flood_epsilon_inline(z0.copy(), nb.int32(nrows), nb.int32(ncols))
print(f'inline   {time.perf_counter() - st:.3f}s')

# ── nbmdsa ────────────────────────────────────────────────────────────────────

st = time.perf_counter()
for _ in range(RUNS):
    pf(z0.copy(), mask, nb_fn, heap, queue)
print(f'nbmdsa   {time.perf_counter() - st:.3f}s')
