"""
Coupled LEM: uplift + nonlinear hillslope diffusion + Hergarten TL-SPL.

Steps per timestep:
  1. Priority flood (ensure drainage)
  2. Build tree + topo order
  3. Accumulate drainage area
  4. Uplift (explicit)
  5. Hillslope diffusion (Ren et al. 2026, implicit)
  6. Hergarten TL-SPL (implicit)
"""

import numpy as np
import matplotlib.pyplot as plt

from nbmdsa.structures.neighbourer    import make_neighbours
from nbmdsa.structures.primitives     import make_heap, make_queue
from nbmdsa.algorithms.priority_flood import make_priority_flood
from nbmdsa.structures.tree           import make_tree
from nbmdsa.algorithms.topo           import topo_order as compute_topo_order
from nbmdsa.algorithms.tree_traversal import tree_traversal_full

from nbmdsa.extra.noise          import perlin_noise
from nbmdsa.extra.lem            import uplift as uplift_mod
from nbmdsa.extra.lem            import hillslope_diffusion as hd
from nbmdsa.extra.lem            import hergarten

import numba as nb
import time

# ── parameters ────────────────────────────────────────────────────────────────

nrows  = 256
ncols  = 256
dx     = 100.0    # cell size [m]
dt     = 5e3      # timestep [yr]
N      = 500     # number of steps

# uplift
U_max  = 1e-3     # peak uplift rate [m/yr]  (zero in southern 1/3)

# hillslope diffusion (nonlinear, Roering 1999)
D2     = 3e-3     # diffusion coefficient [m²/yr]
Sc     = 0.6     # critical slope [m/m]

# fluvial SPL (Hergarten TL)
K_spl  = 1e-5     # erodibility
m_spl  = 0.5      # area exponent
n_spl  = 1.0      # slope exponent

# display
plot_every = 200  # steps between live plot updates (set to 0 to disable)


# ── grid ──────────────────────────────────────────────────────────────────────

nn  = nrows * ncols
rng = np.random.default_rng(42)

z = perlin_noise.generate(nrows, ncols, scale=4.0, octaves=6,
                           persistence=0.5, amplitude=10.0,
                           offset=50.0, seed=42)
z = z.astype(np.float64)

mask = np.ones(nn, dtype=np.uint8)
for j in range(ncols):
    mask[j] = np.uint8(3)   # bottom row = outlets
    z[j]    = 0.0

nb_fn     = make_neighbours(nrows, ncols, d8=True, border='ew')
heap_ops  = make_heap()
queue_ops = make_queue()
pf        = make_priority_flood()

# uplift field: zero in southern 1/3, U_max in northern 2/3
U_field = np.zeros(nn, dtype=np.float64)
split   = nrows // 3
for i in range(nrows):
    if i >= split:
        U_field[i * ncols : (i + 1) * ncols] = U_max

# spatially uniform hillslope params (flat arrays)
D2_arr = np.full(nn, D2, dtype=np.float64)
Sc_arr = np.full(nn, Sc, dtype=np.float64)
phi_fn = hd.make_phi_nonlinear(D2_arr, Sc_arr)

# spatially uniform SPL params (as flat arrays)
K_arr = np.full(nn, K_spl, dtype=np.float64)
m_arr = np.full(nn, m_spl, dtype=np.float64)
n_arr = np.full(nn, n_spl, dtype=np.float64)

# SPL working arrays
A      = np.full(nn, dx * dx, dtype=np.float64)
Q0     = np.zeros(nn, dtype=np.float64)
Qp     = np.zeros(nn, dtype=np.float64)
f_arr  = np.zeros(nn, dtype=np.float64)
dH     = np.zeros(nn, dtype=np.float64)
Qp_sum = np.zeros(nn, dtype=np.float64)
Q0_sum = np.zeros(nn, dtype=np.float64)


# ── catchment kernel ──────────────────────────────────────────────────────────

@nb.njit
def _catchment(idx, parents, child_ptr, child_data, mask, neighbours_fn, A):
    p = parents[idx]
    if p != nb.int64(idx):
        A[p] += A[idx]


# ── main loop ─────────────────────────────────────────────────────────────────

st = time.perf_counter()
for step in range(N):

    if step % 10 == 0:
        z = pf(z.copy(), mask, nb_fn, heap_ops, queue_ops)

    tree  = make_tree(z, nb_fn, mask, mode='par')
    order = compute_topo_order(tree)

    A[:] = dx * dx
    tree_traversal_full(order, tree, _catchment, (A,), direction='up')

    # 1. uplift
    uplift_mod.apply_uplift(order, tree, z, U_field, np.float64(dt))

    # 2. hillslope diffusion
    z = hd.step(z, mask, nrows, ncols, dx, dt, phi_fn, nb_fn,
                laplacian='D8', slope_cap=0.999 * Sc)

    # 3. Hergarten TL-SPL
    hergarten.step(order, tree, z, A, Q0, Qp, f_arr, Qp_sum, Q0_sum, dH,
                   K_arr, m_arr, n_arr, dt, ncols, dx)

    if step % 50 == 0:
        interior = mask > 0
        print(f"step {step:4d}  max_z={z[interior].max():.2f}  mean_z={z[interior].mean():.2f}")

print(f"took {time.perf_counter() - st:.1f}s")


# ── plot ──────────────────────────────────────────────────────────────────────

interior_2d = mask.reshape(nrows, ncols) > 0
z_2d  = z.reshape(nrows, ncols)
A_2d  = np.where(interior_2d, A.reshape(nrows, ncols), np.nan)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(z_2d, cmap='terrain', origin='lower')
axes[0].set_title(f'Elevation  (K={K_spl}, D2={D2}, Sc={Sc}, U={U_max}, dt={dt}, N={N})')
plt.colorbar(im0, ax=axes[0], label='z [m]')

im1 = axes[1].imshow(np.log10(A_2d), cmap='Blues', origin='lower')
axes[1].set_title('log₁₀ drainage area [m²]')
plt.colorbar(im1, ax=axes[1], label='log₁₀(A)')

plt.tight_layout()
plt.show()
