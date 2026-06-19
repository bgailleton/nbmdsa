"""
Yuan et al. (2019) linear decline model — using nbmdsa.extra.lem.yuan.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from nbmdsa.structures.neighbourer    import make_neighbours
from nbmdsa.structures.primitives     import make_heap, make_queue
from nbmdsa.algorithms.priority_flood import make_priority_flood
from nbmdsa.structures.tree           import make_tree
from nbmdsa.algorithms.topo           import topo_order as compute_topo_order
from nbmdsa.algorithms.tree_traversal import tree_traversal_full

from nbmdsa.extra.lem   import yuan, uplift as uplift_mod
from nbmdsa.extra.noise import perlin_noise

import numba as nb

# ── parameters ────────────────────────────────────────────────────────────────

K_spl   = 5e-5   # erodibility
m_spl   = 0.45    # area exponent
n_spl   = 1.0    # slope exponent
G_val   = 0.2    # deposition coefficient (0 = pure DL, large → TL)
U_max   = 1e-3   # peak uplift rate [m/yr]
dt      = 1e3    # time step [yr]
dx      = 100.0  # cell size [m]
N       = 500    # number of time steps
tol     = 1e-3   # Gauss-Seidel convergence tolerance
max_iter= 5000     # maximum GS iterations per timestep
nrows   = 256
ncols   = 256


# ── grid ──────────────────────────────────────────────────────────────────────

nn = nrows * ncols
z  = perlin_noise.generate(nrows, ncols, scale=4.0, octaves=6,
                            persistence=0.5, amplitude=10.0,
                            offset=50.0, seed=42).astype(np.float64)

mask = np.ones(nn, dtype=np.uint8)
for j in range(ncols):
    mask[j] = np.uint8(3)
    z[j]    = 0.0

nb_fn     = make_neighbours(nrows, ncols, d8=True, border='ew')
heap_ops  = make_heap()
queue_ops = make_queue()
pf        = make_priority_flood()

U_field = np.zeros(nn, dtype=np.float64)
split   = nrows // 3
for i in range(nrows):
    if i >= split:
        U_field[i * ncols : (i + 1) * ncols] = U_max

K_arr = np.full(nn, K_spl, dtype=np.float64)
m_arr = np.full(nn, m_spl, dtype=np.float64)
n_arr = np.full(nn, n_spl, dtype=np.float64)
G_arr = np.full(nn, G_val,  dtype=np.float64)

A        = np.full(nn, dx * dx, dtype=np.float64)
h        = np.zeros(nn, dtype=np.float64)
hp       = np.zeros(nn, dtype=np.float64)
elev     = np.zeros(nn, dtype=np.float64)
dh_accum = np.zeros(nn, dtype=np.float64)


@nb.njit
def _catchment(idx, parents, child_ptr, child_data, mask, neighbours_fn, A):
    p = parents[idx]
    if p != nb.int64(idx):
        A[p] += A[idx]


# ── main loop ─────────────────────────────────────────────────────────────────

st = time.perf_counter()
for step in range(N):
    z = pf(z.copy(), mask, nb_fn, heap_ops, queue_ops)

    tree  = make_tree(z, nb_fn, mask, mode='par')
    order = compute_topo_order(tree)

    A[:] = dx * dx
    tree_traversal_full(order, tree, _catchment, (A,), direction='up')

    uplift_mod.apply_uplift(order, tree, z, U_field, np.float64(dt))

    h_t0 = z.copy()
    h[:] = h_t0
    hp[:] = h_t0
    dh_accum[:] = 0.0

    gs_iter = 0
    for gs_iter in range(max_iter):
        err = yuan.step(order, tree, h, hp, elev, dh_accum, A, h_t0,
                        K_arr, m_arr, n_arr, G_arr, dt, ncols, dx, mask)
        if G_val == 0.0 or err < tol * max(h_t0.max(), 1e-10):
            break

    z[:] = h

    if step % 50 == 0:
        interior = mask > 0
        print(f"step {step:4d}  gs_iters={gs_iter+1:2d}  "
              f"max_z={z[interior].max():.3f}  mean_z={z[interior].mean():.3f}")

print('took', time.perf_counter() - st)


# ── plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(z.reshape(nrows, ncols), cmap='terrain', origin='lower')
axes[0].set_title(f'Elevation  Yuan et al. (K={K_spl}, m={m_spl}, G={G_val}, U_max={U_max})')
plt.colorbar(im0, ax=axes[0], label='z [m]')

A_plot = np.where(mask > 0, A, np.nan).reshape(nrows, ncols)
im1 = axes[1].imshow(np.log10(A_plot), cmap='Blues', origin='lower')
axes[1].set_title('log₁₀ drainage area')
plt.colorbar(im1, ax=axes[1], label='log₁₀(cells)')

plt.tight_layout()
plt.show()
