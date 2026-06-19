"""
Yuan et al. (2019) linear decline model — iterative implicit solver.

Erosion rate:   E_i = K * A_i^m * S_i - G * Q_i / A_i
Sediment flux:  Q_i = Σ_upstream(E * cell_area)

This is the Davy-Lague (2009) model solved with a Gauss-Seidel iteration
on the Braun-Willett implicit scheme (FastScape, 2013).

  G = 0   → pure detachment-limited (FastScape, one pass, no iteration)
  G → ∞   → transport-limited end-member

Per Gauss-Seidel iteration:
  1. Compute elev[i] = H_t0[i] + G * upstream_Q[i] / A[i]
         where upstream_Q = cumulative upstream erosion (leaf → outlet sweep)
  2. Implicit SPL update (outlet-first):
         fact = K * dt * A[i]^m / d[i]
         H[i] = (elev[i] + fact * H[receiver]) / (1 + fact)
  3. err = ||H - H_prev|| / sqrt(n)
  4. Repeat until err < tol or max_iter reached.

Uplift is applied explicitly before the iteration (operator splitting).
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from nbmdsa.structures.neighbourer    import make_neighbours
from nbmdsa.structures.primitives     import make_heap, make_queue
from nbmdsa.algorithms.priority_flood import make_priority_flood
from nbmdsa.structures.tree           import make_tree
from nbmdsa.algorithms.topo           import topo_order as compute_topo_order
from nbmdsa.algorithms.tree_traversal import tree_traversal_full


import time

# ── parameters ────────────────────────────────────────────────────────────────

K       = 1e-4   # erodibility
m       = 0.5    # area exponent
n       = 1.    # slope exponent (1 = linear, fast path; else Newton-Raphson per node)
G       = 0.8    # deposition coefficient (0 = pure DL, large → TL)
U_max   = 1e-3   # peak uplift rate [m/yr]
dt      = 1e3    # time step [yr]
dx      = 100.0  # cell size [m]
N       = 500    # number of time steps
tol     = 1e-4   # Gauss-Seidel convergence tolerance (relative to max z)
max_iter= 50     # maximum GS iterations per timestep
nrows   = 256
ncols   = 256


# ── grid ──────────────────────────────────────────────────────────────────────

nn  = nrows * ncols
rng = np.random.default_rng(42)
z   = rng.random(nn) * 0.01

mask = np.ones(nn, dtype=np.uint8)
for j in range(ncols):
    mask[j] = np.uint8(3)
    z[j]    = 0.0

nb_fn     = make_neighbours(nrows, ncols, d8=True, border='ew')
heap_ops  = make_heap()
queue_ops = make_queue()
pf        = make_priority_flood()

# 2D uplift: high in northern 2/3, zero in southern 1/3
U_field = np.zeros(nn, dtype=np.float64)
split   = nrows // 3
for i in range(nrows):
    if i >= split:
        U_field[i * ncols : (i + 1) * ncols] = U_max
Udt_field = U_field * dt


# ── working arrays ────────────────────────────────────────────────────────────

A        = np.full(nn, dx * dx, dtype=np.float64)   # catchment area [m²]
h        = np.zeros(nn, dtype=np.float64)   # current GS estimate
hp       = np.zeros(nn, dtype=np.float64)   # previous GS estimate
elev     = np.zeros(nn, dtype=np.float64)   # effective elevation (with deposition)
dh_accum = np.zeros(nn, dtype=np.float64)   # cumulative upstream erosion


# ── kernels ───────────────────────────────────────────────────────────────────

@nb.njit
def _catchment(idx, parents, child_ptr, child_data, mask, neighbours_fn, A):
    p = parents[idx]
    if p != nb.int64(idx):
        A[p] += A[idx]


@nb.njit
def _uplift(idx, parents, child_ptr, child_data, mask, neighbours_fn, z, Udt_field):
    if mask[idx] != nb.uint8(3):
        z[idx] += Udt_field[idx]


@nb.njit
def _accum_dh(idx, parents, child_ptr, child_data, mask, neighbours_fn, dh):
    """Accumulate erosion from leaves to outlet (leaf → outlet order)."""
    p = parents[idx]
    if p != nb.int64(idx):
        dh[p] += dh[idx]


@nb.njit
def _yuan_sweep(idx, parents, child_ptr, child_data, mask, neighbours_fn,
                h, elev, A, K, m_exp, n_exp, dt, ncols, dx):
    """Implicit SPL update in outlet-first order.

    n=1 (fast path): H_new = (elev + fact * H_rcv) / (1 + fact)
    n≠1:             solve H_new + C*(H_new - H_rcv)^n = elev  via Newton-Raphson
    where fact = K*dt*A^m/d  and  C = K*dt*A^m/d^n  (A in m², d in m)
    """
    p = parents[idx]
    if p == nb.int64(idx):
        return
    if mask[idx] == nb.uint8(0):
        return

    dr  = nb.int64(idx) // nb.int64(ncols) - nb.int64(p) // nb.int64(ncols)
    dc  = nb.int64(idx) %  nb.int64(ncols) - nb.int64(p) %  nb.int64(ncols)
    d   = (nb.float64(1.41421356237) if (dr != 0 and dc != 0) else nb.float64(1.0)) * dx
    h_r = h[p]

    if n_exp == nb.float64(1.0):
        fact  = K * (A[idx] ** m_exp) / d * dt
        h_new = (elev[idx] + fact * h_r) / (nb.float64(1.0) + fact)
    else:
        C     = K * (A[idx] ** m_exp) / (d ** n_exp) * dt
        h_new = elev[idx]                          # initial guess
        for _ in range(nb.int64(20)):
            dh = h_new - h_r
            if dh < nb.float64(0.0):
                dh = nb.float64(0.0)
            f  = h_new + C * (dh ** n_exp) - elev[idx]
            fp = nb.float64(1.0) + n_exp * C * (dh ** (n_exp - nb.float64(1.0)))
            h_new -= f / fp
            if f * f < nb.float64(1e-20):
                break

    h[idx] = h_new if h_new > h_r else h_r


# ── main loop ─────────────────────────────────────────────────────────────────

_K     = np.float64(K)
_m     = np.float64(m)
_n     = np.float64(n)
_dt    = np.float64(dt)
_G     = np.float64(G)
_ncols = np.int64(ncols)
_dx    = np.float64(dx)


st = time.perf_counter()
for step in range(N):
    z = pf(z.copy(), mask, nb_fn, heap_ops, queue_ops)

    tree  = make_tree(z, nb_fn, mask, mode='par')
    order = compute_topo_order(tree)

    A[:] = dx * dx   # each cell contributes dx² m²
    tree_traversal_full(order, tree, _catchment, (A,), direction='up')

    # explicit uplift
    tree_traversal_full(order, tree, _uplift, (z, Udt_field), direction='none')

    h_t0 = z.copy()    # elevation at start of erosion sub-step
    h[:] = h_t0
    hp[:] = h_t0
    dh_accum[:] = 0.0

    for gs_iter in range(max_iter):
        # deposition [m] = G * upstream_sediment_volume [m³] / A [m²]
        # upstream_sediment_volume = Σ(erosion_depth * dx²) = upstream_erosion * dx²
        upstream_erosion = dh_accum - (h_t0 - hp)
        elev[:] = h_t0 + _G * np.where(mask > 0, upstream_erosion * dx * dx / A, 0.0)
        elev[mask == 3] = h_t0[mask == 3]   # outlets: fixed

        # implicit SPL update (outlet-first)
        tree_traversal_full(order, tree, _yuan_sweep,
                            (h, elev, A, _K, _m, _n, _dt, _ncols, _dx),
                            direction='down')

        err = np.sqrt(np.sum((h[mask > 0] - hp[mask > 0]) ** 2) / np.count_nonzero(mask))

        # update hp and dh_accum from current h (Fortran order: hp=h, dh=ht-hp)
        hp[:] = h
        dh_accum[:] = h_t0 - hp          # total erosion from t0 to current h
        tree_traversal_full(order, tree, _accum_dh, (dh_accum,), direction='up')

        if G == 0.0 or n == 1.0 or err < tol * max(h_t0.max(), 1e-10):
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
axes[0].set_title(f'Elevation  Yuan et al. (K={K}, m={m}, G={G}, U_max={U_max})')
plt.colorbar(im0, ax=axes[0], label='z [m]')

A_plot = np.where(mask > 0, A, np.nan).reshape(nrows, ncols)
im1 = axes[1].imshow(np.log10(A_plot), cmap='Blues', origin='lower')
axes[1].set_title('log₁₀ drainage area')
plt.colorbar(im1, ax=axes[1], label='log₁₀(cells)')

plt.tight_layout()
plt.show()
