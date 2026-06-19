"""
Hergarten (2020) transport-limited SPL — minimal implicit solver.

Sediment flux:  Q_i = K * A_i^(m+1) * S_i^n
Evolution:      s_i * dH_i/dt = s_i*U - Q_i + Σ_j Q_j

n=1  — direct three-sweep solver (Sect. 3, Hergarten 2020):

  Sweep 2  (leaves → outlets, direction='up'):
    f_i     = K * A_i^(m+1) / d_i               [A in m², d in m]
    alpha_i = s - dt * Σ Q'_j                   [Eq. 20, s=1]
    den_i   = dt + alpha_i / f_i
    Q'_i    = -alpha_i / den_i                   [Eq. 27]
    Q⁰_i    = (dt * Σ Q⁰_j + alpha_i*(H_i-H_b)) / den_i  [Eq. 26]

  Sweep 3  (outlets → leaves, direction='down'):
    Q_i(t)  = Q⁰_i + Q'_i * dH_b               [Eq. 17]
    H_i(t)  = H_b(t) + Q_i(t) / f_i            [Eq. 23]

n≠1  — outer Newton iteration over the same three sweeps:
    f_eff_i = n * K * A_i^(m+1) / d_i^n * ΔH_i^(n-1)   (linearised at current slope)
    Repeat sweep2_n + sweep3 until ||ΔH|| < outer_tol  (quadratic convergence).
    sweep3 is identical for both cases (H = H_b + Q/f_arr).

Uplift is applied explicitly before the sweeps (operator splitting).
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

K         = 5e-5   # erodibility
m         = 0.45    # area exponent
n         = 1.    # slope exponent (1 = direct solver; else outer Newton iterations)
U_max     = 1e-3   # peak uplift rate [m/yr]
dt        = 1e3    # time step [yr]
dx        = 100.0  # cell size [m]
N         = 5000    # number of time steps
outer_max = 10     # max outer Newton iterations (only used when n≠1)
outer_tol = 1e-6   # outer Newton convergence tolerance [m]
nrows     = 256
ncols     = 256


# ── grid ──────────────────────────────────────────────────────────────────────

nn  = nrows * ncols
rng = np.random.default_rng(42)
z   = rng.random(nn) * 0.01           # small white-noise initial topography

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
Udt_field = U_field * dt


# ── working arrays ────────────────────────────────────────────────────────────

A      = np.full(nn, dx * dx, dtype=np.float64)   # catchment area [m²]
Q0     = np.zeros(nn, dtype=np.float64)
Qp     = np.zeros(nn, dtype=np.float64)
f_arr  = np.zeros(nn, dtype=np.float64)
dH     = np.zeros(nn, dtype=np.float64)
Qp_sum = np.zeros(nn, dtype=np.float64)
Q0_sum = np.zeros(nn, dtype=np.float64)


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
def _sweep2(idx, parents, child_ptr, child_data, mask, neighbours_fn,
            z, A, Q0, Qp, f_arr, Qp_sum, Q0_sum, K, m_exp, dt, ncols, dx):
    """n=1 sweep 2: direct Q⁰ and Q' (leaf → outlet)."""
    if mask[idx] == nb.uint8(0):
        return
    p = parents[idx]
    if p == nb.int64(idx):
        return

    dr = nb.int64(idx) // nb.int64(ncols) - nb.int64(p) // nb.int64(ncols)
    dc = nb.int64(idx) %  nb.int64(ncols) - nb.int64(p) %  nb.int64(ncols)
    d  = (nb.float64(1.41421356237) if (dr != 0 and dc != 0) else nb.float64(1.0)) * dx

    f          = K * (A[idx] ** (m_exp + nb.float64(1.0))) / d
    f_arr[idx] = f
    alpha      = dx * dx - dt * Qp_sum[idx]   # s_i = dx² (Eq. 20)
    den        = dt + alpha / f
    Qp[idx]    = -alpha / den
    Q0[idx]    = (dt * Q0_sum[idx] + alpha * (z[idx] - z[p])) / den
    Qp_sum[p] += Qp[idx]
    Q0_sum[p] += Q0[idx]


@nb.njit
def _sweep2_n(idx, parents, child_ptr, child_data, mask, neighbours_fn,
              z, A, Q0, Qp, f_arr, Qp_sum, Q0_sum, K, m_exp, n_exp, dt, ncols, dx):
    """n≠1 sweep 2: linearise at current slope → effective f, same Q⁰/Q' formulas."""
    if mask[idx] == nb.uint8(0):
        return
    p = parents[idx]
    if p == nb.int64(idx):
        return

    dr = nb.int64(idx) // nb.int64(ncols) - nb.int64(p) // nb.int64(ncols)
    dc = nb.int64(idx) %  nb.int64(ncols) - nb.int64(p) %  nb.int64(ncols)
    d  = (nb.float64(1.41421356237) if (dr != 0 and dc != 0) else nb.float64(1.0)) * dx

    dh = z[idx] - z[p]
    if dh < nb.float64(1e-10) * dx:
        dh = nb.float64(1e-10) * dx          # floor to avoid 0^(n-1) singularity

    # tangent conductance: f_eff = n * K * A^(m+1) / d^n * ΔH^(n-1)
    f_eff      = n_exp * K * (A[idx] ** (m_exp + nb.float64(1.0))) / (d ** n_exp) * (dh ** (n_exp - nb.float64(1.0)))
    f_arr[idx] = f_eff
    alpha      = dx * dx - dt * Qp_sum[idx]   # s_i = dx² (Eq. 20)
    den        = dt + alpha / f_eff
    Qp[idx]    = -alpha / den
    Q0[idx]    = (dt * Q0_sum[idx] + alpha * (z[idx] - z[p])) / den
    Qp_sum[p] += Qp[idx]
    Q0_sum[p] += Q0[idx]


@nb.njit
def _sweep3(idx, parents, child_ptr, child_data, mask, neighbours_fn,
            z, Q0, Qp, f_arr, dH):
    """Sweep 3: update z outlet → leaf.  Identical for n=1 and n≠1."""
    p = parents[idx]
    if p == nb.int64(idx):
        dH[idx] = nb.float64(0.0)
        return
    if mask[idx] == nb.uint8(0):
        return

    Q_i = Q0[idx] + Qp[idx] * dH[p]
    if Q_i < nb.float64(0.0):
        Q_i = nb.float64(0.0)      # flux can't be negative (no uphill transport)
    z_new   = z[p] + Q_i / f_arr[idx]
    dH[idx] = z_new - z[idx]
    z[idx]  = z_new


# ── main loop ─────────────────────────────────────────────────────────────────

_K     = np.float64(K)
_m     = np.float64(m)
_n     = np.float64(n)
_dt    = np.float64(dt)
_ncols = np.int64(ncols)
_dx    = np.float64(dx)


st = time.perf_counter()
for step in range(N):

    if (step+1) % 10 == 0: 
        z = pf(z.copy(), mask, nb_fn, heap_ops, queue_ops)

    tree  = make_tree(z, nb_fn, mask, mode='par')
    order = compute_topo_order(tree)

    A[:] = dx * dx
    tree_traversal_full(order, tree, _catchment, (A,), direction='up')

    tree_traversal_full(order, tree, _uplift, (z, Udt_field), direction='none')

    if n == 1.0:
        # ── direct three-sweep solver (n=1) ──────────────────────────────────
        Qp_sum[:] = 0.0
        Q0_sum[:] = 0.0
        tree_traversal_full(order, tree, _sweep2,
                            (z, A, Q0, Qp, f_arr, Qp_sum, Q0_sum, _K, _m, _dt, _ncols, _dx),
                            direction='up')
        dH[:] = 0.0
        tree_traversal_full(order, tree, _sweep3,
                            (z, Q0, Qp, f_arr, dH),
                            direction='down')
    else:
        # ── outer Newton iterations (n≠1) ────────────────────────────────────
        z_prev = z.copy()
        for outer in range(outer_max):
            Qp_sum[:] = 0.0
            Q0_sum[:] = 0.0
            tree_traversal_full(order, tree, _sweep2_n,
                                (z, A, Q0, Qp, f_arr, Qp_sum, Q0_sum, _K, _m, _n, _dt, _ncols, _dx),
                                direction='up')
            dH[:] = 0.0
            tree_traversal_full(order, tree, _sweep3,
                                (z, Q0, Qp, f_arr, dH),
                                direction='down')
            interior = mask > 0
            err = np.sqrt(np.sum((z[interior] - z_prev[interior]) ** 2) / interior.sum())
            z_prev[:] = z
            if err < outer_tol:
                break

    if step % 50 == 0:
        interior = mask > 0
        iters_str = '' if n == 1.0 else f'  outer={outer+1}'
        print(f"step {step:4d}  max_z={z[interior].max():.3f}  mean_z={z[interior].mean():.3f}{iters_str}")
print('took', time.perf_counter() - st)


# ── plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(z.reshape(nrows, ncols), cmap='terrain', origin='lower')
axes[0].set_title(f'Elevation  Hergarten TL (K={K}, m={m}, n={n}, U_max={U_max})')
plt.colorbar(im0, ax=axes[0], label='z [m]')

A_plot = np.where(mask > 0, A, np.nan).reshape(nrows, ncols)
im1 = axes[1].imshow(np.log10(A_plot), cmap='Blues', origin='lower')
axes[1].set_title('log₁₀ drainage area')
plt.colorbar(im1, ax=axes[1], label='log₁₀(cells)')

plt.tight_layout()
plt.show()
