"""
Hergarten (2020) transport-limited SPL implicit solver.

Sediment flux:  Q_i = K_i * A_i^(m_i+1) * S_i^(n_i)
Evolution:      s_i * dH_i/dt = s_i*U_i - Q_i + Σ_j Q_j

All parameters (K, m, n) are flat float64 arrays (spatially variable).

n_i == 1  per node: direct three-sweep solution.
n_i != 1  per node: linearised conductance (outer Newton iterations required).

The caller is responsible for the outer Newton loop when n varies spatially
(check convergence on ||dH|| and repeat step() until satisfied).
"""

import numba as nb
import numpy as np

from nbmdsa.algorithms.tree_traversal import tree_traversal_full


@nb.njit
def _sweep2(idx, parents, child_ptr, child_data, mask, neighbours_fn,
            z, A, Q0, Qp, f_arr, Qp_sum, Q0_sum, K_arr, m_arr, n_arr, dt, ncols, dx):
    """
    Leaf-to-outlet sweep.  n_arr[idx]==1: direct; else linearised at current slope.
    """
    if mask[idx] == nb.uint8(0):
        return
    p = parents[idx]
    if p == nb.int64(idx):
        return

    dr = nb.int64(idx) // nb.int64(ncols) - nb.int64(p) // nb.int64(ncols)
    dc = nb.int64(idx) %  nb.int64(ncols) - nb.int64(p) %  nb.int64(ncols)
    d  = (nb.float64(1.41421356237) if (dr != 0 and dc != 0) else nb.float64(1.0)) * dx

    K_i = K_arr[idx]
    m_i = m_arr[idx]
    n_i = n_arr[idx]

    if n_i == nb.float64(1.0):
        f = K_i * (A[idx] ** (m_i + nb.float64(1.0))) / d
    else:
        dh = z[idx] - z[p]
        if dh < nb.float64(1e-10) * dx:
            dh = nb.float64(1e-10) * dx
        f = n_i * K_i * (A[idx] ** (m_i + nb.float64(1.0))) / (d ** n_i) * (dh ** (n_i - nb.float64(1.0)))

    f_arr[idx] = f
    alpha      = dx * dx - dt * Qp_sum[idx]
    den        = dt + alpha / f
    Qp[idx]    = -alpha / den
    Q0[idx]    = (dt * Q0_sum[idx] + alpha * (z[idx] - z[p])) / den
    Qp_sum[p] += Qp[idx]
    Q0_sum[p] += Q0[idx]


@nb.njit
def _sweep3(idx, parents, child_ptr, child_data, mask, neighbours_fn,
            z, Q0, Qp, f_arr, dH):
    """Outlet-to-leaf sweep: update z.  Identical for all n."""
    p = parents[idx]
    if p == nb.int64(idx):
        dH[idx] = nb.float64(0.0)
        return
    if mask[idx] == nb.uint8(0):
        return

    Q_i = Q0[idx] + Qp[idx] * dH[p]
    if Q_i < nb.float64(0.0):
        Q_i = nb.float64(0.0)
    z_new   = z[p] + Q_i / f_arr[idx]
    dH[idx] = z_new - z[idx]
    z[idx]  = z_new


def step(order, tree, z, A, Q0, Qp, f_arr, Qp_sum, Q0_sum, dH,
         K_arr, m_arr, n_arr, dt, ncols, dx):
    """
    One implicit Hergarten timestep.

    Arrays Q0, Qp, f_arr, Qp_sum, Q0_sum, dH are working buffers (modified in-place).
    z is updated in-place.
    """
    _dt    = np.float64(dt)
    _ncols = np.int64(ncols)
    _dx    = np.float64(dx)

    Qp_sum[:] = 0.0
    Q0_sum[:] = 0.0
    tree_traversal_full(order, tree, _sweep2,
                        (z, A, Q0, Qp, f_arr, Qp_sum, Q0_sum,
                         K_arr, m_arr, n_arr, _dt, _ncols, _dx),
                        direction='up')
    dH[:] = 0.0
    tree_traversal_full(order, tree, _sweep3,
                        (z, Q0, Qp, f_arr, dH),
                        direction='down')
