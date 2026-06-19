"""
Yuan et al. (2019) / Davy-Lague iterative DL+deposition solver.

Erosion rate:   E_i = K_i * A_i^(m_i) * S_i^(n_i) - G_i * Q_i / A_i
Sediment flux:  Q_i = Σ_upstream(E * dx²)

All parameters (K, m, n, G) are flat float64 arrays (spatially variable).

  G_i = 0   → detachment-limited (no iteration needed)
  G_i → ∞   → transport-limited end-member

Gauss-Seidel outer loop: call step() repeatedly until err < tol.
step() returns the RMS elevation change from the previous GS iteration.
"""

import numba as nb
import numpy as np

from nbmdsa.algorithms.tree_traversal import tree_traversal_full


@nb.njit
def _accum_dh(idx, parents, child_ptr, child_data, mask, neighbours_fn, dh):
    p = parents[idx]
    if p != nb.int64(idx):
        dh[p] += dh[idx]


@nb.njit
def _yuan_sweep(idx, parents, child_ptr, child_data, mask, neighbours_fn,
                h, elev, A, K_arr, m_arr, n_arr, dt, ncols, dx):
    """
    Implicit SPL update outlet-first.

    n_i == 1  (fast path): H = (elev + fact*H_rcv) / (1 + fact)
    n_i != 1: Newton-Raphson solving H + C*(H-H_rcv)^n = elev
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

    K_i = K_arr[idx]
    m_i = m_arr[idx]
    n_i = n_arr[idx]

    if n_i == nb.float64(1.0):
        fact  = K_i * (A[idx] ** m_i) / d * dt
        h_new = (elev[idx] + fact * h_r) / (nb.float64(1.0) + fact)
    else:
        C     = K_i * (A[idx] ** m_i) / (d ** n_i) * dt
        h_new = elev[idx]
        for _ in range(nb.int64(20)):
            dh = h_new - h_r
            if dh < nb.float64(0.0):
                dh = nb.float64(0.0)
            f  = h_new + C * (dh ** n_i) - elev[idx]
            fp = nb.float64(1.0) + n_i * C * (dh ** (n_i - nb.float64(1.0)))
            h_new -= f / fp
            if f * f < nb.float64(1e-20):
                break

    h[idx] = h_new if h_new > h_r else h_r


@nb.njit
def _yuan_sweep_sed(idx, parents, child_ptr, child_data, mask, neighbours_fn,
                    h, elev, A, K_sed_arr, K_b_arr, h_sed_t0, h_t0,
                    m_arr, n_arr, dt, ncols, dx):
    """
    Implicit SPL update with sediment-depth-weighted erodibility.

    K_eff = K_sed if erosion stays within h_sed, K_b if pure bedrock,
    weighted average if the erosion straddles the sediment/bedrock interface.
    h_sed_t0 and h_t0 are frozen at the start of the timestep.
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

    m_i = m_arr[idx]
    n_i = n_arr[idx]

    # effective K weighted by how much of the erosion cuts into bedrock
    dz_est = h_t0[idx] - h_r
    hs     = h_sed_t0[idx]
    K_sed_i = K_sed_arr[idx]
    K_b_i   = K_b_arr[idx]
    if dz_est <= nb.float64(0.0) or hs >= dz_est:
        K_i = K_sed_i
    elif hs <= nb.float64(0.0):
        K_i = K_b_i
    else:
        K_i = (hs * K_sed_i + (dz_est - hs) * K_b_i) / dz_est

    if n_i == nb.float64(1.0):
        fact  = K_i * (A[idx] ** m_i) / d * dt
        h_new = (elev[idx] + fact * h_r) / (nb.float64(1.0) + fact)
    else:
        C     = K_i * (A[idx] ** m_i) / (d ** n_i) * dt
        h_new = elev[idx]
        for _ in range(nb.int64(20)):
            dh = h_new - h_r
            if dh < nb.float64(0.0):
                dh = nb.float64(0.0)
            f  = h_new + C * (dh ** n_i) - elev[idx]
            fp = nb.float64(1.0) + n_i * C * (dh ** (n_i - nb.float64(1.0)))
            h_new -= f / fp
            if f * f < nb.float64(1e-20):
                break

    h[idx] = h_new if h_new > h_r else h_r


def step_with_sed(order, tree, h, hp, elev, dh_accum, A, h_t0, h_sed_t0,
                  K_sed_arr, K_b_arr, m_arr, n_arr, G_arr, dt, ncols, dx, mask):
    """
    One Gauss-Seidel iteration of the Yuan solver with sediment-layer erodibility.

    h_sed_t0 : sediment thickness frozen at the start of the timestep [m]
    K_sed_arr: erodibility for the sediment layer
    K_b_arr  : erodibility for bedrock

    h_sed_t0 is read-only here; caller updates h_sed after GS convergence:
        h_sed[:] = np.maximum(0, h_sed - np.maximum(0, h_t0_before - h_after))

    Returns RMS elevation change ||h - hp|| / sqrt(n_interior).
    """
    _dt    = np.float64(dt)
    _dx    = np.float64(dx)
    _ncols = np.int64(ncols)

    upstream_erosion = np.maximum(0.0, dh_accum - (h_t0 - hp))
    interior = mask > 0
    elev[interior]  = h_t0[interior] + G_arr[interior] * upstream_erosion[interior] * dx * dx / A[interior]
    elev[~interior] = h_t0[~interior]
    elev[mask == 3] = h_t0[mask == 3]

    tree_traversal_full(order, tree, _yuan_sweep_sed,
                        (h, elev, A, K_sed_arr, K_b_arr, h_sed_t0, h_t0,
                         m_arr, n_arr, _dt, _ncols, _dx),
                        direction='down')

    n_interior = int(interior.sum())
    err = np.sqrt(np.sum((h[interior] - hp[interior]) ** 2) / n_interior) if n_interior > 0 else 0.0

    hp[:] = h
    dh_accum[:] = h_t0 - hp
    tree_traversal_full(order, tree, _accum_dh, (dh_accum,), direction='up')

    return err


def step(order, tree, h, hp, elev, dh_accum, A, h_t0,
         K_arr, m_arr, n_arr, G_arr, dt, ncols, dx, mask):
    """
    One Gauss-Seidel iteration of the Yuan solver.

    h       : current elevation estimate (modified in-place)
    hp      : previous GS estimate (updated in-place after sweep)
    elev    : working buffer for effective elevation (modified in-place)
    dh_accum: cumulative upstream erosion buffer (modified in-place)
    h_t0    : elevation at start of timestep (read-only)
    G_arr   : deposition coefficient per cell

    Returns RMS elevation change ||h - hp|| / sqrt(n_interior).
    """
    _dt    = np.float64(dt)
    _dx    = np.float64(dx)
    _ncols = np.int64(ncols)

    upstream_erosion = np.maximum(0.0, dh_accum - (h_t0 - hp))
    interior = mask > 0
    elev[interior]  = h_t0[interior] + G_arr[interior] * upstream_erosion[interior] * dx * dx / A[interior]
    elev[~interior] = h_t0[~interior]
    elev[mask == 3] = h_t0[mask == 3]

    tree_traversal_full(order, tree, _yuan_sweep,
                        (h, elev, A, K_arr, m_arr, n_arr, _dt, _ncols, _dx),
                        direction='down')

    n_interior = int(interior.sum())
    err = np.sqrt(np.sum((h[interior] - hp[interior]) ** 2) / n_interior) if n_interior > 0 else 0.0

    hp[:] = h
    dh_accum[:] = h_t0 - hp
    tree_traversal_full(order, tree, _accum_dh, (dh_accum,), direction='up')

    return err
