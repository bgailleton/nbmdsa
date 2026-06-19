"""
Landslide triggering (Culmann/Densmore) and sediment runout (Carretier nonlocal).

Campforts et al. (2020) HyLands formulation.

trigger()
    Identifies unstable cells, stochastically selects critical cells, expands
    each landslide via Culmann stack, erodes z in-place, returns sed_depth map
    + per-event trigger nodes and total eroded height.

runout_landslides()
    Routes each trigger's sediment flux downstream using nonlinear nonlocal
    deposition (Carretier et al. 2016) and deposits it into z.

    Deposition fraction per cell:
        f_dep = clip( (1 - (S/Sc)²) / lambda_dist,  0, 1 )

    lambda_dist > 1 → longer runout (less deposition per cell)
    lambda_dist < 1 → shorter runout (more deposition per cell)
    lambda_dist = 1 → standard HyLands behaviour
"""

import numpy as np
import numba as nb

from nbmdsa.algorithms.tree_traversal import tree_traversal_full
from nbmdsa.algorithms.mfd_traversal  import mfd_traversal_full


# ── slope / hillslope-height helpers ─────────────────────────────────────────

@nb.njit
def _sfd_beta_hs(order, parents, child_ptr, child_data, z, mask, ncols, dx,
                 beta_arr, hs_arr):
    """
    SFD: β from the tree receiver (steepest-descent parent); H_s = max over
    tree donors (children).  Requires tree built with mode='full'.
    """
    for oi in range(nb.int64(len(order))):
        idx = order[oi]
        if mask[idx] == nb.uint8(0):
            continue
        p = parents[idx]
        if p == nb.int64(idx):
            beta_arr[idx] = nb.float64(0.0)
            hs_arr[idx]   = nb.float64(0.0)
            continue
        dr = nb.int64(idx) // nb.int64(ncols) - nb.int64(p) // nb.int64(ncols)
        dc = nb.int64(idx) %  nb.int64(ncols) - nb.int64(p) %  nb.int64(ncols)
        d  = (nb.float64(1.41421356237) if (dr != 0 and dc != 0)
              else nb.float64(1.0)) * dx
        dz = z[idx] - z[p]
        beta_arr[idx] = np.arctan(dz / d if dz > nb.float64(0.0)
                                  else nb.float64(0.0))
        hs = nb.float64(0.0)
        c0 = child_ptr[idx]
        c1 = child_ptr[idx + nb.int64(1)]
        for ci in range(c0, c1):
            dz2 = z[child_data[ci]] - z[idx]
            if dz2 > hs:
                hs = dz2
        hs_arr[idx] = hs


@nb.njit
def _mfd_beta_hs(z, mask, nb_fn, ncols, dx, beta_arr, hs_arr):
    """
    MFD: β = steepest descent over all lower neighbours; H_s = max elevation
    above any upslope neighbour.
    """
    nbuf = np.empty(nb.int64(8), nb.int64)
    for i in range(nb.int64(len(z))):
        if mask[i] == nb.uint8(0):
            continue
        nb_fn(nb.int64(i), nbuf)
        max_s = nb.float64(0.0)
        hs    = nb.float64(0.0)
        for k in range(nb.int64(8)):
            j = nbuf[k]
            if j < nb.int64(0) or mask[j] == nb.uint8(0):
                continue
            dr = nb.int64(i) // nb.int64(ncols) - nb.int64(j) // nb.int64(ncols)
            dc = nb.int64(i) %  nb.int64(ncols) - nb.int64(j) %  nb.int64(ncols)
            d  = (nb.float64(1.41421356237) if (dr != 0 and dc != 0)
                  else nb.float64(1.0)) * dx
            dz_d = z[i] - z[j]
            if dz_d > nb.float64(0.0) and dz_d / d > max_s:
                max_s = dz_d / d
            dz_u = z[j] - z[i]
            if dz_u > hs:
                hs = dz_u
        beta_arr[i] = np.arctan(max_s)
        hs_arr[i]   = hs


# ── landslide expansion ───────────────────────────────────────────────────────

@nb.njit
def _expand_sfd(z, mask, parents, child_ptr, child_data, ncols, dx,
                crit_idx, theta_c, tan_phi, sed_depth):
    """Culmann stack expansion following tree donors only (SFD).

    Failure zone defined by theta_c; residual slope set to phi so the
    post-slide surface is stable (beta == phi, not beta == theta_c > phi).
    """
    n      = nb.int64(len(z))
    z_c    = z[crit_idx]
    row_c  = nb.int64(crit_idx) // nb.int64(ncols)
    col_c  = nb.int64(crit_idx) %  nb.int64(ncols)
    tan_tc = np.tan(theta_c)

    vis   = np.zeros(n, nb.uint8)
    stack = np.empty(n, nb.int64)
    top   = nb.int64(0)

    stack[top] = nb.int64(crit_idx)
    top += nb.int64(1)
    vis[crit_idx] = nb.uint8(1)
    total = nb.float64(0.0)

    while top > nb.int64(0):
        top   -= nb.int64(1)
        idx    = stack[top]
        row_i  = nb.int64(idx) // nb.int64(ncols)
        col_i  = nb.int64(idx) %  nb.int64(ncols)
        d_h    = np.sqrt(nb.float64((row_i - row_c) ** 2 +
                                    nb.float64((col_i - col_c) ** 2))) * dx
        z_pln  = z_c + d_h * tan_phi   # residual slope at phi, not theta_c
        eroded = z[idx] - z_pln
        if eroded > nb.float64(0.0):
            sed_depth[idx] += eroded
            total          += eroded
            z[idx]          = z_pln

        c0 = child_ptr[idx]
        c1 = child_ptr[idx + nb.int64(1)]
        for ci in range(c0, c1):
            j = child_data[ci]
            if mask[j] == nb.uint8(0) or vis[j] != nb.uint8(0):
                continue
            row_j = nb.int64(j) // nb.int64(ncols)
            col_j = nb.int64(j) %  nb.int64(ncols)
            dj    = np.sqrt(nb.float64((row_j - row_c) ** 2 +
                                       nb.float64((col_j - col_c) ** 2))) * dx
            if z[j] > z_c + dj * tan_tc:   # expansion criterion unchanged
                vis[j]     = nb.uint8(1)
                stack[top] = nb.int64(j)
                top       += nb.int64(1)
    return total


@nb.njit
def _expand_mfd(z, mask, nb_fn, ncols, dx, crit_idx, theta_c, tan_phi, sed_depth):
    """Culmann stack expansion over all upslope neighbours (MFD).

    Failure zone defined by theta_c; residual slope set to phi so the
    post-slide surface is stable (beta == phi, not beta == theta_c > phi).
    """
    n      = nb.int64(len(z))
    z_c    = z[crit_idx]
    row_c  = nb.int64(crit_idx) // nb.int64(ncols)
    col_c  = nb.int64(crit_idx) %  nb.int64(ncols)
    tan_tc = np.tan(theta_c)

    vis   = np.zeros(n, nb.uint8)
    stack = np.empty(n, nb.int64)
    nbuf  = np.empty(nb.int64(8), nb.int64)
    top   = nb.int64(0)

    stack[top] = nb.int64(crit_idx)
    top += nb.int64(1)
    vis[crit_idx] = nb.uint8(1)
    total = nb.float64(0.0)

    while top > nb.int64(0):
        top   -= nb.int64(1)
        idx    = stack[top]
        row_i  = nb.int64(idx) // nb.int64(ncols)
        col_i  = nb.int64(idx) %  nb.int64(ncols)
        d_h    = np.sqrt(nb.float64((row_i - row_c) ** 2 +
                                    nb.float64((col_i - col_c) ** 2))) * dx
        z_pln  = z_c + d_h * tan_phi   # residual slope at phi, not theta_c
        eroded = z[idx] - z_pln
        if eroded > nb.float64(0.0):
            sed_depth[idx] += eroded
            total          += eroded
            z[idx]          = z_pln

        nb_fn(nb.int64(idx), nbuf)
        for k in range(nb.int64(8)):
            j = nbuf[k]
            if j < nb.int64(0) or mask[j] == nb.uint8(0) or vis[j] != nb.uint8(0):
                continue
            row_j = nb.int64(j) // nb.int64(ncols)
            col_j = nb.int64(j) %  nb.int64(ncols)
            dj    = np.sqrt(nb.float64((row_j - row_c) ** 2 +
                                       nb.float64((col_j - col_c) ** 2))) * dx
            if z[j] > z_c + dj * tan_tc:   # expansion criterion unchanged
                vis[j]     = nb.uint8(1)
                stack[top] = nb.int64(j)
                top       += nb.int64(1)
    return total


# ── runout kernels ────────────────────────────────────────────────────────────

@nb.njit
def _runout_sfd_kernel(idx, parents, child_ptr, child_data, mask, neighbours_fn,
                        Q_sed, dep_map, z_orig, dx, Sc_sq, lam, tan_phi, ncols):
    if mask[idx] == nb.uint8(0):
        return
    Q_total = Q_sed[idx]
    if Q_total <= nb.float64(0.0):
        return
    p = parents[idx]
    if p == nb.int64(idx):
        dep_map[idx] += Q_total / (dx * dx)
        return

    # angle-of-repose ceiling from already-processed upstream donors (children)
    max_z_new = nb.float64(1e18)
    c0 = child_ptr[idx]
    c1 = child_ptr[idx + nb.int64(1)]
    for ci in range(c0, c1):
        j = child_data[ci]
        if mask[j] == nb.uint8(0):
            continue
        dr = nb.int64(idx) // nb.int64(ncols) - nb.int64(j) // nb.int64(ncols)
        dc = nb.int64(idx) %  nb.int64(ncols) - nb.int64(j) %  nb.int64(ncols)
        d  = (nb.float64(1.41421356237) if (dr != 0 and dc != 0)
              else nb.float64(1.0)) * dx
        ceiling = z_orig[j] + dep_map[j] + d * tan_phi
        if ceiling < max_z_new:
            max_z_new = ceiling
    max_dep_h = max_z_new - z_orig[idx]
    if max_dep_h < nb.float64(0.0):
        max_dep_h = nb.float64(0.0)

    dr  = nb.int64(idx) // nb.int64(ncols) - nb.int64(p) // nb.int64(ncols)
    dc  = nb.int64(idx) %  nb.int64(ncols) - nb.int64(p) %  nb.int64(ncols)
    d   = (nb.float64(1.41421356237) if (dr != 0 and dc != 0)
           else nb.float64(1.0)) * dx
    S   = max(nb.float64(0.0), (z_orig[idx] - z_orig[p]) / d)
    f_dep = (nb.float64(1.0) - S * S / Sc_sq) / lam
    if f_dep < nb.float64(0.0):
        f_dep = nb.float64(0.0)
    if f_dep > nb.float64(1.0):
        f_dep = nb.float64(1.0)

    dep_h        = min(Q_total * f_dep / (dx * dx), max_dep_h)
    dep_map[idx] += dep_h
    Q_sed[p]     += Q_total - dep_h * dx * dx


@nb.njit
def _runout_mfd_kernel(idx, z_orig, mask, nb_fn,
                        Q_sed, dep_map, dx, Sc_sq, lam, tan_phi, ncols):
    if mask[idx] == nb.uint8(0):
        return
    Q_total = Q_sed[idx]
    if Q_total <= nb.float64(0.0):
        return

    nbuf = np.empty(nb.int64(8), nb.int64)
    nb_fn(nb.int64(idx), nbuf)

    # pass 1: max_S for f_dep (z_orig) + repose ceiling from upslope neighbors
    max_S     = nb.float64(0.0)
    max_z_new = nb.float64(1e18)
    has_down  = nb.uint8(0)

    for k in range(nb.int64(8)):
        j = nbuf[k]
        if j < nb.int64(0) or mask[j] == nb.uint8(0):
            continue
        dr = nb.int64(idx) // nb.int64(ncols) - nb.int64(j) // nb.int64(ncols)
        dc = nb.int64(idx) %  nb.int64(ncols) - nb.int64(j) %  nb.int64(ncols)
        d  = (nb.float64(1.41421356237) if (dr != 0 and dc != 0)
              else nb.float64(1.0)) * dx
        dz = z_orig[idx] - z_orig[j]
        if dz > nb.float64(0.0):
            s = dz / d
            if s > max_S:
                max_S = s
            has_down = nb.uint8(1)
        else:
            ceiling = z_orig[j] + dep_map[j] + d * tan_phi
            if ceiling < max_z_new:
                max_z_new = ceiling

    max_dep_h = max_z_new - z_orig[idx]
    if max_dep_h < nb.float64(0.0):
        max_dep_h = nb.float64(0.0)

    if has_down == nb.uint8(0):
        dep_map[idx] += min(Q_total / (dx * dx), max_dep_h)
        return

    f_dep = (nb.float64(1.0) - max_S * max_S / Sc_sq) / lam
    if f_dep < nb.float64(0.0):
        f_dep = nb.float64(0.0)
    if f_dep > nb.float64(1.0):
        f_dep = nb.float64(1.0)

    dep_h         = min(Q_total * f_dep / (dx * dx), max_dep_h)
    dep_map[idx] += dep_h
    Q_out         = Q_total - dep_h * dx * dx

    if Q_out <= nb.float64(0.0):
        return

    # pass 2: route Q_out using post-deposit elevation of idx,
    # only to originally-downslope (= not-yet-processed) neighbors
    z_new     = z_orig[idx] + dep_h
    slopes    = np.empty(nb.int64(8), nb.float64)
    slope_sum = nb.float64(0.0)

    for k in range(nb.int64(8)):
        slopes[k] = nb.float64(0.0)
        j = nbuf[k]
        if j < nb.int64(0) or mask[j] == nb.uint8(0):
            continue
        if z_orig[j] >= z_orig[idx]:   # originally upslope = already processed
            continue
        dr = nb.int64(idx) // nb.int64(ncols) - nb.int64(j) // nb.int64(ncols)
        dc = nb.int64(idx) %  nb.int64(ncols) - nb.int64(j) %  nb.int64(ncols)
        d  = (nb.float64(1.41421356237) if (dr != 0 and dc != 0)
              else nb.float64(1.0)) * dx
        slopes[k] = (z_new - z_orig[j]) / d   # always > 0 since z_new > z_orig[idx] > z_orig[j]
        slope_sum += slopes[k]

    if slope_sum <= nb.float64(0.0):
        dep_map[idx] += Q_out / (dx * dx)
        return

    for k in range(nb.int64(8)):
        if slopes[k] > nb.float64(0.0):
            Q_sed[nbuf[k]] += Q_out * slopes[k] / slope_sum


# ── public API ────────────────────────────────────────────────────────────────

def trigger(z, mask, nb_fn, order, tree, dx, C, phi, rho, t_LS, dt, ncols,
            g=9.81, sfd=True):
    """
    Culmann landslide trigger.

    Modifies z in-place (erodes failed cells to the sliding plane).

    sfd=True  requires tree built with mode='full'.
    sfd=False uses all-neighbour MFD slope and expansion.

    Parameters
    ----------
    z, mask         : flat grid arrays
    nb_fn           : neighbours closure
    order           : topological order (leaves-first) from compute_topo_order
    tree            : Tree namedtuple; needs mode='full' when sfd=True
    dx              : cell size [m]
    C               : cohesion [Pa]
    phi             : angle of internal friction [rad]
    rho             : rock density [kg m-3]
    t_LS            : landslide return time [yr]
    dt              : timestep [yr]
    ncols           : number of grid columns
    g               : gravity [m s-2]
    sfd             : routing mode

    Returns
    -------
    sed_depth        : float64[:] eroded thickness per cell [m]
    trigger_nodes    : int64[:]   trigger-cell index per event
    trigger_heights  : float64[:] sum of eroded depths per event [m]
    """
    n        = len(z)
    beta_arr = np.zeros(n, np.float64)
    hs_arr   = np.zeros(n, np.float64)

    if sfd:
        _sfd_beta_hs(order, tree.parents, tree.child_ptr, tree.child_data,
                     z, mask, np.int64(ncols), np.float64(dx), beta_arr, hs_arr)
    else:
        _mfd_beta_hs(z, mask, nb_fn, np.int64(ncols), np.float64(dx),
                     beta_arr, hs_arr)

    cos_phi  = np.cos(phi)
    unstable = (beta_arr > phi) & (mask > 0)
    denom    = 1.0 - np.cos(beta_arr - phi)
    valid    = unstable & (denom > 1e-12)

    Hc        = np.full(n, np.inf, np.float64)
    Hc[valid] = (4.0 * C / (rho * g)) * (
        np.sin(beta_arr[valid]) * cos_phi / denom[valid])

    p_LS        = np.zeros(n, np.float64)
    p_LS[valid] = np.clip(hs_arr[valid] / np.maximum(Hc[valid], 1e-10),
                          0.0, 1.0)

    r         = np.random.random(n)
    triggered = valid & (r < p_LS * (dt / t_LS))
    critical  = np.where(triggered)[0].astype(np.int64)

    sed_depth = np.zeros(n, np.float64)
    t_nodes   = []
    t_heights = []

    tan_phi = np.float64(np.tan(phi))
    for c in critical:
        theta_c = (beta_arr[c] + phi) / 2.0
        if sfd:
            h = _expand_sfd(z, mask, tree.parents, tree.child_ptr, tree.child_data,
                             np.int64(ncols), np.float64(dx),
                             nb.int64(c), theta_c, tan_phi, sed_depth)
        else:
            h = _expand_mfd(z, mask, nb_fn, np.int64(ncols), np.float64(dx),
                             nb.int64(c), theta_c, tan_phi, sed_depth)
        if h > 0.0:
            t_nodes.append(int(c))
            t_heights.append(float(h))

    return (sed_depth,
            np.array(t_nodes,   dtype=np.int64),
            np.array(t_heights, dtype=np.float64))


def runout_landslides(trigger_nodes, trigger_heights, z, h_sed, mask, nb_fn,
                       order, tree, mfd_order, dx, phi, ncols,
                       lambda_dist=1.0, sfd=True):
    """
    Route landslide sediment and deposit with nonlinear nonlocal transport.

    Modifies z in-place (adds deposited sediment thickness).

    Parameters
    ----------
    trigger_nodes    : int64[:]   from trigger()
    trigger_heights  : float64[:] from trigger() — eroded depth-sum per event [m]
    z, mask          : flat grid arrays
    nb_fn            : neighbours closure
    order            : topo order for tree traversal (sfd=True)
    tree             : Tree namedtuple
    mfd_order        : MFD topo order from mfd_topo_order() (sfd=False only)
    dx               : cell size [m]
    phi              : angle of internal friction [rad]; S_c = tan(phi)
    ncols            : number of grid columns
    lambda_dist      : transport-length scaling (default 1.0)
    sfd              : True → SFD tree routing; False → MFD routing

    Returns
    -------
    dep_map : float64[:] deposited thickness per cell [m]
    """
    if len(trigger_nodes) == 0:
        return np.zeros(len(z), np.float64)

    n       = len(z)
    tan_phi = np.float64(np.tan(phi))
    Sc_sq   = np.float64(tan_phi ** 2)
    dx_f    = np.float64(dx)
    lam     = np.float64(lambda_dist)
    ncols_  = np.int64(ncols)

    Q_sed   = np.zeros(n, np.float64)
    dep_map = np.zeros(n, np.float64)

    for nd, ht in zip(trigger_nodes, trigger_heights):
        Q_sed[nd] += ht * dx * dx

    z_orig = z.copy()

    if sfd:
        tree_traversal_full(order, tree, _runout_sfd_kernel,
                            (Q_sed, dep_map, z_orig, dx_f, Sc_sq, lam, tan_phi, ncols_),
                            direction='up')
    else:
        mfd_traversal_full(mfd_order, z_orig, mask, nb_fn, _runout_mfd_kernel,
                           (Q_sed, dep_map, dx_f, Sc_sq, lam, tan_phi, ncols_),
                           direction='up')

    z     += dep_map
    h_sed += dep_map   # deposits are pure sediment
    return dep_map


# ── stochastic particle runout ─────────────────────────────────────────────

@nb.njit
def _run_particles(z, mask, nb_fn, sed_depth, dep_map, dx, Sc_sq, ncols,
                   h_particle, h_base, p_power, lam, max_steps, seed):
    if seed >= nb.int64(0):
        np.random.seed(seed)

    nbuf    = np.empty(nb.int64(8), nb.int64)
    weights = np.empty(nb.int64(8), nb.float64)
    n       = nb.int64(len(z))

    for start_idx in range(n):
        if mask[start_idx] == nb.uint8(0) or sed_depth[start_idx] <= nb.float64(0.0):
            continue

        h_total = sed_depth[start_idx]
        n_parts = nb.int64(h_total / h_particle)

        for pi in range(n_parts + nb.int64(1)):
            if pi < n_parts:
                h_carry = h_particle
            else:
                h_carry = h_total - nb.float64(n_parts) * h_particle
                if h_carry <= nb.float64(1e-12):
                    break

            pos = nb.int64(start_idx)

            for _step in range(max_steps):
                if h_carry <= nb.float64(1e-12):
                    break

                # compute max local slope from pre-deposit z (outlets excluded)
                nb_fn(nb.int64(pos), nbuf)
                max_S = nb.float64(0.0)
                for k in range(nb.int64(8)):
                    j = nbuf[k]
                    if j < nb.int64(0) or mask[j] == nb.uint8(0) or mask[j] == nb.uint8(3):
                        continue
                    dr = nb.int64(pos) // nb.int64(ncols) - nb.int64(j) // nb.int64(ncols)
                    dc = nb.int64(pos) %  nb.int64(ncols) - nb.int64(j) %  nb.int64(ncols)
                    d  = (nb.float64(1.41421356237) if (dr != nb.int64(0) and dc != nb.int64(0))
                          else nb.float64(1.0)) * dx
                    dz = z[pos] - z[j]
                    if dz > nb.float64(0.0):
                        s = dz / d
                        if s > max_S:
                            max_S = s

                # step a: stochastic baseline deposit (lam scales transport like step b)
                d_rand = np.random.random() * h_base / lam
                if d_rand > h_carry:
                    d_rand = h_carry
                dep_map[pos] += d_rand
                z[pos]       += d_rand
                h_carry      -= d_rand

                if h_carry <= nb.float64(1e-12):
                    break

                # step b: slope-based deposit
                f_dep = (nb.float64(1.0) - max_S * max_S / Sc_sq) / lam
                if f_dep < nb.float64(0.0):
                    f_dep = nb.float64(0.0)
                if f_dep > nb.float64(1.0):
                    f_dep = nb.float64(1.0)
                d_slope       = f_dep * h_carry
                dep_map[pos] += d_slope
                z[pos]       += d_slope
                h_carry      -= d_slope

                if h_carry <= nb.float64(1e-12):
                    break

                # if already at outlet, dump rest here and stop (mass conserved)
                if mask[pos] == nb.uint8(3):
                    dep_map[pos] += h_carry
                    z[pos]       += h_carry
                    h_carry       = nb.float64(0.0)
                    break

                # step c: D8 routing — (slope^p_power * Uniform) argmax, using updated z
                # outlets excluded as routing targets: particles stay in the interior
                best_w = nb.float64(-1.0)
                best_j = nb.int64(-1)

                for k in range(nb.int64(8)):
                    weights[k] = nb.float64(0.0)
                    j = nbuf[k]
                    if j < nb.int64(0) or mask[j] == nb.uint8(0) or mask[j] == nb.uint8(3):
                        continue
                    dr = nb.int64(pos) // nb.int64(ncols) - nb.int64(j) // nb.int64(ncols)
                    dc = nb.int64(pos) %  nb.int64(ncols) - nb.int64(j) %  nb.int64(ncols)
                    d  = (nb.float64(1.41421356237) if (dr != nb.int64(0) and dc != nb.int64(0))
                          else nb.float64(1.0)) * dx
                    dz = z[pos] - z[j]   # post-deposit elevation
                    if dz > nb.float64(0.0):
                        slope = dz / d
                        w = (slope ** p_power) * np.random.random()
                        weights[k] = w
                        if w > best_w:
                            best_w = w
                            best_j = nb.int64(j)

                if best_j < nb.int64(0):
                    # no downslope neighbor: deposit rest here
                    dep_map[pos] += h_carry
                    z[pos]       += h_carry
                    h_carry       = nb.float64(0.0)
                    break

                pos = best_j

            # max_steps exhausted: deposit remainder at current pos
            if h_carry > nb.float64(1e-12):
                dep_map[pos] += h_carry
                z[pos]       += h_carry


def runout_particles(sed_depth, z, h_sed, mask, nb_fn, dx, phi, ncols,
                     h_particle=1.0, h_base=0.1, p_power=1.0,
                     lambda_dist=1.0, max_steps=1000, seed=-1):
    """
    Stochastic particle runout for landslide sediment.

    Each cell with sed_depth > 0 emits floor(sed_depth/h_particle) particles
    of height h_particle plus one remainder particle.

    Per step each particle:
      a) deposits Uniform(0, h_base) unconditionally
      b) deposits clip((1 - (max_S/Sc)²) / lambda_dist, 0, 1) * h_carry
      c) routes D8 to argmax((slope^p_power) * Uniform(0,1)) using live z

    z is modified in-place; dep_map is returned.

    Parameters
    ----------
    sed_depth   : float64[:] eroded depth per cell [m]
    z           : float64[:] elevation, modified in-place
    mask        : uint8[:]
    nb_fn       : neighbours closure
    dx          : cell size [m]
    phi         : angle of internal friction [rad]; Sc = tan(phi)
    ncols       : grid columns
    h_particle  : max sediment per particle [m]
    h_base      : max unconditional deposit per step [m]
    p_power     : slope exponent for routing weight (0=uniform, large=steepest)
    lambda_dist : transport-length scaling (>1 longer runout, <1 shorter)
    max_steps   : max steps per particle before forced deposit
    seed        : RNG seed for reproducibility (-1 = unseeded)

    Returns
    -------
    dep_map : float64[:] deposited thickness per cell [m]
    """
    if float(sed_depth.max()) <= 0.0:
        return np.zeros(len(z), np.float64)

    dep_map = np.zeros(len(z), np.float64)
    Sc_sq   = np.float64(np.tan(phi) ** 2)

    _run_particles(z, mask, nb_fn, sed_depth, dep_map,
                   np.float64(dx), Sc_sq, np.int64(ncols),
                   np.float64(h_particle), np.float64(h_base),
                   np.float64(p_power), np.float64(lambda_dist),
                   np.int64(max_steps), np.int64(seed))

    h_sed += dep_map   # deposits are pure sediment
    return dep_map
