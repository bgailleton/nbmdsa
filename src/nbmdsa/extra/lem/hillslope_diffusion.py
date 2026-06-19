"""
Ren et al. (2026) Modular Implicit Method for hillslope sediment transport.

Governing equation:
    ∂z/∂t = φ·∇²z + ∇φ·∇z                    (Eq. 17)

φ is lagged (t-1), making the system linear per timestep (Eq. 18).
Solved with a Gauss-Seidel numba kernel driven by the same neighbourer
used for flow routing — periodic/nodata/border behaviour is automatic.

Neighbourer slot layout (buf[8]):
    0=topleft  1=top  2=topright  3=left  4=right
    5=bottomleft  6=bottom  7=bottomright
    -1 = invalid (wall or nodata)

Boundary conditions:
    mask == 1  →  interior (solved)
    mask != 1  →  Dirichlet (z fixed, outlets and inactive cells)
    invalid neighbour slot  →  zero-flux Neumann
"""

import numpy as np
import numba as nb


# ── phi factories ─────────────────────────────────────────────────────────────

def make_phi_linear(D):
    """Linear slope-dependent (Culling).  q_s = -D·∇z."""
    D = np.asarray(D, dtype=np.float64)
    def phi_fn(z, grad_mag):
        return np.broadcast_to(D, z.shape).copy()
    return phi_fn


def make_phi_nonlinear(D, Sc):
    """Nonlinear slope-dependent (Roering 1999).  q_s = -D·∇z / (1-(|∇z|/Sc)²)."""
    D  = np.asarray(D,  dtype=np.float64)
    Sc = np.asarray(Sc, dtype=np.float64)
    def phi_fn(z, grad_mag):
        ratio = np.clip(grad_mag / Sc, 0.0, 0.9999)
        return D / (1.0 - ratio ** 2)
    return phi_fn


def make_phi_depth_slope(D, h):
    """Depth-slope product (Mudd & Furbish 2007).  q_s = -D·h·cosθ·∇z.

    h  : mutable (nrows*ncols,) float64 array of soil thickness [m].
         Update externally each step via  h += z_old - z_new.
    """
    D = np.asarray(D, dtype=np.float64)
    def phi_fn(z, grad_mag):
        cos_theta = 1.0 / np.sqrt(1.0 + grad_mag ** 2)
        return D * h * cos_theta
    return phi_fn


def make_phi_nonlinear_depth_slope(eta, beta, Sc, h):
    """Nonlinear depth-slope (Roering 2008).
    D4(h) = η·(1 - exp(-β·h·cosθ)),  q_s = -D4·∇z / (1-(|∇z|/Sc)²).

    h  : mutable (nrows*ncols,) float64 array of soil thickness [m].
    """
    eta  = np.asarray(eta,  dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    Sc   = np.asarray(Sc,   dtype=np.float64)
    def phi_fn(z, grad_mag):
        cos_theta = 1.0 / np.sqrt(1.0 + grad_mag ** 2)
        D4    = eta * (1.0 - np.exp(-beta * h * cos_theta))
        ratio = np.clip(grad_mag / Sc, 0.0, 0.9999)
        return D4 / (1.0 - ratio ** 2)
    return phi_fn


def make_phi_hybrid(D1, D2, Sc):
    """Hybrid linear/nonlinear (Sect. 5.1).
    Linear φ=D1 where |∇z| < S_T, nonlinear above.
    S_T = Sc·√(1 - D1/D2)  (Eqs. 33-34).
    """
    D1 = np.asarray(D1, dtype=np.float64)
    D2 = np.asarray(D2, dtype=np.float64)
    Sc = np.asarray(Sc, dtype=np.float64)
    St = Sc * np.sqrt(np.clip(1.0 - D1 / D2, 0.0, None))
    def phi_fn(z, grad_mag):
        ratio  = np.clip(grad_mag / Sc, 0.0, 0.9999)
        phi_nl = D2 / (1.0 - ratio ** 2)
        return np.where(grad_mag < St, D1, phi_nl)
    return phi_fn


# ── neighbourer-based gradient ────────────────────────────────────────────────

@nb.njit
def _gradient(arr, nn, dx, neighbours_fn):
    """Central-difference gradient via neighbourer.  Returns (gx, gy) flat arrays.

    gx = ∂arr/∂x  (column / east-west direction, slots 3=west 4=east)
    gy = ∂arr/∂y  (row / north-south direction,  slots 1=north 6=south)
    Falls back to one-sided when one neighbour is invalid (-1).
    """
    buf = np.empty(8, dtype=nb.int64)
    gx  = np.zeros(nn, dtype=nb.float64)
    gy  = np.zeros(nn, dtype=nb.float64)

    for idx in range(nn):
        neighbours_fn(idx, buf)
        w = buf[3]   # west
        e = buf[4]   # east
        if w >= 0 and e >= 0:
            gx[idx] = (arr[e] - arr[w]) / (nb.float64(2.0) * dx)
        elif e >= 0:
            gx[idx] = (arr[e] - arr[idx]) / dx
        elif w >= 0:
            gx[idx] = (arr[idx] - arr[w]) / dx

        n = buf[1]   # north (top)
        s = buf[6]   # south (bottom)
        if n >= 0 and s >= 0:
            gy[idx] = (arr[s] - arr[n]) / (nb.float64(2.0) * dx)
        elif s >= 0:
            gy[idx] = (arr[s] - arr[idx]) / dx
        elif n >= 0:
            gy[idx] = (arr[idx] - arr[n]) / dx

    return gx, gy


# ── Gauss-Seidel kernel ───────────────────────────────────────────────────────

@nb.njit
def _gs_sweep(z, phi, dphi_dx, dphi_dy, mask, z_t0,
              nn, ncols, dx, dt, neighbours_fn, use_d8, color):
    """One red-black half-sweep (color=0: (i+j)%2==0, color=1: odd).

    Laplacian: cardinal slots (1,3,4,6) + diagonal slots (0,2,5,7) for D-8.
    Convection ∇φ·∇z: upwind differencing — guarantees diagonal dominance.
    Red-black ordering eliminates directional sweep bias.
    Invalid slot (-1): zero-flux Neumann. Fixed node (mask!=1): Dirichlet.
    """
    buf    = np.empty(8, dtype=nb.int64)
    dx2    = dx * dx
    sum_sq = nb.float64(0.0)
    count  = nb.int64(0)

    if use_d8:
        w_ctr  = nb.float64(20.0) / (nb.float64(6.0) * dx2)
        w_card = nb.float64(4.0)  / (nb.float64(6.0) * dx2)
        w_diag = nb.float64(1.0)  / (nb.float64(6.0) * dx2)
    else:
        w_ctr  = nb.float64(4.0) / dx2
        w_card = nb.float64(1.0) / dx2
        w_diag = nb.float64(0.0)

    for idx in range(nn):
        if mask[idx] != nb.uint8(1):
            continue
        if (nb.int64(idx) // nb.int64(ncols) + nb.int64(idx) % nb.int64(ncols)) % nb.int64(2) != nb.int64(color):
            continue

        neighbours_fn(idx, buf)
        phi_ij = phi[idx]
        px     = dphi_dx[idx]
        py     = dphi_dy[idx]

        # upwind adds |px|/dx + |py|/dx to diagonal → unconditional diagonal dominance
        a_diag = (nb.float64(1.0) / dt
                  + phi_ij * w_ctr
                  + (abs(px) + abs(py)) / dx)
        rhs = z_t0[idx] / dt

        # north (slot 1)
        # Laplacian: -phi*w_card
        # upwind convection: py≥0 → info from south, north gets 0
        #                    py<0  → info from north, north gets py/dx (negative)
        nb_ = buf[1]
        if nb_ >= 0:
            conv = nb.float64(0.0) if py >= nb.float64(0.0) else py / dx
            rhs -= (-phi_ij * w_card + conv) * z[nb_]

        # south (slot 6)
        # upwind: py≥0 → info from south, south gets -py/dx (negative)
        #         py<0  → info from north, south gets 0
        nb_ = buf[6]
        if nb_ >= 0:
            conv = -py / dx if py >= nb.float64(0.0) else nb.float64(0.0)
            rhs -= (-phi_ij * w_card + conv) * z[nb_]

        # west (slot 3)
        # upwind: px≥0 → info from east, west gets 0
        #         px<0  → info from west, west gets px/dx (negative)
        nb_ = buf[3]
        if nb_ >= 0:
            conv = nb.float64(0.0) if px >= nb.float64(0.0) else px / dx
            rhs -= (-phi_ij * w_card + conv) * z[nb_]

        # east (slot 4)
        # upwind: px≥0 → info from east, east gets -px/dx (negative)
        #         px<0  → info from west, east gets 0
        nb_ = buf[4]
        if nb_ >= 0:
            conv = -px / dx if px >= nb.float64(0.0) else nb.float64(0.0)
            rhs -= (-phi_ij * w_card + conv) * z[nb_]

        # diagonals (D-8 only, Laplacian only)
        if use_d8:
            nb_ = buf[0]
            if nb_ >= 0:
                rhs += phi_ij * w_diag * z[nb_]
            nb_ = buf[2]
            if nb_ >= 0:
                rhs += phi_ij * w_diag * z[nb_]
            nb_ = buf[5]
            if nb_ >= 0:
                rhs += phi_ij * w_diag * z[nb_]
            nb_ = buf[7]
            if nb_ >= 0:
                rhs += phi_ij * w_diag * z[nb_]

        z_new   = rhs / a_diag
        dz      = z_new - z[idx]
        z[idx]  = z_new
        sum_sq += dz * dz
        count  += nb.int64(1)

    if count == 0:
        return nb.float64(0.0)
    return (sum_sq / nb.float64(count)) ** nb.float64(0.5)


# ── solver ────────────────────────────────────────────────────────────────────

def step(z_flat, mask, nrows, ncols, dx, dt, phi_fn, neighbours_fn,
         laplacian='D8', slope_cap=None, max_iter=200, tol=1e-6):
    """
    One implicit hillslope diffusion timestep (Gauss-Seidel solver).

    Parameters
    ----------
    z_flat        : flat float64 array (nrows*ncols,)
    mask          : flat uint8 array — 1=interior, else Dirichlet (fixed)
    nrows, ncols  : int
    dx            : float — cell size [m]
    dt            : float — timestep [yr]
    phi_fn        : callable (z_flat, grad_mag_flat) -> phi_flat, from make_phi_*
    neighbours_fn : neighbourer from make_neighbours(nrows, ncols, d8=..., border=...)
    laplacian     : 'D8' (nine-point) or 'D4' (five-point)
    slope_cap     : float or None — clamp |∇z| before computing φ
    max_iter      : int   — max GS iterations
    tol           : float — RMS(dz) convergence threshold [m]

    Returns
    -------
    flat float64 array (nrows*ncols,)
    """
    nn   = nrows * ncols
    z_t0 = z_flat.copy()
    z    = z_t0.copy()

    # lagged gradient of z via neighbourer
    gx, gy   = _gradient(z_t0, nn, np.float64(dx), neighbours_fn)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    if slope_cap is not None:
        grad_mag = np.minimum(grad_mag, float(slope_cap))

    # phi and its gradient — phi_fn operates on flat arrays
    phi              = phi_fn(z_t0, grad_mag)
    dphi_dx, dphi_dy = _gradient(phi, nn, np.float64(dx), neighbours_fn)

    use_d8  = laplacian == 'D8'
    _dx     = np.float64(dx)
    _dt     = np.float64(dt)
    _ncols  = np.int64(ncols)
    for _ in range(max_iter):
        _gs_sweep(z, phi, dphi_dx, dphi_dy, mask, z_t0,
                  nn, _ncols, _dx, _dt, neighbours_fn, use_d8, 0)
        err = _gs_sweep(z, phi, dphi_dx, dphi_dy, mask, z_t0,
                        nn, _ncols, _dx, _dt, neighbours_fn, use_d8, 1)
        if err < tol:
            break

    return z
