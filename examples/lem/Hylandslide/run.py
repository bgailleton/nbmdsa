"""
Yuan SPL + HyLands-style landslides on a periodic EW mountain–foreland landscape.

Domain
------
  EW periodic, south row (row=0) open outlets.
  Mountain zone  (rows split..nrows): U = +1e-3 m/yr  (uplift)
  Foreland zone  (rows 0..split)    : U = -1e-4 m/yr  (subsidence)

Processes per timestep
----------------------
  1. Priority flood
  2. Build tree (mode='full' for SFD landslide trigger)
  3. Drainage area accumulation
  4. Uplift
  5. Yuan SPL (Gauss-Seidel)
  6. Landslide trigger  → sed_depth, trigger_nodes, trigger_heights
  7. Landslide runout   → deposit sediment downstream
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import numba as nb

from nbmdsa.structures.neighbourer    import make_neighbours
from nbmdsa.structures.primitives     import make_heap, make_queue
from nbmdsa.algorithms.priority_flood import make_priority_flood
from nbmdsa.structures.tree           import make_tree
from nbmdsa.algorithms.topo           import topo_order as compute_topo_order
from nbmdsa.algorithms.tree_traversal import tree_traversal_full
from nbmdsa.algorithms.mfd_traversal  import mfd_topo_order

from nbmdsa.extra.lem        import yuan, uplift as uplift_mod
from nbmdsa.extra.lem.yuan   import step_with_sed
from nbmdsa.extra.noise import perlin_noise

from hylandslide import trigger, runout_landslides, runout_particles

# ── parameters ────────────────────────────────────────────────────────────────

nrows  = 256
ncols  = 256
dx     = 30.0        # cell size [m]
dt     = 500.0        # timestep [yr]
N      = 2000         # number of timesteps

# fluvial (Yuan)
K_b      = 1e-4    # bedrock erodibility
K_sed    = 2e-3    # sediment erodibility (typically higher than bedrock)
m_spl    = 0.45
n_spl    = 1.0
G_val    = 1.
tol      = 1e-3
max_iter = 200

# uplift
U_mtn   = 5e-3        # mountain uplift [m/yr]
U_fore  = 0       # foreland subsidence [m/yr]

# landslide
C       = 15e4        # cohesion [Pa]
phi_deg = 30.0        # internal friction angle [degrees]
phi     = np.radians(phi_deg)
rho     = 2700.0      # rock density [kg m-3]
t_LS    = 1e6         # landslide return time [yr]
lam        = 1      # lambda_dist: > 1 = longer runout
sfd_ls     = False   # True = SFD trigger + routing; False = MFD
use_particles = True # True = stochastic particle runout; False = MFD/SFD
h_particle = 0.1     # sediment height per particle [m]
h_base     = 0.001    # unconditional deposit per step [m]
p_power    = 1.     # slope weighting exponent for D8 routing

# ── grid initialisation ───────────────────────────────────────────────────────

nn = nrows * ncols
z  = perlin_noise.generate(nrows, ncols, scale=15, octaves=6,
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

# uplift field
split   = nrows // 2
U_field = np.empty(nn, dtype=np.float64)
for i in range(nrows):
    row_U = U_mtn if i >= split else U_fore
    U_field[i * ncols:(i + 1) * ncols] = row_U
# outlets subside with the foreland — do NOT zero them out, otherwise PF
# unconditionally floods the foreland back to z=0 every step

# flat arrays for Yuan SPL
K_b_arr   = np.full(nn, K_b,   dtype=np.float64)
K_sed_arr = np.full(nn, K_sed, dtype=np.float64)
m_arr     = np.full(nn, m_spl, dtype=np.float64)
n_arr     = np.full(nn, n_spl, dtype=np.float64)
G_arr     = np.full(nn, G_val, dtype=np.float64)

h_sed = np.zeros(nn, dtype=np.float64)   # sediment thickness [m]

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


# ── live figure ───────────────────────────────────────────────────────────────

interior_2d = mask.reshape(nrows, ncols) > 0

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.ion()

_z0   = np.where(interior_2d, z.reshape(nrows, ncols), np.nan)
_A0   = np.where(interior_2d, A.reshape(nrows, ncols), np.nan)
_sd0  = np.zeros((nrows, ncols))

im0 = axes[0].imshow(_z0,            cmap='terrain', origin='lower')
im1 = axes[1].imshow(np.log10(_A0),  cmap='Blues',   origin='lower')
im2 = axes[2].imshow(_sd0,           cmap='Reds',    origin='lower', vmin=0, vmax=1)

plt.colorbar(im0, ax=axes[0], label='z [m]')
plt.colorbar(im1, ax=axes[1], label='log₁₀(m²)')
plt.colorbar(im2, ax=axes[2], label='eroded depth [m]')

for ax in axes:
    ax.axhline(split, color='red', lw=1, ls='--')

plt.tight_layout()
fig.show()

# ── main loop ─────────────────────────────────────────────────────────────────

n_ls_total    = 0
vol_ls_total  = 0.0
sed_depth     = np.zeros(nn, np.float64)
t_nodes       = np.empty(0, np.int64)
step          = 0
st            = time.perf_counter()

window_steps  = max(1, int(100e3 / dt))   # steps per 100 kyr window
cum_sed_depth = np.zeros(nn, np.float64)
n_ls_window   = 0
window_step   = 0

while True:
    z = pf(z.copy(), mask, nb_fn, heap_ops, queue_ops)

    # mode='full' for SFD landslide child-lookup; fine for Yuan too
    tree  = make_tree(z, nb_fn, mask, mode='full')
    order = compute_topo_order(tree)

    A[:] = dx * dx
    tree_traversal_full(order, tree, _catchment, (A,), direction='up')

    z += U_field * dt   # applies to ALL cells including outlets (subsidence)
    z[mask == 3] = 0

    # Yuan SPL
    h_t0      = z.copy()
    h_sed_t0  = h_sed.copy()
    h[:] = h_t0
    hp[:] = h_t0
    dh_accum[:] = 0.0
    gs_iter = 0
    for gs_iter in range(max_iter):
        err = step_with_sed(order, tree, h, hp, elev, dh_accum, A, h_t0, h_sed_t0,
                            K_sed_arr, K_b_arr, m_arr, n_arr, G_arr, dt, ncols, dx, mask)
        if G_val == 0.0 or err < tol * max(h_t0.max(), 1e-10):
            break
    z[:] = h
    # reduce sediment layer by whatever was eroded
    h_sed[:] = np.maximum(0.0, h_sed - np.maximum(0.0, h_t0 - z))

    # landslide trigger
    mfd_ord = None if sfd_ls else mfd_topo_order(z, mask, nb_fn)
    sed_depth, t_nodes, t_heights = trigger(
        z, mask, nb_fn, order, tree, dx, C, phi, rho, t_LS, dt, ncols,
        sfd=sfd_ls)

    # landslides entrain both sediment and bedrock; reduce h_sed accordingly
    h_sed[:] = np.maximum(0.0, h_sed - sed_depth)

    n_ls          = len(t_nodes)
    n_ls_total   += n_ls
    n_ls_window  += n_ls
    cum_sed_depth += sed_depth
    window_step  += 1
    if window_step >= window_steps:
        window_step   = 0
        cum_sed_depth[:] = 0.0
        n_ls_window   = 0

    if n_ls > 0:
        vol_ls_total += float(t_heights.sum()) * dx * dx
        if use_particles:
            runout_particles(sed_depth, z, h_sed, mask, nb_fn, dx, phi, ncols,
                             h_particle=h_particle, h_base=h_base,
                             p_power=p_power, lambda_dist=lam)
        else:
            runout_landslides(
                t_nodes, t_heights, z, h_sed, mask, nb_fn,
                order, tree, mfd_ord, dx, phi, ncols,
                lambda_dist=lam, sfd=sfd_ls)

    if step % 100 == 0:
        interior = mask > 0
        elapsed  = time.perf_counter() - st
        print(f"step {step:5d}  t={step*dt/1e3:.1f} kyr  gs={gs_iter+1:3d}  "
              f"max_z={z[interior].max():.1f}  mean_z={z[interior].mean():.1f}  "
              f"ls={n_ls:3d}  cum_ls={n_ls_total:5d}  "
              f"cum_vol={vol_ls_total/1e6:.3f} km³  wall={elapsed:.0f}s")

    # if step % 5 == 0:
    #     add = np.random.rand(nn) * K_b/10 + K_b
    #     K_b_arr = add
    #     add = np.random.rand(nn) * K_sed/10 + K_sed
    #     K_sed_arr = add

    #     K_b_arr[K_b_arr<=1e-6] = K_b
    #     K_sed_arr[K_sed_arr<=1e-6] = K_sed

    if step % 5 == 0:
        z_2d  = np.where(interior_2d, z.reshape(nrows, ncols), np.nan)
        A_2d  = np.where(interior_2d, A.reshape(nrows, ncols), np.nan)
        sd_2d = cum_sed_depth.reshape(nrows, ncols)

        im0.set_data(z_2d)
        im0.set_clim(np.nanmin(z_2d), np.nanmax(z_2d))
        im1.set_data(np.log10(A_2d))
        im1.set_clim(np.nanmin(np.log10(A_2d[A_2d > 0])),
                     np.nanmax(np.log10(A_2d)))
        im2.set_data(sd_2d)
        if sd_2d.max() > 0:
            im2.set_clim(0, sd_2d.max())

        axes[0].set_title(f'Elevation  t={step*dt/1e3:.1f} kyr')
        axes[2].set_title(f'Cumul. LS erosion last 100 kyr  ({n_ls_window} events)')

        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.01)

    step += 1
