"""
Perlin noise generator (2D, multi-octave).

Returns a flat float64 array of shape (nrows*ncols,).
"""

import numpy as np
import numba as nb


# ── core Perlin primitives ────────────────────────────────────────────────────

@nb.njit(inline='always')
def _fade(t):
    return t * t * t * (t * (t * nb.float64(6.0) - nb.float64(15.0)) + nb.float64(10.0))


@nb.njit(inline='always')
def _lerp(t, a, b):
    return a + t * (b - a)


@nb.njit(inline='always')
def _grad2(h, x, y):
    """2D gradient from hash — 4 gradient directions."""
    hh = h & nb.int64(3)
    if hh == nb.int64(0):
        return  x + y
    elif hh == nb.int64(1):
        return -x + y
    elif hh == nb.int64(2):
        return  x - y
    else:
        return -x - y


@nb.njit
def _perlin2(x, y, perm):
    """Single-octave 2D Perlin noise, output in [-1, 1]."""
    xi = nb.int64(np.floor(x)) & nb.int64(255)
    yi = nb.int64(np.floor(y)) & nb.int64(255)
    xf = x - np.floor(x)
    yf = y - np.floor(y)

    u = _fade(xf)
    v = _fade(yf)

    aa = perm[perm[xi    ] + yi    ]
    ab = perm[perm[xi    ] + yi + 1]
    ba = perm[perm[xi + 1] + yi    ]
    bb = perm[perm[xi + 1] + yi + 1]

    return _lerp(v,
                 _lerp(u, _grad2(aa,  xf,       yf      ),
                           _grad2(ba,  xf - 1.0, yf      )),
                 _lerp(u, _grad2(ab,  xf,       yf - 1.0),
                           _grad2(bb,  xf - 1.0, yf - 1.0)))


# ── full-grid kernel ──────────────────────────────────────────────────────────

@nb.njit(parallel=True)
def _fill_grid(out, nrows, ncols,
               scale_x, scale_y,
               origin_x, origin_y,
               octaves, persistence, lacunarity,
               perm):
    """Fill flat output array with multi-octave Perlin noise."""
    for i in nb.prange(nrows):
        for j in range(ncols):
            val   = nb.float64(0.0)
            amp   = nb.float64(1.0)
            freq  = nb.float64(1.0)
            norm  = nb.float64(0.0)
            for _ in range(octaves):
                px = (origin_x + nb.float64(j) / nb.float64(ncols)) * scale_x * freq
                py = (origin_y + nb.float64(i) / nb.float64(nrows)) * scale_y * freq
                val  += amp * _perlin2(px, py, perm)
                norm += amp
                amp  *= persistence
                freq *= lacunarity
            out[i * ncols + j] = val / norm


# ── public API ────────────────────────────────────────────────────────────────

def generate(nrows, ncols,
             scale=4.0,
             octaves=1,
             persistence=0.5,
             lacunarity=2.0,
             amplitude=1.0,
             offset=0.0,
             origin=(0.0, 0.0),
             seed=None):
    """
    2D multi-octave Perlin noise on a (nrows, ncols) grid.

    Parameters
    ----------
    nrows, ncols  : int — grid dimensions
    scale         : float or (float, float) — spatial frequency along x and y.
                    scale=1 → one full noise period across the grid;
                    scale=4 → four periods (more features).
    octaves       : int   — number of octave layers
    persistence   : float — amplitude decay per octave  (0 < p < 1, typically 0.5)
    lacunarity    : float — frequency multiplier per octave (typically 2.0)
    amplitude     : float — output is multiplied by this after normalisation
    offset        : float — constant added to the output
    origin        : (float, float) — (x, y) offset into noise space (for tiling/patching)
    seed          : int or None — seed for the permutation table shuffle

    Returns
    -------
    flat float64 array, shape (nrows*ncols,), values roughly in
    [offset - amplitude, offset + amplitude]
    """
    rng = np.random.default_rng(seed)

    # build doubled permutation table (avoids modulo in inner loop)
    p = np.arange(256, dtype=np.int64)
    rng.shuffle(p)
    perm = np.empty(512, dtype=np.int64)
    perm[:256] = p
    perm[256:] = p

    if np.ndim(scale) == 0:
        sx = sy = float(scale)
    else:
        sx, sy = float(scale[0]), float(scale[1])

    out = np.empty(nrows * ncols, dtype=np.float64)
    _fill_grid(out, nrows, ncols,
               np.float64(sx), np.float64(sy),
               np.float64(origin[0]), np.float64(origin[1]),
               np.int64(octaves),
               np.float64(persistence),
               np.float64(lacunarity),
               perm)

    return out * amplitude + offset
