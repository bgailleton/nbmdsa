"""
White noise generator.

Returns a flat float64 array of shape (nrows*ncols,).
"""

import numpy as np


def generate(nrows, ncols,
             amplitude=1.0,
             offset=0.0,
             seed=None):
    """
    Parameters
    ----------
    nrows, ncols : int
    amplitude    : float — half-range of the noise (values in [offset-amplitude, offset+amplitude])
    offset       : float — mean value added to the noise
    seed         : int or None

    Returns
    -------
    flat float64 array, shape (nrows*ncols,)
    """
    rng = np.random.default_rng(seed)
    return (rng.random(nrows * ncols) * 2.0 - 1.0) * amplitude + offset
