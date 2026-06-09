"""Tests for priority_flood_epsilon.

3x3 and 3x5 grids — small enough to verify by hand.
"""

import numpy as np
import numba as nb

from nbmdsa.structures.neighbourer   import make_neighbours
from nbmdsa.structures.primitives    import make_heap, make_queue
from nbmdsa.algorithms.priority_flood import make_priority_flood


def _run(nrows, ncols, z, mask):
    nb_fn = make_neighbours(nrows, ncols, d8=True, border='normal')
    pf    = make_priority_flood(nb.float64)
    return pf(z.copy(), mask, nb_fn, make_heap(nb.float64), make_queue(nb.int64))


# ── depression filling ────────────────────────────────────────────────────────

def test_single_depression_filled():
    # 3x3, all borders are outlets (z=10), center is depression (z=1)
    # after fill: center >= nextafter(10.0, max)
    z    = np.full(9, 10.0)
    z[4] = 1.0
    mask = np.full(9, 3, dtype=np.uint8)
    mask[4] = 1

    result = _run(3, 3, z, mask)

    assert result[4] > 1.0
    assert result[4] == np.nextafter(np.float64(10.0), np.finfo(np.float64).max)
    assert np.all(result[[0,1,2,3,5,6,7,8]] == 10.0)


def test_above_outlets_unchanged():
    # interior cell higher than outlets — no filling needed
    z    = np.full(9, 5.0)
    z[4] = 10.0
    mask = np.full(9, 3, dtype=np.uint8)
    mask[4] = 1

    result = _run(3, 3, z, mask)
    assert result[4] == 10.0


def test_multiple_depressions():
    # 3x5: two interior depressions at idx 6 and 8, nodata at idx 7
    #   col:  0  1  2  3  4
    # row 0:  3  3  3  3  3   z=10
    # row 1:  3  1  0  1  3   z=10,1,0,2,10
    # row 2:  3  3  3  3  3   z=10
    nrows, ncols = 3, 5
    z    = np.full(15, 10.0)
    z[6] = 1.0   # depression
    z[7] = 0.0   # nodata placeholder
    z[8] = 2.0   # depression
    mask = np.full(15, 3, dtype=np.uint8)
    mask[6] = 1
    mask[7] = 0   # nodata
    mask[8] = 1

    result = _run(nrows, ncols, z, mask)

    eps = np.nextafter(np.float64(10.0), np.finfo(np.float64).max)
    assert result[6] == eps
    assert result[8] == eps
    assert result[7] == 0.0   # nodata untouched


def test_nodata_cells_never_modified():
    # fill an entire interior with nodata — outlets should not propagate through
    z    = np.full(9, 10.0)
    mask = np.full(9, 3, dtype=np.uint8)
    mask[4] = 0   # nodata center

    result = _run(3, 3, z, mask)
    assert result[4] == 10.0


def test_outlet_cells_not_modified():
    # outlet cells (mask=3) seed the heap but their z must stay unchanged
    z    = np.full(9, 5.0)
    mask = np.full(9, 3, dtype=np.uint8)
    mask[4] = 1

    result = _run(3, 3, z, mask)
    for i in [0, 1, 2, 3, 5, 6, 7, 8]:
        assert result[i] == 5.0
