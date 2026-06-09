"""Closure-based grid neighbourer — no structref, fully inlineable.

``make_neighbours(nrows, ncols, d8, border, indexing)`` returns a single
``@nb.njit`` function ``neighbours(idx, buf)``.

Buffer layout (always 8 slots):
  0=topleft  1=top  2=topright  3=left  4=right  5=bottomleft  6=bottom  7=bottomright

Invalid slots (border cells in normal mode, diagonals in D4) are ``-1``.

1D: ``idx`` is ``int64``, ``buf`` is ``int64[8]``.
2D: ``idx`` is ``(row, col)`` tuple, ``buf`` is ``int64[8, 2]``;
    invalid slots have ``buf[k, 0] == buf[k, 1] == -1``.
"""

from functools import lru_cache

import numba as nb


@lru_cache(maxsize=None)
def make_neighbours(nrows, ncols, d8=True, border='normal', indexing='1d'):
    """Return a ``@nb.njit`` closure ``neighbours(idx, buf) -> None``."""
    if indexing == '1d':
        return _make_1d(nrows, ncols, d8, border)
    elif indexing == '2d':
        return _make_2d(nrows, ncols, d8, border)
    else:
        raise ValueError(f"indexing must be '1d' or '2d', got {indexing!r}")


# ── 1D ────────────────────────────────────────────────────────────────────────

def _make_1d(nrows, ncols, d8, border):
    if border == 'normal':
        if d8:
            @nb.njit
            def neighbours(idx, buf):
                row = idx // ncols
                col = idx % ncols
                buf[0] = (row - 1) * ncols + (col - 1) if row > 0 and col > 0 else nb.int64(-1)
                buf[1] = (row - 1) * ncols + col        if row > 0             else nb.int64(-1)
                buf[2] = (row - 1) * ncols + (col + 1) if row > 0 and col < ncols - 1 else nb.int64(-1)
                buf[3] = row * ncols + (col - 1)        if col > 0             else nb.int64(-1)
                buf[4] = row * ncols + (col + 1)        if col < ncols - 1     else nb.int64(-1)
                buf[5] = (row + 1) * ncols + (col - 1) if row < nrows - 1 and col > 0 else nb.int64(-1)
                buf[6] = (row + 1) * ncols + col        if row < nrows - 1     else nb.int64(-1)
                buf[7] = (row + 1) * ncols + (col + 1) if row < nrows - 1 and col < ncols - 1 else nb.int64(-1)
            return neighbours
        else:
            @nb.njit
            def neighbours(idx, buf):
                row = idx // ncols
                col = idx % ncols
                buf[0] = nb.int64(-1)
                buf[1] = (row - 1) * ncols + col if row > 0         else nb.int64(-1)
                buf[2] = nb.int64(-1)
                buf[3] = row * ncols + (col - 1) if col > 0         else nb.int64(-1)
                buf[4] = row * ncols + (col + 1) if col < ncols - 1 else nb.int64(-1)
                buf[5] = nb.int64(-1)
                buf[6] = (row + 1) * ncols + col if row < nrows - 1 else nb.int64(-1)
                buf[7] = nb.int64(-1)
            return neighbours

    elif border == 'ew':
        if d8:
            @nb.njit
            def neighbours(idx, buf):
                row = idx // ncols
                col = idx % ncols
                lc  = (col - 1) % ncols
                rc  = (col + 1) % ncols
                buf[0] = (row - 1) * ncols + lc if row > 0         else nb.int64(-1)
                buf[1] = (row - 1) * ncols + col if row > 0        else nb.int64(-1)
                buf[2] = (row - 1) * ncols + rc if row > 0         else nb.int64(-1)
                buf[3] = row * ncols + lc
                buf[4] = row * ncols + rc
                buf[5] = (row + 1) * ncols + lc if row < nrows - 1 else nb.int64(-1)
                buf[6] = (row + 1) * ncols + col if row < nrows - 1 else nb.int64(-1)
                buf[7] = (row + 1) * ncols + rc if row < nrows - 1 else nb.int64(-1)
            return neighbours
        else:
            @nb.njit
            def neighbours(idx, buf):
                row = idx // ncols
                col = idx % ncols
                lc  = (col - 1) % ncols
                rc  = (col + 1) % ncols
                buf[0] = nb.int64(-1)
                buf[1] = (row - 1) * ncols + col if row > 0         else nb.int64(-1)
                buf[2] = nb.int64(-1)
                buf[3] = row * ncols + lc
                buf[4] = row * ncols + rc
                buf[5] = nb.int64(-1)
                buf[6] = (row + 1) * ncols + col if row < nrows - 1 else nb.int64(-1)
                buf[7] = nb.int64(-1)
            return neighbours

    elif border == 'ns':
        if d8:
            @nb.njit
            def neighbours(idx, buf):
                row = idx // ncols
                col = idx % ncols
                tr  = (row - 1) % nrows
                br  = (row + 1) % nrows
                buf[0] = tr * ncols + (col - 1) if col > 0         else nb.int64(-1)
                buf[1] = tr * ncols + col
                buf[2] = tr * ncols + (col + 1) if col < ncols - 1 else nb.int64(-1)
                buf[3] = row * ncols + (col - 1) if col > 0        else nb.int64(-1)
                buf[4] = row * ncols + (col + 1) if col < ncols - 1 else nb.int64(-1)
                buf[5] = br * ncols + (col - 1) if col > 0         else nb.int64(-1)
                buf[6] = br * ncols + col
                buf[7] = br * ncols + (col + 1) if col < ncols - 1 else nb.int64(-1)
            return neighbours
        else:
            @nb.njit
            def neighbours(idx, buf):
                row = idx // ncols
                col = idx % ncols
                tr  = (row - 1) % nrows
                br  = (row + 1) % nrows
                buf[0] = nb.int64(-1)
                buf[1] = tr * ncols + col
                buf[2] = nb.int64(-1)
                buf[3] = row * ncols + (col - 1) if col > 0         else nb.int64(-1)
                buf[4] = row * ncols + (col + 1) if col < ncols - 1 else nb.int64(-1)
                buf[5] = nb.int64(-1)
                buf[6] = br * ncols + col
                buf[7] = nb.int64(-1)
            return neighbours

    else:
        raise ValueError(f"border must be 'normal', 'ew', or 'ns', got {border!r}")


# ── 2D ────────────────────────────────────────────────────────────────────────

def _make_2d(nrows, ncols, d8, border):
    if border == 'normal':
        if d8:
            @nb.njit
            def neighbours(idx, buf):
                row, col = idx
                if row > 0 and col > 0:           buf[0, 0] = row - 1; buf[0, 1] = col - 1
                else:                              buf[0, 0] = nb.int64(-1); buf[0, 1] = nb.int64(-1)
                if row > 0:                        buf[1, 0] = row - 1; buf[1, 1] = col
                else:                              buf[1, 0] = nb.int64(-1); buf[1, 1] = nb.int64(-1)
                if row > 0 and col < ncols - 1:   buf[2, 0] = row - 1; buf[2, 1] = col + 1
                else:                              buf[2, 0] = nb.int64(-1); buf[2, 1] = nb.int64(-1)
                if col > 0:                        buf[3, 0] = row;     buf[3, 1] = col - 1
                else:                              buf[3, 0] = nb.int64(-1); buf[3, 1] = nb.int64(-1)
                if col < ncols - 1:               buf[4, 0] = row;     buf[4, 1] = col + 1
                else:                              buf[4, 0] = nb.int64(-1); buf[4, 1] = nb.int64(-1)
                if row < nrows - 1 and col > 0:   buf[5, 0] = row + 1; buf[5, 1] = col - 1
                else:                              buf[5, 0] = nb.int64(-1); buf[5, 1] = nb.int64(-1)
                if row < nrows - 1:               buf[6, 0] = row + 1; buf[6, 1] = col
                else:                              buf[6, 0] = nb.int64(-1); buf[6, 1] = nb.int64(-1)
                if row < nrows - 1 and col < ncols - 1: buf[7, 0] = row + 1; buf[7, 1] = col + 1
                else:                              buf[7, 0] = nb.int64(-1); buf[7, 1] = nb.int64(-1)
            return neighbours
        else:
            @nb.njit
            def neighbours(idx, buf):
                row, col = idx
                buf[0, 0] = nb.int64(-1); buf[0, 1] = nb.int64(-1)
                if row > 0:               buf[1, 0] = row - 1; buf[1, 1] = col
                else:                     buf[1, 0] = nb.int64(-1); buf[1, 1] = nb.int64(-1)
                buf[2, 0] = nb.int64(-1); buf[2, 1] = nb.int64(-1)
                if col > 0:               buf[3, 0] = row;     buf[3, 1] = col - 1
                else:                     buf[3, 0] = nb.int64(-1); buf[3, 1] = nb.int64(-1)
                if col < ncols - 1:       buf[4, 0] = row;     buf[4, 1] = col + 1
                else:                     buf[4, 0] = nb.int64(-1); buf[4, 1] = nb.int64(-1)
                buf[5, 0] = nb.int64(-1); buf[5, 1] = nb.int64(-1)
                if row < nrows - 1:       buf[6, 0] = row + 1; buf[6, 1] = col
                else:                     buf[6, 0] = nb.int64(-1); buf[6, 1] = nb.int64(-1)
                buf[7, 0] = nb.int64(-1); buf[7, 1] = nb.int64(-1)
            return neighbours

    elif border == 'ew':
        if d8:
            @nb.njit
            def neighbours(idx, buf):
                row, col = idx
                lc = (col - 1) % ncols
                rc = (col + 1) % ncols
                if row > 0:         buf[0, 0] = row - 1; buf[0, 1] = lc
                else:               buf[0, 0] = nb.int64(-1); buf[0, 1] = nb.int64(-1)
                if row > 0:         buf[1, 0] = row - 1; buf[1, 1] = col
                else:               buf[1, 0] = nb.int64(-1); buf[1, 1] = nb.int64(-1)
                if row > 0:         buf[2, 0] = row - 1; buf[2, 1] = rc
                else:               buf[2, 0] = nb.int64(-1); buf[2, 1] = nb.int64(-1)
                buf[3, 0] = row; buf[3, 1] = lc
                buf[4, 0] = row; buf[4, 1] = rc
                if row < nrows - 1: buf[5, 0] = row + 1; buf[5, 1] = lc
                else:               buf[5, 0] = nb.int64(-1); buf[5, 1] = nb.int64(-1)
                if row < nrows - 1: buf[6, 0] = row + 1; buf[6, 1] = col
                else:               buf[6, 0] = nb.int64(-1); buf[6, 1] = nb.int64(-1)
                if row < nrows - 1: buf[7, 0] = row + 1; buf[7, 1] = rc
                else:               buf[7, 0] = nb.int64(-1); buf[7, 1] = nb.int64(-1)
            return neighbours
        else:
            @nb.njit
            def neighbours(idx, buf):
                row, col = idx
                lc = (col - 1) % ncols
                rc = (col + 1) % ncols
                buf[0, 0] = nb.int64(-1); buf[0, 1] = nb.int64(-1)
                if row > 0:         buf[1, 0] = row - 1; buf[1, 1] = col
                else:               buf[1, 0] = nb.int64(-1); buf[1, 1] = nb.int64(-1)
                buf[2, 0] = nb.int64(-1); buf[2, 1] = nb.int64(-1)
                buf[3, 0] = row; buf[3, 1] = lc
                buf[4, 0] = row; buf[4, 1] = rc
                buf[5, 0] = nb.int64(-1); buf[5, 1] = nb.int64(-1)
                if row < nrows - 1: buf[6, 0] = row + 1; buf[6, 1] = col
                else:               buf[6, 0] = nb.int64(-1); buf[6, 1] = nb.int64(-1)
                buf[7, 0] = nb.int64(-1); buf[7, 1] = nb.int64(-1)
            return neighbours

    elif border == 'ns':
        if d8:
            @nb.njit
            def neighbours(idx, buf):
                row, col = idx
                tr = (row - 1) % nrows
                br = (row + 1) % nrows
                if col > 0:         buf[0, 0] = tr; buf[0, 1] = col - 1
                else:               buf[0, 0] = nb.int64(-1); buf[0, 1] = nb.int64(-1)
                buf[1, 0] = tr; buf[1, 1] = col
                if col < ncols - 1: buf[2, 0] = tr; buf[2, 1] = col + 1
                else:               buf[2, 0] = nb.int64(-1); buf[2, 1] = nb.int64(-1)
                if col > 0:         buf[3, 0] = row; buf[3, 1] = col - 1
                else:               buf[3, 0] = nb.int64(-1); buf[3, 1] = nb.int64(-1)
                if col < ncols - 1: buf[4, 0] = row; buf[4, 1] = col + 1
                else:               buf[4, 0] = nb.int64(-1); buf[4, 1] = nb.int64(-1)
                if col > 0:         buf[5, 0] = br; buf[5, 1] = col - 1
                else:               buf[5, 0] = nb.int64(-1); buf[5, 1] = nb.int64(-1)
                buf[6, 0] = br; buf[6, 1] = col
                if col < ncols - 1: buf[7, 0] = br; buf[7, 1] = col + 1
                else:               buf[7, 0] = nb.int64(-1); buf[7, 1] = nb.int64(-1)
            return neighbours
        else:
            @nb.njit
            def neighbours(idx, buf):
                row, col = idx
                tr = (row - 1) % nrows
                br = (row + 1) % nrows
                buf[0, 0] = nb.int64(-1); buf[0, 1] = nb.int64(-1)
                buf[1, 0] = tr; buf[1, 1] = col
                buf[2, 0] = nb.int64(-1); buf[2, 1] = nb.int64(-1)
                if col > 0:         buf[3, 0] = row; buf[3, 1] = col - 1
                else:               buf[3, 0] = nb.int64(-1); buf[3, 1] = nb.int64(-1)
                if col < ncols - 1: buf[4, 0] = row; buf[4, 1] = col + 1
                else:               buf[4, 0] = nb.int64(-1); buf[4, 1] = nb.int64(-1)
                buf[5, 0] = nb.int64(-1); buf[5, 1] = nb.int64(-1)
                buf[6, 0] = br; buf[6, 1] = col
                buf[7, 0] = nb.int64(-1); buf[7, 1] = nb.int64(-1)
            return neighbours

    else:
        raise ValueError(f"border must be 'normal', 'ew', or 'ns', got {border!r}")
