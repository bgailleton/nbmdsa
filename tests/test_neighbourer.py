"""Tests for make_neighbours — 3x4 grid to avoid square-symmetry bugs.

Grid layout (nrows=3, ncols=4):
  col:  0   1   2   3
row 0:  0   1   2   3
row 1:  4   5   6   7
row 2:  8   9  10  11
"""

import numpy as np
import numba as nb
import pytest

from nbmdsa.structures.neighbourer import make_neighbours

NROWS, NCOLS = 3, 4


# ── helpers ───────────────────────────────────────────────────────────────────

def _call_1d(fn, idx):
    """Call a 1D neighbours closure from JIT context, return buf as numpy."""
    @nb.njit
    def _run(fn, idx):
        buf = np.empty(8, nb.int64)
        fn(idx, buf)
        return buf
    return _run(fn, nb.int64(idx))


def _call_2d(fn, row, col):
    """Call a 2D neighbours closure from JIT context, return buf as numpy."""
    @nb.njit
    def _run(fn, row, col):
        buf = np.empty((8, 2), nb.int64)
        fn((row, col), buf)
        return buf
    return _run(fn, nb.int64(row), nb.int64(col))


# ── 1D normal D8 ──────────────────────────────────────────────────────────────

class TestNormal1DD8:
    fn = make_neighbours(NROWS, NCOLS, d8=True, border='normal', indexing='1d')

    def test_center(self):
        # idx=5, row=1, col=1 — all 8 neighbours exist
        buf = _call_1d(self.fn, 5)
        assert list(buf) == [0, 1, 2, 4, 6, 8, 9, 10]

    def test_top_left_corner(self):
        # idx=0, row=0, col=0 — only right, bottom, bottomright valid
        buf = _call_1d(self.fn, 0)
        assert list(buf) == [-1, -1, -1, -1, 1, -1, 4, 5]

    def test_top_right_corner(self):
        # idx=3, row=0, col=3
        buf = _call_1d(self.fn, 3)
        assert list(buf) == [-1, -1, -1, 2, -1, 6, 7, -1]

    def test_bottom_left_corner(self):
        # idx=8, row=2, col=0
        buf = _call_1d(self.fn, 8)
        assert list(buf) == [-1, 4, 5, -1, 9, -1, -1, -1]

    def test_top_edge(self):
        # idx=1, row=0, col=1 — top row, no top neighbours
        buf = _call_1d(self.fn, 1)
        assert list(buf) == [-1, -1, -1, 0, 2, 4, 5, 6]


# ── 1D normal D4 ──────────────────────────────────────────────────────────────

class TestNormal1DD4:
    fn = make_neighbours(NROWS, NCOLS, d8=False, border='normal', indexing='1d')

    def test_diagonals_always_invalid(self):
        # center idx=5 — diagonals (slots 0,2,5,7) always -1
        buf = _call_1d(self.fn, 5)
        assert buf[0] == -1 and buf[2] == -1 and buf[5] == -1 and buf[7] == -1

    def test_center_cardinals(self):
        buf = _call_1d(self.fn, 5)
        assert list(buf) == [-1, 1, -1, 4, 6, -1, 9, -1]

    def test_corner_cardinals(self):
        buf = _call_1d(self.fn, 0)
        assert list(buf) == [-1, -1, -1, -1, 1, -1, 4, -1]


# ── 1D EW (east-west periodic) ────────────────────────────────────────────────

class TestEW1DD8:
    fn = make_neighbours(NROWS, NCOLS, d8=True, border='ew', indexing='1d')

    def test_left_edge_wraps(self):
        # idx=4, row=1, col=0 — left/topleft/bottomleft wrap to col=3
        buf = _call_1d(self.fn, 4)
        assert list(buf) == [3, 0, 1, 7, 5, 11, 8, 9]

    def test_right_edge_wraps(self):
        # idx=7, row=1, col=3 — right/topright/bottomright wrap to col=0
        buf = _call_1d(self.fn, 7)
        assert list(buf) == [2, 3, 0, 6, 4, 10, 11, 8]

    def test_top_row_still_invalid(self):
        # idx=0, row=0, col=0 — top neighbours -1 despite EW
        buf = _call_1d(self.fn, 0)
        assert buf[0] == -1 and buf[1] == -1 and buf[2] == -1

    def test_interior_unchanged(self):
        # idx=5, row=1, col=1 — not on EW border, same as normal
        buf_ew     = _call_1d(self.fn, 5)
        buf_normal = _call_1d(
            make_neighbours(NROWS, NCOLS, d8=True, border='normal', indexing='1d'), 5
        )
        assert list(buf_ew) == list(buf_normal)


# ── 1D NS (north-south periodic) ──────────────────────────────────────────────

class TestNS1DD8:
    fn = make_neighbours(NROWS, NCOLS, d8=True, border='ns', indexing='1d')

    def test_top_row_wraps(self):
        # idx=1, row=0, col=1 — top wraps to row=2
        buf = _call_1d(self.fn, 1)
        assert list(buf) == [8, 9, 10, 0, 2, 4, 5, 6]

    def test_bottom_row_wraps(self):
        # idx=9, row=2, col=1 — bottom wraps to row=0
        buf = _call_1d(self.fn, 9)
        assert list(buf) == [4, 5, 6, 8, 10, 0, 1, 2]

    def test_left_col_still_invalid(self):
        # idx=1, row=0, col=1 — left col border for diagonals respected
        buf = _call_1d(self.fn, 4)  # row=1, col=0
        assert buf[0] == -1 and buf[3] == -1 and buf[5] == -1


# ── 2D normal D8 ──────────────────────────────────────────────────────────────

class TestNormal2DD8:
    fn = make_neighbours(NROWS, NCOLS, d8=True, border='normal', indexing='2d')

    def test_center(self):
        buf = _call_2d(self.fn, 1, 1)
        expected = [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)]
        for k, (r, c) in enumerate(expected):
            assert buf[k, 0] == r and buf[k, 1] == c, f"slot {k}"

    def test_top_left_corner(self):
        buf = _call_2d(self.fn, 0, 0)
        # slots 0,1,2,3 invalid; slot 4=(0,1); slot 5 invalid; slot 6=(1,0); slot 7=(1,1)
        for k in (0, 1, 2, 3, 5):
            assert buf[k, 0] == -1 and buf[k, 1] == -1, f"slot {k} should be -1"
        assert buf[4, 0] == 0 and buf[4, 1] == 1
        assert buf[6, 0] == 1 and buf[6, 1] == 0
        assert buf[7, 0] == 1 and buf[7, 1] == 1


# ── 2D EW D4 (mixed options smoke test) ───────────────────────────────────────

class TestEW2DD4:
    fn = make_neighbours(NROWS, NCOLS, d8=False, border='ew', indexing='2d')

    def test_left_edge_wraps_no_diagonals(self):
        # row=1, col=0 — left wraps to col=3, diagonals always -1
        buf = _call_2d(self.fn, 1, 0)
        assert buf[0, 0] == -1  # topleft diagonal
        assert buf[2, 0] == -1  # topright diagonal
        assert buf[5, 0] == -1  # bottomleft diagonal
        assert buf[7, 0] == -1  # bottomright diagonal
        assert buf[3, 0] == 1 and buf[3, 1] == 3  # left wraps
        assert buf[4, 0] == 1 and buf[4, 1] == 1  # right normal
