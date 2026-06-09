"""Grid neighbourer types for 2-D regular grids.

Usage (Python side):
    g = make_grid_nb(nrows, ncols, border='normal', indexing='1dr', d8=False, mask=None)
    buf = np.empty(8, np.int64)
    count = g.neighbours(flat_index, buf)         # 1D types — writes into buf, returns count
    count, nbrs = g.get_neighbour_array(flat_index)  # 1D types — allocates; use for convenience
    buf_r, buf_c = np.empty(8, np.int64), np.empty(8, np.int64)
    count = g.neighbours(r, c, buf_r, buf_c)     # 2D types

--- Available types ---

Nine types are provided, one per combination of border condition and indexing scheme.
They all expose the same methods; only argument/return types differ between 1D and 2D.

  border   indexing  | 1D row-major (1DR) | 1D col-major (1DC) | 2D (row, col)
  ───────────────────┼────────────────────┼────────────────────┼──────────────
  normal             | GridNbNorm1DR      | GridNbNorm1DC      | GridNbNorm2D
  ew  (col wraps)    | GridNbEW1DR        | GridNbEW1DC        | GridNbEW2D
  ns  (row wraps)    | GridNbNS1DR        | GridNbNS1DC        | GridNbNS2D

border:   'normal' = closed edges.  'ew' = left/right columns connect.
          'ns' = top/bottom rows connect.
indexing: '1dr' = flat index row-major (flat = row*ncols + col).
          '1dc' = flat index col-major (flat = col*nrows + row).
          '2d'  = explicit (row, col) pairs as input and output.
d8:       False (default) = D4 neighbours only (N/S/E/W).
          True  = D8, diagonals included.
mask:     optional bool array (nrows, ncols) marking active nodes; None = all active.
          Inactive nodes are never returned as neighbours but can still be queried.

--- Methods ---

1D types:
    .neighbours(flat, buf)        -> count: int64   (writes neighbours into buf[0:count])
    .get_neighbour_array(flat)    -> (count: int64, nbrs: int64[8])  (allocates)
    .is_active(flat)              -> bool
    .north/N/top(flat)       -> int64   (-1 if border or masked)
    .south/S/bottom, .east/E/right, .west/W/left               (same)
    .northwest/NW/topleft, .northeast/NE/topright              (same)
    .southwest/SW/bottomleft, .southeast/SE/bottomright        (same)
    .to_rowmaj(flat) / .to_colmaj(flat) / .to_2d(flat)

2D types:
    .neighbours(row, col, buf_r, buf_c)   -> count: int64   (writes into buf_r/buf_c[0:count])
    .get_neighbour_array(row, col)        -> (count: int64, rows: int64[8], cols: int64[8])
    .is_active(row, col)                  -> bool
    .north(row, col)         -> (int64, int64)   ((-1,-1) if border or masked)
    ... same direction aliases as 1D ...
    .to_rowmaj(row, col) / .to_colmaj(row, col) / .to_2d(row, col)

Fields (accessible on Python proxy): .nrows, .ncols, .d8

--- File structure (numba structref) ---

Numba structref is the recommended way to define JIT-compiled structs. The pattern
used here has three layers per type:

  1. TypeClass  (e.g. GridNbNorm1DRType) — registered with @structref.register;
     tells numba the struct exists and what fields it has.

  2. TypeInstance  (e.g. GridNbNorm1DRInst) — a concrete numba type object created
     from the class. This is what you pass to structref.new() inside @njit to
     allocate a new struct in compiled code.

  3. Proxy class  (e.g. GridNbNorm1DR) — a Python-side StructRefProxy subclass
     that wraps a native struct so it can be passed back to Python after a @njit
     call. All user-facing methods live here (delegating to thin @nb.njit wrappers,
     since struct fields and overload_method methods are not directly callable from
     plain Python).

Methods shared across multiple types are registered in loops using overload_method,
which avoids rewriting the same function nine times.
"""

import numpy as np
import numba as nb
from numba.experimental import structref
from numba.core import types
from numba.core.extending import overload_method

# ── Fields ───────────────────────────────────────────────────────────────────
# Shared field layout for all 9 types. All types carry the same 4 fields;
# the type class (not a runtime flag) encodes border condition and indexing scheme.

def _fields():
    return [
        ('nrows', types.int64),
        ('ncols', types.int64),
        ('mask',  types.bool_[::1]),
        ('d8',    types.boolean),
        ('dx',    types.float64),
        ('dy',    types.float64),
    ]

# ── Type class registrations ─────────────────────────────────────────────────
# Each of the 9 combinations gets its own class so numba can specialise the
# compiled code per (border, indexing) pair at compile time — no runtime
# dispatch overhead. preprocess_fields strips literal types (required boilerplate).

@structref.register
class GridNbNorm1DRType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

@structref.register
class GridNbEW1DRType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

@structref.register
class GridNbNS1DRType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

@structref.register
class GridNbNorm1DCType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

@structref.register
class GridNbEW1DCType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

@structref.register
class GridNbNS1DCType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

@structref.register
class GridNbNorm2DType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

@structref.register
class GridNbEW2DType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

@structref.register
class GridNbNS2DType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

# ── Type instances ───────────────────────────────────────────────────────────
# One concrete numba type per class, created by calling the class with the field
# list. These *Inst objects are what you pass to structref.new() inside @njit.

GridNbNorm1DRInst = GridNbNorm1DRType(_fields())
GridNbEW1DRInst   = GridNbEW1DRType(_fields())
GridNbNS1DRInst   = GridNbNS1DRType(_fields())

GridNbNorm1DCInst = GridNbNorm1DCType(_fields())
GridNbEW1DCInst   = GridNbEW1DCType(_fields())
GridNbNS1DCInst   = GridNbNS1DCType(_fields())

GridNbNorm2DInst  = GridNbNorm2DType(_fields())
GridNbEW2DInst    = GridNbEW2DType(_fields())
GridNbNS2DInst    = GridNbNS2DType(_fields())

# Convenience tuples used below to register the same overload impl across
# several types at once, avoiding copy-paste.
_NORM_TYPES = (GridNbNorm1DRType, GridNbNorm1DCType, GridNbNorm2DType)
_EW_TYPES   = (GridNbEW1DRType,   GridNbEW1DCType,   GridNbEW2DType)
_NS_TYPES   = (GridNbNS1DRType,   GridNbNS1DCType,   GridNbNS2DType)
_1DR_TYPES  = (GridNbNorm1DRType, GridNbEW1DRType,   GridNbNS1DRType)
_1DC_TYPES  = (GridNbNorm1DCType, GridNbEW1DCType,   GridNbNS1DCType)
_2D_TYPES   = (GridNbNorm2DType,  GridNbEW2DType,    GridNbNS2DType)
_1D_TYPES   = _1DR_TYPES + _1DC_TYPES
_ALL_TYPES  = _1D_TYPES + _2D_TYPES

# ── Border resolution ────────────────────────────────────────────────────────
# Internal method _resolve(row, col, dr, dc) -> (nr, nc) steps one cell in
# direction (dr, dc) and applies the border rule. Returns (-1, -1) when the
# step lands outside the grid (or outside the non-periodic axis for EW/NS).
# Three implementations, each registered on the matching border-condition group.

def _ol_resolve_normal(self, row, col, dr, dc):
    def impl(self, row, col, dr, dc):
        nr = row + dr
        nc = col + dc
        if nr < 0 or nr >= self.nrows or nc < 0 or nc >= self.ncols:
            return nb.int64(-1), nb.int64(-1)
        return nr, nc
    return impl

def _ol_resolve_ew(self, row, col, dr, dc):
    def impl(self, row, col, dr, dc):
        nr = row + dr
        if nr < 0 or nr >= self.nrows:
            return nb.int64(-1), nb.int64(-1)
        nc = (col + dc + self.ncols) % self.ncols
        return nr, nc
    return impl

def _ol_resolve_ns(self, row, col, dr, dc):
    def impl(self, row, col, dr, dc):
        nc = col + dc
        if nc < 0 or nc >= self.ncols:
            return nb.int64(-1), nb.int64(-1)
        nr = (row + dr + self.nrows) % self.nrows
        return nr, nc
    return impl

for _T in _NORM_TYPES:
    overload_method(_T, '_resolve')(_ol_resolve_normal)
for _T in _EW_TYPES:
    overload_method(_T, '_resolve')(_ol_resolve_ew)
for _T in _NS_TYPES:
    overload_method(_T, '_resolve')(_ol_resolve_ns)

# ── Index conversion helpers (internal, for 1D types) ────────────────────────
# _flat_to_rc / _rc_to_flat translate between flat 1D index and (row, col).
# The formula differs between row-major (flat = row*ncols + col) and col-major
# (flat = col*nrows + row). 2D types skip these — they work in (row, col) natively.
# The mask is always stored row-major, so all mask lookups use row*ncols+col.

def _ol_1dr_flat_to_rc(self, flat):
    def impl(self, flat):
        return flat // self.ncols, flat % self.ncols
    return impl

def _ol_1dc_flat_to_rc(self, flat):
    def impl(self, flat):
        return flat % self.nrows, flat // self.nrows
    return impl

def _ol_1dr_rc_to_flat(self, row, col):
    def impl(self, row, col):
        return row * self.ncols + col
    return impl

def _ol_1dc_rc_to_flat(self, row, col):
    def impl(self, row, col):
        return col * self.nrows + row
    return impl

for _T in _1DR_TYPES:
    overload_method(_T, '_flat_to_rc')(_ol_1dr_flat_to_rc)
    overload_method(_T, '_rc_to_flat')(_ol_1dr_rc_to_flat)
for _T in _1DC_TYPES:
    overload_method(_T, '_flat_to_rc')(_ol_1dc_flat_to_rc)
    overload_method(_T, '_rc_to_flat')(_ol_1dc_rc_to_flat)

# ── is_active ────────────────────────────────────────────────────────────────
# Checks mask[row*ncols+col]. For 1D types the flat index is first converted to
# (row, col) so the row-major mask lookup is always consistent.

def _ol_is_active_1d(self, flat_idx):
    def impl(self, flat_idx):
        row, col = self._flat_to_rc(flat_idx)
        return self.mask[row * self.ncols + col]
    return impl

def _ol_is_active_2d(self, row, col):
    def impl(self, row, col):
        return self.mask[row * self.ncols + col]
    return impl

for _T in _1D_TYPES:
    overload_method(_T, 'is_active')(_ol_is_active_1d)
for _T in _2D_TYPES:
    overload_method(_T, 'is_active')(_ol_is_active_2d)

# ── Cross-scheme index conversion (public) ───────────────────────────────────
# to_rowmaj / to_colmaj / to_2d let callers convert an index from the neighbourer's
# native scheme to any other scheme — useful when mixing a 1DC neighbourer with
# a row-major data array, for example. Each is a trivial arithmetic formula;
# three variants per method (one per indexing scheme), registered via loops.

def _ol_to_rowmaj_1dr(self, flat):
    def impl(self, flat): return flat
    return impl

def _ol_to_rowmaj_1dc(self, flat):
    def impl(self, flat):
        row = flat % self.nrows
        col = flat // self.nrows
        return row * self.ncols + col
    return impl

def _ol_to_rowmaj_2d(self, row, col):
    def impl(self, row, col): return row * self.ncols + col
    return impl

def _ol_to_colmaj_1dr(self, flat):
    def impl(self, flat):
        row = flat // self.ncols
        col = flat % self.ncols
        return col * self.nrows + row
    return impl

def _ol_to_colmaj_1dc(self, flat):
    def impl(self, flat): return flat
    return impl

def _ol_to_colmaj_2d(self, row, col):
    def impl(self, row, col): return col * self.nrows + row
    return impl

def _ol_to_2d_1dr(self, flat):
    def impl(self, flat): return flat // self.ncols, flat % self.ncols
    return impl

def _ol_to_2d_1dc(self, flat):
    def impl(self, flat): return flat % self.nrows, flat // self.nrows
    return impl

def _ol_to_2d_2d(self, row, col):
    def impl(self, row, col): return row, col
    return impl

for _T in _1DR_TYPES:
    overload_method(_T, 'to_rowmaj')(_ol_to_rowmaj_1dr)
    overload_method(_T, 'to_colmaj')(_ol_to_colmaj_1dr)
    overload_method(_T, 'to_2d')(_ol_to_2d_1dr)
for _T in _1DC_TYPES:
    overload_method(_T, 'to_rowmaj')(_ol_to_rowmaj_1dc)
    overload_method(_T, 'to_colmaj')(_ol_to_colmaj_1dc)
    overload_method(_T, 'to_2d')(_ol_to_2d_1dc)
for _T in _2D_TYPES:
    overload_method(_T, 'to_rowmaj')(_ol_to_rowmaj_2d)
    overload_method(_T, 'to_colmaj')(_ol_to_colmaj_2d)
    overload_method(_T, 'to_2d')(_ol_to_2d_2d)

# ── Direction methods ─────────────────────────────────────────────────────────
# _make_dir_1d / _make_dir_2d are factories: they return an overload function
# for one specific (dr, dc) offset, capturing the offset as a closure constant.
# This avoids writing 8 nearly-identical functions by hand.
# _DIR_OFFSETS maps the 8 canonical names to their (dr, dc) offsets;
# _DIR_ALIASES maps every alias (N, top, topleft, …) to the canonical name.
# The loop at the end registers all 8 primaries + 16 aliases on all 9 types.

def _make_dir_1d(dr, dc):
    def _ol(self, flat_idx):
        def impl(self, flat_idx):
            row, col = self._flat_to_rc(flat_idx)
            nr, nc = self._resolve(row, col, dr, dc)
            if nr < 0 or not self.mask[nr * self.ncols + nc]:
                return nb.int64(-1)
            return self._rc_to_flat(nr, nc)
        return impl
    return _ol

def _make_dir_2d(dr, dc):
    def _ol(self, row, col):
        def impl(self, row, col):
            nr, nc = self._resolve(row, col, dr, dc)
            if nr < 0 or not self.mask[nr * self.ncols + nc]:
                return nb.int64(-1), nb.int64(-1)
            return nr, nc
        return impl
    return _ol

_DIR_OFFSETS = {
    'north':     (-1,  0), 'south':     ( 1,  0),
    'east':      ( 0,  1), 'west':      ( 0, -1),
    'northwest': (-1, -1), 'northeast': (-1,  1),
    'southwest': ( 1, -1), 'southeast': ( 1,  1),
}
_DIR_ALIASES = {
    'N': 'north',     'S': 'south',     'E': 'east',      'W': 'west',
    'NW': 'northwest','NE': 'northeast','SW': 'southwest','SE': 'southeast',
    'top': 'north',   'bottom': 'south','right': 'east',  'left': 'west',
    'topleft': 'northwest', 'topright': 'northeast',
    'bottomleft': 'southwest', 'bottomright': 'southeast',
}

_dir1d = {name: _make_dir_1d(dr, dc) for name, (dr, dc) in _DIR_OFFSETS.items()}
_dir2d = {name: _make_dir_2d(dr, dc) for name, (dr, dc) in _DIR_OFFSETS.items()}

for _T in _1D_TYPES:
    for _name, _ol in _dir1d.items():
        overload_method(_T, _name)(_ol)
    for _alias, _canon in _DIR_ALIASES.items():
        overload_method(_T, _alias)(_dir1d[_canon])
for _T in _2D_TYPES:
    for _name, _ol in _dir2d.items():
        overload_method(_T, _name)(_ol)
    for _alias, _canon in _DIR_ALIASES.items():
        overload_method(_T, _alias)(_dir2d[_canon])

# ── neighbours / get_neighbour_array ─────────────────────────────────────────
# neighbours(flat, buf) — writes into a caller-supplied int64[8] buffer, returns
# count. Zero allocation in the hot path.
# get_neighbour_array(flat) — allocates a fresh int64[8] and returns (count, arr).
# Use get_neighbour_array for convenience; use neighbours in performance-critical loops.
# The d8 block is guarded by an if so diagonal checks are skipped when d8=False.

def _ol_neighbours_1d(self, flat_idx, buf):
    def impl(self, flat_idx, buf):
        row, col = self._flat_to_rc(flat_idx)
        count = nb.int64(0)
        nr, nc = self._resolve(row, col, -1, 0)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            buf[count] = self._rc_to_flat(nr, nc); count += 1
        nr, nc = self._resolve(row, col, 1, 0)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            buf[count] = self._rc_to_flat(nr, nc); count += 1
        nr, nc = self._resolve(row, col, 0, 1)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            buf[count] = self._rc_to_flat(nr, nc); count += 1
        nr, nc = self._resolve(row, col, 0, -1)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            buf[count] = self._rc_to_flat(nr, nc); count += 1
        if self.d8:
            nr, nc = self._resolve(row, col, -1, -1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                buf[count] = self._rc_to_flat(nr, nc); count += 1
            nr, nc = self._resolve(row, col, -1, 1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                buf[count] = self._rc_to_flat(nr, nc); count += 1
            nr, nc = self._resolve(row, col, 1, -1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                buf[count] = self._rc_to_flat(nr, nc); count += 1
            nr, nc = self._resolve(row, col, 1, 1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                buf[count] = self._rc_to_flat(nr, nc); count += 1
        return count
    return impl

def _ol_neighbours_2d(self, row, col, buf_r, buf_c):
    def impl(self, row, col, buf_r, buf_c):
        count = nb.int64(0)
        nr, nc = self._resolve(row, col, -1, 0)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            buf_r[count] = nr; buf_c[count] = nc; count += 1
        nr, nc = self._resolve(row, col, 1, 0)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            buf_r[count] = nr; buf_c[count] = nc; count += 1
        nr, nc = self._resolve(row, col, 0, 1)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            buf_r[count] = nr; buf_c[count] = nc; count += 1
        nr, nc = self._resolve(row, col, 0, -1)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            buf_r[count] = nr; buf_c[count] = nc; count += 1
        if self.d8:
            nr, nc = self._resolve(row, col, -1, -1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                buf_r[count] = nr; buf_c[count] = nc; count += 1
            nr, nc = self._resolve(row, col, -1, 1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                buf_r[count] = nr; buf_c[count] = nc; count += 1
            nr, nc = self._resolve(row, col, 1, -1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                buf_r[count] = nr; buf_c[count] = nc; count += 1
            nr, nc = self._resolve(row, col, 1, 1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                buf_r[count] = nr; buf_c[count] = nc; count += 1
        return count
    return impl

def _ol_get_neighbour_array_1d(self, flat_idx):
    def impl(self, flat_idx):
        out = np.empty(nb.int64(8), nb.int64)
        count = self.neighbours(flat_idx, out)
        return count, out
    return impl

def _ol_get_neighbour_array_2d(self, row, col):
    def impl(self, row, col):
        out_r = np.empty(nb.int64(8), nb.int64)
        out_c = np.empty(nb.int64(8), nb.int64)
        count = self.neighbours(row, col, out_r, out_c)
        return count, out_r, out_c
    return impl

for _T in _1D_TYPES:
    overload_method(_T, 'neighbours')(_ol_neighbours_1d)
    overload_method(_T, 'get_neighbour_array')(_ol_get_neighbour_array_1d)
for _T in _2D_TYPES:
    overload_method(_T, 'neighbours')(_ol_neighbours_2d)
    overload_method(_T, 'get_neighbour_array')(_ol_get_neighbour_array_2d)

# ── distances and neighbours_dist ────────────────────────────────────────────
# distances(idx) returns (count, float64[8]) — same ordering/masking as neighbours.
# neighbours_dist(idx) returns (count, int64[8], float64[8]) — both in one pass.
# N/S distance = dy, E/W = dx, diagonals = sqrt(dx²+dy²).

def _ol_distances_1d(self, flat_idx):
    def impl(self, flat_idx):
        row, col = self._flat_to_rc(flat_idx)
        out_d = np.empty(nb.int64(8), nb.float64)
        count = nb.int64(0)
        diag = (self.dx * self.dx + self.dy * self.dy) ** 0.5
        nr, nc = self._resolve(row, col, -1, 0)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out_d[count] = self.dy; count += 1
        nr, nc = self._resolve(row, col, 1, 0)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out_d[count] = self.dy; count += 1
        nr, nc = self._resolve(row, col, 0, 1)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out_d[count] = self.dx; count += 1
        nr, nc = self._resolve(row, col, 0, -1)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out_d[count] = self.dx; count += 1
        if self.d8:
            nr, nc = self._resolve(row, col, -1, -1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out_d[count] = diag; count += 1
            nr, nc = self._resolve(row, col, -1, 1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out_d[count] = diag; count += 1
            nr, nc = self._resolve(row, col, 1, -1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out_d[count] = diag; count += 1
            nr, nc = self._resolve(row, col, 1, 1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out_d[count] = diag; count += 1
        return count, out_d
    return impl

def _ol_distances_2d(self, row, col):
    def impl(self, row, col):
        out_d = np.empty(nb.int64(8), nb.float64)
        count = nb.int64(0)
        diag = (self.dx * self.dx + self.dy * self.dy) ** 0.5
        nr, nc = self._resolve(row, col, -1, 0)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out_d[count] = self.dy; count += 1
        nr, nc = self._resolve(row, col, 1, 0)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out_d[count] = self.dy; count += 1
        nr, nc = self._resolve(row, col, 0, 1)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out_d[count] = self.dx; count += 1
        nr, nc = self._resolve(row, col, 0, -1)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out_d[count] = self.dx; count += 1
        if self.d8:
            nr, nc = self._resolve(row, col, -1, -1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out_d[count] = diag; count += 1
            nr, nc = self._resolve(row, col, -1, 1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out_d[count] = diag; count += 1
            nr, nc = self._resolve(row, col, 1, -1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out_d[count] = diag; count += 1
            nr, nc = self._resolve(row, col, 1, 1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out_d[count] = diag; count += 1
        return count, out_d
    return impl

def _ol_neighbours_dist_1d(self, flat_idx):
    def impl(self, flat_idx):
        row, col = self._flat_to_rc(flat_idx)
        out   = np.empty(nb.int64(8), nb.int64)
        out_d = np.empty(nb.int64(8), nb.float64)
        count = nb.int64(0)
        diag = (self.dx * self.dx + self.dy * self.dy) ** 0.5
        nr, nc = self._resolve(row, col, -1, 0)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out[count] = self._rc_to_flat(nr, nc); out_d[count] = self.dy; count += 1
        nr, nc = self._resolve(row, col, 1, 0)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out[count] = self._rc_to_flat(nr, nc); out_d[count] = self.dy; count += 1
        nr, nc = self._resolve(row, col, 0, 1)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out[count] = self._rc_to_flat(nr, nc); out_d[count] = self.dx; count += 1
        nr, nc = self._resolve(row, col, 0, -1)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out[count] = self._rc_to_flat(nr, nc); out_d[count] = self.dx; count += 1
        if self.d8:
            nr, nc = self._resolve(row, col, -1, -1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out[count] = self._rc_to_flat(nr, nc); out_d[count] = diag; count += 1
            nr, nc = self._resolve(row, col, -1, 1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out[count] = self._rc_to_flat(nr, nc); out_d[count] = diag; count += 1
            nr, nc = self._resolve(row, col, 1, -1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out[count] = self._rc_to_flat(nr, nc); out_d[count] = diag; count += 1
            nr, nc = self._resolve(row, col, 1, 1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out[count] = self._rc_to_flat(nr, nc); out_d[count] = diag; count += 1
        return count, out, out_d
    return impl

def _ol_neighbours_dist_2d(self, row, col):
    def impl(self, row, col):
        out_r = np.empty(nb.int64(8), nb.int64)
        out_c = np.empty(nb.int64(8), nb.int64)
        out_d = np.empty(nb.int64(8), nb.float64)
        count = nb.int64(0)
        diag = (self.dx * self.dx + self.dy * self.dy) ** 0.5
        nr, nc = self._resolve(row, col, -1, 0)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out_r[count] = nr; out_c[count] = nc; out_d[count] = self.dy; count += 1
        nr, nc = self._resolve(row, col, 1, 0)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out_r[count] = nr; out_c[count] = nc; out_d[count] = self.dy; count += 1
        nr, nc = self._resolve(row, col, 0, 1)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out_r[count] = nr; out_c[count] = nc; out_d[count] = self.dx; count += 1
        nr, nc = self._resolve(row, col, 0, -1)
        if nr >= 0 and self.mask[nr * self.ncols + nc]:
            out_r[count] = nr; out_c[count] = nc; out_d[count] = self.dx; count += 1
        if self.d8:
            nr, nc = self._resolve(row, col, -1, -1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out_r[count] = nr; out_c[count] = nc; out_d[count] = diag; count += 1
            nr, nc = self._resolve(row, col, -1, 1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out_r[count] = nr; out_c[count] = nc; out_d[count] = diag; count += 1
            nr, nc = self._resolve(row, col, 1, -1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out_r[count] = nr; out_c[count] = nc; out_d[count] = diag; count += 1
            nr, nc = self._resolve(row, col, 1, 1)
            if nr >= 0 and self.mask[nr * self.ncols + nc]:
                out_r[count] = nr; out_c[count] = nc; out_d[count] = diag; count += 1
        return count, out_r, out_c, out_d
    return impl

for _T in _1D_TYPES:
    overload_method(_T, 'distances')(_ol_distances_1d)
    overload_method(_T, 'neighbours_dist')(_ol_neighbours_dist_1d)
for _T in _2D_TYPES:
    overload_method(_T, 'distances')(_ol_distances_2d)
    overload_method(_T, 'neighbours_dist')(_ol_neighbours_dist_2d)

# ── Proxy classes and boxing ─────────────────────────────────────────────────
# One StructRefProxy subclass per type. define_boxing tells numba how to wrap a
# native struct back into a Python object when it crosses the @njit boundary.
# Methods are injected later via the mixin (see below).

class GridNbNorm1DR(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

class GridNbEW1DR(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

class GridNbNS1DR(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

class GridNbNorm1DC(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

class GridNbEW1DC(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

class GridNbNS1DC(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

class GridNbNorm2D(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

class GridNbEW2D(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

class GridNbNS2D(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

structref.define_boxing(GridNbNorm1DRType, GridNbNorm1DR)
structref.define_boxing(GridNbEW1DRType,   GridNbEW1DR)
structref.define_boxing(GridNbNS1DRType,   GridNbNS1DR)
structref.define_boxing(GridNbNorm1DCType, GridNbNorm1DC)
structref.define_boxing(GridNbEW1DCType,   GridNbEW1DC)
structref.define_boxing(GridNbNS1DCType,   GridNbNS1DC)
structref.define_boxing(GridNbNorm2DType,  GridNbNorm2D)
structref.define_boxing(GridNbEW2DType,    GridNbEW2D)
structref.define_boxing(GridNbNS2DType,    GridNbNS2D)

# ── @njit wrappers ───────────────────────────────────────────────────────────
# Struct fields and overload_method methods cannot be called directly on a proxy
# object from Python — they only exist in compiled code. Each thin @nb.njit
# wrapper compiles once on first call and is reused for every subsequent Python-
# side call. The mixin delegates to these rather than duplicating the njit logic.

@nb.njit
def _nb_nrows(g):      return g.nrows
@nb.njit
def _nb_ncols(g):      return g.ncols
@nb.njit
def _nb_d8(g):         return g.d8
@nb.njit
def _nb_dx(g):         return g.dx
@nb.njit
def _nb_dy(g):         return g.dy
@nb.njit
def _nb_is_active_1d(g, flat):        return g.is_active(flat)
@nb.njit
def _nb_is_active_2d(g, row, col):    return g.is_active(row, col)
@nb.njit
def _nb_neighbours_1d(g, flat, buf):        return g.neighbours(flat, buf)
@nb.njit
def _nb_neighbours_2d(g, row, col, br, bc): return g.neighbours(row, col, br, bc)
@nb.njit
def _nb_get_neighbour_array_1d(g, flat):       return g.get_neighbour_array(flat)
@nb.njit
def _nb_get_neighbour_array_2d(g, row, col):   return g.get_neighbour_array(row, col)
@nb.njit
def _nb_north_1d(g, flat):            return g.north(flat)
@nb.njit
def _nb_south_1d(g, flat):            return g.south(flat)
@nb.njit
def _nb_east_1d(g, flat):             return g.east(flat)
@nb.njit
def _nb_west_1d(g, flat):             return g.west(flat)
@nb.njit
def _nb_northwest_1d(g, flat):        return g.northwest(flat)
@nb.njit
def _nb_northeast_1d(g, flat):        return g.northeast(flat)
@nb.njit
def _nb_southwest_1d(g, flat):        return g.southwest(flat)
@nb.njit
def _nb_southeast_1d(g, flat):        return g.southeast(flat)
@nb.njit
def _nb_north_2d(g, row, col):        return g.north(row, col)
@nb.njit
def _nb_south_2d(g, row, col):        return g.south(row, col)
@nb.njit
def _nb_east_2d(g, row, col):         return g.east(row, col)
@nb.njit
def _nb_west_2d(g, row, col):         return g.west(row, col)
@nb.njit
def _nb_northwest_2d(g, row, col):    return g.northwest(row, col)
@nb.njit
def _nb_northeast_2d(g, row, col):    return g.northeast(row, col)
@nb.njit
def _nb_southwest_2d(g, row, col):    return g.southwest(row, col)
@nb.njit
def _nb_southeast_2d(g, row, col):    return g.southeast(row, col)
@nb.njit
def _nb_to_rowmaj_1d(g, flat):        return g.to_rowmaj(flat)
@nb.njit
def _nb_to_colmaj_1d(g, flat):        return g.to_colmaj(flat)
@nb.njit
def _nb_to_2d_1d(g, flat):            return g.to_2d(flat)
@nb.njit
def _nb_to_rowmaj_2d(g, row, col):    return g.to_rowmaj(row, col)
@nb.njit
def _nb_to_colmaj_2d(g, row, col):    return g.to_colmaj(row, col)
@nb.njit
def _nb_to_2d_2d(g, row, col):        return g.to_2d(row, col)
@nb.njit
def _nb_distances_1d(g, flat):        return g.distances(flat)
@nb.njit
def _nb_distances_2d(g, row, col):    return g.distances(row, col)
@nb.njit
def _nb_neighbours_dist_1d(g, flat):       return g.neighbours_dist(flat)
@nb.njit
def _nb_neighbours_dist_2d(g, row, col):   return g.neighbours_dist(row, col)

# ── Proxy mixins ─────────────────────────────────────────────────────────────
# Two mixins — one for 1D types, one for 2D — provide the Python-side method
# interface by forwarding to the @njit wrappers above. Aliases (N, top, NW, …)
# are plain Python delegates here; they do not need separate @njit wrappers.
# The mixin is injected into the proxy __bases__ after the classes are defined,
# because define_boxing must run before the mixin alters the MRO.

class _GridNb1DProxyMixin:
    @property
    def nrows(self): return _nb_nrows(self)
    @property
    def ncols(self): return _nb_ncols(self)
    @property
    def d8(self): return _nb_d8(self)
    @property
    def dx(self): return _nb_dx(self)
    @property
    def dy(self): return _nb_dy(self)

    def is_active(self, flat):                       return _nb_is_active_1d(self, flat)
    def neighbours(self, flat, buf):                 return _nb_neighbours_1d(self, flat, buf)
    def get_neighbour_array(self, flat):             return _nb_get_neighbour_array_1d(self, flat)
    def distances(self, flat):                       return _nb_distances_1d(self, flat)
    def neighbours_dist(self, flat):                 return _nb_neighbours_dist_1d(self, flat)

    def north(self, flat):          return _nb_north_1d(self, flat)
    def south(self, flat):          return _nb_south_1d(self, flat)
    def east(self, flat):           return _nb_east_1d(self, flat)
    def west(self, flat):           return _nb_west_1d(self, flat)
    def northwest(self, flat):      return _nb_northwest_1d(self, flat)
    def northeast(self, flat):      return _nb_northeast_1d(self, flat)
    def southwest(self, flat):      return _nb_southwest_1d(self, flat)
    def southeast(self, flat):      return _nb_southeast_1d(self, flat)

    def N(self, flat):              return _nb_north_1d(self, flat)
    def S(self, flat):              return _nb_south_1d(self, flat)
    def E(self, flat):              return _nb_east_1d(self, flat)
    def W(self, flat):              return _nb_west_1d(self, flat)
    def NW(self, flat):             return _nb_northwest_1d(self, flat)
    def NE(self, flat):             return _nb_northeast_1d(self, flat)
    def SW(self, flat):             return _nb_southwest_1d(self, flat)
    def SE(self, flat):             return _nb_southeast_1d(self, flat)

    def top(self, flat):            return _nb_north_1d(self, flat)
    def bottom(self, flat):         return _nb_south_1d(self, flat)
    def right(self, flat):          return _nb_east_1d(self, flat)
    def left(self, flat):           return _nb_west_1d(self, flat)
    def topleft(self, flat):        return _nb_northwest_1d(self, flat)
    def topright(self, flat):       return _nb_northeast_1d(self, flat)
    def bottomleft(self, flat):     return _nb_southwest_1d(self, flat)
    def bottomright(self, flat):    return _nb_southeast_1d(self, flat)

    def to_rowmaj(self, flat):      return _nb_to_rowmaj_1d(self, flat)
    def to_colmaj(self, flat):      return _nb_to_colmaj_1d(self, flat)
    def to_2d(self, flat):          return _nb_to_2d_1d(self, flat)


class _GridNb2DProxyMixin:
    @property
    def nrows(self): return _nb_nrows(self)
    @property
    def ncols(self): return _nb_ncols(self)
    @property
    def d8(self): return _nb_d8(self)
    @property
    def dx(self): return _nb_dx(self)
    @property
    def dy(self): return _nb_dy(self)

    def is_active(self, row, col):                        return _nb_is_active_2d(self, row, col)
    def neighbours(self, row, col, buf_r, buf_c):         return _nb_neighbours_2d(self, row, col, buf_r, buf_c)
    def get_neighbour_array(self, row, col):              return _nb_get_neighbour_array_2d(self, row, col)
    def distances(self, row, col):                        return _nb_distances_2d(self, row, col)
    def neighbours_dist(self, row, col):                  return _nb_neighbours_dist_2d(self, row, col)

    def north(self, row, col):       return _nb_north_2d(self, row, col)
    def south(self, row, col):       return _nb_south_2d(self, row, col)
    def east(self, row, col):        return _nb_east_2d(self, row, col)
    def west(self, row, col):        return _nb_west_2d(self, row, col)
    def northwest(self, row, col):   return _nb_northwest_2d(self, row, col)
    def northeast(self, row, col):   return _nb_northeast_2d(self, row, col)
    def southwest(self, row, col):   return _nb_southwest_2d(self, row, col)
    def southeast(self, row, col):   return _nb_southeast_2d(self, row, col)

    def N(self, row, col):           return _nb_north_2d(self, row, col)
    def S(self, row, col):           return _nb_south_2d(self, row, col)
    def E(self, row, col):           return _nb_east_2d(self, row, col)
    def W(self, row, col):           return _nb_west_2d(self, row, col)
    def NW(self, row, col):          return _nb_northwest_2d(self, row, col)
    def NE(self, row, col):          return _nb_northeast_2d(self, row, col)
    def SW(self, row, col):          return _nb_southwest_2d(self, row, col)
    def SE(self, row, col):          return _nb_southeast_2d(self, row, col)

    def top(self, row, col):         return _nb_north_2d(self, row, col)
    def bottom(self, row, col):      return _nb_south_2d(self, row, col)
    def right(self, row, col):       return _nb_east_2d(self, row, col)
    def left(self, row, col):        return _nb_west_2d(self, row, col)
    def topleft(self, row, col):     return _nb_northwest_2d(self, row, col)
    def topright(self, row, col):    return _nb_northeast_2d(self, row, col)
    def bottomleft(self, row, col):  return _nb_southwest_2d(self, row, col)
    def bottomright(self, row, col): return _nb_southeast_2d(self, row, col)

    def to_rowmaj(self, row, col):   return _nb_to_rowmaj_2d(self, row, col)
    def to_colmaj(self, row, col):   return _nb_to_colmaj_2d(self, row, col)
    def to_2d(self, row, col):       return _nb_to_2d_2d(self, row, col)


# Inject mixin into proxy classes
for _cls in (GridNbNorm1DR, GridNbEW1DR, GridNbNS1DR,
             GridNbNorm1DC, GridNbEW1DC, GridNbNS1DC):
    _cls.__bases__ = (_GridNb1DProxyMixin,) + _cls.__bases__

for _cls in (GridNbNorm2D, GridNbEW2D, GridNbNS2D):
    _cls.__bases__ = (_GridNb2DProxyMixin,) + _cls.__bases__

# ── @njit constructors ───────────────────────────────────────────────────────
# One thin constructor per type: allocates a new struct and sets its four fields.
# Called by the Python factory below; can also be called directly inside @njit
# when you need to build a neighbourer entirely in compiled code.

@nb.njit
def _ctor_norm_1dr(nrows, ncols, mask, d8, dx, dy):
    h = structref.new(GridNbNorm1DRInst)
    h.nrows = nrows; h.ncols = ncols; h.mask = mask; h.d8 = d8; h.dx = dx; h.dy = dy
    return h

@nb.njit
def _ctor_ew_1dr(nrows, ncols, mask, d8, dx, dy):
    h = structref.new(GridNbEW1DRInst)
    h.nrows = nrows; h.ncols = ncols; h.mask = mask; h.d8 = d8; h.dx = dx; h.dy = dy
    return h

@nb.njit
def _ctor_ns_1dr(nrows, ncols, mask, d8, dx, dy):
    h = structref.new(GridNbNS1DRInst)
    h.nrows = nrows; h.ncols = ncols; h.mask = mask; h.d8 = d8; h.dx = dx; h.dy = dy
    return h

@nb.njit
def _ctor_norm_1dc(nrows, ncols, mask, d8, dx, dy):
    h = structref.new(GridNbNorm1DCInst)
    h.nrows = nrows; h.ncols = ncols; h.mask = mask; h.d8 = d8; h.dx = dx; h.dy = dy
    return h

@nb.njit
def _ctor_ew_1dc(nrows, ncols, mask, d8, dx, dy):
    h = structref.new(GridNbEW1DCInst)
    h.nrows = nrows; h.ncols = ncols; h.mask = mask; h.d8 = d8; h.dx = dx; h.dy = dy
    return h

@nb.njit
def _ctor_ns_1dc(nrows, ncols, mask, d8, dx, dy):
    h = structref.new(GridNbNS1DCInst)
    h.nrows = nrows; h.ncols = ncols; h.mask = mask; h.d8 = d8; h.dx = dx; h.dy = dy
    return h

@nb.njit
def _ctor_norm_2d(nrows, ncols, mask, d8, dx, dy):
    h = structref.new(GridNbNorm2DInst)
    h.nrows = nrows; h.ncols = ncols; h.mask = mask; h.d8 = d8; h.dx = dx; h.dy = dy
    return h

@nb.njit
def _ctor_ew_2d(nrows, ncols, mask, d8, dx, dy):
    h = structref.new(GridNbEW2DInst)
    h.nrows = nrows; h.ncols = ncols; h.mask = mask; h.d8 = d8; h.dx = dx; h.dy = dy
    return h

@nb.njit
def _ctor_ns_2d(nrows, ncols, mask, d8, dx, dy):
    h = structref.new(GridNbNS2DInst)
    h.nrows = nrows; h.ncols = ncols; h.mask = mask; h.d8 = d8; h.dx = dx; h.dy = dy
    return h

_CTORS = {
    ('normal', '1dr'): _ctor_norm_1dr,
    ('ew',     '1dr'): _ctor_ew_1dr,
    ('ns',     '1dr'): _ctor_ns_1dr,
    ('normal', '1dc'): _ctor_norm_1dc,
    ('ew',     '1dc'): _ctor_ew_1dc,
    ('ns',     '1dc'): _ctor_ns_1dc,
    ('normal', '2d'):  _ctor_norm_2d,
    ('ew',     '2d'):  _ctor_ew_2d,
    ('ns',     '2d'):  _ctor_ns_2d,
}

# ── Public factory ───────────────────────────────────────────────────────────

def make_grid_nb(nrows: int, ncols: int, border: str = 'normal',
                 indexing: str = '1dr', d8: bool = False,
                 mask: np.ndarray = None, dx: float = 1.0, dy: float = 1.0):
    """Create a grid neighbourer.

    border:   'normal' | 'ew' (periodic left-right) | 'ns' (periodic top-bottom)
    indexing: '1dr' (1D row-major) | '1dc' (1D col-major) | '2d' (row, col)
    d8:       True to include diagonal neighbours (default False = D4 only)
    mask:     bool array shape (nrows, ncols) or (nrows*ncols,); None = all active
    dx:       physical cell spacing in the x (column) direction (default 1.0)
    dy:       physical cell spacing in the y (row) direction (default 1.0)
    """
    key = (border.lower(), indexing.lower())
    if key not in _CTORS:
        raise ValueError(f"Unknown border={border!r} or indexing={indexing!r}")
    if mask is None:
        m = np.ones(nrows * ncols, dtype=np.bool_)
    else:
        m = np.asarray(mask, dtype=np.bool_).ravel()
        if m.size != nrows * ncols:
            raise ValueError("mask size must equal nrows * ncols")
    return _CTORS[key](nb.int64(nrows), nb.int64(ncols), m, d8,
                       nb.float64(dx), nb.float64(dy))
