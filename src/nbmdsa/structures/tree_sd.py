"""Steepest-descent tree structures built from a 2-D elevation grid.

Each node's parent is its neighbour with the steepest downward slope (dz/distance).
A node with no lower neighbour is its own parent (parent[i] == i) — it is a root.

Three variants trading memory for query speed:

  SteepTreeImpl  — fully implicit.  Stores only z.  parent/children/is_root
                   are computed on the fly and require the neighbourer (g) at
                   query time.  Zero extra memory beyond z itself.

  SteepTreePar   — parent explicit.  Stores z + parents[n].  parent(idx) and
                   is_root(idx) are O(1).  children(idx, g) still scans via the
                   neighbourer (each neighbour whose parent is idx).

  SteepTreeFull  — fully explicit.  Stores z + parents[n] + CSR children arrays
                   (ptr[n+1] + flat[?]).  All queries are O(1) or O(k children);
                   no neighbourer needed at query time.

API signature differences are intentional and reflect the storage trade-off:

  variant       | parent(idx)   | children(idx)   | is_root(idx)
  ──────────────┼───────────────┼─────────────────┼──────────────
  SteepTreeImpl | parent(idx,g) | children(idx,g) | is_root(idx,g)
  SteepTreePar  | parent(idx)   | children(idx,g) | is_root(idx)
  SteepTreeFull | parent(idx)   | children(idx)   | is_root(idx)

Common methods (same signature on all three):
    .get_z(idx)  -> float64     elevation at node idx
    .size()      -> int64       total number of nodes (= len(z))

Numba type instances (for structref.new() inside @njit):
    SteepTreeImplInst, SteepTreeParInst, SteepTreeFullInst

Factories (Python-side, accept any 1D neighbourer):
    make_steep_tree_impl(z)        — wraps z, no computation
    make_steep_tree_par(z, g)      — builds parents array
    make_steep_tree_full(z, g)     — builds parents + CSR children
"""

import numpy as np
import numba as nb
from numba.experimental import structref
from numba.core import types
from numba.core.extending import overload_method

# ── Fields ────────────────────────────────────────────────────────────────────

def _fields_impl():
    return [('z', types.float64[::1])]

def _fields_par():
    return [('z', types.float64[::1]), ('parents', types.int64[::1])]

def _fields_full():
    return [
        ('z',             types.float64[::1]),
        ('parents',       types.int64[::1]),
        ('children_ptr',  types.int64[::1]),
        ('children_flat', types.int64[::1]),
    ]

# ── Type class registrations ──────────────────────────────────────────────────

@structref.register
class SteepTreeImplType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

@structref.register
class SteepTreeParType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

@structref.register
class SteepTreeFullType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

# ── Type instances ────────────────────────────────────────────────────────────

SteepTreeImplInst = SteepTreeImplType(_fields_impl())
SteepTreeParInst  = SteepTreeParType(_fields_par())
SteepTreeFullInst = SteepTreeFullType(_fields_full())

# ── Core steepest-parent computation ─────────────────────────────────────────
# Standalone @njit function so it can be called from both overload_method impls
# and from the @njit builders below.  Polymorphic in g — numba specialises per
# neighbourer type.  Returns idx itself when no strictly lower neighbour exists.

@nb.njit
def _steep_parent(z, idx, g):
    cnt, nbrs, dists = g.neighbours_dist(idx)
    best_slope = nb.float64(0.0)
    best = idx
    for k in range(cnt):
        nb_idx = nbrs[k]
        dz = z[idx] - z[nb_idx]
        if dz > 0.0:
            slope = dz / dists[k]
            if slope > best_slope:
                best_slope = slope
                best = nb_idx
    return best

# ── SteepTreeImpl overloads ───────────────────────────────────────────────────

@overload_method(SteepTreeImplType, 'parent')
def _ol_impl_parent(self, idx, g):
    def impl(self, idx, g):
        return _steep_parent(self.z, idx, g)
    return impl

@overload_method(SteepTreeImplType, 'is_root')
def _ol_impl_is_root(self, idx, g):
    def impl(self, idx, g):
        return _steep_parent(self.z, idx, g) == idx
    return impl

@overload_method(SteepTreeImplType, 'children')
def _ol_impl_children(self, idx, g):
    def impl(self, idx, g):
        cnt, nbrs = g.get_neighbour_array(idx)
        out = np.empty(nb.int64(8), nb.int64)
        count = nb.int64(0)
        for k in range(cnt):
            nb_idx = nbrs[k]
            if _steep_parent(self.z, nb_idx, g) == idx:
                out[count] = nb_idx
                count += 1
        return count, out
    return impl

@overload_method(SteepTreeImplType, 'parent_of')
def _ol_impl_parent_of(self, idx, g):
    def impl(self, idx, g):
        return _steep_parent(self.z, idx, g)
    return impl

@overload_method(SteepTreeImplType, 'get_z')
def _ol_impl_get_z(self, idx):
    def impl(self, idx): return self.z[idx]
    return impl

@overload_method(SteepTreeImplType, 'size')
def _ol_impl_size(self):
    def impl(self): return nb.int64(len(self.z))
    return impl

# ── SteepTreePar overloads ────────────────────────────────────────────────────

@overload_method(SteepTreeParType, 'parent')
def _ol_par_parent(self, idx):
    def impl(self, idx): return self.parents[idx]
    return impl

@overload_method(SteepTreeParType, 'is_root')
def _ol_par_is_root(self, idx):
    def impl(self, idx): return self.parents[idx] == idx
    return impl

@overload_method(SteepTreeParType, 'children')
def _ol_par_children(self, idx, g):
    # check each neighbour: if its stored parent is idx, it is a child
    def impl(self, idx, g):
        cnt, nbrs = g.get_neighbour_array(idx)
        out = np.empty(nb.int64(8), nb.int64)
        count = nb.int64(0)
        for k in range(cnt):
            nb_idx = nbrs[k]
            if self.parents[nb_idx] == idx:
                out[count] = nb_idx
                count += 1
        return count, out
    return impl

@overload_method(SteepTreeParType, 'parent_of')
def _ol_par_parent_of(self, idx, g):
    def impl(self, idx, g): return self.parents[idx]
    return impl

@overload_method(SteepTreeParType, 'get_z')
def _ol_par_get_z(self, idx):
    def impl(self, idx): return self.z[idx]
    return impl

@overload_method(SteepTreeParType, 'size')
def _ol_par_size(self):
    def impl(self): return nb.int64(len(self.z))
    return impl

# ── SteepTreeFull overloads ───────────────────────────────────────────────────

@overload_method(SteepTreeFullType, 'parent')
def _ol_full_parent(self, idx):
    def impl(self, idx): return self.parents[idx]
    return impl

@overload_method(SteepTreeFullType, 'is_root')
def _ol_full_is_root(self, idx):
    def impl(self, idx): return self.parents[idx] == idx
    return impl

@overload_method(SteepTreeFullType, 'children')
def _ol_full_children(self, idx):
    def impl(self, idx):
        start = self.children_ptr[idx]
        end   = self.children_ptr[idx + 1]
        count = end - start
        out = np.empty(nb.int64(8), nb.int64)
        for k in range(count):
            out[k] = self.children_flat[start + k]
        return count, out
    return impl

@overload_method(SteepTreeFullType, 'parent_of')
def _ol_full_parent_of(self, idx, g):
    def impl(self, idx, g): return self.parents[idx]
    return impl

@overload_method(SteepTreeFullType, 'get_z')
def _ol_full_get_z(self, idx):
    def impl(self, idx): return self.z[idx]
    return impl

@overload_method(SteepTreeFullType, 'size')
def _ol_full_size(self):
    def impl(self): return nb.int64(len(self.z))
    return impl

# ── Proxy classes and boxing ──────────────────────────────────────────────────

class SteepTreeImpl(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

class SteepTreePar(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

class SteepTreeFull(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

structref.define_boxing(SteepTreeImplType, SteepTreeImpl)
structref.define_boxing(SteepTreeParType,  SteepTreePar)
structref.define_boxing(SteepTreeFullType, SteepTreeFull)

# ── @njit wrappers ────────────────────────────────────────────────────────────

@nb.njit
def _nb_impl_parent(t, idx, g):   return t.parent(idx, g)
@nb.njit
def _nb_impl_is_root(t, idx, g):  return t.is_root(idx, g)
@nb.njit
def _nb_impl_children(t, idx, g): return t.children(idx, g)
@nb.njit
def _nb_par_parent(t, idx):       return t.parent(idx)
@nb.njit
def _nb_par_is_root(t, idx):      return t.is_root(idx)
@nb.njit
def _nb_par_children(t, idx, g):  return t.children(idx, g)
@nb.njit
def _nb_full_parent(t, idx):      return t.parent(idx)
@nb.njit
def _nb_full_is_root(t, idx):     return t.is_root(idx)
@nb.njit
def _nb_full_children(t, idx):    return t.children(idx)
@nb.njit
def _nb_get_z(t, idx):            return t.get_z(idx)
@nb.njit
def _nb_size(t):                  return t.size()
@nb.njit
def _nb_get_parents_field(t):     return t.parents
@nb.njit
def _nb_get_z_field(t):           return t.z
@nb.njit
def _nb_parent_of(t, idx, g): return t.parent_of(idx, g)

# ── Proxy mixins ──────────────────────────────────────────────────────────────

class _SteepTreeImplProxyMixin:
    def parent(self, idx, g):    return _nb_impl_parent(self, idx, g)
    def parent_of(self, idx, g): return _nb_parent_of(self, idx, g)
    def is_root(self, idx, g):   return _nb_impl_is_root(self, idx, g)
    def children(self, idx, g):  return _nb_impl_children(self, idx, g)
    def get_z(self, idx):        return _nb_get_z(self, idx)
    def size(self):              return _nb_size(self)
    def get_z_array(self):       return _nb_get_z_field(self)
    def get_parents(self, g):    return _build_parents(_nb_get_z_field(self), g)

class _SteepTreeParProxyMixin:
    def parent(self, idx):       return _nb_par_parent(self, idx)
    def parent_of(self, idx, g): return _nb_parent_of(self, idx, g)
    def is_root(self, idx):      return _nb_par_is_root(self, idx)
    def children(self, idx, g):  return _nb_par_children(self, idx, g)
    def get_z(self, idx):        return _nb_get_z(self, idx)
    def size(self):              return _nb_size(self)
    def get_z_array(self):       return _nb_get_z_field(self)
    def get_parents(self):       return _nb_get_parents_field(self)

class _SteepTreeFullProxyMixin:
    def parent(self, idx):       return _nb_full_parent(self, idx)
    def parent_of(self, idx, g): return _nb_parent_of(self, idx, g)
    def is_root(self, idx):      return _nb_full_is_root(self, idx)
    def children(self, idx):     return _nb_full_children(self, idx)
    def get_z(self, idx):        return _nb_get_z(self, idx)
    def size(self):              return _nb_size(self)
    def get_z_array(self):       return _nb_get_z_field(self)
    def get_parents(self):       return _nb_get_parents_field(self)

SteepTreeImpl.__bases__ = (_SteepTreeImplProxyMixin,) + SteepTreeImpl.__bases__
SteepTreePar.__bases__  = (_SteepTreeParProxyMixin,)  + SteepTreePar.__bases__
SteepTreeFull.__bases__ = (_SteepTreeFullProxyMixin,) + SteepTreeFull.__bases__

# ── @njit builders ────────────────────────────────────────────────────────────
# _build_parents and _build_full are polymorphic in g — numba compiles a
# specialisation per neighbourer type at first call.

@nb.njit
def _build_parents(z, g):
    n = len(z)
    parents = np.empty(n, nb.int64)
    for i in range(n):
        parents[i] = _steep_parent(z, nb.int64(i), g)
    return parents

@nb.njit
def _build_full(z, g):
    n = len(z)
    parents = np.empty(n, nb.int64)
    for i in range(n):
        parents[i] = _steep_parent(z, nb.int64(i), g)
    # count children per node
    counts = np.zeros(n, nb.int64)
    for i in range(n):
        p = parents[i]
        if p != i:
            counts[p] += 1
    # prefix-sum -> ptr
    ptr = np.zeros(n + 1, nb.int64)
    for i in range(n):
        ptr[i + 1] = ptr[i] + counts[i]
    # fill flat children array
    total = ptr[n]
    flat = np.empty(total, nb.int64)
    pos = ptr[:n].copy()
    for i in range(n):
        p = parents[i]
        if p != i:
            flat[pos[p]] = i
            pos[p] += 1
    return parents, ptr, flat

@nb.njit
def _ctor_impl(z):
    t = structref.new(SteepTreeImplInst)
    t.z = z
    return t

@nb.njit
def _ctor_par(z, parents):
    t = structref.new(SteepTreeParInst)
    t.z = z
    t.parents = parents
    return t

@nb.njit
def _ctor_full(z, parents, ptr, flat):
    t = structref.new(SteepTreeFullInst)
    t.z = z
    t.parents = parents
    t.children_ptr  = ptr
    t.children_flat = flat
    return t

# ── Public factories ──────────────────────────────────────────────────────────

def make_steep_tree_impl(z: np.ndarray) -> SteepTreeImpl:
    """Wrap z without precomputing anything. parent/children require g at query time."""
    return _ctor_impl(np.asarray(z, dtype=np.float64).ravel())

def make_steep_tree_par(z: np.ndarray, g) -> SteepTreePar:
    """Build and store the parents array. children still require g at query time."""
    z_flat = np.asarray(z, dtype=np.float64).ravel()
    parents = _build_parents(z_flat, g)
    return _ctor_par(z_flat, parents)

def make_steep_tree_full(z: np.ndarray, g) -> SteepTreeFull:
    """Build and store parents + CSR children. No neighbourer needed at query time."""
    z_flat = np.asarray(z, dtype=np.float64).ravel()
    parents, ptr, flat = _build_full(z_flat, g)
    return _ctor_full(z_flat, parents, ptr, flat)

def make_tree(z: np.ndarray, g=None, variant: str = 'par'):
    """Unified factory. variant: 'impl' | 'par' (default) | 'full'."""
    if variant == 'impl':
        return make_steep_tree_impl(z)
    if variant == 'par':
        return make_steep_tree_par(z, g)
    if variant == 'full':
        return make_steep_tree_full(z, g)
    raise ValueError(f"variant must be 'impl', 'par', or 'full', got {variant!r}")
