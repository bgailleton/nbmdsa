"""Bounded priority queues — keep the K best elements.

MinHeapBd keeps the K *smallest* scores; MaxHeapBd keeps the K *largest*.
When full, emplace() evicts the current worst element only if the new score
is strictly better, otherwise returns False (no insertion).

Internal structure is the *opposite* of the name — MinHeapBd is backed by
a max-heap so the worst kept element (the largest) sits at the root for O(1)
eviction; MaxHeapBd is backed by a min-heap for the same reason.

As a consequence, top()/top_score()/top_idx() return the *worst* of the
currently kept elements (useful as a rejection threshold).

drain_sorted() fills the output arrays backward so the result is always
ascending for Min* and descending for Max*, consistent with the other variants.

Public factories:
    make_min_heap_bd(max_size, dtype)
    make_max_heap_bd(max_size, dtype)
    make_min_heap_bd_from(indices, scores)
    make_max_heap_bd_from(indices, scores)

Methods (same API as St/Dy):
    heap.emplace(idx, score) -> bool   # False if full and score not better
    heap.top()               -> (idx, score)   # worst kept element
    heap.top_score()         -> score
    heap.top_idx()           -> idx
    heap.pop()               # removes worst kept element
    heap.is_empty()          -> bool
    heap.reset()
    heap.reserve(n)          # no-op (API compatibility)
    heap.shrink_to_fit()     # no-op (API compatibility)
    heap.drain_sorted(out_indices, out_scores)
    heap.indices_view()      -> int64[:]
    heap.scores_view()       -> dtype[:]

Fields: heap.size, heap.max_size
"""

import numpy as np
import numba as nb
from numba.experimental import structref
from numba.core import types
from numba.core.extending import overload_method

from nbmdsa.structures._heap_common import _HeapProxyMixin

# ── Type class registration ───────────────────────────────────────────────────

@structref.register
class MinHeapBdType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(t)) for name, t in fields)

@structref.register
class MaxHeapBdType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(t)) for name, t in fields)

# ── Type instances ────────────────────────────────────────────────────────────

def _fields(score_nb_type):
    return [
        ('indices',  types.int64[::1]),
        ('scores',   types.Array(score_nb_type, 1, 'C')),
        ('size',     types.int64),
        ('max_size', types.int64),
    ]

MinHeapBdF32 = MinHeapBdType(_fields(types.float32))
MinHeapBdF64 = MinHeapBdType(_fields(types.float64))
MinHeapBdI32 = MinHeapBdType(_fields(types.int32))
MinHeapBdI64 = MinHeapBdType(_fields(types.int64))
MinHeapBdU8  = MinHeapBdType(_fields(types.uint8))

MaxHeapBdF32 = MaxHeapBdType(_fields(types.float32))
MaxHeapBdF64 = MaxHeapBdType(_fields(types.float64))
MaxHeapBdI32 = MaxHeapBdType(_fields(types.int32))
MaxHeapBdI64 = MaxHeapBdType(_fields(types.int64))
MaxHeapBdU8  = MaxHeapBdType(_fields(types.uint8))

# ── Sift operations ───────────────────────────────────────────────────────────
# MinHeapBd is backed by a max-heap; MaxHeapBd by a min-heap.

@overload_method(MinHeapBdType, '_sift_up')
def _ol_min_bd_sift_up(self, i):
    def impl(self, i):
        while i > 0:
            p = (i - 1) >> 1
            if self.scores[p] < self.scores[i]:
                self.indices[p], self.indices[i] = self.indices[i], self.indices[p]
                self.scores[p],  self.scores[i]  = self.scores[i],  self.scores[p]
                i = p
            else:
                break
    return impl

@overload_method(MaxHeapBdType, '_sift_up')
def _ol_max_bd_sift_up(self, i):
    def impl(self, i):
        while i > 0:
            p = (i - 1) >> 1
            if self.scores[p] > self.scores[i]:
                self.indices[p], self.indices[i] = self.indices[i], self.indices[p]
                self.scores[p],  self.scores[i]  = self.scores[i],  self.scores[p]
                i = p
            else:
                break
    return impl

@overload_method(MinHeapBdType, '_sift_down')
def _ol_min_bd_sift_down(self, i):
    def impl(self, i):
        while True:
            best = i
            left = 2 * i + 1
            right = left + 1
            if left < self.size and self.scores[left] > self.scores[best]:
                best = left
            if right < self.size and self.scores[right] > self.scores[best]:
                best = right
            if best == i:
                break
            self.indices[i], self.indices[best] = self.indices[best], self.indices[i]
            self.scores[i],  self.scores[best]  = self.scores[best],  self.scores[i]
            i = best
    return impl

@overload_method(MaxHeapBdType, '_sift_down')
def _ol_max_bd_sift_down(self, i):
    def impl(self, i):
        while True:
            best = i
            left = 2 * i + 1
            right = left + 1
            if left < self.size and self.scores[left] < self.scores[best]:
                best = left
            if right < self.size and self.scores[right] < self.scores[best]:
                best = right
            if best == i:
                break
            self.indices[i], self.indices[best] = self.indices[best], self.indices[i]
            self.scores[i],  self.scores[best]  = self.scores[best],  self.scores[i]
            i = best
    return impl

# ── Type-specific emplace (eviction condition differs) ────────────────────────

@overload_method(MinHeapBdType, 'emplace')
def _ol_min_bd_emplace(self, idx, score):
    def impl(self, idx, score):
        if self.size < self.max_size:
            pos = self.size
            self.indices[pos] = idx
            self.scores[pos]  = score
            self.size += 1
            self._sift_up(pos)
            return True
        if score < self.scores[0]:
            self.indices[0] = idx
            self.scores[0]  = score
            self._sift_down(0)
            return True
        return False
    return impl

@overload_method(MaxHeapBdType, 'emplace')
def _ol_max_bd_emplace(self, idx, score):
    def impl(self, idx, score):
        if self.size < self.max_size:
            pos = self.size
            self.indices[pos] = idx
            self.scores[pos]  = score
            self.size += 1
            self._sift_up(pos)
            return True
        if score > self.scores[0]:
            self.indices[0] = idx
            self.scores[0]  = score
            self._sift_down(0)
            return True
        return False
    return impl

# ── Shared methods ────────────────────────────────────────────────────────────

def _ol_top(self):
    def impl(self):
        return self.indices[0], self.scores[0]
    return impl

def _ol_top_score(self):
    def impl(self):
        return self.scores[0]
    return impl

def _ol_top_idx(self):
    def impl(self):
        return self.indices[0]
    return impl

def _ol_pop(self):
    def impl(self):
        last = self.size - 1
        self.indices[0] = self.indices[last]
        self.scores[0]  = self.scores[last]
        self.size = last
        if last > 0:
            self._sift_down(0)
    return impl

def _ol_is_empty(self):
    def impl(self):
        return self.size == 0
    return impl

def _ol_reset(self):
    def impl(self):
        self.size = nb.int64(0)
    return impl

def _ol_reserve_noop(self, n):
    def impl(self, n):
        pass
    return impl

def _ol_shrink_noop(self):
    def impl(self):
        pass
    return impl

def _ol_drain_sorted(self, out_indices, out_scores):
    # internal heap is reversed relative to name, so pop order is also reversed;
    # fill backward to produce ascending (Min*) or descending (Max*) output
    def impl(self, out_indices, out_scores):
        i = self.size - nb.int64(1)
        while not self.is_empty():
            out_indices[i], out_scores[i] = self.top()
            self.pop()
            i -= 1
    return impl

def _ol_indices_view(self):
    def impl(self):
        return self.indices[:self.size]
    return impl

def _ol_scores_view(self):
    def impl(self):
        return self.scores[:self.size]
    return impl

for _T in (MinHeapBdType, MaxHeapBdType):
    overload_method(_T, 'top')(_ol_top)
    overload_method(_T, 'top_score')(_ol_top_score)
    overload_method(_T, 'top_idx')(_ol_top_idx)
    overload_method(_T, 'pop')(_ol_pop)
    overload_method(_T, 'is_empty')(_ol_is_empty)
    overload_method(_T, 'reset')(_ol_reset)
    overload_method(_T, 'reserve')(_ol_reserve_noop)
    overload_method(_T, 'shrink_to_fit')(_ol_shrink_noop)
    overload_method(_T, 'drain_sorted')(_ol_drain_sorted)
    overload_method(_T, 'indices_view')(_ol_indices_view)
    overload_method(_T, 'scores_view')(_ol_scores_view)

# ── Proxy classes and boxing ──────────────────────────────────────────────────

class MinHeapBd(_HeapProxyMixin, structref.StructRefProxy):
    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)

class MaxHeapBd(_HeapProxyMixin, structref.StructRefProxy):
    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)

structref.define_boxing(MinHeapBdType, MinHeapBd)
structref.define_boxing(MaxHeapBdType, MaxHeapBd)

# ── Constructor factories ─────────────────────────────────────────────────────

def _make_empty_ctor(nb_heap_type, np_score_dtype):
    @nb.njit
    def _ctor(n):
        h = structref.new(nb_heap_type)
        h.indices  = np.empty(n, nb.int64)
        h.scores   = np.empty(n, np_score_dtype)
        h.size     = nb.int64(0)
        h.max_size = nb.int64(n)
        return h
    return _ctor

def _make_from_ctor(nb_heap_type):
    @nb.njit
    def _ctor(indices, scores):
        n = len(indices)
        h = structref.new(nb_heap_type)
        h.indices  = indices.copy()
        h.scores   = scores.copy()
        h.size     = nb.int64(n)
        h.max_size = nb.int64(n)
        for i in range((n - 2) // 2, -1, -1):
            h._sift_down(i)
        return h
    return _ctor

_SCORE_CONFIG = [
    (np.float32, MinHeapBdF32, MaxHeapBdF32),
    (np.float64, MinHeapBdF64, MaxHeapBdF64),
    (np.int32,   MinHeapBdI32, MaxHeapBdI32),
    (np.int64,   MinHeapBdI64, MaxHeapBdI64),
    (np.uint8,   MinHeapBdU8,  MaxHeapBdU8),
]

_min_empty_ctors: dict = {}
_max_empty_ctors: dict = {}
_min_from_ctors:  dict = {}
_max_from_ctors:  dict = {}

for _np_dtype, _min_nb, _max_nb in _SCORE_CONFIG:
    _key = np.dtype(_np_dtype)
    _min_empty_ctors[_key] = _make_empty_ctor(_min_nb, _np_dtype)
    _max_empty_ctors[_key] = _make_empty_ctor(_max_nb, _np_dtype)
    _min_from_ctors[_key]  = _make_from_ctor(_min_nb)
    _max_from_ctors[_key]  = _make_from_ctor(_max_nb)

# ── Public Python-side factories ──────────────────────────────────────────────

def make_min_heap_bd(max_size: int, dtype=np.float64):
    """Return an empty MinHeapBd keeping the K smallest scores."""
    return _min_empty_ctors[np.dtype(dtype)](max_size)

def make_max_heap_bd(max_size: int, dtype=np.float64):
    """Return an empty MaxHeapBd keeping the K largest scores."""
    return _max_empty_ctors[np.dtype(dtype)](max_size)

def make_min_heap_bd_from(indices: np.ndarray, scores: np.ndarray):
    """Build a MinHeapBd from existing arrays (heapifies in O(n))."""
    return _min_from_ctors[np.dtype(scores.dtype)](indices, scores)

def make_max_heap_bd_from(indices: np.ndarray, scores: np.ndarray):
    """Build a MaxHeapBd from existing arrays (heapifies in O(n))."""
    return _max_from_ctors[np.dtype(scores.dtype)](indices, scores)
