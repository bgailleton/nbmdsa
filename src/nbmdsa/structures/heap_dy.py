"""Dynamic (auto-growing) priority queues backed by a binary heap.

Drop-in replacement for MinHeapSt/MaxHeapSt with no fixed capacity.
Backing arrays grow automatically; reserve() and shrink_to_fit() are
functional here (no-ops on the static variant for API compatibility).

Growth rule:
    increment = min(capacity, max_chunk)
    new_cap   = capacity + max(1, increment)

max_chunk defaults to 2**62 (pure doubling). Set e.g. max_chunk=10_000
to prevent large single allocations.

Public factories:
    make_min_heap_dy(initial_capacity, dtype, max_chunk)
    make_max_heap_dy(initial_capacity, dtype, max_chunk)
    make_min_heap_dy_from(indices, scores, max_chunk)
    make_max_heap_dy_from(indices, scores, max_chunk)

Methods (identical API to static variants):
    heap.emplace(idx, score) -> bool   # always True; grows if needed
    heap.top()               -> (idx, score)
    heap.top_score()         -> score
    heap.top_idx()           -> idx
    heap.pop()
    heap.is_empty()          -> bool
    heap.reset()
    heap.reserve(n)          # grow capacity to at least n
    heap.shrink_to_fit()     # reallocate down to size
    heap.drain_sorted(out_indices, out_scores)
    heap.indices_view()      -> int64[:]
    heap.scores_view()       -> dtype[:]

Fields: heap.size, heap.capacity, heap.max_chunk
"""

import numpy as np
import numba as nb
import numba.np.numpy_support as _nps
from numba.experimental import structref
from numba.core import types
from numba.core.extending import overload_method

from nbmdsa.structures._heap_common import _HeapProxyMixin

_DEFAULT_MAX_CHUNK = nb.int64(1 << 62)

# ── Type class registration ───────────────────────────────────────────────────

@structref.register
class MinHeapDyType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(t)) for name, t in fields)

@structref.register
class MaxHeapDyType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(t)) for name, t in fields)

# ── Type instances ────────────────────────────────────────────────────────────

def _fields(score_nb_type):
    return [
        ('indices',   types.int64[::1]),
        ('scores',    types.Array(score_nb_type, 1, 'C')),
        ('size',      types.int64),
        ('capacity',  types.int64),
        ('max_chunk', types.int64),
    ]

MinHeapDyF32 = MinHeapDyType(_fields(types.float32))
MinHeapDyF64 = MinHeapDyType(_fields(types.float64))
MinHeapDyI32 = MinHeapDyType(_fields(types.int32))
MinHeapDyI64 = MinHeapDyType(_fields(types.int64))
MinHeapDyU8  = MinHeapDyType(_fields(types.uint8))

MaxHeapDyF32 = MaxHeapDyType(_fields(types.float32))
MaxHeapDyF64 = MaxHeapDyType(_fields(types.float64))
MaxHeapDyI32 = MaxHeapDyType(_fields(types.int32))
MaxHeapDyI64 = MaxHeapDyType(_fields(types.int64))
MaxHeapDyU8  = MaxHeapDyType(_fields(types.uint8))

# ── Sift operations ───────────────────────────────────────────────────────────

@overload_method(MinHeapDyType, '_sift_up')
def _ol_min_sift_up(self, i):
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

@overload_method(MaxHeapDyType, '_sift_up')
def _ol_max_sift_up(self, i):
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

@overload_method(MinHeapDyType, '_sift_down')
def _ol_min_sift_down(self, i):
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

@overload_method(MaxHeapDyType, '_sift_down')
def _ol_max_sift_down(self, i):
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

# ── Grow / reserve / shrink (need score dtype from type at overload time) ─────

def _ol_grow(self):
    score_np_dtype = _nps.as_dtype(self.field_dict['scores'].dtype)
    def impl(self):
        increment = min(self.capacity, self.max_chunk)
        new_cap = self.capacity + max(nb.int64(1), increment)
        new_idx = np.empty(new_cap, nb.int64)
        new_scr = np.empty(new_cap, score_np_dtype)
        new_idx[:self.size] = self.indices[:self.size]
        new_scr[:self.size] = self.scores[:self.size]
        self.indices  = new_idx
        self.scores   = new_scr
        self.capacity = new_cap
    return impl

def _ol_reserve(self, n):
    score_np_dtype = _nps.as_dtype(self.field_dict['scores'].dtype)
    def impl(self, n):
        if n <= self.capacity:
            return
        new_idx = np.empty(n, nb.int64)
        new_scr = np.empty(n, score_np_dtype)
        new_idx[:self.size] = self.indices[:self.size]
        new_scr[:self.size] = self.scores[:self.size]
        self.indices  = new_idx
        self.scores   = new_scr
        self.capacity = nb.int64(n)
    return impl

def _ol_shrink_to_fit(self):
    score_np_dtype = _nps.as_dtype(self.field_dict['scores'].dtype)
    def impl(self):
        n = max(self.size, nb.int64(1))
        if n == self.capacity:
            return
        new_idx = np.empty(n, nb.int64)
        new_scr = np.empty(n, score_np_dtype)
        new_idx[:self.size] = self.indices[:self.size]
        new_scr[:self.size] = self.scores[:self.size]
        self.indices  = new_idx
        self.scores   = new_scr
        self.capacity = n
    return impl

# ── Shared methods ────────────────────────────────────────────────────────────

def _ol_emplace(self, idx, score):
    def impl(self, idx, score):
        if self.size == self.capacity:
            self._grow()
        pos = self.size
        self.indices[pos] = idx
        self.scores[pos]  = score
        self.size += 1
        self._sift_up(pos)
        return True
    return impl

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

def _ol_drain_sorted(self, out_indices, out_scores):
    def impl(self, out_indices, out_scores):
        i = nb.int64(0)
        while not self.is_empty():
            out_indices[i], out_scores[i] = self.top()
            self.pop()
            i += 1
    return impl

def _ol_indices_view(self):
    def impl(self):
        return self.indices[:self.size]
    return impl

def _ol_scores_view(self):
    def impl(self):
        return self.scores[:self.size]
    return impl

for _T in (MinHeapDyType, MaxHeapDyType):
    overload_method(_T, '_grow')(_ol_grow)
    overload_method(_T, 'emplace')(_ol_emplace)
    overload_method(_T, 'top')(_ol_top)
    overload_method(_T, 'top_score')(_ol_top_score)
    overload_method(_T, 'top_idx')(_ol_top_idx)
    overload_method(_T, 'pop')(_ol_pop)
    overload_method(_T, 'is_empty')(_ol_is_empty)
    overload_method(_T, 'reset')(_ol_reset)
    overload_method(_T, 'reserve')(_ol_reserve)
    overload_method(_T, 'shrink_to_fit')(_ol_shrink_to_fit)
    overload_method(_T, 'drain_sorted')(_ol_drain_sorted)
    overload_method(_T, 'indices_view')(_ol_indices_view)
    overload_method(_T, 'scores_view')(_ol_scores_view)

# ── Proxy classes and boxing ──────────────────────────────────────────────────

class MinHeapDy(_HeapProxyMixin, structref.StructRefProxy):
    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)

class MaxHeapDy(_HeapProxyMixin, structref.StructRefProxy):
    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)

structref.define_boxing(MinHeapDyType, MinHeapDy)
structref.define_boxing(MaxHeapDyType, MaxHeapDy)

# ── Constructor factories ─────────────────────────────────────────────────────

def _make_empty_ctor(nb_heap_type, np_score_dtype):
    @nb.njit
    def _ctor(initial_capacity, max_chunk):
        h = structref.new(nb_heap_type)
        h.indices   = np.empty(initial_capacity, nb.int64)
        h.scores    = np.empty(initial_capacity, np_score_dtype)
        h.size      = nb.int64(0)
        h.capacity  = nb.int64(initial_capacity)
        h.max_chunk = nb.int64(max_chunk)
        return h
    return _ctor

def _make_from_ctor(nb_heap_type):
    @nb.njit
    def _ctor(indices, scores, max_chunk):
        n = len(indices)
        h = structref.new(nb_heap_type)
        h.indices   = indices.copy()
        h.scores    = scores.copy()
        h.size      = nb.int64(n)
        h.capacity  = nb.int64(n)
        h.max_chunk = nb.int64(max_chunk)
        for i in range((n - 2) // 2, -1, -1):
            h._sift_down(i)
        return h
    return _ctor

_SCORE_CONFIG = [
    (np.float32, MinHeapDyF32, MaxHeapDyF32),
    (np.float64, MinHeapDyF64, MaxHeapDyF64),
    (np.int32,   MinHeapDyI32, MaxHeapDyI32),
    (np.int64,   MinHeapDyI64, MaxHeapDyI64),
    (np.uint8,   MinHeapDyU8,  MaxHeapDyU8),
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

def make_min_heap_dy(initial_capacity: int = 16, dtype=np.float64, max_chunk: int = None):
    """Return an empty MinHeapDy."""
    mc = _DEFAULT_MAX_CHUNK if max_chunk is None else nb.int64(max_chunk)
    return _min_empty_ctors[np.dtype(dtype)](initial_capacity, mc)

def make_max_heap_dy(initial_capacity: int = 16, dtype=np.float64, max_chunk: int = None):
    """Return an empty MaxHeapDy."""
    mc = _DEFAULT_MAX_CHUNK if max_chunk is None else nb.int64(max_chunk)
    return _max_empty_ctors[np.dtype(dtype)](initial_capacity, mc)

def make_min_heap_dy_from(indices: np.ndarray, scores: np.ndarray, max_chunk: int = None):
    """Build a MinHeapDy from existing arrays (heapifies in O(n))."""
    mc = _DEFAULT_MAX_CHUNK if max_chunk is None else nb.int64(max_chunk)
    return _min_from_ctors[np.dtype(scores.dtype)](indices, scores, mc)

def make_max_heap_dy_from(indices: np.ndarray, scores: np.ndarray, max_chunk: int = None):
    """Build a MaxHeapDy from existing arrays (heapifies in O(n))."""
    mc = _DEFAULT_MAX_CHUNK if max_chunk is None else nb.int64(max_chunk)
    return _max_from_ctors[np.dtype(scores.dtype)](indices, scores, mc)
