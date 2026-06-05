"""Static (fixed-capacity) priority queues backed by a binary heap.

Provides MinHeapSt and MaxHeapSt variants, each available in five score
dtypes: float32, float64, int32, int64, uint8.  All types are usable
inside @njit functions.

Public factories (work in Python and njit):
    make_min_heap_st(max_size, dtype)
    make_max_heap_st(max_size, dtype)
    make_min_heap_st_from(indices, scores)   # heapifies in O(n)
    make_max_heap_st_from(indices, scores)

Methods available on any heap instance inside @njit:
    heap.emplace(idx, score) -> bool   # False if full
    heap.top()               -> (idx, score)
    heap.pop()
    heap.is_empty()          -> bool
    heap.indices_view()      -> int64[:]   # read-only slice [:size]
    heap.scores_view()       -> dtype[:]   # read-only slice [:size]

Fields (direct access in @njit):
    heap.size      current number of elements
    heap.max_size  capacity
"""

import numpy as np
import numba as nb
from numba.experimental import structref
from numba.core import types
from numba.core.extending import overload_method

# ── Type class registration ───────────────────────────────────────────────────

@structref.register
class MinHeapStType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(t)) for name, t in fields)

@structref.register
class MaxHeapStType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(t)) for name, t in fields)

# ── Type instances (one per score dtype, per variant) ─────────────────────────

def _fields(score_nb_type):
    return [
        ('indices', types.int64[::1]),
        ('scores',  types.Array(score_nb_type, 1, 'C')),
        ('size',    types.int64),
        ('max_size', types.int64),
    ]

MinHeapStF32 = MinHeapStType(_fields(types.float32))
MinHeapStF64 = MinHeapStType(_fields(types.float64))
MinHeapStI32 = MinHeapStType(_fields(types.int32))
MinHeapStI64 = MinHeapStType(_fields(types.int64))
MinHeapStU8  = MinHeapStType(_fields(types.uint8))

MaxHeapStF32 = MaxHeapStType(_fields(types.float32))
MaxHeapStF64 = MaxHeapStType(_fields(types.float64))
MaxHeapStI32 = MaxHeapStType(_fields(types.int32))
MaxHeapStI64 = MaxHeapStType(_fields(types.int64))
MaxHeapStU8  = MaxHeapStType(_fields(types.uint8))

# ── Sift operations (comparison differs between min and max) ──────────────────

@overload_method(MinHeapStType, '_sift_up')
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

@overload_method(MaxHeapStType, '_sift_up')
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

@overload_method(MinHeapStType, '_sift_down')
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

@overload_method(MaxHeapStType, '_sift_down')
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

# ── Shared methods (registered for both variants) ─────────────────────────────

def _ol_emplace(self, idx, score):
    def impl(self, idx, score):
        if self.size >= self.max_size:
            return False
        pos = self.size
        self.indices[pos] = idx
        self.scores[pos] = score
        self.size += 1
        self._sift_up(pos)
        return True
    return impl

def _ol_top(self):
    def impl(self):
        return self.indices[0], self.scores[0]
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

def _ol_indices_view(self):
    def impl(self):
        return self.indices[:self.size]
    return impl

def _ol_scores_view(self):
    def impl(self):
        return self.scores[:self.size]
    return impl

def _ol_reset(self):
    def impl(self):
        self.size = nb.int64(0)
    return impl

for _T in (MinHeapStType, MaxHeapStType):
    overload_method(_T, 'emplace')(_ol_emplace)
    overload_method(_T, 'top')(_ol_top)
    overload_method(_T, 'pop')(_ol_pop)
    overload_method(_T, 'is_empty')(_ol_is_empty)
    overload_method(_T, 'indices_view')(_ol_indices_view)
    overload_method(_T, 'scores_view')(_ol_scores_view)
    overload_method(_T, 'reset')(_ol_reset)

# ── Python-side proxy classes and boxing ─────────────────────────────────────
# Without a registered StructRefProxy + define_boxing, numba cannot convert a
# structref back to a Python object (boxing), causing TypeError on return from
# any @njit function called from Python.

@nb.njit
def _nb_emplace(h, idx, score):
    return h.emplace(idx, score)

@nb.njit
def _nb_top(h):
    return h.top()

@nb.njit
def _nb_pop(h):
    h.pop()

@nb.njit
def _nb_is_empty(h):
    return h.is_empty()

@nb.njit
def _nb_reset(h):
    h.reset()

@nb.njit
def _nb_size(h):
    return h.size


class MinHeapSt(structref.StructRefProxy):
    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)

    def emplace(self, idx, score):
        return _nb_emplace(self, idx, score)

    def top(self):
        return _nb_top(self)

    def pop(self):
        _nb_pop(self)

    def is_empty(self):
        return _nb_is_empty(self)

    def reset(self):
        _nb_reset(self)

    @property
    def size(self):
        return _nb_size(self)


class MaxHeapSt(structref.StructRefProxy):
    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)

    def emplace(self, idx, score):
        return _nb_emplace(self, idx, score)

    def top(self):
        return _nb_top(self)

    def pop(self):
        _nb_pop(self)

    def is_empty(self):
        return _nb_is_empty(self)

    def reset(self):
        _nb_reset(self)

    @property
    def size(self):
        return _nb_size(self)


structref.define_boxing(MinHeapStType, MinHeapSt)
structref.define_boxing(MaxHeapStType, MaxHeapSt)

# ── Constructor factories ─────────────────────────────────────────────────────
# Each call to _make_*_ctor produces a distinct @njit function with the
# heap type and numpy dtype captured as compile-time closure constants.

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

# (dtype, min_type_instance, max_type_instance)
_SCORE_CONFIG = [
    (np.float32, MinHeapStF32, MaxHeapStF32),
    (np.float64, MinHeapStF64, MaxHeapStF64),
    (np.int32,   MinHeapStI32, MaxHeapStI32),
    (np.int64,   MinHeapStI64, MaxHeapStI64),
    (np.uint8,   MinHeapStU8,  MaxHeapStU8),
]

_min_empty_ctors: dict = {}
_max_empty_ctors: dict = {}
_min_from_ctors: dict  = {}
_max_from_ctors: dict  = {}

for _np_dtype, _min_nb, _max_nb in _SCORE_CONFIG:
    _key = np.dtype(_np_dtype)
    _min_empty_ctors[_key] = _make_empty_ctor(_min_nb, _np_dtype)
    _max_empty_ctors[_key] = _make_empty_ctor(_max_nb, _np_dtype)
    _min_from_ctors[_key]  = _make_from_ctor(_min_nb)
    _max_from_ctors[_key]  = _make_from_ctor(_max_nb)

# ── Public Python-side factories ──────────────────────────────────────────────

def make_min_heap_st(max_size: int, dtype=np.float64):
    """Return an empty MinHeapSt with given capacity and score dtype."""
    return _min_empty_ctors[np.dtype(dtype)](max_size)

def make_max_heap_st(max_size: int, dtype=np.float64):
    """Return an empty MaxHeapSt with given capacity and score dtype."""
    return _max_empty_ctors[np.dtype(dtype)](max_size)

def make_min_heap_st_from(indices: np.ndarray, scores: np.ndarray):
    """Build a MinHeapSt from existing index/score arrays (heapifies in O(n))."""
    return _min_from_ctors[np.dtype(scores.dtype)](indices, scores)

def make_max_heap_st_from(indices: np.ndarray, scores: np.ndarray):
    """Build a MaxHeapSt from existing index/score arrays (heapifies in O(n))."""
    return _max_from_ctors[np.dtype(scores.dtype)](indices, scores)
