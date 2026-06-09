"""Bounded (static-capacity) FIFO queue backed by a circular buffer.

One TypeClass (QueueType) covers all element dtypes via multiple TypeInstances.
Pre-built instances for standard dtypes: QueueF32, QueueF64, QueueI32, QueueI64, QueueU8.
For a custom numba scalar type use make_queue_type(nb_scalar_type, np_dtype) which
returns (TypeInstance, @njit_ctor); the TypeInstance can be used in structref.new()
inside @njit, and the ctor is the Python-side factory.

Methods (usable in @njit and on the proxy):
    q.enqueue(val) -> bool   # False if full
    q.dequeue()    -> val    # undefined behaviour on empty queue
    q.peek()       -> val    # front element, no removal
    q.is_empty()   -> bool
    q.is_full()    -> bool
    q.reset()                # head=tail=size=0, data unchanged
    q.length()     -> int64

Field (read-only after construction): q.capacity

Factories:
    make_queue(capacity, dtype=np.int64)
    make_queue_type(nb_scalar_type, np_dtype) -> (TypeInstance, ctor)
"""

import numpy as np
import numba as nb
from numba.experimental import structref
from numba.core import types
from numba.core.extending import overload_method

# ── Type class ────────────────────────────────────────────────────────────────

@structref.register
class QueueType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

def _fields_queue(elem_nb_type):
    return [
        ('data',     types.Array(elem_nb_type, 1, 'C')),
        ('head',     types.int64),
        ('tail',     types.int64),
        ('size',     types.int64),
        ('capacity', types.int64),
    ]

# ── Pre-built TypeInstances ───────────────────────────────────────────────────

QueueF32 = QueueType(_fields_queue(types.float32))
QueueF64 = QueueType(_fields_queue(types.float64))
QueueI32 = QueueType(_fields_queue(types.int32))
QueueI64 = QueueType(_fields_queue(types.int64))
QueueU8  = QueueType(_fields_queue(types.uint8))

# ── Overload methods ──────────────────────────────────────────────────────────

@overload_method(QueueType, 'enqueue')
def _ol_enqueue(self, val):
    def impl(self, val):
        if self.size >= self.capacity:
            return False
        self.data[self.tail] = val
        self.tail = (self.tail + nb.int64(1)) % self.capacity
        self.size += nb.int64(1)
        return True
    return impl

@overload_method(QueueType, 'dequeue')
def _ol_dequeue(self):
    def impl(self):
        val = self.data[self.head]
        self.head = (self.head + nb.int64(1)) % self.capacity
        self.size -= nb.int64(1)
        return val
    return impl

@overload_method(QueueType, 'peek')
def _ol_peek(self):
    def impl(self):
        return self.data[self.head]
    return impl

@overload_method(QueueType, 'is_empty')
def _ol_is_empty(self):
    def impl(self):
        return self.size == nb.int64(0)
    return impl

@overload_method(QueueType, 'is_full')
def _ol_is_full(self):
    def impl(self):
        return self.size >= self.capacity
    return impl

@overload_method(QueueType, 'reset')
def _ol_reset(self):
    def impl(self):
        self.head = nb.int64(0)
        self.tail = nb.int64(0)
        self.size = nb.int64(0)
    return impl

@overload_method(QueueType, 'length')
def _ol_length(self):
    def impl(self):
        return self.size
    return impl

# ── Proxy class and boxing ────────────────────────────────────────────────────

class Queue(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

structref.define_boxing(QueueType, Queue)

# ── @njit wrappers ────────────────────────────────────────────────────────────

@nb.njit
def _nb_q_enqueue(q, val): return q.enqueue(val)
@nb.njit
def _nb_q_dequeue(q):      return q.dequeue()
@nb.njit
def _nb_q_peek(q):         return q.peek()
@nb.njit
def _nb_q_is_empty(q):     return q.is_empty()
@nb.njit
def _nb_q_is_full(q):      return q.is_full()
@nb.njit
def _nb_q_reset(q):        q.reset()
@nb.njit
def _nb_q_length(q):       return q.length()

# ── Proxy mixin ───────────────────────────────────────────────────────────────

class _QueueProxyMixin:
    def enqueue(self, val): return _nb_q_enqueue(self, val)
    def dequeue(self):      return _nb_q_dequeue(self)
    def peek(self):         return _nb_q_peek(self)
    def is_empty(self):     return _nb_q_is_empty(self)
    def is_full(self):      return _nb_q_is_full(self)
    def reset(self):        _nb_q_reset(self)
    def length(self):       return _nb_q_length(self)

Queue.__bases__ = (_QueueProxyMixin,) + Queue.__bases__

# ── Constructor factory ───────────────────────────────────────────────────────

def _make_queue_ctor(nb_inst, np_dtype):
    @nb.njit
    def _ctor(capacity):
        q = structref.new(nb_inst)
        q.data     = np.empty(capacity, np_dtype)
        q.head     = nb.int64(0)
        q.tail     = nb.int64(0)
        q.size     = nb.int64(0)
        q.capacity = nb.int64(capacity)
        return q
    return _ctor

_QUEUE_CONFIG = [
    (np.float32, QueueF32),
    (np.float64, QueueF64),
    (np.int32,   QueueI32),
    (np.int64,   QueueI64),
    (np.uint8,   QueueU8),
]

_queue_ctors: dict = {}
for _np_dtype, _nb_inst in _QUEUE_CONFIG:
    _queue_ctors[np.dtype(_np_dtype)] = _make_queue_ctor(_nb_inst, _np_dtype)

# ── Public factories ──────────────────────────────────────────────────────────

def make_queue(capacity: int, dtype=np.int64) -> Queue:
    """Return an empty Queue with the given capacity and element dtype."""
    return _queue_ctors[np.dtype(dtype)](capacity)

def make_queue_type(nb_scalar_type, np_dtype):
    """Create a QueueType instance for an arbitrary numba scalar type.

    Returns (TypeInstance, ctor) where TypeInstance can be passed to
    structref.new() inside @njit and ctor(capacity) -> Queue from Python.
    """
    inst = QueueType(_fields_queue(nb_scalar_type))
    ctor = _make_queue_ctor(inst, np_dtype)
    return inst, ctor
