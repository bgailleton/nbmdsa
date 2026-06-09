"""Bounded (static-capacity) LIFO stack.

One TypeClass (StackType) covers all element dtypes via multiple TypeInstances.
Pre-built instances for standard dtypes: StackF32, StackF64, StackI32, StackI64, StackU8.
For a custom numba scalar type use make_stack_type(nb_scalar_type, np_dtype) which
returns (TypeInstance, @njit_ctor); the TypeInstance can be used in structref.new()
inside @njit, and the ctor is the Python-side factory.

Methods (usable in @njit and on the proxy):
    s.push(val) -> bool   # False if full
    s.pop()     -> val    # undefined behaviour on empty stack
    s.peek()    -> val    # top element, no removal
    s.is_empty() -> bool
    s.is_full()  -> bool
    s.reset()              # top=0, data unchanged
    s.length()  -> int64

Field (read-only after construction): s.capacity

Factories:
    make_stack(capacity, dtype=np.int64)
    make_stack_type(nb_scalar_type, np_dtype) -> (TypeInstance, ctor)
"""

import numpy as np
import numba as nb
from numba.experimental import structref
from numba.core import types
from numba.core.extending import overload_method

# ── Type class ────────────────────────────────────────────────────────────────

@structref.register
class StackType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

def _fields_stack(elem_nb_type):
    return [
        ('data',     types.Array(elem_nb_type, 1, 'C')),
        ('top',      types.int64),
        ('capacity', types.int64),
    ]

# ── Pre-built TypeInstances ───────────────────────────────────────────────────

StackF32 = StackType(_fields_stack(types.float32))
StackF64 = StackType(_fields_stack(types.float64))
StackI32 = StackType(_fields_stack(types.int32))
StackI64 = StackType(_fields_stack(types.int64))
StackU8  = StackType(_fields_stack(types.uint8))

# ── Overload methods ──────────────────────────────────────────────────────────

@overload_method(StackType, 'push')
def _ol_push(self, val):
    def impl(self, val):
        if self.top >= self.capacity:
            return False
        self.data[self.top] = val
        self.top += nb.int64(1)
        return True
    return impl

@overload_method(StackType, 'pop')
def _ol_pop(self):
    def impl(self):
        self.top -= nb.int64(1)
        return self.data[self.top]
    return impl

@overload_method(StackType, 'peek')
def _ol_peek(self):
    def impl(self):
        return self.data[self.top - nb.int64(1)]
    return impl

@overload_method(StackType, 'is_empty')
def _ol_is_empty(self):
    def impl(self):
        return self.top == nb.int64(0)
    return impl

@overload_method(StackType, 'is_full')
def _ol_is_full(self):
    def impl(self):
        return self.top >= self.capacity
    return impl

@overload_method(StackType, 'reset')
def _ol_reset(self):
    def impl(self):
        self.top = nb.int64(0)
    return impl

@overload_method(StackType, 'length')
def _ol_length(self):
    def impl(self):
        return self.top
    return impl

# ── Proxy class and boxing ────────────────────────────────────────────────────

class Stack(structref.StructRefProxy):
    def __new__(cls, *args): return structref.StructRefProxy.__new__(cls, *args)

structref.define_boxing(StackType, Stack)

# ── @njit wrappers ────────────────────────────────────────────────────────────

@nb.njit
def _nb_s_push(s, val):  return s.push(val)
@nb.njit
def _nb_s_pop(s):        return s.pop()
@nb.njit
def _nb_s_peek(s):       return s.peek()
@nb.njit
def _nb_s_is_empty(s):   return s.is_empty()
@nb.njit
def _nb_s_is_full(s):    return s.is_full()
@nb.njit
def _nb_s_reset(s):      s.reset()
@nb.njit
def _nb_s_length(s):     return s.length()

# ── Proxy mixin ───────────────────────────────────────────────────────────────

class _StackProxyMixin:
    def push(self, val):  return _nb_s_push(self, val)
    def pop(self):        return _nb_s_pop(self)
    def peek(self):       return _nb_s_peek(self)
    def is_empty(self):   return _nb_s_is_empty(self)
    def is_full(self):    return _nb_s_is_full(self)
    def reset(self):      _nb_s_reset(self)
    def length(self):     return _nb_s_length(self)

Stack.__bases__ = (_StackProxyMixin,) + Stack.__bases__

# ── Constructor factory ───────────────────────────────────────────────────────

def _make_stack_ctor(nb_inst, np_dtype):
    @nb.njit
    def _ctor(capacity):
        s = structref.new(nb_inst)
        s.data     = np.empty(capacity, np_dtype)
        s.top      = nb.int64(0)
        s.capacity = nb.int64(capacity)
        return s
    return _ctor

_STACK_CONFIG = [
    (np.float32, StackF32),
    (np.float64, StackF64),
    (np.int32,   StackI32),
    (np.int64,   StackI64),
    (np.uint8,   StackU8),
]

_stack_ctors: dict = {}
for _np_dtype, _nb_inst in _STACK_CONFIG:
    _stack_ctors[np.dtype(_np_dtype)] = _make_stack_ctor(_nb_inst, _np_dtype)

# ── Public factories ──────────────────────────────────────────────────────────

def make_stack(capacity: int, dtype=np.int64) -> Stack:
    """Return an empty Stack with the given capacity and element dtype."""
    return _stack_ctors[np.dtype(dtype)](capacity)

def make_stack_type(nb_scalar_type, np_dtype):
    """Create a StackType instance for an arbitrary numba scalar type.

    Returns (TypeInstance, ctor) where TypeInstance can be passed to
    structref.new() inside @njit and ctor(capacity) -> Stack from Python.
    """
    inst = StackType(_fields_stack(nb_scalar_type))
    ctor = _make_stack_ctor(inst, np_dtype)
    return inst, ctor
