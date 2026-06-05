import numpy as np
import numba as nb
import pytest

from nbmdsa.structures.heap_st import (
    make_min_heap_st,
    make_max_heap_st,
    make_min_heap_st_from,
    MinHeapStF64,
)
from numba.experimental import structref


def test_min_heap_order():
    h = make_min_heap_st(10)
    for idx, score in enumerate([3.0, 1.0, 4.0, 1.5, 2.0]):
        h.emplace(idx, score)
    out = []
    while not h.is_empty():
        _, s = h.top()
        out.append(s)
        h.pop()
    assert out == sorted(out)


def test_max_heap_order():
    h = make_max_heap_st(10)
    for idx, score in enumerate([3.0, 1.0, 4.0, 1.5, 2.0]):
        h.emplace(idx, score)
    out = []
    while not h.is_empty():
        _, s = h.top()
        out.append(s)
        h.pop()
    assert out == sorted(out, reverse=True)


def test_emplace_full():
    h = make_min_heap_st(3)
    assert h.emplace(0, 1.0)
    assert h.emplace(1, 2.0)
    assert h.emplace(2, 3.0)
    assert not h.emplace(3, 4.0)


def test_reset():
    h = make_min_heap_st(5)
    for i in range(5):
        h.emplace(i, float(i + 10))
    h.reset()
    assert h.is_empty()
    h.emplace(0, 99.0)
    _, s = h.top()
    assert s == 99.0
    assert h.size == 1


def test_from_arrays():
    indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    scores  = np.array([5.0, 2.0, 8.0, 1.0, 3.0], dtype=np.float64)
    h = make_min_heap_st_from(indices, scores)
    out = []
    while not h.is_empty():
        _, s = h.top()
        out.append(s)
        h.pop()
    assert out == sorted(out)


def test_njit_usage():
    @nb.njit
    def drain(n):
        h = structref.new(MinHeapStF64)
        h.indices  = np.empty(n, nb.int64)
        h.scores   = np.empty(n, nb.float64)
        h.size     = nb.int64(0)
        h.max_size = nb.int64(n)
        scores_in = np.array([3.0, 1.0, 4.0, 1.5, 2.0])
        for i in range(len(scores_in)):
            h.emplace(nb.int64(i), scores_in[i])
        out = np.empty(len(scores_in), nb.float64)
        for i in range(len(scores_in)):
            _, s = h.top()
            out[i] = s
            h.pop()
        return out

    result = drain(5)
    assert list(result) == sorted(result)
