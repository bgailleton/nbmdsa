"""Tests for heap, queue, stack, deque, union-find.
All operations exercised from @nb.njit to verify the inlining path.
"""

import numpy as np
import numba as nb

from nbmdsa.structures.primitives import (
    make_heap, make_queue, make_stack, make_deque, make_union_find,
)


# ── MinHeap ───────────────────────────────────────────────────────────────────

class TestMinHeap:
    ops = make_heap(nb.float64)

    def test_min_order(self):
        emplace, pop, top, is_empty = self.ops

        @nb.njit
        def _run(emplace, pop, top):
            sc = np.empty(16, nb.float64)
            ix = np.empty(16, nb.int64)
            sz = nb.int64(0)
            for v in (5.0, 3.0, 8.0, 1.0, 4.0):
                sz = emplace(sc, ix, sz, v, nb.int64(0))
            out = np.empty(5, nb.float64)
            for k in range(5):
                s, _ = top(sc, ix)
                out[k] = s
                sz = pop(sc, ix, sz)
            return out

        result = _run(emplace, pop, top)
        assert list(result) == sorted([5.0, 3.0, 8.0, 1.0, 4.0])

    def test_is_empty(self):
        emplace, pop, top, is_empty = self.ops

        @nb.njit
        def _run(emplace, pop, top, is_empty):
            sc = np.empty(4, nb.float64)
            ix = np.empty(4, nb.int64)
            sz = nb.int64(0)
            e0 = is_empty(sz)
            sz = emplace(sc, ix, sz, nb.float64(1.0), nb.int64(0))
            e1 = is_empty(sz)
            sz = pop(sc, ix, sz)
            e2 = is_empty(sz)
            return e0, e1, e2

        e0, e1, e2 = _run(emplace, pop, top, is_empty)
        assert e0 and not e1 and e2

    def test_index_tracks_score(self):
        emplace, pop, top, is_empty = self.ops

        @nb.njit
        def _run(emplace, pop, top):
            sc = np.empty(4, nb.float64)
            ix = np.empty(4, nb.int64)
            sz = nb.int64(0)
            sz = emplace(sc, ix, sz, nb.float64(9.0), nb.int64(9))
            sz = emplace(sc, ix, sz, nb.float64(1.0), nb.int64(1))
            sz = emplace(sc, ix, sz, nb.float64(5.0), nb.int64(5))
            s0, i0 = top(sc, ix); sz = pop(sc, ix, sz)
            s1, i1 = top(sc, ix); sz = pop(sc, ix, sz)
            s2, i2 = top(sc, ix); sz = pop(sc, ix, sz)
            return i0, i1, i2

        i0, i1, i2 = _run(emplace, pop, top)
        assert (i0, i1, i2) == (1, 5, 9)


# ── Queue ─────────────────────────────────────────────────────────────────────

class TestQueue:
    ops = make_queue(nb.int64)

    def test_fifo_order(self):
        enqueue, dequeue, peek, is_empty = self.ops

        @nb.njit
        def _run(enqueue, dequeue):
            data = np.empty(8, nb.int64)
            head = tail = size = nb.int64(0)
            for v in (10, 20, 30):
                tail, size = enqueue(data, head, tail, size, nb.int64(v))
            out = np.empty(3, nb.int64)
            for k in range(3):
                item, head, size = dequeue(data, head, tail, size)
                out[k] = item
            return out

        assert list(_run(enqueue, dequeue)) == [10, 20, 30]

    def test_peek_does_not_consume(self):
        enqueue, dequeue, peek, is_empty = self.ops

        @nb.njit
        def _run(enqueue, dequeue, peek):
            data = np.empty(4, nb.int64)
            head = tail = size = nb.int64(0)
            tail, size = enqueue(data, head, tail, size, nb.int64(42))
            p = peek(data, head)
            item, head, size = dequeue(data, head, tail, size)
            return p, item, size

        p, item, sz = _run(enqueue, dequeue, peek)
        assert p == 42 and item == 42 and sz == 0

    def test_wrap_around(self):
        enqueue, dequeue, peek, is_empty = self.ops

        @nb.njit
        def _run(enqueue, dequeue):
            data = np.empty(4, nb.int64)
            head = tail = size = nb.int64(0)
            for v in (1, 2, 3):
                tail, size = enqueue(data, head, tail, size, nb.int64(v))
            _, head, size = dequeue(data, head, tail, size)
            _, head, size = dequeue(data, head, tail, size)
            tail, size = enqueue(data, head, tail, size, nb.int64(4))
            tail, size = enqueue(data, head, tail, size, nb.int64(5))
            out = np.empty(3, nb.int64)
            for k in range(3):
                item, head, size = dequeue(data, head, tail, size)
                out[k] = item
            return out

        assert list(_run(enqueue, dequeue)) == [3, 4, 5]


# ── Stack ─────────────────────────────────────────────────────────────────────

class TestStack:
    ops = make_stack(nb.int64)

    def test_lifo_order(self):
        push, pop, peek, is_empty = self.ops

        @nb.njit
        def _run(push, pop):
            data = np.empty(8, nb.int64)
            sz = nb.int64(0)
            for v in (1, 2, 3):
                sz = push(data, sz, nb.int64(v))
            out = np.empty(3, nb.int64)
            for k in range(3):
                item, sz = pop(data, sz)
                out[k] = item
            return out

        assert list(_run(push, pop)) == [3, 2, 1]

    def test_peek_does_not_consume(self):
        push, pop, peek, is_empty = self.ops

        @nb.njit
        def _run(push, pop, peek):
            data = np.empty(4, nb.int64)
            sz = nb.int64(0)
            sz = push(data, sz, nb.int64(7))
            p = peek(data, sz)
            item, sz = pop(data, sz)
            return p, item, sz

        p, item, sz = _run(push, pop, peek)
        assert p == 7 and item == 7 and sz == 0


# ── Deque ─────────────────────────────────────────────────────────────────────

class TestDeque:
    ops = make_deque(nb.int64)

    def test_push_back_pop_front_is_queue(self):
        push_front, push_back, pop_front, pop_back, peek_front, peek_back, is_empty = self.ops

        @nb.njit
        def _run(push_back, pop_front):
            data = np.empty(8, nb.int64)
            head = tail = size = nb.int64(0)
            for v in (1, 2, 3):
                tail, size = push_back(data, head, tail, size, nb.int64(v))
            out = np.empty(3, nb.int64)
            for k in range(3):
                item, head, size = pop_front(data, head, tail, size)
                out[k] = item
            return out

        assert list(_run(push_back, pop_front)) == [1, 2, 3]

    def test_push_front(self):
        push_front, push_back, pop_front, pop_back, peek_front, peek_back, is_empty = self.ops

        @nb.njit
        def _run(push_front, pop_front):
            data = np.empty(8, nb.int64)
            head = tail = size = nb.int64(0)
            for v in (1, 2, 3):
                head, size = push_front(data, head, tail, size, nb.int64(v))
            out = np.empty(3, nb.int64)
            for k in range(3):
                item, head, size = pop_front(data, head, tail, size)
                out[k] = item
            return out

        assert list(_run(push_front, pop_front)) == [3, 2, 1]

    def test_mixed_ends(self):
        push_front, push_back, pop_front, pop_back, peek_front, peek_back, is_empty = self.ops

        @nb.njit
        def _run(push_front, push_back, pop_front, pop_back):
            data = np.empty(8, nb.int64)
            head = tail = size = nb.int64(0)
            tail, size = push_back(data,  head, tail, size, nb.int64(2))
            head, size = push_front(data, head, tail, size, nb.int64(1))
            tail, size = push_back(data,  head, tail, size, nb.int64(3))
            a, head, size = pop_front(data, head, tail, size)
            b, tail, size = pop_back(data,  head, tail, size)
            c, head, size = pop_front(data, head, tail, size)
            return a, b, c

        a, b, c = _run(push_front, push_back, pop_front, pop_back)
        assert (a, b, c) == (1, 3, 2)


# ── UnionFind ─────────────────────────────────────────────────────────────────

class TestUnionFind:
    ops = make_union_find()

    def test_initially_disconnected(self):
        find, union, connected = self.ops

        @nb.njit
        def _run(connected):
            parent = np.arange(5, dtype=nb.int64)
            rank   = np.zeros(5, nb.int64)
            return connected(parent, nb.int64(0), nb.int64(1))

        assert not _run(connected)

    def test_union_connects(self):
        find, union, connected = self.ops

        @nb.njit
        def _run(union, connected):
            parent = np.arange(5, dtype=nb.int64)
            rank   = np.zeros(5, nb.int64)
            union(parent, rank, nb.int64(0), nb.int64(1))
            union(parent, rank, nb.int64(1), nb.int64(2))
            c01 = connected(parent, nb.int64(0), nb.int64(1))
            c02 = connected(parent, nb.int64(0), nb.int64(2))
            c03 = connected(parent, nb.int64(0), nb.int64(3))
            return c01, c02, c03

        c01, c02, c03 = _run(union, connected)
        assert c01 and c02 and not c03
