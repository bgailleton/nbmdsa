"""Closure-based primitive data structures — no structref, fully inlineable.

Return-new-size convention: mutable scalars (size, head, tail) are returned
rather than mutated — state stays explicit in the caller.

State arrays are always caller-allocated; closures capture only dtype for
lru_cache memoisation (Numba specialises on actual array types at call time).
"""

from collections import namedtuple
from functools import lru_cache

import numba as nb


# ── Min-Heap (index-score pairs) ──────────────────────────────────────────────

_MinHeap = namedtuple('MinHeap', ['emplace', 'pop', 'top', 'is_empty'])


@lru_cache(maxsize=None)
def make_heap(score_dtype=nb.float64, idx_dtype=nb.int64):
    """Return MinHeap closures specialised to score/idx dtypes.

    State: scores[capacity], indices[capacity], size: int64
    """
    @nb.njit
    def _sift_up(scores, indices, i):
        while i > nb.int64(0):
            p = (i - nb.int64(1)) >> nb.int64(1)
            if scores[p] > scores[i]:
                scores[p], scores[i] = scores[i], scores[p]
                indices[p], indices[i] = indices[i], indices[p]
                i = p
            else:
                break

    @nb.njit
    def _sift_down(scores, indices, size, i):
        while True:
            best = i
            l = nb.int64(2) * i + nb.int64(1)
            r = l + nb.int64(1)
            if l < size and scores[l] < scores[best]:
                best = l
            if r < size and scores[r] < scores[best]:
                best = r
            if best == i:
                break
            scores[i], scores[best] = scores[best], scores[i]
            indices[i], indices[best] = indices[best], indices[i]
            i = best

    @nb.njit
    def emplace(scores, indices, size, score, idx):
        scores[size] = score
        indices[size] = idx
        _sift_up(scores, indices, size)
        return size + nb.int64(1)

    @nb.njit
    def pop(scores, indices, size):
        size = size - nb.int64(1)
        scores[0] = scores[size]
        indices[0] = indices[size]
        _sift_down(scores, indices, size, nb.int64(0))
        return size

    @nb.njit
    def top(scores, indices):
        return scores[0], indices[0]

    @nb.njit
    def is_empty(size):
        return size == nb.int64(0)

    return _MinHeap(emplace, pop, top, is_empty)


# ── Queue (circular buffer FIFO) ──────────────────────────────────────────────

_Queue = namedtuple('Queue', ['enqueue', 'dequeue', 'peek', 'is_empty'])


@lru_cache(maxsize=None)
def make_queue(dtype=nb.int64):
    """Return Queue closures.

    State: data[capacity], head: int64, tail: int64, size: int64
    enqueue -> (tail, size)
    dequeue -> (item, head, size)
    """
    @nb.njit
    def enqueue(data, head, tail, size, item):
        data[tail] = item
        tail = (tail + nb.int64(1)) % nb.int64(len(data))
        return tail, size + nb.int64(1)

    @nb.njit
    def dequeue(data, head, tail, size):
        item = data[head]
        head = (head + nb.int64(1)) % nb.int64(len(data))
        return item, head, size - nb.int64(1)

    @nb.njit
    def peek(data, head):
        return data[head]

    @nb.njit
    def is_empty(size):
        return size == nb.int64(0)

    return _Queue(enqueue, dequeue, peek, is_empty)


# ── Stack ─────────────────────────────────────────────────────────────────────

_Stack = namedtuple('Stack', ['push', 'pop', 'peek', 'is_empty'])


@lru_cache(maxsize=None)
def make_stack(dtype=nb.int64):
    """Return Stack closures.

    State: data[capacity], size: int64
    push -> size
    pop  -> (item, size)
    """
    @nb.njit
    def push(data, size, item):
        data[size] = item
        return size + nb.int64(1)

    @nb.njit
    def pop(data, size):
        size = size - nb.int64(1)
        return data[size], size

    @nb.njit
    def peek(data, size):
        return data[size - nb.int64(1)]

    @nb.njit
    def is_empty(size):
        return size == nb.int64(0)

    return _Stack(push, pop, peek, is_empty)


# ── Deque (double-ended queue, circular buffer) ───────────────────────────────

_Deque = namedtuple('Deque', [
    'push_front', 'push_back',
    'pop_front',  'pop_back',
    'peek_front', 'peek_back',
    'is_empty',
])


@lru_cache(maxsize=None)
def make_deque(dtype=nb.int64):
    """Return Deque closures.

    State: data[capacity], head: int64, tail: int64, size: int64
    push_front -> (head, size)
    push_back  -> (tail, size)
    pop_front  -> (item, head, size)
    pop_back   -> (item, tail, size)
    """
    @nb.njit
    def push_back(data, head, tail, size, item):
        data[tail] = item
        tail = (tail + nb.int64(1)) % nb.int64(len(data))
        return tail, size + nb.int64(1)

    @nb.njit
    def push_front(data, head, tail, size, item):
        n = nb.int64(len(data))
        head = (head - nb.int64(1)) % n
        data[head] = item
        return head, size + nb.int64(1)

    @nb.njit
    def pop_front(data, head, tail, size):
        item = data[head]
        head = (head + nb.int64(1)) % nb.int64(len(data))
        return item, head, size - nb.int64(1)

    @nb.njit
    def pop_back(data, head, tail, size):
        n = nb.int64(len(data))
        tail = (tail - nb.int64(1)) % n
        return data[tail], tail, size - nb.int64(1)

    @nb.njit
    def peek_front(data, head):
        return data[head]

    @nb.njit
    def peek_back(data, tail):
        n = nb.int64(len(data))
        return data[(tail - nb.int64(1)) % n]

    @nb.njit
    def is_empty(size):
        return size == nb.int64(0)

    return _Deque(push_front, push_back, pop_front, pop_back,
                  peek_front, peek_back, is_empty)


# ── Union-Find (path halving + union by rank) ─────────────────────────────────

_UnionFind = namedtuple('UnionFind', ['find', 'union', 'connected'])


@lru_cache(maxsize=None)
def make_union_find():
    """Return UnionFind closures (always int64).

    State: parent[n]: int64, rank[n]: int64
    Initialise: parent[i] = i, rank[i] = 0
    """
    @nb.njit
    def find(parent, i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path halving
            i = parent[i]
        return i

    @nb.njit
    def union(parent, rank, i, j):
        ri = find(parent, i)
        rj = find(parent, j)
        if ri == rj:
            return
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += nb.int64(1)

    @nb.njit
    def connected(parent, i, j):
        return find(parent, i) == find(parent, j)

    return _UnionFind(find, union, connected)
