"""Shared njit wrappers and proxy mixin for all heap variants."""

import numba as nb
import numpy as np


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

@nb.njit
def _nb_top_score(h):
    return h.top_score()

@nb.njit
def _nb_top_idx(h):
    return h.top_idx()

@nb.njit
def _nb_reserve(h, n):
    h.reserve(n)

@nb.njit
def _nb_shrink_to_fit(h):
    h.shrink_to_fit()

@nb.njit
def _nb_drain_sorted(h, out_indices, out_scores):
    h.drain_sorted(out_indices, out_scores)


class _HeapProxyMixin:
    """Python-side method interface shared by all heap proxy classes."""

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

    def top_score(self):
        return _nb_top_score(self)

    def top_idx(self):
        return _nb_top_idx(self)

    def reserve(self, n):
        _nb_reserve(self, n)

    def shrink_to_fit(self):
        _nb_shrink_to_fit(self)

    def drain_sorted(self, out_indices, out_scores):
        _nb_drain_sorted(self, out_indices, out_scores)
