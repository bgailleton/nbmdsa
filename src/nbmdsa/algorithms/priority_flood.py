"""Priority-flood + epsilon (Barnes 2014).

Factory pattern: ``make_priority_flood(score_dtype)`` returns a ``@nb.njit``
function specialised to the score dtype.

``z`` and ``score_dtype`` must match (e.g. both float64 or both float32).

Mask convention:
  0 — nodata: never visited, never enqueued, skipped in neighbour loops
  1 — normal node
  3 — outlet: seeds the open priority queue instead of grid borders
"""

from functools import lru_cache

import numpy as np
import numba as nb


@lru_cache(maxsize=None)
def make_priority_flood(score_dtype=nb.float64):
    import numpy as _np
    _np_dtype = _np.float32 if score_dtype == nb.float32 else _np.float64
    _INF = score_dtype(_np.finfo(_np_dtype).max)

    @nb.njit
    def priority_flood_epsilon(z, mask, neighbours_fn, heap_ops, queue_ops):
        emplace, h_pop, h_top, h_empty = heap_ops
        enqueue, dequeue, q_peek, q_empty = queue_ops

        n      = nb.int64(len(z))
        INF    = _INF
        closed = np.zeros(n, nb.bool_)
        nbuf   = np.empty(nb.int64(8), nb.int64)

        h_scores  = np.empty(n, score_dtype)
        h_indices = np.empty(n, nb.int64)
        h_size    = nb.int64(0)

        q_data = np.empty(n, nb.int64)
        q_head = nb.int64(0)
        q_tail = nb.int64(0)
        q_size = nb.int64(0)

        for i in range(n):
            if mask[i] == nb.uint8(3):
                closed[i] = True
                h_size = emplace(h_scores, h_indices, h_size,
                                 score_dtype(z[i]), nb.int64(i))

        while not h_empty(h_size) or not q_empty(q_size):
            if not q_empty(q_size):
                if h_empty(h_size):
                    use_pit = True
                else:
                    top_score, _ = h_top(h_scores, h_indices)
                    use_pit = top_score >= score_dtype(z[q_peek(q_data, q_head)])
            else:
                use_pit = False

            if use_pit:
                c, q_head, q_size = dequeue(q_data, q_head, q_tail, q_size)
            else:
                _, c = h_top(h_scores, h_indices)
                h_size = h_pop(h_scores, h_indices, h_size)

            c_z = score_dtype(z[c])

            neighbours_fn(nb.int64(c), nbuf)
            for k in range(nb.int64(8)):
                nb_idx = nbuf[k]
                if nb_idx == nb.int64(-1):
                    continue
                if mask[nb_idx] == nb.uint8(0):
                    continue
                if closed[nb_idx]:
                    continue
                closed[nb_idx] = True
                eps_z = np.nextafter(c_z, INF)
                if score_dtype(z[nb_idx]) <= eps_z:
                    z[nb_idx] = eps_z
                    q_tail, q_size = enqueue(q_data, q_head, q_tail, q_size, nb_idx)
                else:
                    h_size = emplace(h_scores, h_indices, h_size,
                                     score_dtype(z[nb_idx]), nb_idx)

        return z

    return priority_flood_epsilon
