"""Priority-flood + epsilon, fully inlined.

No structref, no neighbourer, no abstractions. Single @njit function with raw
arrays and inline heap/queue logic. D8, normal borders, float32 heap scores.
Exists purely to isolate algorithm speed from structref overhead.
"""

import numpy as np
import numba as nb


@nb.njit
def _sift_up(h_score, h_idx, i):
    while i > nb.int64(0):
        p = (i - nb.int64(1)) >> nb.int64(1)
        if h_score[p] > h_score[i]:
            h_score[p], h_score[i] = h_score[i], h_score[p]
            h_idx[p],   h_idx[i]   = h_idx[i],   h_idx[p]
            i = p
        else:
            break


@nb.njit
def _sift_down(h_score, h_idx, size, i):
    while True:
        best = i
        l = nb.int64(2) * i + nb.int64(1)
        r = l + nb.int64(1)
        if l < size and h_score[l] < h_score[best]:
            best = l
        if r < size and h_score[r] < h_score[best]:
            best = r
        if best == i:
            break
        h_score[i], h_score[best] = h_score[best], h_score[i]
        h_idx[i],   h_idx[best]   = h_idx[best],   h_idx[i]
        i = best


@nb.njit
def priority_flood_epsilon_inline(z_f64, nrows, ncols):
    n = nrows * ncols

    z = np.empty(n, nb.float32)
    for i in range(n):
        z[i] = nb.float32(z_f64[i])

    closed = np.zeros(n, nb.bool_)

    h_score = np.empty(n, nb.float32)
    h_idx   = np.empty(n, nb.int64)
    h_size  = nb.int64(0)

    q_data = np.empty(n, nb.int64)
    q_head = nb.int64(0)
    q_tail = nb.int64(0)
    q_size = nb.int64(0)

    INF = nb.float32(3.4028235e+38)

    for row in range(nrows):
        for col in range(ncols):
            if row == 0 or row == nrows - 1 or col == 0 or col == ncols - 1:
                idx = row * ncols + col
                closed[idx] = True
                h_score[h_size] = z[idx]
                h_idx[h_size]   = idx
                _sift_up(h_score, h_idx, h_size)
                h_size += nb.int64(1)

    while h_size > nb.int64(0) or q_size > nb.int64(0):
        if q_size > nb.int64(0) and (h_size == nb.int64(0) or h_score[0] >= z[q_data[q_head]]):
            c   = q_data[q_head]
            q_head = (q_head + nb.int64(1)) % n
            q_size -= nb.int64(1)
            c_z = z[c]
        else:
            c_z = h_score[0]
            c   = h_idx[0]
            h_size -= nb.int64(1)
            h_score[0] = h_score[h_size]
            h_idx[0]   = h_idx[h_size]
            _sift_down(h_score, h_idx, h_size, nb.int64(0))

        row = c // ncols
        col = c % ncols

        for dr, dc in ((-1, -1), (-1, 0), (-1, 1),
                       ( 0, -1),           ( 0, 1),
                       ( 1, -1), ( 1, 0), ( 1, 1)):
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                continue
            nb_idx = nr * ncols + nc
            if closed[nb_idx]:
                continue
            closed[nb_idx] = True
            eps_z = np.nextafter(c_z, INF)
            if z[nb_idx] <= eps_z:
                z[nb_idx] = eps_z
                q_data[q_tail] = nb_idx
                q_tail = (q_tail + nb.int64(1)) % n
                q_size += nb.int64(1)
            else:
                h_score[h_size] = z[nb_idx]
                h_idx[h_size]   = nb_idx
                _sift_up(h_score, h_idx, h_size)
                h_size += nb.int64(1)

    out = np.empty(n, nb.float64)
    for i in range(n):
        out[i] = nb.float64(z[i])
    return out
