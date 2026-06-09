"""Depression filling via Priority-Flood + Epsilon (Barnes et al. 2014).

Raises every pit cell to just above the lowest spill point into the
surrounding terrain, using std::nextafter-style epsilon increments.
The result has no flat areas: every cell has a strictly lower neighbour
toward the outlet.

Reference:
    Barnes R., Lehman C., Mulla D. (2014). Priority-flood: An optimal
    depression-filling and watershed-labeling algorithm for digital elevation
    models. Computers & Geosciences 62, 117-127.

Algorithm uses two structures:
    open  — min-heap of unvisited cells, seeded with the border
    pit   — FIFO queue of cells inside a depression being filled

A cell goes to pit (not open) when its elevation is ≤ nextafter(c_z, +inf),
i.e. it would form a flat or a reversed gradient relative to its processor.
Its elevation is raised to nextafter(c_z, +inf) so a strict downslope
gradient is always maintained.

Public API:
    priority_flood_epsilon(z, g) -> float64[N]
        z: 1-D float64 elevation array (row-major), modified in-place copy
        g: any 1-D grid neighbourer (normal / EW / NS, D4 / D8)
        Returns the filled elevation array.
"""

import numpy as np
import numba as nb
from numba.experimental import structref

from nbmdsa.structures.heap_dy import MinHeapDyF32
from nbmdsa.structures.queue_st import QueueI64

# ── Core @njit implementation ─────────────────────────────────────────────────

@nb.njit
def _priority_flood_epsilon(z, g, base_mask):
    n = nb.int64(len(z))

    closed = np.zeros(n, nb.bool_)
    nbrs   = np.empty(nb.int64(8), nb.int64)

    open_h = structref.new(MinHeapDyF32)
    open_h.indices   = np.empty(nb.int64(1024), nb.int64)
    open_h.scores    = np.empty(nb.int64(1024), nb.float32)
    open_h.size      = nb.int64(0)
    open_h.capacity  = nb.int64(1024)
    open_h.max_chunk = nb.int64(1 << 62)

    pit_q = structref.new(QueueI64)
    pit_q.data     = np.empty(n, nb.int64)
    pit_q.head     = nb.int64(0)
    pit_q.tail     = nb.int64(0)
    pit_q.size     = nb.int64(0)
    pit_q.capacity = n

    for i in range(n):
        if base_mask[i]:
            closed[i] = True
            open_h.emplace(nb.int64(i), nb.float32(z[i]))

    INF = nb.float32(3.4028235e+38)

    while not open_h.is_empty() or not pit_q.is_empty():
        if (not pit_q.is_empty() and not open_h.is_empty()
                and open_h.top_score() == nb.float32(z[pit_q.peek()])):
            c, c_z = open_h.top()
            open_h.pop()
        elif not pit_q.is_empty():
            c   = pit_q.dequeue()
            c_z = nb.float32(z[c])
        else:
            c, c_z = open_h.top()
            open_h.pop()

        cnt = g.neighbours(c, nbrs)
        for k in range(cnt):
            nb_idx = nbrs[k]
            if closed[nb_idx]:
                continue
            closed[nb_idx] = True
            eps_z = np.nextafter(c_z, INF)
            if nb.float32(z[nb_idx]) <= eps_z:
                z[nb_idx] = nb.float64(eps_z)
                pit_q.enqueue(nb_idx)
            else:
                open_h.emplace(nb_idx, nb.float32(z[nb_idx]))

    return z

# ── Public Python wrapper ─────────────────────────────────────────────────────

def priority_flood_epsilon(z: np.ndarray, g, base_mask: np.ndarray) -> np.ndarray:
    """Fill depressions with epsilon gradients (Barnes et al. 2014).

    base_mask: bool[N] — cells seeded as outlets into the priority queue.
               Typically the grid border, but any set of known outlets works.

    Returns a new float64 array; the input is not modified.
    Build the steepest-descent tree from the returned array to get a
    depression-free drainage network.
    """
    z_out = np.array(z, dtype=np.float64).ravel()
    return _priority_flood_epsilon(z_out, g, np.asarray(base_mask, dtype=np.bool_))
