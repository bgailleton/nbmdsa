"""Numba-accelerated data structures."""
# ruff: noqa: F401

from nbmdsa.structures.heap_st import (
    MinHeapStType, MaxHeapStType,
    MinHeapSt, MaxHeapSt,
    MinHeapStF32, MinHeapStF64, MinHeapStI32, MinHeapStI64, MinHeapStU8,
    MaxHeapStF32, MaxHeapStF64, MaxHeapStI32, MaxHeapStI64, MaxHeapStU8,
    make_min_heap_st, make_max_heap_st,
    make_min_heap_st_from, make_max_heap_st_from,
)
from nbmdsa.structures.heap_dy import (
    MinHeapDyType, MaxHeapDyType,
    MinHeapDy, MaxHeapDy,
    MinHeapDyF32, MinHeapDyF64, MinHeapDyI32, MinHeapDyI64, MinHeapDyU8,
    MaxHeapDyF32, MaxHeapDyF64, MaxHeapDyI32, MaxHeapDyI64, MaxHeapDyU8,
    make_min_heap_dy, make_max_heap_dy,
    make_min_heap_dy_from, make_max_heap_dy_from,
)
from nbmdsa.structures.grid_nb import (
    GridNbNorm1DRType, GridNbEW1DRType, GridNbNS1DRType,
    GridNbNorm1DCType, GridNbEW1DCType, GridNbNS1DCType,
    GridNbNorm2DType,  GridNbEW2DType,  GridNbNS2DType,
    GridNbNorm1DRInst, GridNbEW1DRInst, GridNbNS1DRInst,
    GridNbNorm1DCInst, GridNbEW1DCInst, GridNbNS1DCInst,
    GridNbNorm2DInst,  GridNbEW2DInst,  GridNbNS2DInst,
    GridNbNorm1DR, GridNbEW1DR, GridNbNS1DR,
    GridNbNorm1DC, GridNbEW1DC, GridNbNS1DC,
    GridNbNorm2D,  GridNbEW2D,  GridNbNS2D,
    make_grid_nb,
)
from nbmdsa.structures.tree_sd import (
    SteepTreeImplType, SteepTreeParType, SteepTreeFullType,
    SteepTreeImplInst, SteepTreeParInst, SteepTreeFullInst,
    SteepTreeImpl, SteepTreePar, SteepTreeFull,
    make_steep_tree_impl, make_steep_tree_par, make_steep_tree_full,
    make_tree,
)
from nbmdsa.structures.queue_st import (
    QueueType,
    QueueF32, QueueF64, QueueI32, QueueI64, QueueU8,
    Queue,
    make_queue, make_queue_type,
)
from nbmdsa.structures.stack_st import (
    StackType,
    StackF32, StackF64, StackI32, StackI64, StackU8,
    Stack,
    make_stack, make_stack_type,
)
from nbmdsa.structures.heap_bd import (
    MinHeapBdType, MaxHeapBdType,
    MinHeapBd, MaxHeapBd,
    MinHeapBdF32, MinHeapBdF64, MinHeapBdI32, MinHeapBdI64, MinHeapBdU8,
    MaxHeapBdF32, MaxHeapBdF64, MaxHeapBdI32, MaxHeapBdI64, MaxHeapBdU8,
    make_min_heap_bd, make_max_heap_bd,
    make_min_heap_bd_from, make_max_heap_bd_from,
)

__all__ = [
    # static
    "MinHeapStType", "MaxHeapStType",
    "MinHeapSt", "MaxHeapSt",
    "MinHeapStF32", "MinHeapStF64", "MinHeapStI32", "MinHeapStI64", "MinHeapStU8",
    "MaxHeapStF32", "MaxHeapStF64", "MaxHeapStI32", "MaxHeapStI64", "MaxHeapStU8",
    "make_min_heap_st", "make_max_heap_st",
    "make_min_heap_st_from", "make_max_heap_st_from",
    # dynamic
    "MinHeapDyType", "MaxHeapDyType",
    "MinHeapDy", "MaxHeapDy",
    "MinHeapDyF32", "MinHeapDyF64", "MinHeapDyI32", "MinHeapDyI64", "MinHeapDyU8",
    "MaxHeapDyF32", "MaxHeapDyF64", "MaxHeapDyI32", "MaxHeapDyI64", "MaxHeapDyU8",
    "make_min_heap_dy", "make_max_heap_dy",
    "make_min_heap_dy_from", "make_max_heap_dy_from",
    # bounded
    "MinHeapBdType", "MaxHeapBdType",
    "MinHeapBd", "MaxHeapBd",
    "MinHeapBdF32", "MinHeapBdF64", "MinHeapBdI32", "MinHeapBdI64", "MinHeapBdU8",
    "MaxHeapBdF32", "MaxHeapBdF64", "MaxHeapBdI32", "MaxHeapBdI64", "MaxHeapBdU8",
    "make_min_heap_bd", "make_max_heap_bd",
    "make_min_heap_bd_from", "make_max_heap_bd_from",
    # grid neighbourers
    "GridNbNorm1DRType", "GridNbEW1DRType", "GridNbNS1DRType",
    "GridNbNorm1DCType", "GridNbEW1DCType", "GridNbNS1DCType",
    "GridNbNorm2DType",  "GridNbEW2DType",  "GridNbNS2DType",
    "GridNbNorm1DRInst", "GridNbEW1DRInst", "GridNbNS1DRInst",
    "GridNbNorm1DCInst", "GridNbEW1DCInst", "GridNbNS1DCInst",
    "GridNbNorm2DInst",  "GridNbEW2DInst",  "GridNbNS2DInst",
    "GridNbNorm1DR", "GridNbEW1DR", "GridNbNS1DR",
    "GridNbNorm1DC", "GridNbEW1DC", "GridNbNS1DC",
    "GridNbNorm2D",  "GridNbEW2D",  "GridNbNS2D",
    "make_grid_nb",
    # queue
    "QueueType",
    "QueueF32", "QueueF64", "QueueI32", "QueueI64", "QueueU8",
    "Queue",
    "make_queue", "make_queue_type",
    # stack
    "StackType",
    "StackF32", "StackF64", "StackI32", "StackI64", "StackU8",
    "Stack",
    "make_stack", "make_stack_type",
    # steepest-descent trees
    "SteepTreeImplType", "SteepTreeParType", "SteepTreeFullType",
    "SteepTreeImplInst", "SteepTreeParInst",  "SteepTreeFullInst",
    "SteepTreeImpl",     "SteepTreePar",       "SteepTreeFull",
    "make_steep_tree_impl", "make_steep_tree_par", "make_steep_tree_full",
    "make_tree",
]
