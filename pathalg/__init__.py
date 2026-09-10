"""Path algebra computations for commuting squares of subfactors.

Quick start::

    from pathalg import CommutingSquare, Level

    sq = CommutingSquare(V00=..., V01=..., V10=..., V11=...,
                         h0=..., v0=..., h1=..., v1=..., u_cells=...)
    sq.verify()                       # unitarity, bi-unitarity, weights

    lv = Level(sq, n=3)               # everything represented on H_4
    lv.verify()                       # W unitary, trace, Jones, commuting square

    x = lv.matrix_unit(xi, eta)                  # in A_{3,1}
    y = lv.matrix_unit(alpha, beta, which='K')   # in A_{3,0}, pulled back to H_4
    lv.expectation(x, 2, 1)                      # E onto A_{2,1}
    lv.relative_commutant(0, 1)                  # basis of A_{0,1}' cap A_{3,1}
"""

from .backend import ArrayBackend, NumpyBackend, SympyBackend, get_backend, gpu_available
from .paths import BlockMap, Op, PathSpace
from .square import CheckResult, CommutingSquare, Graph
from .tower import Level
from .weights import pf_bipartite

__all__ = [
    "CommutingSquare",
    "Graph",
    "CheckResult",
    "Level",
    "PathSpace",
    "Op",
    "BlockMap",
    "SympyBackend",
    "ArrayBackend",
    "NumpyBackend",
    "get_backend",
    "gpu_available",
    "pf_bipartite",
]
