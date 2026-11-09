"""Relative commutants along the 1-parameter family of 4 x 4 Hadamard matrices.

Run with ``python hadamard4_demo.py``.

The family
----------
Haagerup showed that every 4 x 4 complex Hadamard matrix is equivalent to one of

        [ 1   1   1   1 ]
        [ 1   q  -1  -q ]         q = i e^{ia},
        [ 1  -1   1  -1 ]
        [ 1  -q  -1   q ]

(F_4^(1)(a) in Tadej-Zyczkowski, "A concise guide to complex Hadamard matrices",
arXiv:quant-ph/0512154; pathalg.examples.f4_family.)  a = 0 is the Fourier matrix
F_4 and a = pi/2 is F_2 (x) F_2, and 0 <= a <= pi/2 covers every class.  Each
matrix is a bi-unitary connection on the spin model square of
pathalg.examples.hadamard, so each a gives a subfactor N < M of index 4.

What is computed
----------------
By Ocneanu compactness, for n >= 1,

        N' cap M_(n-1) = A_(0,1)' cap A_(n,0)
        M' cap M_(n-1) = A_(1,1)' cap A_(n,0)          (n >= 2)

are the two sides of the standard invariant: closed walks of length 2n at * on
the principal graph, and of length 2n - 2 on the dual principal graph.

What to look for
----------------
Index 4 leaves only affine and infinite graphs, and the run lands on these:

  * a a rational multiple of pi: D_(m+2)^(1), the chain with a fork at each end,
    where m = 2 ord(q^4) = 2 * denominator(2 a / pi).  That is the star
    D_4^(1) with dimensions 4^(n-1) at both ends of the family, m = 2.
  * otherwise: D_infinity, one fork, with dimensions binomial(2n - 1, n) =
    1, 3, 10, 35, 126, ...

The second fork is at distance m, so a rational a follows the D_infinity row
until n = m and then leaves it by exactly 1.  The dual side agrees.  Note that
n = 2 is 3 and not 2: the whole family has an intermediate subfactor of index 2,
which is the F_2 (x) F_2 shape of the matrix surviving the deformation.

The rule for m is an observation of this code, tested by the runs recorded in
the README.  It is consistent with the quantum permutation groups reported for
this family (Banica-Bichon, "Quantum groups acting on 4 points",
arXiv:math/0703118, classifies the possibilities): a twist of a dihedral group
when q has finite order, and O_(-1)(2) when it does not.

Section 5 recomputes the same numbers with none of this library: for the magic
unitary P_ij = Proj(H_i / H_j), dim Fix(u^(x)p) is the multiplicity of the
eigenvalue 1 of the N^p x N^p matrix T_p with entries tr(P_(i1 j1) ... P_(ip jp))
(see Banica, "Truncation and duality results for Hopf image algebras",
arXiv:1404.3544).

Numerics
--------
Two cautions, both visible in the margin column of the tables.

  * The largest discarded cosine is cos(pi / m), so the cut gets harder as m
    grows, and near a special value it is cos(2 (a - a_special)).  Nothing here
    resolves a parameter within about 0.02 of a low order special value: those
    rows are reported as unresolved, not guessed.
  * method="iterative" is used on purpose.  At n = 7 method="auto" can pick the
    dense route, and under complex64 its eigh spreads the eigenvalue 1 cluster
    past the fixed cut: at a = pi/5 that returned 204 rather than 1716, with a
    discarded cosine of 0.99999.  The iterative route gets 1716, 40 times faster.
"""

from __future__ import annotations

import time

import numpy as np
import sympy as sp

from pathalg import InsufficientMemory, Level, gpu_available
from pathalg.examples import f4_family, hadamard

# ------------------------------------------------------------------- 0. knobs

# The parameters a.  Rational multiples of pi are the special points; anything
# else is D_infinity, as long as it is not within 0.02 or so of a special point.
PARAMETERS = [sp.Integer(0), sp.pi / 2, sp.pi / 4, sp.pi / 3, sp.pi / 6, sp.pi / 8,
              sp.pi / 5, sp.Integer(1)]

# Where the tower computes -- see ah_demo.py for what each choice means.
#   "numpy", "cupy", "cupy64", or "auto" (cupy64 with a GPU, else numpy)
BACKEND = "auto"

# Relative commutants are computed for n = 1 .. MAX_N.  n = 7 takes seconds per
# parameter on a GPU; drop to 6 on a CPU.
MAX_N = 7

# The parameter used for the exact level checks of section 2.
SAMPLE = sp.pi / 4
LEVEL_N = 2

# Section 4 goes up to this p.  T_p costs N^(2p + 2) complex numbers on the way.
CHECK_P = 5

# A discarded cosine above this is too close to 1 to call the cut.
MARGIN = 0.999


# ------------------------------------------------- 1. the graphs and the prediction


def predicted_m(a):
    """``m`` of ``D_(m+2)^(1)`` for a rational multiple of pi, else ``None``."""
    r = sp.sympify(a) / sp.pi
    return 2 * int((2 * r).q) if r.is_rational else None


def fork_graph(m, reach):
    """Edges of ``D_(m+2)^(1)``, or for ``m=None`` enough of ``D_infinity``.

    The root is the fork tip ``"*"``; ``reach`` is how far walks will get.
    """
    last = reach + 1 if m is None else m - 1
    edges = [("*", 1), ("*'", 1)] + [(i, i + 1) for i in range(1, last)]
    if m is not None:
        edges += [(last, "x"), (last, "x'")]
    return edges


def closed_walks(edges, start, length):
    """Number of closed walks of ``length`` at ``start``."""
    verts = list(dict.fromkeys(v for e in edges for v in e))
    index = {v: i for i, v in enumerate(verts)}
    adj = sp.zeros(len(verts), len(verts))
    for x, y in edges:
        adj[index[x], index[y]] += 1
        adj[index[y], index[x]] += 1
    return int((adj ** length)[index[start], index[start]])


def predicted_dims(a, levels):
    """What the graph of ``a`` predicts for ``dim N' cap M_(n-1)``."""
    edges = fork_graph(predicted_m(a), max(levels))
    return [closed_walks(edges, "*", 2 * n) for n in levels]


def graph_name(a):
    m = predicted_m(a)
    return "D_inf" if m is None else f"D_{m + 2}^(1)"


# ------------------------------------------------------- 2. the relative commutants


def commutant_dimensions(square, sub, levels):
    """``[dim A_sub' cap A_(n,0) for n in levels]`` and the worst margin met.

    Raises ``RuntimeError`` rather than report a level whose cut is not clear.
    """
    dims, worst_lo, worst_hi = [], 0.0, 1.0
    for n in levels:
        lv = Level(square, n)
        # the index is 4, so a level is at most about 4 times the one before
        block = 4 * dims[-1] + 16 if dims else None
        dim = lv.relative_commutant_dimension(sub, (n, 0), method="iterative", block=block)
        lo, hi = lv._last_angle_gap
        if lo > MARGIN:
            raise RuntimeError(f"n = {n}: discarded cosine {lo:.6f} against kept {hi:.6f}")
        dims.append(dim)
        worst_lo, worst_hi = max(worst_lo, lo), min(worst_hi, hi)
    return dims, worst_lo, worst_hi


def table(backend, sub, levels, shift=0):
    """One row per parameter; returns ``{a: dims}`` for the rows that finished."""
    print("  {:>8}  {:>10}  ".format("a", "graph")
          + "".join(f"{f'n={n}':>7}" for n in levels) + "  margin          time")
    results = {}
    for a in PARAMETERS:
        start = time.time()
        label = f"  {str(a):>8}  {graph_name(a):>10}  "
        try:
            dims, lo, hi = commutant_dimensions(hadamard(f4_family(a), backend=backend),
                                                sub, levels)
        except (RuntimeError, InsufficientMemory) as exc:
            print(f"{label}unresolved: {exc}")
            continue
        results[a] = dims
        want = predicted_dims(a, [n - shift for n in levels])
        mark = "" if dims == want else f"   <- the graph says {want}"
        print(label + "".join(f"{d:>7}" for d in dims)
              + f"  {lo:.4f} / {hi:.5f}  {time.time() - start:5.1f}s{mark}", flush=True)
    return results


# ------------------------------------------------ 3. the same numbers, independently


def fixed_points(H, pmax):
    """``dim Fix(u^(x)p)`` for ``p = 1 .. pmax``, with no path algebra at all.

    ``H`` is a numpy Hadamard matrix.  The products ``P_(i1 j1) ... P_(ip jp)``
    are built up one factor at a time, indexed by ``(i1, j1, i2, j2, ...)``.
    """
    N = H.shape[0]
    xi = H[:, None, :] / H[None, :, :]                       # xi[i, j] = H_i / H_j
    P = (xi[..., :, None] * xi[..., None, :].conj() / N).reshape(N * N, N, N)
    out = []
    prod = np.eye(N, dtype=complex)[None]
    for p in range(1, pmax + 1):
        prod = (prod[:, None] @ P).reshape(-1, N, N)
        order = list(range(0, 2 * p, 2)) + list(range(1, 2 * p, 2))
        T = np.trace(prod, axis1=1, axis2=2).reshape([N] * (2 * p)) / N
        T = T.transpose(order).reshape(N ** p, N ** p)
        out.append(int((abs(np.linalg.eigvalsh(T) - 1) < 1e-8).sum()))
    return out


# -------------------------------------------------------------- the demo


def main():
    backend = BACKEND
    if backend == "auto":
        backend = "cupy64" if gpu_available() else "numpy"

    print("=" * 72)
    print("1. the family and its connections, checked exactly")
    print("=" * 72)
    print(f"  q = i e^(ia), backend {backend} for sections 3 and 4\n")
    for a in PARAMETERS:
        square = hadamard(f4_family(a))
        checks = square.verify(verbose=False)
        q4 = sp.exp(4 * sp.I * a)
        print(f"  a = {str(a):>5}:  bi-unitary {all(checks)},  q^4 = {q4},  "
              f"predicted graph {graph_name(a)}")

    print("\n" + "=" * 72)
    print(f"2. one level, exactly: a = {SAMPLE}, n = {LEVEL_N}")
    print("=" * 72)
    square = hadamard(f4_family(SAMPLE))
    print(f"  {square}")
    level = Level(square, LEVEL_N)
    print(f"  {level}, blocks {level.H.block_dims()}\n")
    for result in level.verify(verbose=False):
        print(" ", result)

    print("\n" + "=" * 72)
    print("3. dim N' cap M_(n-1) = dim A_(0,1)' cap A_(n,0)")
    print("=" * 72)
    print("  The margin is the largest discarded cosine against the smallest kept one.\n")
    primal = table(backend, (0, 1), list(range(1, MAX_N + 1)))

    print("\n" + "=" * 72)
    print("4. dim M' cap M_(n-1) = dim A_(1,1)' cap A_(n,0)")
    print("=" * 72)
    print("  The dual side, so shifted by one level: n = 2 is the trivial M' cap M.\n")
    table(backend, (1, 1), list(range(2, MAX_N + 1)), shift=1)

    print("\n" + "=" * 72)
    print("5. the same numbers with no path algebra: multiplicity of 1 in T_p")
    print("=" * 72)
    print()
    for a in PARAMETERS:
        H = np.array(f4_family(a).evalf(30).tolist(), dtype=complex)
        direct = fixed_points(H, CHECK_P)
        row = primal.get(a)
        if row is None:
            verdict = "no row in section 3 to compare"
        else:
            common = min(len(row), CHECK_P)
            verdict = "agrees" if row[:common] == direct[:common] else "DISAGREES"
        print(f"  a = {str(a):>5}:  {direct}   {verdict}", flush=True)

    print("\ndone.")


if __name__ == "__main__":
    main()
