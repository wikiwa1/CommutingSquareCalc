"""The Asaeda-Haagerup commuting square, worked through the same way as demo.py.

Run with ``python ah_demo.py``.

Reference
---------
The connection comes from

    D. Bisch and J. Cáceres, "New hyperfinite subfactors with infinite depth",
    arXiv:2509.00227, https://arxiv.org/abs/2509.00227

Section 5, "A 1-parameter family of commuting squares for 4-stars", constructs
a one-parameter family of bi-unitary connections on the 4-stars S(i,i,j,j);
``S1122.txt`` is the case S(1,1,2,2), i = 1 and j = 2, whose index is the
(5 + sqrt(17)) / 2 entry of the table there.  The parameter s and the angle t
below are the ones of that section.  Theorem 5.4 uses the family to show that
there are irreducible hyperfinite A_infinity subfactors at this index.

The square
----------
The data is the one in ``S1122.txt``: a 3 x 4 initial graph

        a2 --- a1
               |
        A  --- a1, b1, c1, d1
               |
        d2 --- d1

used for all four sides of the square, together with a one parameter family of
connections.  The graph norm squared is exactly the Asaeda-Haagerup index

        delta^2 = (5 + sqrt(17)) / 2 = 4.56155...

so if the connection is bi-unitary the tower it generates is a candidate for the
Asaeda-Haagerup subfactor, and the higher relative commutant dimensions

        dim A_(0,1)' cap A_(n,0)                   (Ocneanu compactness)

are the invariant to look at.

The connection is bi-unitary for every s, but it is not flat for every s: only
finitely many parameters are expected to give the Asaeda-Haagerup subfactor,
and the rest give A_infinity.  So the Catalan numbers 1, 2, 5, 14, 42, 132 are
the *generic* answer here, not the target -- a run that reproduces them all the
way up is reporting A_infinity, i.e. this s is not one of the special ones.
Asaeda-Haagerup is 4-supertransitive, so its sequence agrees with Catalan
through n = 4 as well; the two only part company at n = 5, where loops through
the branch point of the principal graph push the dimension strictly above 42.
Everything below n = 5 is consistent with both, so n = 5 is the only row of the
table that decides anything.

The knobs are all at the top.  Section 4 is the expensive one, and it is the
reason the tower has a GPU backend: n = 6 takes 815 s in numpy and 5 s in
complex64 on the card.
"""

from __future__ import annotations

import time

import sympy as sp

from pathalg import CommutingSquare, Level, gpu_available

# ------------------------------------------------------------------- 0. knobs

# The parameter of the connection.  Every value in [0, 2 pi) gives a bi-unitary
# connection; only finitely many are expected to be flat and give Asaeda-
# Haagerup, the rest giving A_infinity.  So this is the knob to sweep.  Pick a
# value where cos(s) and exp(i s / 2) stay closed form to run exactly.
S = sp.pi / 3

# Where the tower computes.  The square itself is always built exactly; this
# only sets the arithmetic of the linear algebra.
#
#   "sympy"   exact, and far too slow past about n = 3
#   "numpy"   complex128 on the CPU
#   "cupy"    complex128 on the GPU
#   "cupy64"  complex64 on the GPU -- the fast one.  Single precision is safe
#             here because every decision is a principal angle that lands
#             either near 0.96 or at 1.0; section 4 prints the margin so you
#             can watch it.
#   "auto"    "cupy64" if this machine has a usable GPU, else "numpy".  The
#             default, so the same file runs everywhere; only the time differs.
BACKEND = "auto"

# Level used for sections 2 and 3 (the illustrative one).
LEVEL_N = 2

# Higher relative commutants are computed for n = 1 .. MAX_N.  Cost grows fast:
# dim H_(n+1) runs 10, 28, 44, 128, 200, 584 and the outer commutant that gets
# cut down has dimension 6, 22, 96, 434, 1976, 9010.  n = 7 does not fit in
# memory under any backend -- see the README.
MAX_N = 6


# ------------------------------------------------- 1. the parameters of S1122
#
#   delta = sqrt((5 + sqrt(17)) / 2)
#   w1 = e^{i s}
#   t  = (s - arccos(cos(s) / delta^2)) / 2,      w2 = e^{i t}
#   z1 = (delta (1 + w1) + delta^2 (w2 - w1/w2)) / (2 (1 - delta^2))
#   z2 = (delta (1 - w1) + delta^2 (w2 + w1/w2)) / (2 (1 - delta^2))
#
# arccos is not something the exact machinery can work with, so w2 is written
# out in radicals instead.  With c = cos(s) / delta^2 and phi = arccos(c) we
# have phi/2 in [0, pi/2], hence
#
#   e^{-i phi / 2} = cos(phi/2) - i sin(phi/2)
#                  = sqrt((1 + c) / 2) - i sqrt((1 - c) / 2)
#
# and w2 = e^{i s / 2} e^{-i phi / 2}.  This is an identity, valid for every s.

DELTA = sp.sqrt((5 + sp.sqrt(17)) / 2)


def ah_params(s, delta=DELTA):
    """Return ``(w1, w2, z1, z2)`` for the parameter ``s``, in closed form."""
    s = sp.sympify(s)
    w1 = sp.exp(sp.I * s)
    c = sp.cos(s) / delta ** 2
    w2 = sp.exp(sp.I * s / 2) * (sp.sqrt((1 + c) / 2) - sp.I * sp.sqrt((1 - c) / 2))
    denom = 2 * (1 - delta ** 2)
    z1 = (delta * (1 + w1) + delta ** 2 * (w2 - w1 / w2)) / denom
    z2 = (delta * (1 - w1) + delta ** 2 * (w2 + w1 / w2)) / denom
    return w1, w2, z1, z2


# ------------------------------------------------------------- 2. the square
#
#         a ---- b          a in V00 (top left)     b in V01 (top right)
#         |      |          c in V10 (bottom left)  d in V11 (bottom right)
#         c ---- d
#
# and the cells carry v^{c,b}_{a,d}, so the keys below are (a, b, c, d) exactly
# as written in S1122.txt.  The u half is derived by the renormalisation rule.

EDGES_TOP = [("a2", "a1"), ("A", "a1"), ("A", "b1"), ("A", "c1"), ("A", "d1"),
             ("d2", "d1")]                                     # V00 - V01/V10
EDGES_BOTTOM = [("a1", "a2"), ("a1", "A"), ("b1", "A"), ("c1", "A"),
                ("d1", "A"), ("d1", "d2")]                     # V10/V01 - V11


def ah_cells(s, delta=DELTA):
    """The 22 cells of the connection, as ``v^{c,b}_{a,d}`` keyed by (a,b,c,d)."""
    w1, w2, z1, z2 = ah_params(s, delta)
    r = sp.sqrt(delta ** 2 - 1) / delta
    cells = {
        ("A", "b1", "a1", "A"): sp.Integer(1),
        ("A", "c1", "a1", "A"): sp.Integer(1),
        ("A", "d1", "a1", "A"): sp.Integer(1),
        ("A", "a1", "b1", "A"): sp.Integer(1),
        ("A", "a1", "c1", "A"): sp.Integer(1),
        ("A", "a1", "d1", "A"): sp.Integer(-1),
        ("A", "d1", "b1", "A"): 1 / w2,
        ("A", "d1", "c1", "A"): -1 / w2,
        ("A", "b1", "d1", "A"): w2 / w1,
        ("A", "c1", "d1", "A"): -w2 / w1,
        ("A", "b1", "b1", "A"): 1 / z1,
        ("A", "c1", "b1", "A"): 1 / z2,
        ("A", "b1", "c1", "A"): -z2 / w1,
        ("A", "c1", "c1", "A"): z1 / w1,
        ("a2", "a1", "a1", "a2"): -1 / delta,
        ("a2", "a1", "a1", "A"): r,
        ("A", "a1", "a1", "a2"): r,
        ("A", "a1", "a1", "A"): 1 / delta,
        ("d2", "d1", "d1", "d2"): -1 / delta,
        ("d2", "d1", "d1", "A"): r,
        ("A", "d1", "d1", "d2"): r,
        ("A", "d1", "d1", "A"): 1 / delta,
    }
    return {k: sp.expand(sp.sympify(val)) for k, val in cells.items()}


def ah_square(s=S, delta=DELTA, **kwargs):
    """The Asaeda-Haagerup commuting square at parameter ``s``."""
    return CommutingSquare(
        V00=["a2", "A", "d2"],
        V01=["a1", "b1", "c1", "d1"],
        V10=["a1", "b1", "c1", "d1"],
        V11=["a2", "A", "d2"],
        h0=EDGES_TOP,        # V00 - V01
        v0=EDGES_TOP,        # V00 - V10
        h1=EDGES_BOTTOM,     # V10 - V11
        v1=EDGES_BOTTOM,     # V01 - V11
        v_cells=ah_cells(s, delta),
        **kwargs,
    )


# -------------------------------------------------------------- the demo


def catalan(n):
    return int(sp.binomial(2 * n, n)) // (n + 1)


def closed_walks(edges, start, length):
    """Number of closed walks of ``length`` at ``start`` in a bipartite graph."""
    verts = sorted({v for e in edges for v in e})
    index = {v: i for i, v in enumerate(verts)}
    adj = sp.zeros(len(verts), len(verts))
    for x, y in edges:
        adj[index[x], index[y]] += 1
        adj[index[y], index[x]] += 1
    return int((adj ** length)[index[start], index[start]])


def main():
    backend = BACKEND
    if backend == "auto":
        backend = "cupy64" if gpu_available() else "numpy"

    print("=" * 72)
    print("1. the Asaeda-Haagerup square and its connection")
    print("=" * 72)
    print(f"  parameter s = {S}   (= {float(S):.6f}),  backend {backend}")
    w1, w2, z1, z2 = ah_params(S)
    for name, val in (("w1", w1), ("w2", w2), ("z1", z1), ("z2", z2)):
        print(f"    {name} = {complex(sp.N(val)):.6f}")

    square = ah_square(S, backend=backend)
    print()
    print(square)
    print("  Perron-Frobenius weights:")
    for tag in ("00", "01", "10", "11"):
        print(f"    V{tag}: {square.mu[tag]}")
    print(f"  beta_h^2 = {sp.nsimplify(sp.N(square.beta_h ** 2), [sp.sqrt(17)])}"
          f"  (= {float(sp.N(square.beta_h ** 2)):.9f}),"
          f"   Asaeda-Haagerup index = {float(sp.N((5 + sp.sqrt(17)) / 2)):.9f}")
    print()
    for result in square.verify(verbose=False):
        print(" ", result)

    print("\n  a few cells, with the u half derived from v:")
    for cell in list(square.cells())[:6]:
        print(f"    {cell}:  v = {complex(sp.N(square.v(*cell))):.6f}"
              f"   u = {complex(sp.N(square.u(*cell))):.6f}")

    # ------------------------------------------------------------ 2. a level
    level = Level(square, LEVEL_N, backend=backend)
    print("\n" + "=" * 72)
    print(f"2. level n = {LEVEL_N}: everything represented on H_{LEVEL_N + 1}")
    print("=" * 72)
    print(f"  {level}")
    print(f"  H_{LEVEL_N + 1} runs through "
          f"{' - '.join('V' + t for t in level.H.types)}")
    print(f"  K_{LEVEL_N + 1} runs through "
          f"{' - '.join('V' + t for t in level.K.types)}")
    print(f"  dim H_{LEVEL_N + 1} = {len(level.H.paths)}")
    print(f"  blocks (by end vertex): {level.H.block_dims()}")
    print()
    for result in level.verify(verbose=False):
        print(" ", result)

    # -------------------------------------------------------- 3. some elements
    print("\n" + "=" * 72)
    print("3. elements of A_(n,1), A_(n-1,1) and A_(n,0), all acting on H_(n+1)")
    print("=" * 72)

    blk = max(level.H.blocks.values(), key=len)
    x = level.matrix_unit(blk[0], blk[1])
    print(f"  x = E[{blk[0]} , {blk[1]}]  in A_({LEVEL_N},1)")
    print(f"    trace {x.trace()}")

    sub = max(level.H.level_blocks[LEVEL_N].values(), key=len)
    y = level.matrix_unit(sub[0], sub[0])          # diagonal, so its trace is a weight
    print(f"  y = E[{sub[0]} , {sub[0]}]  in A_({LEVEL_N - 1},1),  trace {y.trace()}")

    ksub = max(level.K.level_blocks[LEVEL_N].values(), key=len)
    z = level.matrix_unit(ksub[0], ksub[-1], which="K")
    print(f"  z = E[{ksub[0]} , {ksub[-1]}]  in A_({LEVEL_N},0) pulled back by W,"
          f"  trace {z.trace()}")

    print("\n  they live in one representation, so they compose:")
    print(f"    tr(x x*)  = {(x * x.H).trace()}")
    print(f"    tr(x y)   = {(x * y).trace()}")
    print(f"    tr(z z*)  = {(z * z.H).trace()}")
    print(f"    [y, z] = 0 ?  {(y * z - z * y).is_zero()}")

    print("\n  conditional expectations and the commuting square condition:")
    e_h = level.expectation(z, LEVEL_N - 1, 1)
    e_k = level.expectation(z, LEVEL_N - 1, 0)
    print(f"    tr(E_(A_({LEVEL_N - 1},1)) z) = {e_h.trace()}   (= tr z)")
    print(f"    E_(A_({LEVEL_N - 1},1)) z == E_(A_({LEVEL_N - 1},0)) z ?  "
          f"{(e_h - e_k).is_zero()}")

    print("\n  Jones projections:")
    for k in range(2, LEVEL_N + 1):
        f = level.jones_H(k)
        print(f"    f_{k}: projection {(f * f - f).is_zero()},  trace {f.trace()}"
              f"   (1 / beta_v^2 = {float(sp.N(1 / square.beta_v ** 2)):.9f})")

    print("\n  relative commutants inside A_({0},1):".format(LEVEL_N))
    for p, q in level.available_algebras():
        basis = level.relative_commutant(p, q)
        print(f"    dim A_({p},{q})' cap A_({LEVEL_N},1) = {level.span_dimension(basis)}")

    # ------------------------------------------- 4. the higher relative commutants
    print("\n" + "=" * 72)
    print("4. higher relative commutants: A_infinity or Asaeda-Haagerup?")
    print("=" * 72)
    print("  dim A_(0,1)' cap A_(n,0) is the number of loops of length 2n at *")
    print("  in the principal graph.  Catalan is the A_infinity answer, and it")
    print("  is what a generic, non flat s gives.  Asaeda-Haagerup is")
    print("  4-supertransitive, so it agrees with Catalan through n = 4 too;")
    print("  the two separate at n = 5, where A_infinity gives exactly 42 and")
    print("  Asaeda-Haagerup strictly more.  Only that row distinguishes them.")
    print("  The walks column counts closed walks of length 2n at A in the")
    print("  initial graph, for scale.  The margin is the largest discarded")
    print("  principal angle cosine against the smallest kept one; the cut is")
    print("  unambiguous as long as those two stay far apart.")
    print()
    header = ("n", "dim H_(n+1)", "dim A01' cap An0", "Catalan(n)",
              "walks at A", "margin", "time")
    print("    {:>3}  {:>11}  {:>16}  {:>10}  {:>10}  {:>17}  {:>8}".format(*header))
    for n in range(1, MAX_N + 1):
        start = time.time()
        lv = Level(square, n, backend=backend)
        dim = lv.span_dimension(lv.relative_commutant_in((0, 1), (n, 0)))
        elapsed = time.time() - start
        flat = closed_walks(EDGES_TOP, "A", 2 * n)
        mark = ""
        if n >= 5:
            mark = "  <- A_infinity" if dim == catalan(n) else "  <- branches"
        lo, hi = getattr(lv, "_last_angle_gap", (0.0, 1.0))
        print(f"    {n:>3}  {len(lv.H.paths):>11}  {dim:>16}"
              f"  {catalan(n):>10}  {flat:>10}  {lo:.4f} / {hi:.6f}"
              f"  {elapsed:6.1f}s{mark}")

    if MAX_N < 5:
        print()
        print("  MAX_N < 5, so nothing above separates A_infinity from Asaeda-")
        print("  Haagerup; raise MAX_N to 5 for the row that does.")

    print("\ndone.")


if __name__ == "__main__":
    main()
