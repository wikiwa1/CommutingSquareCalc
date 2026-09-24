"""The 4-star S(1,1,3,3) commuting square, worked through the same way as ah_demo.py.

Run with ``python s1133_demo.py``.

Reference
---------
    D. Bisch and J. Cáceres, "New hyperfinite subfactors with infinite depth",
    arXiv:2509.00227, https://arxiv.org/abs/2509.00227

Section 5, "A 1-parameter family of commuting squares for 4-stars", constructs
a one-parameter family of bi-unitary connections on the 4-stars S(i,i,j,j).
This file is the case with two arms of length 1 and two of length 3, whose
index is the 3 + sqrt(3) entry of the table there; Theorem 5.4 uses the family
to show that there are irreducible hyperfinite A_infinity subfactors at this
index.  ah_demo.py is the neighbouring case S(1,1,2,2).

The square
----------
The initial graph, with the vertex names of S1122.txt extended along the two
long arms (V00 = black = {a2, A, d2}, V01 = white = {a1, a3, b1, c1, d1, d3}):

        a3 --- a2 --- a1
                      |
                A --- a1, b1, c1, d1
                      |
        d3 --- d2 --- d1

used for all four sides of the square.  Its norm squared is 3 + sqrt(3).

The connection
--------------
Section 5 writes u^(A,A), the only 4 x 4 summand of u, rows and columns
(a1, b1, c1, d1), as

        [  alpha1       alpha2         alpha2         alpha3    ]
        [  alpha2       xi z1          beta z2        alpha2 z3 ]
        [  alpha2      -beta w /z2     xi w /z1      -alpha2 z3 ]
        [ -alpha3       alpha2 w /z3  -alpha2 w /z3   alpha1    ]

with w = e^{is} the parameter, z3 = e^{it}, and z1, z2, t the formulas of that
section.  The magnitudes are forced by the Perron-Frobenius weights mu:

        alpha1 = mu(a3) / mu(A)          alpha2 = sqrt(mu(a1) mu(b1)) / mu(A)
        alpha3 = mu(a1) / mu(A)          xi = beta = mu(b1) / mu(A)

alpha2, alpha3 and beta because the summands v^(x1,y1), x != y, are 1 x 1;
xi = beta because the short arms make v^(b1,b1) and v^(c1,c1) 1 x 1 as well;
alpha1 from the 2 x 2 summand v^(a1,a1), whose two diagonal entries must have
the same modulus.  The last row is the orthogonal complement written the way
S1122.txt writes it (first entry -alpha3), and with that same choice this
template reproduces the u^(A,A) of S1122.txt to 1e-16.

The paper does not write out the rest, so it is filled in here as forced: on
each long arm x in {a, d} the 1 x 1 summands u^(A,x2) and u^(x2,A) are 1, and
the 2 x 2 summand u^(x2,x2) on (x1, x3) is

        [ -mu(x3)/mu(x2)                  sqrt(mu(x1) mu(x3))/mu(x2) ]
        [  sqrt(mu(x1) mu(x3))/mu(x2)     mu(x3)/mu(x2)              ]

the minus sign being what makes v^(x1,x1) unitary (S1122.txt has the same sign
on v^(a1,a1)).  That is all 28 cells; section 1 checks unitarity and
bi-unitarity, and at every s tried both hold.

What to look for
----------------
The connection is bi-unitary for every s but flat for at most finitely many:
generic s give A_infinity, whose higher relative commutants

        dim A_(0,1)' cap A_(n,0)                   (Ocneanu compactness)

are the Catalan numbers 1, 2, 5, 14, 42, 132, ...  Finite depth subfactors
exist at index 3 + sqrt(3), so a special s would show up as the first n at
which the sequence leaves Catalan -- the depth of the branch point of its
principal graph.  A run that stays on Catalan is reporting A_infinity.
"""

from __future__ import annotations

import time

import sympy as sp

from pathalg import CommutingSquare, InsufficientMemory, Level, gpu_available

# ------------------------------------------------------------------- 0. knobs

# The parameter w = e^{is} of the connection.  Every value gives a bi-unitary
# connection.  Pick one where cos(s) and exp(i s / 2) stay closed form if you
# want to run exactly.
# You get the GHJ subfactor for S = sp.pi / 2 or 3* sp.pi/2
S = sp.pi /2
# Where the tower computes -- see ah_demo.py for what each choice means.
#   "sympy", "numpy", "cupy", "cupy64", or "auto" (cupy64 with a GPU, else numpy)
BACKEND = "auto"

# Level used for sections 2 and 3 (the illustrative one).
LEVEL_N = 2

# Higher relative commutants are computed for n = 1 .. MAX_N.  The library
# picks the dense or the iterative method per level and refuses, before
# allocating, anything that would not fit; the table then says so.
MAX_N = 5

# When a level does not fit on the GPU, retry it on the CPU if it fits in RAM.
CPU_FALLBACK = True


# ------------------------------------------------------------- 1. the graph
#
#         a ---- b          a in V00 (top left)     b in V01 (top right)
#         |      |          c in V10 (bottom left)  d in V11 (bottom right)
#         c ---- d
#
# The cells below are keyed (a, b, c, d) as in S1122.txt, but they hold u, not
# v: u^{a,d}_{c,b}, row c, column b.  The v half is derived by the
# renormalisation rule.

V_BLACK = ["a2", "A", "d2"]
V_WHITE = ["a1", "a3", "b1", "c1", "d1", "d3"]

EDGES_TOP = [("a2", "a1"), ("a2", "a3"),
             ("A", "a1"), ("A", "b1"), ("A", "c1"), ("A", "d1"),
             ("d2", "d1"), ("d2", "d3")]                       # V00 - V01/V10
EDGES_BOTTOM = [(white, black) for black, white in EDGES_TOP]  # V10/V01 - V11


# ------------------------------------------------ 2. the parameters of Section 5
#
#   w  = e^{is}
#   t  = (s - arccos((beta^2 - xi^2 + alpha1^2 cos s) / alpha3^2)) / 2,  z3 = e^{it}
#   z1 = -(alpha1 (1 + w) + alpha3 (z3 - w conj(z3))) / (2 xi)
#   z2 = -(alpha1 (1 - w) + alpha3 (z3 + w conj(z3))) / (2 beta)
#
# Here beta = xi, so the arccos argument is c = (alpha1 / alpha3)^2 cos s, and as
# in ah_demo.py z3 is written in radicals rather than through arccos:
# with phi = arccos(c), phi/2 lies in [0, pi/2], so
#
#   z3 = e^{is/2} e^{-i phi/2} = e^{is/2} ( sqrt((1 + c)/2) - i sqrt((1 - c)/2) ).
#
# For S(1,1,2,2) the same c is cos(s) / delta^2, which is the t of S1122.txt.


def s1133_params(mu_black, mu_white, s):
    """``dict`` of alpha1..3, xi, beta, w, z1, z2, z3 for the parameter ``s``."""
    s = sp.sympify(s)
    mA = mu_black["A"]
    alpha1 = mu_white["a3"] / mA
    alpha2 = sp.sqrt(mu_white["a1"] * mu_white["b1"]) / mA
    alpha3 = mu_white["a1"] / mA
    xi = beta = mu_white["b1"] / mA
    w = sp.exp(sp.I * s)
    c = (alpha1 / alpha3) ** 2 * sp.cos(s)
    z3 = sp.exp(sp.I * s / 2) * (sp.sqrt((1 + c) / 2) - sp.I * sp.sqrt((1 - c) / 2))
    z1 = -(alpha1 * (1 + w) + alpha3 * (z3 - w / z3)) / (2 * xi)
    z2 = -(alpha1 * (1 - w) + alpha3 * (z3 + w / z3)) / (2 * beta)
    return dict(alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, xi=xi, beta=beta,
                w=w, z1=z1, z2=z2, z3=z3)


# ---------------------------------------------------------- 3. the connection


def s1133_cells(mu_black, mu_white, s):
    """The 28 cells of the connection, as ``u^{a,d}_{c,b}`` keyed by (a, b, c, d)."""
    p = s1133_params(mu_black, mu_white, s)
    al1, al2, al3 = p["alpha1"], p["alpha2"], p["alpha3"]
    xi, beta, w, z1, z2, z3 = p["xi"], p["beta"], p["w"], p["z1"], p["z2"], p["z3"]

    # u^(A,A): rows c, columns b, both in the order a1, b1, c1, d1
    u_AA = [
        [al1, al2, al2, al3],
        [al2, xi * z1, beta * z2, al2 * z3],
        [al2, -beta * w / z2, xi * w / z1, -al2 * z3],
        [-al3, al2 * w / z3, -al2 * w / z3, al1],
    ]
    order = ["a1", "b1", "c1", "d1"]
    cells = {}
    for i, c in enumerate(order):
        for j, b in enumerate(order):
            cells[("A", b, c, "A")] = u_AA[i][j]

    # the two long arms, x1 - x2 - x3 for x in {a, d}
    for x in ("a", "d"):
        x1, x2, x3 = x + "1", x + "2", x + "3"
        diag = mu_white[x3] / mu_black[x2]
        off = sp.sqrt(mu_white[x1] * mu_white[x3]) / mu_black[x2]
        cells[("A", x1, x1, x2)] = sp.Integer(1)     # u^(A,x2), 1 x 1
        cells[(x2, x1, x1, "A")] = sp.Integer(1)     # u^(x2,A), 1 x 1
        cells[(x2, x1, x1, x2)] = -diag              # u^(x2,x2), 2 x 2 on (x1, x3)
        cells[(x2, x3, x1, x2)] = off
        cells[(x2, x1, x3, x2)] = off
        cells[(x2, x3, x3, x2)] = diag

    return {k: sp.expand(v) for k, v in cells.items()}


def s1133_square(s=S, **kwargs):
    """The S(1,1,3,3) commuting square at parameter ``s``.

    The cells depend on the Perron-Frobenius weights, which the square computes
    itself, so they are handed over as a callable that is evaluated once the
    weights are known.
    """
    table = {}

    def u_cells(square, a, b, c, d):
        if not table:
            table.update(s1133_cells(square.mu["00"], square.mu["01"], s))
        return table[(a, b, c, d)]

    return CommutingSquare(
        V00=V_BLACK,
        V01=V_WHITE,
        V10=V_WHITE,
        V11=V_BLACK,
        h0=EDGES_TOP,        # V00 - V01
        v0=EDGES_TOP,        # V00 - V10
        h1=EDGES_BOTTOM,     # V10 - V11
        v1=EDGES_BOTTOM,     # V01 - V11
        u_cells=u_cells,
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


def higher_relative_commutant(square, n, backend, block=None):
    """``(level, dim A_(0,1)' cap A_(n,0))``, falling back to the CPU if allowed."""

    def progress(p):
        print(f"    {n:>3}  ... round {p['round']}: {p['found']} found so far, "
              f"block {p['block']}, next cosine {p['next']:.4f}, "
              f"{p['seconds']:.0f}s", end="\r", flush=True)

    def compute(lv):
        try:
            return lv.relative_commutant_dimension((0, 1), (n, 0), block=block,
                                                   progress=progress)
        finally:
            print(" " * 78, end="\r", flush=True)

    lv = Level(square, n, backend=backend)
    try:
        return lv, compute(lv)
    except InsufficientMemory as exc:
        if not (CPU_FALLBACK and lv.backend.device == "GPU"):
            raise
        print(f"    {n:>3}  does not fit on the GPU ({exc.needed / 1e9:.1f} GB needed, "
              f"{exc.available / 1e9:.1f} GB free); retrying on the CPU, "
              "which is much slower")
        del lv
        lv = Level(square, n, backend="numpy")
        return lv, compute(lv)


def main():
    backend = BACKEND
    if backend == "auto":
        backend = "cupy64" if gpu_available() else "numpy"

    print("=" * 72)
    print("1. the S(1,1,3,3) square and its connection")
    print("=" * 72)
    print(f"  parameter s = {S}   (= {float(S):.6f}),  backend {backend}")

    square = s1133_square(S, backend=backend)
    print()
    print(square)
    print("  Perron-Frobenius weights:")
    for tag in ("00", "01", "10", "11"):
        print(f"    V{tag}: {square.mu[tag]}")
    print(f"  beta_h^2 = {sp.nsimplify(sp.N(square.beta_h ** 2), [sp.sqrt(3)])}"
          f"  (= {float(sp.N(square.beta_h ** 2)):.9f}),"
          f"   3 + sqrt(3) = {float(sp.N(3 + sp.sqrt(3))):.9f}")

    params = s1133_params(square.mu["00"], square.mu["01"], S)
    print("\n  the parameters of Section 5:")
    for name, val in params.items():
        v = complex(sp.N(val))
        extra = f"   |{name}| = {abs(v):.9f}" if name in ("w", "z1", "z2", "z3") else ""
        print(f"    {name:>6} = {v:.6f}{extra}")
    print()
    for result in square.verify(verbose=False):
        print(" ", result)

    cells = list(square.cells())
    print(f"\n  all {len(cells)} cells, u as given and v derived from it:")
    for cell in cells:
        print(f"    {str(cell):26}  u = {complex(sp.N(square.u(*cell))):.6f}"
              f"   v = {complex(sp.N(square.v(*cell))):.6f}")

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

    print(f"\n  relative commutants inside A_({LEVEL_N},1):")
    for p, q in level.available_algebras():
        basis = level.relative_commutant(p, q)
        print(f"    dim A_({p},{q})' cap A_({LEVEL_N},1) = {level.span_dimension(basis)}")

    # ------------------------------------------- 4. the higher relative commutants
    print("\n" + "=" * 72)
    print("4. higher relative commutants: A_infinity or a special parameter?")
    print("=" * 72)
    print("  dim A_(0,1)' cap A_(n,0) is the number of loops of length 2n at *")
    print("  in the principal graph.  Catalan is the A_infinity answer, which is")
    print("  what a generic, non flat s gives.  The first n at which a row")
    print("  leaves Catalan would locate the branch point of the principal graph")
    print("  of a special s.  The walks column counts closed walks of length 2n")
    print("  at A in the initial graph, for scale.  The margin is the largest")
    print("  discarded principal angle cosine against the smallest kept one; the")
    print("  cut is unambiguous as long as those two stay far apart.")
    print()
    header = ("n", "dim H_(n+1)", "dim A01' cap An0", "Catalan(n)",
              "walks at A", "margin", "time", "method")
    print("    {:>3}  {:>11}  {:>16}  {:>10}  {:>10}  {:>17}  {:>8}  {}".format(*header))
    previous = None
    for n in range(1, MAX_N + 1):
        block = None if previous is None else int(3.6 * previous) + 16
        start = time.time()
        try:
            lv, dim = higher_relative_commutant(square, n, backend, block)
        except InsufficientMemory as exc:
            print(f"    {n:>3}  skipped: {exc}")
            break  # every larger n needs more still
        elapsed = time.time() - start
        previous = dim
        flat = closed_walks(EDGES_TOP, "A", 2 * n)
        mark = "" if dim == catalan(n) else "  <- leaves Catalan"
        lo, hi = getattr(lv, "_last_angle_gap", (0.0, 1.0))
        info = getattr(lv, "_last_intersection", {})
        how = info.get("method", "-")
        if lv.backend.name != "cupy" and backend.startswith("cupy"):
            how += " (CPU)"
        print(f"    {n:>3}  {len(lv.H.paths):>11}  {dim:>16}"
              f"  {catalan(n):>10}  {flat:>10}  {lo:.4f} / {hi:.6f}"
              f"  {elapsed:7.1f}s  {how}{mark}")
        del lv

    print("\ndone.")


if __name__ == "__main__":
    main()
