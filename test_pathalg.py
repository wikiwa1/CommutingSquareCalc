"""Test suite for :mod:`pathalg`.

Run with ``python test_pathalg.py`` (no pytest needed), or ``python
test_pathalg.py --slow`` to include the exact-arithmetic cases that take
minutes.

The interesting tests are the ones with an *external* answer:

* ``test_dynkin_biunitarity`` -- Ocneanu's published connection on ``A_n``
  satisfies both unitarity and bi-unitarity as this library computes them,
  which independently pins down the renormalisation convention.
* ``test_hadamard_higher_relative_commutants`` -- the spin model of the Fourier
  matrix ``F_N`` is the ``Z/N`` group-subfactor commuting square, whose higher
  relative commutants have dimensions ``1, N, N^2, ...``.
* ``test_f4_family_relative_commutants`` -- the ``4 x 4`` Hadamard family agrees
  with a count that uses no path algebra, the multiplicity of 1 in ``T_p``.
* ``test_dynkin_higher_relative_commutants`` -- for the flat ``A_n``
  connection, ``dim A_{0,1}' cap A_{n,0}`` is the number of closed walks of
  length ``2n`` at the endpoint of the Dynkin diagram.

The rest are internal consistency: the cheap relative-commutant basis against a
brute force linear solve, ``W`` unitary, the Markov trace, the Jones relations,
the commuting square condition, and generation.
"""

from __future__ import annotations

import sys
import time
import traceback

import sympy as sp

from pathalg import CommutingSquare, Level, Op, pf_bipartite
from pathalg.examples import (
    dynkin_A,
    f4_family,
    fourier_matrix,
    hadamard,
    path_graph_bipartition,
)

SLOW = "--slow" in sys.argv

_FAILURES = []
_RUN = 0


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    global _RUN
    _RUN += 1
    t = time.time()
    try:
        fn()
        print(f"  ok    {fn.__name__:46s} {time.time() - t:6.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        _FAILURES.append((fn.__name__, exc, traceback.format_exc()))
        print(f"  FAIL  {fn.__name__:46s} {time.time() - t:6.1f}s  {exc}", flush=True)


# --------------------------------------------------------------------- weights


def test_pf_weights():
    cases = {
        "A3": (sp.Matrix([[1], [1]]), sp.sqrt(2)),
        "A4": (sp.Matrix([[1, 0], [1, 1]]), (1 + sp.sqrt(5)) / 2),
        "A5": (sp.Matrix([[1, 0], [1, 1], [0, 1]]), sp.sqrt(3)),
        "E6": (sp.Matrix([[1, 0, 0], [1, 1, 1], [0, 1, 0]]), None),
        "E8": (sp.Matrix([[1, 0, 0, 0], [1, 1, 0, 1], [0, 1, 1, 0], [0, 0, 1, 0]]), None),
    }
    for name, (L, expected) in cases.items():
        beta, mu_r, mu_c = pf_bipartite(L)
        if expected is not None:
            check(
                sp.simplify(beta - expected) == 0,
                f"{name}: graph norm {beta} != {expected}",
            )
        r, c = sp.Matrix(mu_r), sp.Matrix(mu_c)
        check(
            sp.simplify(sp.expand(L * c - beta * r)) == sp.zeros(L.rows, 1),
            f"{name}: L mu_col != beta mu_row",
        )
        check(
            sp.simplify(sp.expand(L.T * r - beta * c)) == sp.zeros(L.cols, 1),
            f"{name}: L^T mu_row != beta mu_col",
        )
        check(all(sp.N(x) > 0 for x in mu_r + mu_c), f"{name}: weights not positive")


def test_pf_weights_single_field():
    """Every weight must be written over one number field, or zero tests break."""
    L = sp.Matrix([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 1, 0]])  # D6
    beta, mu_r, mu_c = pf_bipartite(L)
    gens = set()
    for x in [beta] + mu_r + mu_c:
        gens |= sp.sympify(x).atoms(sp.CRootOf)
    check(len(gens) <= 1, f"D6 weights span {len(gens)} different CRootOf generators")


# ------------------------------------------------------------- the connections


def test_dynkin_biunitarity():
    """Ocneanu's A_n connection is unitary *and* bi-unitary as computed here."""
    for n in (3, 4, 5):
        sq = dynkin_A(n)
        for res in (sq.check_inclusion_matrices(), sq.check_weights(),
                    sq.check_unitarity(), sq.check_biunitarity()):
            check(res.ok, f"A_{n}: {res!r}")


def test_dynkin_other_sign():
    """The conjugate connection (sign = -1) is bi-unitary too."""
    sq = dynkin_A(3, sign=-1)
    check(sq.check_unitarity().ok, "A_3 (sign -1): u not unitary")
    check(sq.check_biunitarity().ok, "A_3 (sign -1): v not unitary")


def test_hadamard_biunitarity():
    for N in (2, 3, 4):
        sq = hadamard(fourier_matrix(N))
        check(sq.check_unitarity().ok, f"F_{N}: u not unitary")
        check(sq.check_biunitarity().ok, f"F_{N}: v not unitary")
        check(
            sp.simplify(sq.beta_h - sp.sqrt(N)) == 0
            and sp.simplify(sq.beta_v - sp.sqrt(N)) == 0,
            f"F_{N}: graph norms should both be sqrt({N})",
        )


def test_v_cells_round_trip():
    """Supplying ``v_cells`` must reproduce the same square as supplying ``u_cells``."""
    sq = dynkin_A(3)
    v_cells = {cell: sq.v(*cell) for cell in sq.cells()}
    even, odd, edges = path_graph_bipartition(3)
    other = CommutingSquare(
        V00=even, V01=odd, V10=odd, V11=even,
        h0=edges, v0=edges,
        h1=[(j, i) for (i, j) in edges], v1=[(j, i) for (i, j) in edges],
        v_cells=v_cells,
    )
    for cell in sq.cells():
        check(
            sq.backend.is_zero(sq.u(*cell) - other.u(*cell)),
            f"u disagrees at {cell} when the square is built from v_cells",
        )
    check(other.check_unitarity().ok, "round trip square: u not unitary")


def test_non_cell_rejected():
    sq = dynkin_A(3)
    try:
        CommutingSquare(
            V00=sq.V["00"], V01=sq.V["01"], V10=sq.V["10"], V11=sq.V["11"],
            h0=sq.graphs["h0"].edges, v0=sq.graphs["v0"].edges,
            h1=sq.graphs["h1"].edges, v1=sq.graphs["v1"].edges,
            u_cells={("nope", 2, 2, 1): 1},
        )
    except ValueError:
        return
    raise AssertionError("an invalid cell key was accepted")


# ---------------------------------------------------------------- path spaces


def test_path_space_shapes():
    """H_3 and K_3 must look exactly as specified."""
    lv = Level(hadamard(fourier_matrix(2)), 2)
    check(lv.H.types == ("00", "01", "11", "01"), f"H_3 types are {lv.H.types}")
    check(lv.K.types == ("00", "10", "00", "01"), f"K_3 types are {lv.K.types}")
    lv3 = Level(hadamard(fourier_matrix(2)), 3)
    check(lv3.H.types == ("00", "01", "11", "01", "11"), f"H_4 types {lv3.H.types}")
    check(lv3.K.types == ("00", "10", "00", "10", "11"), f"K_4 types {lv3.K.types}")
    for n in range(6):
        lv = Level(hadamard(fourier_matrix(2)), n)
        check(
            lv.H.types[-1] == lv.K.types[-1],
            f"n={n}: H and K end on different vertex sets",
        )
        check(
            lv.H.block_dims() == lv.K.block_dims(),
            f"n={n}: H and K block dimensions differ",
        )


def test_matrix_units_need_only_common_endpoint():
    """The path algebra is over the Bratteli diagram: sources may differ."""
    lv = Level(dynkin_A(3), 1)
    xi, eta = (1, 2, 1), (3, 2, 1)
    check(xi[0] != eta[0], "test needs two paths with different sources")
    e = Op.matrix_unit(lv.H, xi, eta)
    check(not e.is_zero(), "matrix unit across different sources came out zero")
    check(
        lv.span_dimension(lv.generators(1, 1)) == sum(
            len(v) ** 2 for v in lv.H.blocks.values()
        ),
        "A_(1,1) does not have the dimension of the full path algebra",
    )


# ------------------------------------------------------------- level machinery


def _level_checks(sq, ns, backend=None, label=""):
    for n in ns:
        lv = Level(sq, n, backend=backend)
        results = [
            lv.check_W(), lv.check_trace(), lv.check_expectations(),
            lv.check_commuting_square(), lv.check_generation(),
        ]
        if n >= 2:
            results.append(lv.check_jones())
        for m in range(1, n + 1):
            results.append(lv.check_commuting_square(level=m))
        for res in results:
            check(res.ok, f"{label} n={n}: {res!r}")


def test_levels_hadamard():
    _level_checks(hadamard(fourier_matrix(2)), range(0, 5), label="F2")
    _level_checks(hadamard(fourier_matrix(3)), range(0, 3), label="F3")


def test_levels_dynkin():
    _level_checks(dynkin_A(3), range(0, 3), label="A3")


def test_levels_numpy_backend():
    _level_checks(hadamard(fourier_matrix(4)), (2, 3), backend="numpy", label="F4")
    _level_checks(dynkin_A(4), (2, 3, 4), backend="numpy", label="A4")
    _level_checks(dynkin_A(5), (2, 3), backend="numpy", label="A5")


def test_backends_agree():
    """Exact and numerical backends must produce the same trace and dimensions."""
    sq = dynkin_A(3)
    for n in (2, 3):
        exact, num = Level(sq, n), Level(sq, n, backend="numpy")
        g_e = exact._sample(n, 0, 4)
        g_n = num._sample(n, 0, 4)
        for a, b in zip(g_e, g_n):
            ta = complex(sp.N(a.trace(), 25))
            tb = complex(b.trace())
            check(abs(ta - tb) < 1e-9, f"n={n}: traces differ, {ta} vs {tb}")
        for p, q in exact.available_algebras():
            da = exact.span_dimension(exact.relative_commutant(p, q))
            db = num.span_dimension(num.relative_commutant(p, q))
            check(da == db, f"n={n}: dim A_({p},{q})' differs, {da} vs {db}")


def test_jones_projection_trace():
    sq = dynkin_A(3)
    lv = Level(sq, 4, backend="numpy")
    for k in range(2, 5):
        f = lv.jones_H(k)
        check((f * f - f).is_zero(), f"f_{k} not idempotent")
        check((f - f.H).is_zero(), f"f_{k} not self adjoint")
        got = complex(f.trace())
        want = 1 / complex(sp.N(sq.beta_v ** 2, 25))
        check(abs(got - want) < 1e-9, f"tr(f_{k}) = {got}, expected {want}")


# ------------------------------------------------------- relative commutants


def test_relative_commutant_matches_brute_force():
    for label, sq, ns in [
        ("F2", hadamard(fourier_matrix(2)), (1, 2, 3)),
        ("F3", hadamard(fourier_matrix(3)), (1, 2)),
        ("A3", dynkin_A(3), (1, 2)),
    ]:
        for n in ns:
            lv = Level(sq, n)
            for p, q in lv.available_algebras():
                cheap = lv.relative_commutant(p, q)
                gens = lv.generators(p, q)
                d_cheap = lv.span_dimension(cheap)
                d_brute = lv.span_dimension(lv.commutant_of(gens))
                check(
                    d_cheap == d_brute == len(cheap),
                    f"{label} n={n} A_({p},{q})': cheap basis {len(cheap)} "
                    f"spans {d_cheap}, brute force gives {d_brute}",
                )
                for x in cheap:
                    for g in gens:
                        check(
                            (x * g - g * x).is_zero(),
                            f"{label} n={n} A_({p},{q})': basis element does not commute",
                        )


def test_hadamard_higher_relative_commutants():
    """F_N spin model = the Z/N group subfactor: dims are 1, N, N^2, ..."""
    for N in (2, 3):
        sq = hadamard(fourier_matrix(N))
        for n in range(1, 4 if N == 2 else 3):
            lv = Level(sq, n, backend="numpy")
            got = lv.span_dimension(lv.relative_commutant_in((0, 1), (n, 0)))
            want = N ** (n - 1)
            check(got, "")
            check(got == want, f"F_{N} n={n}: got dim {got}, expected {want}")


def test_f4_family_relative_commutants():
    """F_4^(1)(a) follows D_(m+2)^(1), and T_p counts the same with no path algebra."""
    import numpy as np

    import hadamard4_demo

    levels = [1, 2, 3, 4]
    cases = (
        (sp.Integer(0), [1, 4, 16, 64]),      # F_4
        (sp.pi / 2, [1, 4, 16, 64]),          # F_2 (x) F_2
        (sp.pi / 4, [1, 3, 10, 36]),          # m = 4: the second fork is reached
        (sp.pi / 3, [1, 3, 10, 35]),          # m = 6: not yet
        (sp.Integer(1), [1, 3, 10, 35]),      # D_infinity
    )
    for a, want in cases:
        H = f4_family(a)
        check(H == H.T, f"a={a}: the family is meant to be symmetric")
        sq = hadamard(H, backend="numpy")
        check(all(sq.verify(verbose=False)), f"a={a}: not bi-unitary")
        got = [Level(sq, n).relative_commutant_dimension((0, 1), (n, 0)) for n in levels]
        check(got == want, f"a={a}: got {got}, expected {want}")
        check(hadamard4_demo.predicted_dims(a, levels) == want, f"a={a}: wrong graph")
        direct = hadamard4_demo.fixed_points(
            np.array(H.evalf(30).tolist(), dtype=complex), len(levels)
        )
        check(direct == want, f"a={a}: T_p gives {direct}, expected {want}")

        dual = [Level(sq, n).relative_commutant_dimension((1, 1), (n, 0)) for n in levels[1:]]
        check(dual == want[:-1], f"a={a}: dual side {dual}, expected {want[:-1]}")


def _closed_walks(adj, vertex, length):
    return (adj ** length)[vertex, vertex]


def test_dynkin_higher_relative_commutants():
    """For the flat A_m connection, dim A_(0,1)' cap A_(n,0) counts closed walks.

    Specifically the closed walks of length 2n at the endpoint of A_m, which is
    the known dimension of the n-th higher relative commutant of the A_m
    subfactor.
    """
    for m, ns in ((3, (1, 2, 3, 4)), (4, (1, 2, 3, 4)), (5, (1, 2, 3))):
        adj = sp.Matrix(m, m, lambda i, j: 1 if abs(i - j) == 1 else 0)
        sq = dynkin_A(m)
        for n in ns:
            lv = Level(sq, n, backend="numpy")
            got = lv.span_dimension(lv.relative_commutant_in((0, 1), (n, 0)))
            want = int(_closed_walks(adj, 0, 2 * n))
            check(got == want, f"A_{m} n={n}: got dim {got}, expected {want} walks")


def test_relative_commutant_is_exact_and_in_ambient():
    lv = Level(dynkin_A(3), 2)
    basis = lv.relative_commutant_in((0, 1), (2, 0))
    ambient = lv.generators(2, 0)
    sub = lv.generators(0, 1)
    check(basis, "empty relative commutant")
    for x in basis:
        check(lv.in_span(x, ambient), "element is not in A_(n,0)")
        for g in sub:
            check((x * g - g * x).is_zero(), "element does not commute with A_(0,1)")


# ------------------------------------------------------- expectations & trace


def test_expectation_tower_is_consistent():
    """E onto a smaller algebra factors through the bigger one."""
    lv = Level(dynkin_A(3), 3, backend="numpy")
    for g in lv._sample(3, 1, 6):
        direct = lv.expectation(g, 1, 1)
        staged = lv.expectation(lv.expectation(g, 2, 1), 1, 1)
        check((direct - staged).is_zero(), "E_1 != E_1 . E_2")


def test_trace_of_projections():
    lv = Level(hadamard(fourier_matrix(3)), 2)
    total = sum(
        complex(sp.N(Op.matrix_unit(lv.H, p, p).trace(), 25)) for p in lv.H.paths
    )
    check(abs(total - 1) < 1e-12, f"the minimal projections trace to {total}, not 1")


def test_exact_nullspace_agrees_with_sympy():
    import random

    from pathalg.backend import SympyBackend

    be = SympyBackend()
    random.seed(11)
    for _ in range(8):
        r, c = random.randint(2, 8), random.randint(2, 6)
        A = sp.Matrix(r, c, lambda i, j: sp.Rational(random.randint(-3, 3), random.randint(1, 3)))
        ns = be.nullspace(A)
        check(len(ns) == c - A.rank(), f"nullity {len(ns)} != {c - A.rank()}")
        for v in ns:
            check((A * sp.Matrix(v)).is_zero_matrix, "vector is not in the kernel")
    # algebraic entries, the case that actually matters
    A = sp.Matrix([
        [sp.sqrt(2), 1, sp.exp(sp.I * sp.pi / 4)],
        [2, sp.sqrt(2), sp.sqrt(2) * sp.exp(sp.I * sp.pi / 4)],
    ])
    ns = be.nullspace(A)
    check(len(ns) == 2, f"expected nullity 2, got {len(ns)}")
    for v in ns:
        check(be.mat_is_zero(A * sp.Matrix(v)), "algebraic kernel vector is wrong")


# ------------------------------------------------- the batched / GPU route


def _array_backends():
    """``numpy``, plus the GPU backends when there is a GPU to run them on."""
    from pathalg import gpu_available

    return ["numpy", "cupy", "cupy64"] if gpu_available() else ["numpy"]


def test_intersection_routes_agree():
    """The batched principal-angle route against the original Gram cut-down.

    ``_intersect_by_angles`` is a different algorithm reaching the same
    subspace -- singular values of a k x dim(ambient) matrix rather than the
    kernel of a k x k one -- so agreeing on every case is a real check of both.
    """
    cases = [(dynkin_A(3), (1, 2, 3)), (dynkin_A(4), (1, 2, 3)),
             (hadamard(fourier_matrix(3)), (1, 2))]
    for sq, ns in cases:
        for n in ns:
            lv = Level(sq, n, backend="numpy")
            fast = lv._intersect_by_angles((0, 1), (n, 0))
            slow = lv._cut_down_by_expectation(
                lv.relative_commutant_in((0, 1), (n, 1)), (n, 0)
            )
            a, b = lv.span_dimension(fast), lv.span_dimension(slow)
            check(a == b, f"n={n}: angle route gives {a}, cut-down gives {b}")
            lo, hi = lv._last_angle_gap
            check(hi - lo > 1e-3, f"n={n}: principal angles too close to call")


def test_asaeda_haagerup_sequence():
    """Check the Catalan prefix at s=pi/3, without classifying its depth.

    Asaeda-Haagerup is 5-supertransitive, so levels 1 through 5 do not
    distinguish it from A_infinity. The first distinction is at level 6.
    """
    import ah_demo

    want = [1, 2, 5, 14, 42]
    for name in _array_backends():
        sq = ah_demo.ah_square(sp.pi / 3, backend=name)
        check(sq.verify(verbose=False), f"{name}: AH square failed verification")
        for n in range(1, 6 if SLOW else 5):
            lv = Level(sq, n, backend=name)
            got = lv.span_dimension(lv.relative_commutant_in((0, 1), (n, 0)))
            check(
                got == want[n - 1],
                f"{name} AH n={n}: got dim {got}, expected {want[n - 1]}",
            )


def test_angle_maps_are_adjoint():
    """``<c, F a> = <F* c, a>``: the two halves of the iterative route agree."""
    import numpy as np

    import ah_demo

    lv = Level(ah_demo.ah_square(sp.pi / 3, backend="numpy"), 3, backend="numpy")
    prob = lv._angle_problem((0, 1), (3, 0))
    rng = np.random.default_rng(0)
    a = rng.standard_normal((prob.k, 4)) + 1j * rng.standard_normal((prob.k, 4))
    c = rng.standard_normal((prob.width, 4)) + 1j * rng.standard_normal((prob.width, 4))
    lhs = np.vdot(c, lv._angle_forward(prob, a))
    rhs = np.vdot(lv._angle_backward(prob, c), a)
    check(abs(lhs - rhs) <= 1e-12 * abs(lhs), f"F and F* disagree: {lhs} vs {rhs}")


def test_unit_eigenspace_recovers_known_subspace():
    """A spectrum with a known eigenvalue 1 block, starting from too small a block."""
    import numpy as np

    from pathalg.backend import get_backend
    from pathalg.eigen import unit_eigenspace

    rng = np.random.default_rng(1)
    k, r = 400, 40
    Q, _ = np.linalg.qr(rng.standard_normal((k, k)) + 1j * rng.standard_normal((k, k)))
    N = (Q * np.concatenate([np.ones(r), rng.uniform(0, 0.9, k - r)])) @ Q.conj().T
    be = get_backend("numpy")
    X, info = unit_eigenspace(lambda V: N @ V, k, be, cut=(1 - be.tol) ** 2,
                              block=16, chunk=32)
    check(X.shape[1] == r, f"found {X.shape[1]} directions, expected {r}")
    err = np.linalg.norm(Q[:, :r] @ Q[:, :r].conj().T - X @ X.conj().T, 2)
    check(err < 1e-6, f"wrong subspace, projector error {err:.1e}")
    check(info["block"] > 16, "the block should have had to grow")


def test_iterative_matches_dense():
    """The iterative route gives the dense route's dimension and margin."""
    import ah_demo

    cases = [(dynkin_A(4), (1, 2, 3, 4)), (hadamard(fourier_matrix(3)), (1, 2, 3)),
             (ah_demo.ah_square(sp.pi / 3, backend="numpy"), (3, 4, 5) if SLOW else (3, 4))]
    for sq, ns in cases:
        for n in ns:
            dense = Level(sq, n, backend="numpy")
            iterative = Level(sq, n, backend="numpy")
            a = dense.relative_commutant_dimension((0, 1), (n, 0), method="dense")
            b = iterative.relative_commutant_dimension((0, 1), (n, 0),
                                                       method="iterative", block=32)
            check(a == b, f"n={n}: dense gives {a}, iterative gives {b}")
            check(abs(dense._last_angle_gap[0] - iterative._last_angle_gap[0]) < 1e-4,
                  f"n={n}: margins differ, {dense._last_angle_gap} vs "
                  f"{iterative._last_angle_gap}")
            # and the dimension-only call agrees with building the basis
            basis = iterative.relative_commutant_in((0, 1), (n, 0))
            check(iterative.span_dimension(basis) == b,
                  f"n={n}: the basis spans {iterative.span_dimension(basis)}, not {b}")


def test_memory_refusal():
    """Too little memory is refused up front; enough for iterative only picks it."""
    import ah_demo
    from pathalg import ArrayBackend, InsufficientMemory

    sq = ah_demo.ah_square(sp.pi / 3, backend="numpy")
    sub, amb = (0, 1), (4, 0)

    def level_with(free):
        be = ArrayBackend()
        be.free_memory = lambda: free
        return Level(sq, 4, backend=be)

    probe = level_with(None)
    prob = probe._angle_problem(sub, amb)
    dense = probe._memory_plan(prob, sub, amb, "dense", None)[1]
    small = probe._memory_plan(prob, sub, amb, "iterative", 4)[1]
    grown = probe._iterative_bytes(prob, 20)  # the first growth from a block of 4
    check(small < grown < dense, "the estimates are not ordered as this test assumes")

    try:
        level_with(1e6).relative_commutant_dimension(sub, amb)
        check(False, "1 MB should have been refused")
    except InsufficientMemory as exc:
        check(exc.needed and exc.available == 1e6, "refusal carries no numbers")

    lv = level_with((small + dense) / 2 / 0.85)  # iterative fits, dense does not
    check(lv.relative_commutant_dimension(sub, amb, block=32) == 14, "wrong dimension")
    check(lv._last_intersection["method"] == "iterative", "should have gone iterative")

    # room for the first block but not for growing it: the answer (14) is
    # bigger than a block of 4, so the refusal has to come mid-computation
    try:
        level_with((small + grown) / 2 / 0.85).relative_commutant_dimension(
            sub, amb, method="iterative", block=4)
        check(False, "growing the block should have been refused")
    except InsufficientMemory:
        pass


def test_intersection_same_parameter():
    """s' = s: C_n is all of A_{n,0}, no angles, and ||Lambda||^2 = delta^2."""
    import ah_demo
    from pathalg.intersection import LevelPair, analyse

    sq = ah_demo.ah_square(sp.pi / 3, backend="numpy")
    res = analyse(LevelPair(sq, ah_demo.ah_square(sp.pi / 3, backend="numpy"), 3))
    delta2 = float(sp.N(ah_demo.DELTA ** 2))
    for r in res.records:
        check(r.spectrum.dim_intersection == r.spectrum.dim,
              f"p={r.p}: C_p has dim {r.spectrum.dim_intersection}, not {r.spectrum.dim}")
        check(not r.spectrum.distinct, f"p={r.p}: angles strictly inside (0, 1)")
        check(abs(r.index_estimate - delta2) < 1e-9,
              f"p={r.p}: ||Lambda||^2 = {r.index_estimate}, expected {delta2}")


def test_intersection_hadamard_theorem_B():
    """Bakshi-Guin-Guruprasad, arXiv:2510.14499, Theorem B: index N^2 / |H|.

    ``R_U cap R_V`` for U = F_4 and V = D U is a factor of index 16 / |H| in R,
    |H| = dim(Ad_U(Delta) cap Ad_V(Delta)) = dim C_1.  Here V = D U is the
    library's Hadamard matrix F_4 D (the library's matrix is the paper's U*).
    """
    from pathalg.intersection import LevelPair, analyse

    F = fourier_matrix(4)
    for d, want_h in (([1, 1, 1, -1], 1), ([1, 1, -1, -1], 2), ([1, 1, sp.I, sp.I], 1)):
        pair = LevelPair(hadamard(F, backend="numpy"),
                         hadamard(F * sp.diag(*d), backend="numpy"), 3)
        res = analyse(pair)
        h = res.records[0].spectrum.dim_intersection
        check(h == want_h, f"D={d}: |H| = {h}, expected {want_h}")
        counts = [len(r.spectrum.distinct) for r in res.records]
        check(len(set(counts)) == 1, f"D={d}: the spectrum grows: {counts}")
        for r in res.records:
            check(abs(r.index_estimate - 16 / h) < 1e-9,
                  f"D={d} p={r.p}: ||Lambda||^2 = {r.index_estimate}, expected {16 / h}")
        check(res.records[-1].dobrushin < 1e-9, f"D={d}: not ergodic")


def test_angle_spectra_are_nested():
    """Every eigenvalue of T_p is an eigenvalue of T_(p+1): L^2(A_p1) reduces T."""
    import numpy as np

    import ah_demo
    from pathalg.intersection import LevelPair

    pair = LevelPair(ah_demo.ah_square(sp.pi / 3, backend="numpy"),
                     ah_demo.ah_square(sp.pi / 2, backend="numpy"), 3)
    specs = [pair.angle_spectrum(p) for p in (1, 2, 3)]
    for a, b in zip(specs, specs[1:]):
        big = np.asarray([v for v, _ in b.distinct])
        for v, _ in a.distinct:
            check(np.abs(big - v).min() < 1e-10,
                  f"eigenvalue {v} of level {a.p} missing at level {b.p}")
        check(b.dim_intersection >= a.dim_intersection, "C_p shrank")


def test_gpu_matches_cpu():
    """Same dimensions from the GPU, in both precisions, or skip if no cupy."""
    backends = _array_backends()
    if len(backends) == 1:
        return  # no usable GPU on this machine; nothing to compare
    for sq, ns in ((dynkin_A(4), (1, 2, 3)), (hadamard(fourier_matrix(3)), (1, 2))):
        for n in ns:
            dims = []
            for name in backends:
                lv = Level(sq, n, backend=name)
                dims.append(
                    lv.span_dimension(lv.relative_commutant_in((0, 1), (n, 0)))
                )
            check(
                len(set(dims)) == 1,
                f"n={n}: backends disagree, {dict(zip(backends, dims))}",
            )



# -------------------------------------------------------------- slow / exact


def test_slow_exact_higher_relative_commutants():
    lv = Level(dynkin_A(3), 3)
    got = lv.span_dimension(lv.relative_commutant_in((0, 1), (3, 0)))
    check(got == 4, f"exact A_3 n=3: got {got}, expected 4")


def test_slow_dynkin_A4_variants():
    """The heavier exact cases: A_4 conjugate connection and v_cells round trip."""
    sq = dynkin_A(4, sign=-1)
    check(sq.check_unitarity().ok, "A_4 (sign -1): u not unitary")
    check(sq.check_biunitarity().ok, "A_4 (sign -1): v not unitary")
    base = dynkin_A(4)
    even, odd, edges = path_graph_bipartition(4)
    other = CommutingSquare(
        V00=even, V01=odd, V10=odd, V11=even,
        h0=edges, v0=edges,
        h1=[(j, i) for (i, j) in edges], v1=[(j, i) for (i, j) in edges],
        v_cells={cell: base.v(*cell) for cell in base.cells()},
    )
    for cell in base.cells():
        check(base.backend.is_zero(base.u(*cell) - other.u(*cell)),
              f"A_4: u disagrees at {cell} when built from v_cells")


def test_slow_exact_rank():
    lv = Level(dynkin_A(3), 2, exact_rank=True)
    res = lv.check_generation()
    check(res.ok, f"exact rank generation check: {res!r}")


TESTS = [
    test_pf_weights,
    test_pf_weights_single_field,
    test_dynkin_biunitarity,
    test_dynkin_other_sign,
    test_hadamard_biunitarity,
    test_v_cells_round_trip,
    test_non_cell_rejected,
    test_path_space_shapes,
    test_matrix_units_need_only_common_endpoint,
    test_levels_hadamard,
    test_levels_dynkin,
    test_levels_numpy_backend,
    test_backends_agree,
    test_jones_projection_trace,
    test_relative_commutant_matches_brute_force,
    test_hadamard_higher_relative_commutants,
    test_f4_family_relative_commutants,
    test_dynkin_higher_relative_commutants,
    test_relative_commutant_is_exact_and_in_ambient,
    test_expectation_tower_is_consistent,
    test_trace_of_projections,
    test_exact_nullspace_agrees_with_sympy,
    test_intersection_routes_agree,
    test_asaeda_haagerup_sequence,
    test_angle_maps_are_adjoint,
    test_unit_eigenspace_recovers_known_subspace,
    test_iterative_matches_dense,
    test_memory_refusal,
    test_intersection_same_parameter,
    test_intersection_hadamard_theorem_B,
    test_angle_spectra_are_nested,
    test_gpu_matches_cpu,

    test_slow_exact_higher_relative_commutants,
    test_slow_exact_rank,
]

SLOW_TESTS = [
    test_slow_dynkin_A4_variants,
]


if __name__ == "__main__":
    started = time.time()
    for t in TESTS + (SLOW_TESTS if SLOW else []):
        run(t)
    print(f"\n{_RUN - len(_FAILURES)}/{_RUN} passed in {time.time() - started:.1f}s")
    if not SLOW:
        print("(run with --slow to add the exact-arithmetic cases)")
    if _FAILURES:
        print("\nfailures:")
        for name, _, tb in _FAILURES:
            print(f"\n--- {name} ---\n{tb}")
        sys.exit(1)
