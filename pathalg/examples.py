"""Ready made commuting squares, useful as templates and as regression tests.

``dynkin`` builds Ocneanu's connection on a Dynkin diagram; ``hadamard``
builds the spin model commuting square of a complex Hadamard matrix.  Both are
independently documented in the literature, so running :meth:`CommutingSquare.verify`
on them exercises the unitarity and bi-unitarity machinery against known answers.
"""

from __future__ import annotations

import sympy as sp

from .square import CommutingSquare

__all__ = [
    "path_graph_bipartition",
    "dynkin_A",
    "dynkin",
    "fourier_matrix",
    "f4_family",
    "hadamard",
]


def path_graph_bipartition(n):
    """Vertices ``1..n`` of the Dynkin diagram ``A_n``, split by parity."""
    even = [i for i in range(1, n + 1) if i % 2 == 1]
    odd = [i for i in range(1, n + 1) if i % 2 == 0]
    edges = [(i, j) for i in even for j in odd if abs(i - j) == 1]
    return even, odd, edges


def dynkin(even, odd, edges, coxeter, sign=1, **kwargs):
    """Ocneanu's connection on a bipartite graph with Coxeter number ``coxeter``.

    All four graphs of the square are the same graph, so ``V00 = V11 = even``
    and ``V01 = V10 = odd``.  With the cell drawn as in :mod:`pathalg.square`,

        u^{a,d}_{c,b}  =  eps * delta(c, b)
                        + conj(eps) * sqrt( mu(b) mu(c) / ( mu(a) mu(d) ) ) * delta(a, d)

        eps = i * exp(-i pi / (2 N)),   N = coxeter number

    as in Kawahigashi, *On flatness of Ocneanu's connections on the Dynkin
    diagrams*, section 3 (there written with ``i, l, k, j`` for the top-left,
    top-right, bottom-left, bottom-right corners).  Unitarity rests on
    ``eps^2 + conj(eps)^2 + beta = 0`` with ``beta = 2 cos(pi / N)`` the graph
    norm.  ``sign = -1`` swaps ``eps`` and its conjugate, giving the other
    connection.

    Parameters
    ----------
    even, odd : lists
        The two halves of the bipartition.
    edges : list of pairs
        Edges written ``(even vertex, odd vertex)``.
    coxeter : int
        Coxeter number of the diagram (``n + 1`` for ``A_n``, ``12`` for
        ``E_6``, ``18`` for ``E_7``, ``30`` for ``E_8``, ``2n - 2`` for ``D_n``).
    """
    # i*exp(-i pi / 2N) = exp(i pi (N - 1) / 2N).  Keeping it as a single exp
    # atom matters: sympy combines powers of one exponential automatically, so
    # products of connection entries stay small.  Rewriting it with
    # expand_complex turns it into nested radicals that never collapse again.
    N = sp.Integer(coxeter)
    eps = sp.exp(sign * sp.I * sp.pi * sp.Rational(int(N) - 1, 2 * int(N)))
    eps_bar = sp.exp(-sign * sp.I * sp.pi * sp.Rational(int(N) - 1, 2 * int(N)))
    flipped = [(j, i) for (i, j) in edges]

    def rule(sq, a, b, c, d):
        term = eps if c == b else sp.Integer(0)
        if a == d:
            term += eps_bar * sp.sqrt(
                sq.mu["01"][b] * sq.mu["10"][c] / (sq.mu["00"][a] * sq.mu["11"][d])
            )
        # expand only: radsimp rationalises exp(-i pi q) as 1/exp(i pi q) against
        # its minimal polynomial and turns a two term cell into a huge fraction
        return sp.expand(term)

    return CommutingSquare(
        V00=even,
        V01=odd,
        V10=odd,
        V11=even,
        h0=edges,
        v0=edges,
        h1=flipped,
        v1=flipped,
        u_cells=rule,
        **kwargs,
    )


def dynkin_A(n, sign=1, **kwargs):
    """Ocneanu's connection on the Dynkin diagram ``A_n`` (index ``4 cos^2(pi/(n+1))``)."""
    even, odd, edges = path_graph_bipartition(n)
    return dynkin(even, odd, edges, coxeter=n + 1, sign=sign, **kwargs)


def fourier_matrix(N):
    """The ``N x N`` Fourier matrix, a complex Hadamard matrix."""
    w = sp.exp(2 * sp.pi * sp.I / sp.Integer(N))
    return sp.Matrix(N, N, lambda j, k: sp.expand_complex(sp.simplify(w ** (j * k))))


def f4_family(a):
    """The one-parameter family ``F_4^(1)(a)`` of ``4 x 4`` complex Hadamard matrices.

    With ``q = i e^{ia}``,

        [ 1   1   1   1 ]
        [ 1   q  -1  -q ]
        [ 1  -1   1  -1 ]
        [ 1  -q  -1   q ]

    Haagerup showed every ``4 x 4`` complex Hadamard matrix is equivalent to one
    of these.  ``a = 0`` is the Fourier matrix ``F_4`` and ``a = pi/2`` is
    ``F_2 (x) F_2``; since ``q``, ``-q`` and ``conj(q)`` give equivalent
    matrices, ``0 <= a <= pi/2`` already covers everything.  The matrix is
    symmetric, so the two ways of reading the commuting square agree.
    """
    # a single exp atom, for the reason given in dynkin() above
    q = sp.exp(sp.I * (a + sp.pi / 2))
    return sp.Matrix(
        [
            [1, 1, 1, 1],
            [1, q, -1, -q],
            [1, -1, 1, -1],
            [1, -q, -1, q],
        ]
    )


def hadamard(H, **kwargs):
    """The spin model commuting square of a complex Hadamard matrix ``H``.

    ``H`` is ``N x N`` with all entries of modulus 1 and ``H H* = N``.  The
    square is

        C          <  Delta_N
        ^                ^
        u Delta_N u* <  M_N

    so ``V00`` and ``V11`` are single points and ``V01``, ``V10`` have ``N``
    elements each; every graph is a star, ``beta_h = beta_v = sqrt(N)``, and

        u^{*,*}_{c,b} = H[c, b] / sqrt(N)

    Here bi-unitarity reduces to ``|H[c, b]| = 1``: the renormalisation factor
    is ``sqrt(N)``, so each ``v`` block is the single number ``conj(H[c, b])``.
    """
    N = H.rows
    if H.cols != N:
        raise ValueError("H must be square")
    bs = [f"b{j}" for j in range(N)]
    cs = [f"c{i}" for i in range(N)]
    cells = {
        ("*", bs[j], cs[i], "@"): sp.simplify(H[i, j] / sp.sqrt(N))
        for i in range(N)
        for j in range(N)
    }
    return CommutingSquare(
        V00=["*"],
        V01=bs,
        V10=cs,
        V11=["@"],
        h0=[("*", b) for b in bs],
        v0=[("*", c) for c in cs],
        h1=[(c, "@") for c in cs],
        v1=[(b, "@") for b in bs],
        u_cells=cells,
        **kwargs,
    )
