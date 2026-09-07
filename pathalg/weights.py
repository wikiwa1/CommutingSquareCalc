"""Exact Perron-Frobenius data for a connected simple bipartite graph.

For a bipartite graph with (rows x cols) adjacency matrix ``L`` we form the
symmetric matrix ``B = [[0, L], [L^T, 0]]``.  Its largest eigenvalue ``beta`` is
the graph norm ``||L||``, and splitting the corresponding eigenvector in two
gives Perron-Frobenius weights on the two vertex sets already in the *relative*
normalisation the renormalisation rule for a connection needs:

    L @ mu_col = beta * mu_row          L^T @ mu_row = beta * mu_col

Working with ``B`` rather than with ``L L^T`` is what makes the two halves
consistently normalised without a separate rescaling step.

All exact linear algebra happens inside the number field ``QQ(beta)``, so every
weight comes out as a *polynomial in the single algebraic number* ``beta``.
That keeps the whole weight system in one field, which is what makes later
zero-testing of connection identities reliable (mixing a radical form for one
weight with a ``CRootOf`` form for another defeats ``simplify``).
"""

from __future__ import annotations

import sympy as sp

__all__ = ["pf_bipartite", "prettify_algebraic"]

_X = sp.Symbol("_pf_x")


def prettify_algebraic(expr, target=None):
    """Try to rewrite an algebraic number (typically a ``CRootOf``) in radicals.

    Purely cosmetic, and deliberately conservative: it gives up unless the
    minimal polynomial has degree <= 4 *and* the radical form it finds is free
    of complex cube roots (the casus irreducibilis forms sympy returns for
    degree 3 are far less readable than the ``CRootOf`` they replace).
    Returns the input unchanged on any failure.
    """
    expr = sp.sympify(expr)
    if not expr.has(sp.CRootOf):
        return expr
    if target is None:
        try:
            target = complex(sp.N(expr, 30))
        except (TypeError, ValueError):
            return expr
    try:
        mp = sp.Poly(sp.minimal_polynomial(expr, _X), _X)
        if mp.degree() > 4:
            return expr
        best, best_d = None, None
        for root in sp.roots(mp):
            if root.has(sp.CRootOf) or root.has(sp.I):
                continue
            try:
                d = abs(complex(sp.N(root, 30)) - target)
            except (TypeError, ValueError):
                continue
            if best_d is None or d < best_d:
                best, best_d = root, d
        if best is not None and best_d < 1e-15:
            pretty = sp.radsimp(sp.sqrtdenest(sp.expand(best)))
            if not pretty.has(sp.I):
                return pretty
    except Exception:
        pass
    return expr


def _nullvector_in_field(B, beta_root, base):
    """Exact kernel vector of ``B - beta I``, normalised so entry ``base`` is 1.

    Returns a list of sympy expressions, each a polynomial in ``beta_root``.
    """
    n = B.rows
    try:
        from sympy.polys.matrices import DomainMatrix

        K = sp.QQ.algebraic_field(beta_root)
        beta_K = K.from_sympy(beta_root)
        rows = [
            [K.convert(int(B[i, j])) - (beta_K if i == j else K.zero) for j in range(n)]
            for i in range(n)
        ]
        A = DomainMatrix(rows, (n, n), K)
        ns = A.nullspace()
        for i in range(ns.shape[0]):
            row = [ns[i, j].element for j in range(n)]
            if all(x == K.zero for x in row):
                continue
            if row[base] == K.zero:
                continue
            row = [K.quo(x, row[base]) for x in row]
            # exact residual check, still inside the field
            vec = DomainMatrix([[x] for x in row], (n, 1), K)
            if not all(e.element == K.zero for e in (A * vec).to_list_flat()):
                continue
            return [K.to_sympy(x) for x in row]
    except Exception:
        pass

    # Generic fallback (slower, and the expressions are messier).
    A = B - beta_root * sp.eye(n)
    for vec in A.nullspace():
        if vec[base] != 0:
            return [sp.radsimp(sp.cancel(sp.expand(e / vec[base]))) for e in vec]
    raise ValueError(
        "could not find a Perron-Frobenius eigenvector; is the graph connected?"
    )


def pf_bipartite(L, base: int = 0, exact: bool = True):
    """Perron-Frobenius data of a connected simple bipartite graph.

    Parameters
    ----------
    L : sympy.Matrix
        0/1 adjacency matrix, rows indexed by one vertex set, columns by the
        other.
    base : int
        Index of the row vertex whose weight is normalised to 1.
    exact : bool
        When False, return floating point weights (much faster for big graphs).

    Returns
    -------
    (beta, mu_row, mu_col)
        ``beta`` is the graph norm; ``mu_row`` and ``mu_col`` are lists of
        strictly positive weights with ``mu_row[base] == 1``.
    """
    R, C = L.shape
    n = R + C
    B = sp.zeros(n, n)
    B[0:R, R:n] = L
    B[R:n, 0:R] = L.T

    if not exact:
        import numpy as np

        vals, vecs = np.linalg.eigh(np.array(B.tolist(), dtype=float))
        beta = float(vals[-1])
        w = vecs[:, -1]
        if w[base] < 0:
            w = -w
        w = w / w[base]
        return beta, [float(x) for x in w[:R]], [float(x) for x in w[R:]]

    charpoly = sp.Poly(B.charpoly(_X).as_expr(), _X)
    roots = sp.real_roots(charpoly)
    if not roots:
        raise ValueError("adjacency matrix has no real eigenvalues")
    beta_root = roots[-1]

    w = _nullvector_in_field(B, beta_root, base)

    # Cosmetic pass: only if beta itself has a clean radical form, and then
    # applied uniformly so that every weight stays in the same representation.
    beta = prettify_algebraic(beta_root)
    if beta is not beta_root and beta != beta_root:
        w = [sp.radsimp(sp.expand(e.subs(beta_root, beta))) for e in w]

    for e in w:
        val = complex(sp.N(e, 30))
        if val.real <= 0 or abs(val.imag) > 1e-20:
            raise ValueError(
                "Perron-Frobenius eigenvector is not strictly positive; "
                "the graph is probably not connected"
            )
    return beta, w[:R], w[R:]
