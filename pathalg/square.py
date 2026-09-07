"""The commuting square, its four graphs, and the bi-unitary connection.

Geometry and labelling follow Kawahigashi, *On flatness of Ocneanu's
connections on the Dynkin diagrams*, section 1::

        a ---- b            a in V00   (top left)     b in V01   (top right)
        |      |            c in V10   (bottom left)  d in V11   (bottom right)
        c ---- d

for the square of inclusions

        A00  <  A01
         ^       ^
        A10  <  A11

The four simple bipartite graphs are

    h0 : V00 - V01   (A00 < A01)      v0 : V00 - V10   (A00 < A10)
    h1 : V10 - V11   (A10 < A11)      v1 : V01 - V11   (A01 < A11)

The two halves of the connection are, as in the problem statement,

    u^{a,d}_{c,b} :   [a,b,d]  |-->  sum_c u^{a,d}_{c,b} [a,c,d]
    v^{c,b}_{a,d} :   [c,d,b]  |-->  sum_a v^{c,b}_{a,d} [c,a,b]

so ``u`` for fixed corners ``(a,d)`` is a matrix with rows indexed by ``c`` and
columns by ``b``, and ``v`` for fixed corners ``(c,b)`` has rows indexed by
``a`` and columns by ``d``.  Unitarity of every ``u`` block is the *unitarity*
axiom; unitarity of every ``v`` block is *bi-unitarity*.  The two are tied
together by the renormalisation rule, which in this notation reads

    v^{c,b}_{a,d} = sqrt( mu(a) mu(d) / ( mu(b) mu(c) ) ) * conj( u^{a,d}_{c,b} )

with ``mu`` the Perron-Frobenius weights of the four graphs, normalised
consistently (see :mod:`pathalg.weights`).
"""

from __future__ import annotations

import sympy as sp

from .backend import get_backend
from .weights import pf_bipartite

__all__ = ["Graph", "CommutingSquare", "CheckResult"]

# the four vertex-set tags
CORNERS = ("00", "01", "10", "11")


class Graph:
    """A simple bipartite graph between two ordered vertex lists."""

    def __init__(self, name, rows, cols, edges):
        self.name = name
        self.rows = list(rows)
        self.cols = list(cols)
        self.ri = {v: i for i, v in enumerate(self.rows)}
        self.ci = {v: i for i, v in enumerate(self.cols)}
        if len(self.ri) != len(self.rows) or len(self.ci) != len(self.cols):
            raise ValueError(f"graph {name}: duplicate vertex labels")

        self.edges = set()
        for r, c in edges:
            if r not in self.ri:
                raise ValueError(f"graph {name}: {r!r} is not a row vertex")
            if c not in self.ci:
                raise ValueError(f"graph {name}: {c!r} is not a column vertex")
            self.edges.add((r, c))

        self.adj = sp.Matrix(
            len(self.rows),
            len(self.cols),
            lambda i, j: 1 if (self.rows[i], self.cols[j]) in self.edges else 0,
        )
        self.row_nbrs = {r: [c for c in self.cols if (r, c) in self.edges] for r in self.rows}
        self.col_nbrs = {c: [r for r in self.rows if (r, c) in self.edges] for c in self.cols}

    def has(self, r, c):
        return (r, c) in self.edges

    def __repr__(self):
        return f"<Graph {self.name}: {len(self.rows)}x{len(self.cols)}, {len(self.edges)} edges>"


class CheckResult:
    """Outcome of a verification routine: truthy iff everything passed."""

    def __init__(self, name):
        self.name = name
        self.failures = []
        self.notes = []

    def fail(self, msg):
        self.failures.append(msg)

    def note(self, msg):
        self.notes.append(msg)

    @property
    def ok(self):
        return not self.failures

    def __bool__(self):
        return self.ok

    def __repr__(self):
        head = f"{self.name}: {'OK' if self.ok else 'FAILED'}"
        body = "".join("\n    " + m for m in self.failures[:20])
        if len(self.failures) > 20:
            body += f"\n    ... and {len(self.failures) - 20} more"
        for m in self.notes:
            body += "\n    (note) " + m
        return head + body


class CommutingSquare:
    """A commuting square together with a bi-unitary connection.

    Parameters
    ----------
    V00, V01, V10, V11 : sequences
        Ordered lists of minimal central projections (any hashable labels).
        Labels must be distinct *within* each set; they may repeat across sets.
    h0, v0, h1, v1 : sequences of pairs
        Edges of the four bipartite graphs, each written (row vertex, column
        vertex) in the orientation of the module docstring:
        ``h0 = [(a, b), ...]``, ``v0 = [(a, c), ...]``,
        ``h1 = [(c, d), ...]``, ``v1 = [(b, d), ...]``.
    u_cells, v_cells : dict, optional
        Exactly one of the two must be given.  Keys are cells ``(a, b, c, d)``
        in geometric order (top-left, top-right, bottom-left, bottom-right);
        values are the connection entries ``u^{a,d}_{c,b}`` resp.
        ``v^{c,b}_{a,d}``.  Cells that are omitted are taken to be zero.
    backend : 'sympy' | 'numpy' | backend instance
        Arithmetic used by the verification routines and, by default, by any
        :class:`~pathalg.tower.Level` built on this square.
    exact_weights : bool
        Compute Perron-Frobenius weights exactly (default) or numerically.
    """

    def __init__(
        self,
        V00,
        V01,
        V10,
        V11,
        h0,
        v0,
        h1,
        v1,
        u_cells=None,
        v_cells=None,
        backend="sympy",
        exact_weights=True,
    ):
        self.backend = get_backend(backend)
        self.V = {"00": list(V00), "01": list(V01), "10": list(V10), "11": list(V11)}
        for tag, verts in self.V.items():
            if len(set(verts)) != len(verts):
                raise ValueError(f"V{tag} contains duplicate labels")
            if not verts:
                raise ValueError(f"V{tag} is empty")

        self.graphs = {
            "h0": Graph("h0", self.V["00"], self.V["01"], h0),
            "v0": Graph("v0", self.V["00"], self.V["10"], v0),
            "h1": Graph("h1", self.V["10"], self.V["11"], h1),
            "v1": Graph("v1", self.V["01"], self.V["11"], v1),
        }

        self._compute_weights(exact_weights)
        self._install_connection(u_cells, v_cells)

    # ------------------------------------------------------------------ weights
    def _compute_weights(self, exact):
        beta_h, mu00_h, mu01 = pf_bipartite(self.graphs["h0"].adj, exact=exact)
        beta_v, mu00_v, mu10 = pf_bipartite(self.graphs["v0"].adj, exact=exact)

        for a_h, a_v, name in zip(mu00_h, mu00_v, self.V["00"]):
            gap = complex(sp.N(sp.sympify(a_h) - sp.sympify(a_v), 30))
            if abs(gap) > 1e-12:
                raise ValueError(
                    "the horizontal and vertical graphs induce different "
                    f"Perron-Frobenius weights on V00 at vertex {name!r} "
                    f"({a_h} vs {a_v}); this is not a commuting square"
                )

        self.beta_h = sp.sympify(beta_h)
        self.beta_v = sp.sympify(beta_v)
        self.mu = {
            "00": dict(zip(self.V["00"], map(sp.sympify, mu00_h))),
            "01": dict(zip(self.V["01"], map(sp.sympify, mu01))),
            "10": dict(zip(self.V["10"], map(sp.sympify, mu10))),
        }
        # mu on V11 is then forced; check_weights() cross-checks it against h1.
        mu01_vec = sp.Matrix([self.mu["01"][b] for b in self.V["01"]])
        mu11_vec = (self.graphs["v1"].adj.T * mu01_vec) / self.beta_v
        self.mu["11"] = {
            d: sp.radsimp(sp.expand(mu11_vec[i])) for i, d in enumerate(self.V["11"])
        }

    def weight(self, tag, vertex):
        """Perron-Frobenius weight of ``vertex`` inside the set ``V<tag>``."""
        return self.mu[tag][vertex]

    def beta(self, graph_name):
        """Graph norm: ``h0``/``h1`` share ``beta_h``, ``v0``/``v1`` share ``beta_v``."""
        return self.beta_h if graph_name in ("h0", "h1") else self.beta_v

    # --------------------------------------------------------------------- cells
    def is_cell(self, a, b, c, d):
        g = self.graphs
        return (
            g["h0"].has(a, b)
            and g["v0"].has(a, c)
            and g["v1"].has(b, d)
            and g["h1"].has(c, d)
        )

    def cells(self):
        """Iterate over every valid cell ``(a, b, c, d)``."""
        g = self.graphs
        for a in self.V["00"]:
            for b in g["h0"].row_nbrs[a]:
                for c in g["v0"].row_nbrs[a]:
                    for d in g["v1"].row_nbrs[b]:
                        if g["h1"].has(c, d):
                            yield (a, b, c, d)

    def ratio(self, a, b, c, d):
        """``sqrt( mu(a) mu(d) / ( mu(b) mu(c) ) )``, the renormalisation factor."""
        key = (a, b, c, d)
        r = self._ratio_cache.get(key)
        if r is None:
            be = self.backend
            r = be.simp(
                be.sqrt(
                    self.mu["00"][a]
                    * self.mu["11"][d]
                    / (self.mu["01"][b] * self.mu["10"][c])
                )
            )
            self._ratio_cache[key] = r
        return r

    def _install_connection(self, u_cells, v_cells):
        if (u_cells is None) == (v_cells is None):
            raise ValueError("supply exactly one of u_cells= or v_cells=")
        self._ratio_cache = {}
        be = self.backend

        given = u_cells if u_cells is not None else v_cells
        which = "u" if u_cells is not None else "v"
        if callable(given):
            # a rule f(square, a, b, c, d) -> value, evaluated once the weights
            # are known (formulas for a connection almost always need them)
            given = {cell: given(self, *cell) for cell in self.cells()}
        else:
            for key in given:
                if len(key) != 4 or not self.is_cell(*key):
                    raise ValueError(f"{key!r} is not a valid cell of this square")

        self._u = {}
        self._v = {}
        for cell in self.cells():
            a, b, c, d = cell
            val = be.num(given.get(cell, 0))
            r = self.ratio(a, b, c, d)
            if which == "u":
                self._u[cell] = val
                self._v[cell] = be.simp(r * be.conj(val))
            else:
                self._v[cell] = val
                # r is real and positive, so it is its own conjugate
                self._u[cell] = be.simp(be.conj(val) / r)

    def u(self, a, b, c, d):
        """``u^{a,d}_{c,b}``, addressed by the cell in geometric order."""
        return self._u.get((a, b, c, d), self.backend.num(0))

    def v(self, a, b, c, d):
        """``v^{c,b}_{a,d}``, addressed by the cell in geometric order."""
        return self._v.get((a, b, c, d), self.backend.num(0))

    # ------------------------------------------------------- connection matrices
    def u_indices(self, a, d):
        """``(c_list, b_list)`` for the ``u`` block with corners ``(a, d)``."""
        g = self.graphs
        cs = [c for c in g["v0"].row_nbrs[a] if g["h1"].has(c, d)]
        bs = [b for b in g["h0"].row_nbrs[a] if g["v1"].has(b, d)]
        return cs, bs

    def v_indices(self, c, b):
        """``(a_list, d_list)`` for the ``v`` block with corners ``(c, b)``."""
        g = self.graphs
        as_ = [a for a in g["v0"].col_nbrs[c] if g["h0"].has(a, b)]
        ds = [d for d in g["h1"].row_nbrs[c] if g["v1"].has(b, d)]
        return as_, ds

    def u_matrix(self, a, d):
        """Rows indexed by ``c``, columns by ``b``: ``U[c, b] = u^{a,d}_{c,b}``."""
        cs, bs = self.u_indices(a, d)
        M = self.backend.zeros(len(cs), len(bs))
        for i, c in enumerate(cs):
            for j, b in enumerate(bs):
                M[i, j] = self.u(a, b, c, d)
        return cs, bs, M

    def v_matrix(self, c, b):
        """Rows indexed by ``a``, columns by ``d``: ``V[a, d] = v^{c,b}_{a,d}``."""
        as_, ds = self.v_indices(c, b)
        M = self.backend.zeros(len(as_), len(ds))
        for i, a in enumerate(as_):
            for j, d in enumerate(ds):
                M[i, j] = self.v(a, b, c, d)
        return as_, ds, M

    # ----------------------------------------------------------------- verifying
    def check_weights(self):
        """The eight Perron-Frobenius relations that pin down the weight system."""
        res = CheckResult("Perron-Frobenius weights")
        be = self.backend
        for gname, beta, rtag, ctag in (
            ("h0", self.beta_h, "00", "01"),
            ("v0", self.beta_v, "00", "10"),
            ("h1", self.beta_h, "10", "11"),
            ("v1", self.beta_v, "01", "11"),
        ):
            g = self.graphs[gname]
            for r in g.rows:
                lhs = sp.Add(*[self.mu[ctag][c] for c in g.row_nbrs[r]])
                if not be.is_zero(sp.expand(lhs - beta * self.mu[rtag][r])):
                    res.fail(f"{gname}: sum of mu over neighbours of {r!r} != beta*mu({r!r})")
            for c in g.cols:
                lhs = sp.Add(*[self.mu[rtag][r] for r in g.col_nbrs[c]])
                if not be.is_zero(sp.expand(lhs - beta * self.mu[ctag][c])):
                    res.fail(f"{gname}: sum of mu over neighbours of {c!r} != beta*mu({c!r})")
        res.note(f"beta_h = {self.beta_h}, beta_v = {self.beta_v}")
        return res

    def _check_unitary_family(self, name, blocks):
        res = CheckResult(name)
        be = self.backend
        for label, rows, cols, M in blocks:
            if not rows and not cols:
                continue
            if len(rows) != len(cols):
                res.fail(f"{label}: block is {len(rows)}x{len(cols)}, not square")
                continue
            n = len(rows)
            for D, tag in (
                (be.mul(be.dagger(M), M), "M* M"),
                (be.mul(M, be.dagger(M)), "M M*"),
            ):
                if not be.mat_is_zero(be.sub(D, be.eye(n))):
                    res.fail(f"{label}: {tag} != 1")
        return res

    def check_unitarity(self):
        """Every ``u`` block, for fixed corners ``(a, d)``, is unitary."""
        blocks = []
        for a in self.V["00"]:
            for d in self.V["11"]:
                cs, bs, M = self.u_matrix(a, d)
                blocks.append((f"u(a={a!r}, d={d!r})", cs, bs, M))
        return self._check_unitary_family("unitarity of u", blocks)

    def check_biunitarity(self):
        """Every ``v`` block, for fixed corners ``(c, b)``, is unitary."""
        blocks = []
        for c in self.V["10"]:
            for b in self.V["01"]:
                as_, ds, M = self.v_matrix(c, b)
                blocks.append((f"v(c={c!r}, b={b!r})", as_, ds, M))
        return self._check_unitary_family("bi-unitarity (unitarity of v)", blocks)

    def check_inclusion_matrices(self):
        """``Lam_h0 Lam_v1 == Lam_v0 Lam_h1``.

        Both sides are the inclusion matrix of ``A00 < A11``, so they have to
        agree for the square of graphs to be consistent at all.
        """
        res = CheckResult("inclusion matrices for A00 < A11 agree")
        lhs = self.graphs["h0"].adj * self.graphs["v1"].adj
        rhs = self.graphs["v0"].adj * self.graphs["h1"].adj
        if lhs != rhs:
            res.fail(f"Lam_h0 Lam_v1 =\n{lhs}\nbut Lam_v0 Lam_h1 =\n{rhs}")
        return res

    def verify(self, verbose=True):
        """Run every square-level check and return the list of results."""
        results = [
            self.check_inclusion_matrices(),
            self.check_weights(),
            self.check_unitarity(),
            self.check_biunitarity(),
        ]
        if verbose:
            for r in results:
                print(r)
        return results

    def __repr__(self):
        sizes = ", ".join(f"|V{t}|={len(self.V[t])}" for t in CORNERS)
        return f"<CommutingSquare {sizes}, beta_h={self.beta_h}, beta_v={self.beta_v}>"
