"""One level of the tower: A_{n,1}, A_{n-1,1} and A_{n,0} on H_{n+1}.

Path spaces
-----------
``H_{n+1}`` runs ``V00`` then alternates ``V01, V11, V01, ...`` -- it is the
Bratteli path space of

    A00 < A01 < A11 < A21 < ... < A_{n,1}

so a path with ``k`` edges labels a minimal projection route into the ``k``-th
algebra of that tower, and ``A_{n,1}`` is the algebra of ``H_{n+1}``.

``K_{n+1}`` runs ``V00`` and alternates with ``V10`` for ``n`` edges, then takes
one last horizontal edge into ``V_{n,1}``.  It is the path space of

    A00 < A10 < A20 < ... < A_{n,0} < A_{n,1}

Both end on ``V_{n,1}`` (``V01`` for even ``n``, ``V11`` for odd ``n``), and the
number of paths from ``a`` to ``y`` is in both cases the ``(a, y)`` entry of the
inclusion matrix of ``A00 < A_{n,1}``, so the two spaces match block by block.

The connection unitary
----------------------
A path in ``H_{n+1}`` carries a single *horizontal* edge, at the front.  ``W``
pushes it to the back, one step at a time, through the ``n`` vertical edges.
Writing ``S_k`` for the intermediate space in which the horizontal edge sits at
position ``k+1``, so that ``S_0 = H_{n+1}`` and ``S_n = K_{n+1}``, the move
``S_k -> S_{k+1}`` uses

  * for even ``k``, the ``u`` half:  ``[a, b, d] |-> sum_c u^{a,d}_{c,b} [a, c, d]``
  * for odd ``k``,  the ``v`` half:  ``[c, d, b] |-> sum_a v^{c,b}_{a,d} [c, a, b]``

acting on positions ``(k, k+1, k+2)`` and leaving every other vertex alone.
Each move is unitary because each connection block is, so ``W`` is unitary; and
each move fixes the first and last vertex, so ``W`` is block diagonal.

Markov trace
------------
``tr(E_{xi,eta}) = delta_{xi,eta} mu(r(xi)) / (Z0 * P)`` where ``Z0`` is the sum
of the weights on ``V00`` and ``P`` is the product of the graph norms along the
path space (``beta_h * beta_v^n`` for both ``H_{n+1}`` and ``K_{n+1}``).  This
is the Markov trace of the tower and it is the same on the two spaces, so ``W``
is trace preserving.
"""

from __future__ import annotations

import itertools

import sympy as sp

from .backend import get_backend
from .paths import BlockMap, Op, PathSpace
from .square import CheckResult

__all__ = ["Level"]


class _SpanTracker:
    """Incremental linear-independence test over coordinate vectors.

    Only ever asked rank questions -- "does this operator add a dimension?" --
    which are numerically well conditioned here, so by default it works in
    floating point (modified Gram-Schmidt with one reorthogonalisation pass)
    even under the sympy backend.  Exact Gaussian elimination on cyclotomic
    entries swells the intermediate expressions badly enough that a 27
    dimensional span takes longer than every other computation in the library
    put together.  Pass ``exact=True`` to force it anyway; identities
    (products, adjoints, expectations, commutators) are always exact
    regardless, this only governs dimension counting.
    """

    def __init__(self, backend, exact=False, tol=1e-9):
        self.be = backend
        self.exact = exact
        self.tol = tol
        self.rows = []  # exact: (pivot index, reduced row).  numeric: unit vectors.
        if not exact:
            import numpy as np

            self.np = np

    @property
    def dim(self):
        return len(self.rows)

    def add(self, vec):
        """Reduce ``vec``; keep it and return True if it was independent."""
        return self._add_exact(vec) if self.exact else self._add_numeric(vec)

    def _add_numeric(self, vec):
        np = self.np
        be = self.be
        if hasattr(vec, "ndim"):  # already an array: convert in one go
            v = np.asarray(be.to_host(vec), dtype=complex)
        else:
            v = np.array([complex(be.to_complex(x)) for x in vec], dtype=complex)
        norm0 = np.linalg.norm(v)
        if norm0 <= self.tol:
            return False
        for _ in range(2):  # one reorthogonalisation pass
            for r in self.rows:
                v -= (r.conj() @ v) * r
        norm = np.linalg.norm(v)
        if norm <= self.tol * norm0:
            return False
        self.rows.append(v / norm)
        return True

    def _add_exact(self, vec):
        be = self.be
        v = [be.num(x) for x in vec]
        for piv, r in self.rows:
            if be.is_zero(v[piv]):
                continue
            f = v[piv]
            v = [be.simp(a - f * b) for a, b in zip(v, r)]
        for i, x in enumerate(v):
            if not be.is_zero(x):
                v = [be.simp(a / x) for a in v]
                self.rows.append((i, v))
                return True
        return False


def _H_types(n):
    """Vertex-set tags for H_{n+1}: V00, then alternating V01, V11, ..."""
    types = ["00"]
    for i in range(1, n + 2):
        types.append("01" if i % 2 == 1 else "11")
    return tuple(types)


def _K_types(n):
    """Vertex-set tags for K_{n+1}: V00/V10 alternating, then one horizontal edge."""
    types = ["00" if i % 2 == 0 else "10" for i in range(n + 1)]
    types.append("01" if n % 2 == 0 else "11")
    return tuple(types)


def _stage_types(n, k):
    """Tags for the intermediate space S_k, with the horizontal edge at position k+1."""
    if not 0 <= k <= n:
        raise ValueError(f"stage index {k} out of range 0..{n}")
    types = ["00" if i % 2 == 0 else "10" for i in range(k + 1)]
    head = "01" if k % 2 == 0 else "11"
    types.append(head)
    for i in range(k + 2, n + 2):
        offset = i - (k + 1)
        types.append(head if offset % 2 == 0 else ("11" if head == "01" else "01"))
    return tuple(types)


class Level:
    """Everything at level ``n``, represented on ``H_{n+1}``.

    Parameters
    ----------
    square : CommutingSquare
    n : int
        The level; ``n >= 0``.  ``A_{n,1}`` is the full algebra of ``H_{n+1}``.
    backend : optional
        Overrides the backend of the square (use ``'numpy'`` for speed).
    exact_rank : bool
        Whether dimension counting (:meth:`span_dimension`, :meth:`in_span`,
        :meth:`generated_algebra`, :meth:`check_generation`) uses exact
        arithmetic.  Off by default: those are rank questions, well conditioned
        here, and exact elimination on cyclotomic entries is very slow.  Every
        operator identity is exact either way.
    """

    def __init__(self, square, n, backend=None, exact_rank=False):
        if n < 0:
            raise ValueError("n must be non-negative")
        self.square = square
        self.n = n
        self.exact_rank = exact_rank
        self.backend = get_backend(backend if backend is not None else square.backend)

        # stage 0 is H and stage n is K by construction; take them from the same
        # list so that the `space is self.K` tests elsewhere stay meaningful
        # (at n = 0 the two spaces genuinely coincide and W is the identity)
        self.stages = [PathSpace(square, _stage_types(n, k)) for k in range(n + 1)]
        self.H = self.stages[0]
        self.K = self.stages[n]
        assert self.H.types == _H_types(n) and self.K.types == _K_types(n)

        self._check_block_shapes()
        self.moves = [self._move(k) for k in range(n)]
        self.W = self._build_W()
        self._commutant_cache = {}
        self._exp_plans = {}

    # ------------------------------------------------------------------ set-up
    def _check_block_shapes(self):
        hd, kd = self.H.block_dims(), self.K.block_dims()
        if hd != kd:
            raise ValueError(
                "H_{n+1} and K_{n+1} have different block dimensions:\n"
                f"  H: {hd}\n  K: {kd}\n"
                "The two towers do not end at the same algebra; check the four graphs."
            )

    def _move(self, k):
        """The elementary connection move ``S_k -> S_{k+1}``."""
        src, dst = self.stages[k], self.stages[k + 1]
        sq, be = self.square, self.backend
        # the entries go in one at a time, so on a device backend they are
        # staged on the host first: a per-element write to GPU memory is a
        # kernel launch, and there are tens of thousands of them here
        stage = getattr(be, "array", False) and be.name != "numpy"
        blocks = {}
        for key in src.keys:
            rows, cols = dst.blocks[key], src.blocks[key]
            if stage:
                import numpy as _np

                M = _np.zeros((len(rows), len(cols)), dtype=be.dtype)
            else:
                M = be.zeros(len(rows), len(cols))
            rowpos = {p: i for i, p in enumerate(rows)}
            for j, xi in enumerate(cols):
                left, right = xi[k], xi[k + 2]
                for mid in sq.V[dst.types[k + 1]]:
                    eta = xi[: k + 1] + (mid,) + xi[k + 2 :]
                    i = rowpos.get(eta)
                    if i is None:
                        continue
                    if k % 2 == 0:
                        # left = a in V00, xi[k+1] = b in V01, right = d in V11,
                        # mid = c in V10  ->  u^{a,d}_{c,b}
                        val = sq.u(left, xi[k + 1], mid, right)
                    else:
                        # left = c in V10, xi[k+1] = d in V11, right = b in V01,
                        # mid = a in V00  ->  v^{c,b}_{a,d}
                        val = sq.v(mid, right, left, xi[k + 1])
                    M[i, j] = be.num(val)
            blocks[key] = be.asarray(M) if stage else M
        return BlockMap(src, dst, blocks, be)

    def _build_W(self):
        W = BlockMap.identity(self.H, self.backend)
        for move in self.moves:
            W = move.compose(W)
        return W

    # -------------------------------------------------------------- the algebras
    def _space_and_level(self, p, q):
        """Translate ``A_{p,q}`` into (path space, number of edges)."""
        n = self.n
        if q == 1:
            if not 0 <= p <= n:
                raise ValueError(f"A_({p},1) is not represented at level n={n}")
            return self.H, p + 1
        if q == 0:
            if not 0 <= p <= n:
                raise ValueError(f"A_({p},0) is not represented at level n={n}")
            return self.K, p
        raise ValueError("the second index must be 0 or 1")

    def generators(self, p, q):
        """Matrix units spanning ``A_{p,q}``, as operators on ``H_{n+1}``.

        ``A_{p,1}`` is read off ``H_{n+1}`` directly; ``A_{p,0}`` is read off
        ``K_{n+1}`` and pulled back through ``W``.  ``A_{0,0}`` is the same
        either way.
        """
        space, m = self._space_and_level(p, q)
        gens = []
        for key, plist in space.level_blocks[m].items():
            del key
            for xi, eta in itertools.product(plist, repeat=2):
                gens.append(Op.matrix_unit(space, xi, eta, self.backend))
        if space is self.K:
            gens = [self.W.pull_back(g) for g in gens]
        return gens

    def generating_set(self, p, q):
        """A small multiplicative generating set for ``A_{p,q}``.

        Fixing a reference path ``xi0`` in each block, the units
        ``E_{xi0, xi}`` and ``E_{xi, xi0}`` generate that block, so this has
        ``sum_w (2 d_w - 1)`` elements instead of ``sum_w d_w^2``.  Commuting
        with a generating set is the same as commuting with the algebra, so
        commutant computations should use this rather than
        :meth:`generators`.  It does *not* span, so anything that needs a
        basis (a span or an expectation test) must use :meth:`generators`.
        """
        space, m = self._space_and_level(p, q)
        gens = []
        for plist in space.level_blocks[m].values():
            ref = plist[0]
            for xi in plist:
                gens.append(Op.matrix_unit(space, ref, xi, self.backend))
                if xi != ref:
                    gens.append(Op.matrix_unit(space, xi, ref, self.backend))
        if space is self.K:
            gens = [self.W.pull_back(g) for g in gens]
        return gens

    def matrix_unit(self, xi, eta, which="H"):
        """A single matrix unit, as an operator on ``H_{n+1}``.

        ``which`` selects the picture the paths are written in: ``'H'`` for the
        ``A_{p,1}`` tower, ``'K'`` for the ``A_{p,0}`` tower.
        """
        space = self.H if which == "H" else self.K
        op = Op.matrix_unit(space, tuple(xi), tuple(eta), self.backend)
        return self.W.pull_back(op) if space is self.K else op

    def one(self):
        return Op.identity(self.H, self.backend)

    def to_K(self, op):
        """Move an operator from the ``H`` picture to the ``K`` picture."""
        return self.W.conjugate_op(op)

    def to_H(self, op):
        """Move an operator from the ``K`` picture to the ``H`` picture."""
        return self.W.pull_back(op)

    # ------------------------------------------------------------------- trace
    def trace(self, op):
        """Markov trace, normalised to 1 on the identity."""
        return op.trace()

    # ------------------------------------------------- conditional expectations
    def _expectation_on(self, space, op, m):
        """Trace preserving expectation onto the level-``m`` subalgebra of ``space``."""
        be = self.backend
        L = space.L
        if not 0 <= m <= L:
            raise ValueError(f"level {m} out of range 0..{L}")
        Q = be.num(sp.expand(space.P[L] / space.P[m]))
        mu_end = space.square.mu[space.types[L]]
        mu_lvl = space.square.mu[space.types[m]]
        numpy_path = getattr(be, "array", False)
        blocks = {}
        for w, prefixes, per_end in self._expectation_plan(space, m):
            P = len(prefixes)
            denom = be.num(mu_lvl[w]) * Q
            if numpy_path:
                np = be.np
                acc = np.zeros((P, P), dtype=complex)
                for y, idx in per_end:
                    B = op.blocks.get(y)
                    if B is None:
                        continue
                    acc += complex(be.num(mu_end[y])) * B[
                        idx[:, None, :], idx[None, :, :]
                    ].sum(axis=2)
                coeff = acc / complex(denom)
                for y, idx in per_end:
                    M = blocks.get(y)
                    if M is None:
                        M = be.zeros(len(space.blocks[y]), len(space.blocks[y]))
                        blocks[y] = M
                    M[idx[:, None, :], idx[None, :, :]] = coeff[:, :, None]
                continue

            acc = [[be.num(0)] * P for _ in range(P)]
            for y, idx in per_end:
                B = op.blocks.get(y)
                if B is None:
                    continue
                wy = be.num(mu_end[y])
                for i in range(P):
                    for j in range(P):
                        s = be.num(0)
                        for a, b in zip(idx[i], idx[j]):
                            s = s + B[a, b]
                        acc[i][j] = acc[i][j] + wy * s
            for y, idx in per_end:
                M = blocks.get(y)
                if M is None:
                    M = be.zeros(len(space.blocks[y]), len(space.blocks[y]))
                    blocks[y] = M
                for i in range(P):
                    for j in range(P):
                        if be.is_zero(acc[i][j]):
                            continue
                        c = be.simp(acc[i][j] / denom)
                        for a, b in zip(idx[i], idx[j]):
                            M[a, b] = c
        return Op(space, {k: be.simp_mat(v) for k, v in blocks.items()}, be)

    def _expectation_plan(self, space, m):
        """Index tables for the level-``m`` expectation, built once per level.

        For each vertex ``w`` at level ``m`` this records the prefixes ending
        there and, for each end vertex ``y``, the matrix of positions of
        ``prefix + tail`` inside the block of ``y``.  The expectation is then a
        weighted partial trace over the tail index, which those tables turn
        into array indexing instead of a dictionary lookup per entry.
        """
        key = (id(space), m)
        plan = self._exp_plans.get(key)
        if plan is not None:
            return plan
        plan = []
        for w, prefixes in space.level_blocks[m].items():
            by_end = {}
            for tail in space.tails(m, w):
                by_end.setdefault(tail[-1], []).append(tail)
            per_end = []
            for y, tails in by_end.items():
                idx = [[space.index[p + t[1:]][1] for t in tails] for p in prefixes]
                if getattr(self.backend, "array", False):
                    idx = self.backend.xp.asarray(idx, dtype=int)
                per_end.append((y, idx))
            plan.append((w, prefixes, per_end))
        self._exp_plans[key] = plan
        return plan

    def expectation(self, op, p, q):
        """Trace preserving conditional expectation of ``op`` onto ``A_{p,q}``."""
        space, m = self._space_and_level(p, q)
        if space is self.K:
            return self.W.pull_back(self._expectation_on(space, self.to_K(op), m))
        return self._expectation_on(space, op, m)

    # ------------------------------------------------------- Jones projections
    def jones_H(self, k):
        """Jones projection ``f_k`` for the ``H`` tower, living in the level-``k+1`` algebra.

        ``f_k`` implements the conditional expectation of the level-``k``
        algebra onto the level-``k-1`` algebra.  Valid for ``2 <= k <= n``: the
        step from level 1 to level 2 is ``A01 < A11``, the first inclusion of
        the tower that is a basic construction, so ``f_k`` exists from ``k = 2``
        on.  ``tr(f_k) = 1 / beta_v**2``.
        """
        if not 2 <= k <= self.n:
            raise ValueError(
                f"jones_H(k) needs 2 <= k <= n = {self.n}; "
                "the H tower only becomes a Jones tower from level 1 on"
            )
        return self._jones_on(self.H, k)

    def jones_K(self, k):
        """Jones projection ``f_k`` for the ``K`` tower, as an operator on ``H_{n+1}``.

        Valid for ``2 <= k <= n - 1`` (the last edge of ``K_{n+1}`` is
        horizontal and so does not continue the vertical Jones tower).
        """
        if not 2 <= k <= self.n - 1:
            raise ValueError(f"jones_K(k) needs 2 <= k <= n-1 = {self.n - 1}")
        return self.W.pull_back(self._jones_on(self.K, k))

    def jones(self, j):
        """The Jones projection ``e_j`` in the ``A`` indexing.

        ``e_j`` implements the expectation of ``A_{j,1}`` onto ``A_{j-1,1}``
        and lies in ``A_{j+1,1}``; it is ``jones_H(j + 1)``.  Valid for
        ``1 <= j <= n - 1``.
        """
        return self.jones_H(j + 1)

    def _jones_on(self, space, k):
        be = self.backend
        if space.types[k + 1] != space.types[k - 1]:
            raise ValueError(
                f"levels {k - 1} and {k + 1} of {'-'.join(space.types)} sit on "
                "different vertex sets, so there is no Jones projection there"
            )
        # sum of mu over the neighbours of w at level k equals beta[k-1]*mu(w)
        beta = space.beta[k - 1]
        mu_mid, mu_base = space.square.mu[space.types[k]], space.square.mu[space.types[k - 1]]
        coeffs = {}
        entries = {}
        for sigma in space.level_paths[k - 1]:
            w = sigma[-1]
            mids = space._succ[k - 1][w]
            for x in mids:
                for y in mids:
                    coeff = coeffs.get((w, x, y))
                    if coeff is None:
                        coeff = be.simp(
                            be.sqrt(mu_mid[x] * mu_mid[y]) / (beta * mu_base[w])
                        )
                        coeffs[(w, x, y)] = coeff
                    for tail in space.tails(k + 1, w):
                        xi = sigma + (x, w) + tail[1:]
                        eta = sigma + (y, w) + tail[1:]
                        entries[(xi, eta)] = coeff
        return Op.from_entries(space, entries, be)

    # ------------------------------------------------------ relative commutants
    def relative_commutant(self, p, q):
        """A basis of ``A_{p,q}' \\cap A_{n,1}``, as operators on ``H_{n+1}``.

        Uses the standard description of the commutant of a level algebra
        inside a path algebra.  Splitting the paths ending at ``y`` as
        ``(+)_w (paths to w) (x) (tails w -> y)``, the level-``m`` algebra acts
        as ``(+)_w M(paths to w) (x) 1``, so its commutant is
        ``(+)_w 1 (x) End(tails w -> y)``, with basis

            g = sum over every prefix sigma ending at w of E_{sigma tau, sigma rho}

        one for each vertex ``w`` at level ``m`` and each pair of tails
        ``tau, rho`` from ``w`` to a common end vertex.  The dimension is
        therefore ``sum_{w, y} (# tails from w to y)^2``.  Cross-checked against
        a brute force linear solve in the test suite.
        """
        return self.relative_commutant_in((p, q), (self.n, 1))

    def relative_commutant_in(self, sub, ambient):
        """A basis of ``A_sub' \\cap A_ambient``, both given as ``(p, q)`` pairs.

        ``relative_commutant_in((0, 1), (n, 0))`` is the object in Ocneanu's
        compactness argument -- the higher relative commutant of the subfactor
        the commuting square generates.

        When both algebras are levels of the same path space the cheap basis
        above is used; otherwise (the interesting mixed case, where one lives in
        the ``H`` picture and the other in the ``K`` picture) it falls back to a
        linear solve over the ambient matrix units.
        """
        key = (tuple(sub), tuple(ambient))
        if key in self._commutant_cache:
            return self._commutant_cache[key]

        sub_space, sub_m = self._space_and_level(*sub)
        amb_space, amb_m = self._space_and_level(*ambient)
        if amb_space is not sub_space and tuple(ambient) == (self.n, 1):
            # A_{n,1} is the whole algebra of H_{n+1} *and* of K_{n+1}, so read
            # it off in whichever picture the subalgebra lives in and stay on
            # the cheap path
            amb_space, amb_m = sub_space, sub_space.L
        if sub_space is amb_space and sub_m <= amb_m:
            basis = self._relative_commutant_on(sub_space, sub_m, amb_m)
            if sub_space is self.K:
                basis = [self.W.pull_back(g) for g in basis]
        else:
            # Mixed pictures (one algebra read off H, the other off K).  Rather
            # than solve the commutation relations against a basis of the
            # ambient -- which costs one operator product per (ambient
            # generator, sub generator) pair and is by far the slowest thing in
            # the library -- cut down the commutant inside A_{n,1}, which does
            # have the cheap basis, by
            #
            #     A_sub' cap A_amb = { x in A_sub' cap A_{n,1} : E_amb(x) = x }
            #
            # using that a trace preserving expectation fixes exactly its own
            # range.  That is one expectation per basis element, and the basis
            # is small.
            if getattr(self.backend, "array", False) and sub_space is self.H:
                basis = self._intersect_by_angles(sub, ambient)
            else:
                outer = self.relative_commutant_in(sub, (self.n, 1))
                basis = self._cut_down_by_expectation(outer, ambient)
        self._commutant_cache[key] = basis
        return basis

    def _relative_commutant_on(self, space, m, upto):
        be = self.backend
        basis = []
        for w, sigmas in space.level_blocks[m].items():
            tails = {}
            for tail in space.tails(m, w, upto):
                tails.setdefault(tail[-1], []).append(tail)
            for tlist in tails.values():
                for tau, rho in itertools.product(tlist, repeat=2):
                    op = Op.zero(space, be)
                    for sigma in sigmas:
                        op = op + Op.matrix_unit(
                            space, sigma + tau[1:], sigma + rho[1:], be
                        )
                    basis.append(op)
        return basis

    # ------------------------------------------- the batched intersection route
    #
    # Everything from here to _cut_down_by_expectation is one algorithm, used
    # whenever the blocks are arrays (numpy or cupy).  It computes the same
    # thing as _cut_down_by_expectation and is checked against it in the tests,
    # but it never builds a k x k matrix and never touches a Python scalar in a
    # loop, which is what makes n = 5 and n = 6 reachable.
    #
    # The observation it rests on: the cheap commutant basis
    #
    #     u_i = sum over prefixes sigma of E_{sigma tau, sigma rho}
    #
    # is *orthogonal* for the Markov inner product, with
    # ||u_i||^2 = (#prefixes) mu(y) / (Z0 P).  So after rescaling to unit
    # vectors the problem is a principal angle problem between two subspaces of
    # A_{n,1}: span(u_i), and the ambient A_ambient.  A trace preserving
    # expectation is the orthogonal projection onto the second, so
    #
    #     A_sub' cap A_amb = { x in span(u) : ||E x|| = ||x|| },
    #
    # the singular value 1 subspace of B = (coefficients of E u_i), a
    # k x dim A_amb matrix.  Both dimensions are far smaller than
    # dim A_{n,1}, and B is written one chunk of rows at a time, so the peak
    # memory is set by the chunk rather than by k.

    def _cheap_basis_groups(self, space, m):
        """The cheap commutant basis of the level-``m`` algebra, as index data.

        Returns ``(groups, k, weights)``.  A group is ``(y, idx, start)`` where
        ``idx`` has shape ``(#prefixes, #tails)`` and holds positions inside the
        block of ``y``; element ``t1 * T + t2`` of the group -- basis element
        ``start + t1 * T + t2`` overall -- is
        ``sum_s E_{idx[s, t1], idx[s, t2]}``.  ``weights[i]`` is ``||u_i||^2``.

        The index tables are exactly the ones :meth:`_expectation_plan` already
        builds, and the enumeration order matches
        :meth:`_relative_commutant_on`, so the two bases agree element for
        element.
        """
        be = self.backend
        sq = self.square
        Z0 = sp.Add(*[sq.mu["00"][a] for a in sq.V["00"]])
        norm = Z0 * space.P[space.L]
        mu_end = sq.mu[space.types[space.L]]
        groups, weights, start = [], [], 0
        for w, prefixes, per_end in self._expectation_plan(space, m):
            for y, idx in per_end:
                T = idx.shape[1]
                groups.append((y, idx, start))
                wt = float(sp.N(len(prefixes) * mu_end[y] / norm))
                weights.extend([wt] * (T * T))
                start += T * T
        return groups, start, be.xp.asarray(weights, dtype=be.real_dtype)

    def _stack_chunk(self, space, groups, lo, hi):
        """Blocks of basis elements ``lo:hi``, as ``{y: array (hi-lo, d, d)}``."""
        be, xp = self.backend, self.backend.xp
        stack = {}
        for y, idx, start in groups:
            T = idx.shape[1]
            a, b = max(lo, start), min(hi, start + T * T)
            if a >= b:
                continue
            A = stack.get(y)
            if A is None:
                d = len(space.blocks[y])
                A = be.zeros(hi - lo, d, d)
                stack[y] = A
            loc = xp.arange(a - start, b - start)
            A[xp.arange(a - lo, b - lo)[None, :], idx[:, loc // T], idx[:, loc % T]] = 1
        return stack

    def _congruence_stack(self, bmap, stack):
        """``W X W*`` for a whole stack at once, one batched matmul per block."""
        out = {}
        for key, X in stack.items():
            A = bmap.blocks.get(key)
            if A is not None:
                out[key] = A @ X @ A.conj().T
        return out

    def _expectation_coeffs(self, space, stack, m):
        """Coefficients of ``E_m(x)`` on the matrix units of the level-``m`` algebra.

        Same weighted partial trace as :meth:`_expectation_on`, with a leading
        batch axis and stopping before the result is written back out to a full
        block -- the coefficients are all the inner products need, and there are
        ``dim A_{m}`` of them rather than ``dim A_{n,1}``.
        """
        be, xp = self.backend, self.backend.xp
        L = space.L
        Q = be.num(sp.expand(space.P[L] / space.P[m]))
        mu_end = space.square.mu[space.types[L]]
        mu_lvl = space.square.mu[space.types[m]]
        out = []
        for w, prefixes, per_end in self._expectation_plan(space, m):
            acc = None
            for y, idx in per_end:
                B = stack.get(y)
                if B is None:
                    continue
                gathered = B[:, idx[:, None, :], idx[None, :, :]].sum(axis=3)
                term = xp.asarray(be.num(mu_end[y]), dtype=be.dtype) * gathered
                acc = term if acc is None else acc + term
            if acc is None:
                continue
            out.append((w, acc / xp.asarray(be.num(mu_lvl[w]) * Q, dtype=be.dtype)))
        return out

    def _ambient_unit_weights(self, space, m):
        """``||f||^2`` for a matrix unit ``f`` of the level-``m`` algebra, per ``w``."""
        be = self.backend
        sq = space.square
        Z0 = sp.Add(*[sq.mu["00"][a] for a in sq.V["00"]])
        mu_end = sq.mu[space.types[space.L]]
        out = {}
        for w, prefixes, per_end in self._expectation_plan(space, m):
            total = sp.Integer(0)
            for y, idx in per_end:
                total += idx.shape[1] * mu_end[y]
            out[w] = be.num(total / (Z0 * space.P[space.L]))
        return out

    def _intersect_by_angles(self, sub, ambient, chunk=None):
        """``A_sub' cap A_ambient`` via principal angles.  See the note above."""
        be, xp = self.backend, self.backend.xp
        sub_space, sub_m = self._space_and_level(*sub)
        amb_space, amb_m = self._space_and_level(*ambient)
        groups, k, gram = self._cheap_basis_groups(sub_space, sub_m)
        if k == 0:
            return []

        unit = self._ambient_unit_weights(amb_space, amb_m)
        cols, width = [], 0
        for w, prefixes, _ in self._expectation_plan(amb_space, amb_m):
            cols.append((w, width, len(prefixes) ** 2))
            width += len(prefixes) ** 2

        if chunk is None:
            # keep one chunk of dense blocks under ~256 MB
            per = max(1, sum(len(b) ** 2 for b in sub_space.blocks.values()))
            chunk = max(1, min(k, int(256e6 / (per * xp.dtype(be.dtype).itemsize))))

        B = be.zeros(k, width)
        for lo in range(0, k, chunk):
            hi = min(k, lo + chunk)
            stack = self._stack_chunk(sub_space, groups, lo, hi)
            if amb_space is not sub_space:
                stack = self._congruence_stack(self.W, stack)
            for w, C in self._expectation_coeffs(amb_space, stack, amb_m):
                off = next(o for ww, o, _ in cols if ww == w)
                scale = xp.asarray(be.sqrt(unit[w]), dtype=be.dtype)
                B[lo:hi, off : off + C.shape[1] ** 2] = scale * C.reshape(hi - lo, -1)
            del stack
        B /= xp.sqrt(gram)[:, None]

        # ||E x|| = ||x|| exactly on the singular value 1 subspace of B, and
        # every singular value is at most 1 because E is a projection.  Take
        # that subspace from the eigenvectors of B B*, not from an SVD of B:
        # the right singular vectors are never used and they are the bulk of
        # the work, dim A_amb being far larger than k.  Squaring costs half the
        # digits and there are plenty -- the gap either side of the cut is
        # reported in _last_angle_gap and is of order 0.1, not of order eps.
        w, U = xp.linalg.eigh(B @ B.conj().T)
        w = be.to_host(w).real
        cut = (1 - be.tol) ** 2
        first = int((w < cut).sum())  # eigh is ascending
        keep = w.size - first
        self._last_angle_gap = (
            float(max(w[first - 1], 0.0)) ** 0.5 if first else 0.0,
            float(max(w[first], 0.0)) ** 0.5 if keep else 1.0,
        )
        coeffs = U[:, first:].conj() / xp.sqrt(gram)[:, None]
        return [self._op_from_cheap_coeffs(sub_space, groups, coeffs[:, j])
                for j in range(keep)]

    def _op_from_cheap_coeffs(self, space, groups, coeffs):
        """``sum_i coeffs[i] u_i`` written straight into its blocks."""
        be, xp = self.backend, self.backend.xp
        blocks = {}
        for y, idx, start in groups:
            T = idx.shape[1]
            d = len(space.blocks[y])
            M = blocks.get(y)
            if M is None:
                M = be.zeros(d, d)
                blocks[y] = M
            c = coeffs[start : start + T * T].reshape(T, T)
            flat = (idx[:, :, None] * d + idx[:, None, :]).reshape(-1)
            M.reshape(-1)[flat] += xp.broadcast_to(
                c, (idx.shape[0], T, T)
            ).reshape(-1)
        return Op(space, blocks, be)

    def _cut_down_by_expectation(self, outer, ambient):
        """``{ x in span(outer) : E_ambient(x) = x }``, via a small Gram matrix.

        Writing ``x = sum c_i u_i``, the trace preserving expectation is an
        orthogonal projection for the Markov inner product, so

            || x - E(x) ||^2  =  ||x||^2 - ||E(x)||^2  =  c* (G - H) c

        with ``G_ij = tr(u_i* u_j)`` and ``H_ij = tr(u_i* E(u_j))``.  The answer
        is the kernel of the positive semidefinite form ``G - H``, which is only
        ``len(outer)`` square -- whereas writing the same condition out in
        ambient coordinates would need one row per dimension of ``A_{n,1}``,
        tens of thousands of them by ``n = 5``.  The ``u_i`` are very sparse
        (one entry per prefix), so each pairing is a gather rather than a full
        block product.
        """
        be = self.backend
        k = len(outer)
        if k == 0:
            return []
        tables, norm = self._pairing_tables(outer)
        G = be.zeros(k, k)
        M = be.zeros(k, k)
        for j, u in enumerate(outer):
            col_g = self._pair_all(tables, k, u, norm)
            col_h = self._pair_all(tables, k, self.expectation(u, *ambient), norm)
            for i in range(k):
                G[i, j] = col_g[i]
                M[i, j] = col_g[i] - col_h[i]
        del G
        basis = []
        for coeffs in self._solve_homogeneous(
            [[M[i, j] for j in range(k)] for i in range(k)], k
        ):
            op = Op.zero(self.H, be)
            for c, u in zip(coeffs, outer):
                if not be.is_zero(c):
                    op = op + c * u
            basis.append(op)
        return basis

    def _pairing_tables(self, ops):
        """Support tables for computing ``tr(u_i* X)`` for every ``u_i`` at once.

        ``Op`` stores blocks densely, but the cheap commutant basis has only a
        handful of nonzero entries each, so recording the support once turns the
        whole Gram computation into one gather per target operator.
        """
        be = self.backend
        sq, space = self.square, self.H
        endtag = space.types[-1]
        Z0 = sp.Add(*[sq.mu["00"][a] for a in sq.V["00"]])
        norm = be.num(Z0 * space.P[space.L])
        tables = {}
        for i, op in enumerate(ops):
            for y, blk in op.blocks.items():
                w = sq.mu[endtag][y]
                t = tables.setdefault(y, {"i": [], "r": [], "c": [], "k": []})
                n_y = len(space.blocks[y])
                for r in range(n_y):
                    for c in range(n_y):
                        val = blk[r, c]
                        if be.is_zero(val):
                            continue
                        t["i"].append(i)
                        t["r"].append(r)
                        t["c"].append(c)
                        t["k"].append(be.conj(val) * be.num(w))
        if getattr(be, "array", False):
            np = be.xp
            for t in tables.values():
                t["i"] = np.array(t["i"], dtype=int)
                t["r"] = np.array(t["r"], dtype=int)
                t["c"] = np.array(t["c"], dtype=int)
                t["k"] = np.array(t["k"], dtype=complex)
        return tables, norm

    def _pair_all(self, tables, k, target, norm):
        """``[ tr(u_i* target) for i in range(k) ]``."""
        be = self.backend
        if getattr(be, "array", False):
            np = be.xp
            out = np.zeros(k, dtype=complex)
            for y, t in tables.items():
                B = target.blocks.get(y)
                if B is None:
                    continue
                vals = t["k"] * B[t["r"], t["c"]]
                out += np.bincount(t["i"], weights=vals.real, minlength=k) + 1j * (
                    np.bincount(t["i"], weights=vals.imag, minlength=k)
                )
            return out / complex(norm)
        out = [be.num(0)] * k
        for y, t in tables.items():
            B = target.blocks.get(y)
            if B is None:
                continue
            for i, r, c, coef in zip(t["i"], t["r"], t["c"], t["k"]):
                out[i] = out[i] + coef * B[r, c]
        return [be.simp(x / norm) for x in out]

    def _commutant_by_solving(self, sub_gens, amb_gens):
        """Elements of ``span(amb_gens)`` commuting with every ``sub_gens``."""
        be = self.backend
        if not amb_gens:
            return []
        # one column per ambient generator; the rows stack the entries of
        # [g, s] over every generator s of the subalgebra
        cols = []
        for g in amb_gens:
            col = []
            for s in sub_gens:
                col.extend(g.comm(s).vector())
            cols.append(col)
        rows = [[col[i] for col in cols] for i in range(len(cols[0]))]
        basis = []
        for vec in self._solve_homogeneous(rows, len(amb_gens)):
            op = Op.zero(self.H, be)
            for j, g in enumerate(amb_gens):
                if be.is_zero(vec[j]):
                    continue
                op = op + vec[j] * g
            basis.append(op)
        return basis

    def _solve_homogeneous(self, rows, width):
        """Exact kernel of the homogeneous system with the given rows.

        These systems are hugely overdetermined -- the rank is at most ``width``
        but there are typically ten or more times that many rows -- and the
        exact solve is superlinear in the number of rows.  So a numerically
        independent subset of the rows is picked first and only that is solved
        exactly.

        The shortcut is made sound by checking the answer: ``ker(subset)``
        always contains ``ker(all rows)``, so if every original row annihilates
        the reduced kernel then the two coincide.  In the rare case it does not
        (a row exactly independent but numerically dependent), the full system
        is solved instead.
        """
        be = self.backend
        if width == 0:
            return []
        if not rows:
            return [[be.num(1) if i == j else be.num(0) for i in range(width)]
                    for j in range(width)]

        picker = _SpanTracker(be)
        keep = [r for r in rows if picker.add(r)]

        def solve(selected):
            A = be.zeros(len(selected), width)
            for i, row in enumerate(selected):
                for j, val in enumerate(row):
                    A[i, j] = val
            return be.nullspace(A)

        ns = solve(keep)
        if not ns or len(keep) == len(rows):
            return ns
        residuals = []
        for row in rows:
            for vec in ns:
                residuals.append(sum((a * b for a, b in zip(row, vec)), be.num(0)))
        R = be.zeros(1, len(residuals))
        for i, val in enumerate(residuals):
            R[0, i] = val
        if be.mat_is_zero(R):
            return ns
        return solve(rows)

    def commutant_of(self, ops):
        """Brute force ``{ops}' \\cap A_{n,1}``: solve ``[X, g] = 0`` block by block.

        Slower than :meth:`relative_commutant` but works for an arbitrary set of
        operators.  Returns a basis.
        """
        be = self.backend
        basis = []
        for key in self.H.keys:
            d = len(self.H.blocks[key])
            if d == 0:
                continue
            rows = []
            for g in ops:
                G = g._blk(key)
                # [X, G] = 0, X unknown d x d, flattened row major
                for i in range(d):
                    for j in range(d):
                        row = [be.num(0)] * (d * d)
                        for t in range(d):
                            row[i * d + t] = row[i * d + t] + G[t, j]
                            row[t * d + j] = row[t * d + j] - G[i, t]
                        rows.append(row)
            for vec in self._solve_homogeneous(rows, d * d):
                blk = be.zeros(d, d)
                for i in range(d):
                    for j in range(d):
                        blk[i, j] = vec[i * d + j]
                basis.append(Op(self.H, {key: blk}, be))
        return basis

    # ------------------------------------------------------- linear algebra aids
    def _tracker(self):
        return _SpanTracker(self.backend, exact=self.exact_rank)

    def _coords(self, op):
        """Coordinates of ``op``, as one array when the backend has arrays."""
        if getattr(self.backend, "array", False) and not self.exact_rank:
            return op.vector_array()
        return op.vector()

    def span_dimension(self, ops):
        """Dimension of the linear span of a list of operators."""
        be = self.backend
        if getattr(be, "array", False) and not self.exact_rank and ops:
            # one rank of one stacked matrix, rather than a Gram-Schmidt pass
            # per operator against every operator kept so far
            M = be.xp.stack([op.vector_array() for op in ops])
            return int(be.rank(M))
        tr = self._tracker()
        for op in ops:
            tr.add(self._coords(op))
        return tr.dim

    def in_span(self, op, ops):
        """Is ``op`` in the linear span of ``ops``?"""
        tr = self._tracker()
        for x in ops:
            tr.add(self._coords(x))
        return not tr.add(self._coords(op))

    _in_span = in_span

    def generated_algebra(self, ops, include_identity=True, max_steps=64):
        """A basis of the unital ``*``-algebra generated by ``ops``."""
        seeds = list(ops) + [op.H for op in ops]
        if include_identity:
            seeds = [self.one()] + seeds
        tr = self._tracker()
        basis = []
        for op in seeds:
            if tr.add(self._coords(op)):
                basis.append(op)
        for _ in range(max_steps):
            grew = False
            for x in list(basis):
                for y in list(basis):
                    p = x * y
                    if tr.add(self._coords(p)):
                        basis.append(p)
                        grew = True
            if not grew:
                break
        return basis

    def center(self, basis):
        """The centre of the span of ``basis`` (assumed to be an algebra)."""
        return [op for op in self.commutant_of(basis) if self.in_span(op, basis)]

    # ---------------------------------------------------------------- verifying
    def check_W(self):
        """``W`` is unitary and each elementary move is unitary."""
        res = CheckResult(f"connection unitary W at n={self.n}")
        for k, move in enumerate(self.moves):
            if not move.is_unitary():
                res.fail(f"elementary move S_{k} -> S_{k + 1} is not unitary")
        if not self.W.is_unitary():
            res.fail("W is not unitary")
        res.note(f"dim H_{self.n + 1} = {self.H.dim}, {len(self.H.keys)} blocks")
        return res

    def check_trace(self):
        """The Markov trace is normalised, and ``W`` preserves it."""
        res = CheckResult(f"Markov trace at n={self.n}")
        be = self.backend
        if not be.is_zero(be.num(self.one().trace() - 1)):
            res.fail("trace of the identity is not 1")
        for p, q in self._available_algebras():
            for g in self._sample(p, q, 8):
                lhs, rhs = g.trace(), self.to_K(g).trace()
                if not be.is_zero(be.num(lhs - rhs)):
                    res.fail(f"W does not preserve the trace of a generator of A_({p},{q})")
                    break
        return res

    def check_commuting_square(self, level=None):
        """The commuting square condition at the top of the tower::

            A_{m-1,0}  <  A_{m,0}
                ^             ^
            A_{m-1,1}  <  A_{m,1}

        Tested in the equivalent form ``E_{A_{m-1,1}}(x) == E_{A_{m-1,0}}(x)``
        for every ``x`` in ``A_{m,0}``, which follows from (and implies)
        ``E_{A_{m-1,1}} E_{A_{m,0}} = E_{A_{m-1,0}}``.  ``level=1`` recovers the
        original commuting square that the connection was built from.
        """
        m = self.n if level is None else level
        res = CheckResult(f"commuting square at level {m} (inside n={self.n})")
        if not 1 <= m <= self.n:
            res.note(f"level {m} is not available at n = {self.n}")
            return res
        for g in self.generators(m, 0):
            lhs = self.expectation(g, m - 1, 1)
            rhs = self.expectation(g, m - 1, 0)
            if not (lhs - rhs).is_zero():
                res.fail(
                    f"E onto A_({m - 1},1) and E onto A_({m - 1},0) differ on a "
                    f"generator of A_({m},0)"
                )
                break
        return res

    def check_generation(self):
        """``A_{n-1,1} . A_{n,0}`` spans ``A_{n,1}``.

        For a commuting square whose basic construction is the level above,
        products of the two subalgebras already span everything, so this is
        checked as a single span computation rather than an algebra closure.
        """
        res = CheckResult(f"A_(n-1,1) . A_(n,0) spans A_(n,1) at n={self.n}")
        if self.n < 1:
            res.note("nothing to check at n = 0")
            return res
        want = sum(len(v) ** 2 for v in self.H.blocks.values())
        left = self.generators(self.n - 1, 1)
        right = self.generators(self.n, 0)
        tr = self._tracker()
        for x in left:
            for y in right:
                tr.add(self._coords(x * y))
                if tr.dim == want:
                    return res
        res.fail(f"the products span dimension {tr.dim}, but dim A_(n,1) = {want}")
        return res

    def check_jones(self):
        """Each ``f_k`` is a projection of the right trace and satisfies the TL relations."""
        res = CheckResult(f"Jones projections at n={self.n}")
        be = self.backend
        beta2 = be.num(self.square.beta_v ** 2)
        fs = {}
        for k in range(2, self.n + 1):
            f = self.jones_H(k)
            fs[k] = f
            if not (f * f - f).is_zero():
                res.fail(f"f_{k} is not idempotent")
            if not (f - f.H).is_zero():
                res.fail(f"f_{k} is not self adjoint")
            if not be.is_zero(be.num(f.trace() - 1 / beta2)):
                res.fail(f"tr(f_{k}) != 1/beta_v^2")
        for k in fs:
            for l in fs:
                if abs(k - l) == 1:
                    lhs = fs[k] * fs[l] * fs[k]
                    rhs = (1 / beta2) * fs[k]
                    if not (lhs - rhs).is_zero():
                        res.fail(f"f_{k} f_{l} f_{k} != f_{k}/beta_v^2")
                elif abs(k - l) >= 2:
                    if not (fs[k] * fs[l] - fs[l] * fs[k]).is_zero():
                        res.fail(f"f_{k} and f_{l} do not commute")
        return res

    def check_expectations(self):
        """Expectations are idempotent, trace preserving and land where they should."""
        res = CheckResult(f"conditional expectations at n={self.n}")
        be = self.backend
        sample = self._sample(self.n, 1, 8)
        for p, q in self._available_algebras():
            for g in sample:
                e = self.expectation(g, p, q)
                if not be.is_zero(be.num(e.trace() - g.trace())):
                    res.fail(f"E onto A_({p},{q}) is not trace preserving")
                    break
                if not (self.expectation(e, p, q) - e).is_zero():
                    res.fail(f"E onto A_({p},{q}) is not idempotent")
                    break
        return res

    def _sample(self, p, q, count):
        """A spread out handful of matrix units of ``A_{p,q}``, for spot checks."""
        space, m = self._space_and_level(p, q)
        pairs = [
            (xi, eta)
            for plist in space.level_blocks[m].values()
            for xi, eta in itertools.product(plist, repeat=2)
        ]
        step = max(1, len(pairs) // count)
        picked = pairs[::step][:count]
        ops = [Op.matrix_unit(space, xi, eta, self.backend) for xi, eta in picked]
        return [self.W.pull_back(o) for o in ops] if space is self.K else ops

    def available_algebras(self):
        """Every ``(p, q)`` this level can represent, as arguments for the API."""
        out = [(0, 0)]
        for p in range(self.n + 1):
            out.append((p, 1))
        for p in range(1, self.n + 1):
            out.append((p, 0))
        return out

    _available_algebras = available_algebras

    def verify(self, verbose=True, heavy=True):
        """Run the level checks.  ``heavy=False`` skips the generation check."""
        results = [self.check_W(), self.check_trace(), self.check_expectations()]
        if self.n >= 2:
            results.append(self.check_jones())
        results.append(self.check_commuting_square())
        if heavy:
            results.append(self.check_generation())
        if verbose:
            for r in results:
                print(r)
        return results

    def __repr__(self):
        return f"<Level n={self.n}, dim H_{self.n + 1} = {self.H.dim}, backend {self.backend.name}>"
