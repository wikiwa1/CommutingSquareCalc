"""Path spaces and block-diagonal operators on them.

A path space is described by nothing more than the sequence of vertex sets its
paths run through, e.g. ``('00', '01', '11', '01')`` for H_3.  Consecutive tags
determine which of the four graphs supplies the edge, and in which direction,
so ``H``, ``K`` and every intermediate stage of the connection move all come
from the same constructor.

The algebra of a path space is the path algebra of a Bratteli diagram: its
minimal central projections are the *end* vertices, and the matrix unit
``E_{xi, eta}`` exists whenever ``xi`` and ``eta`` end at the same vertex --
they need not start at the same one.  So a level-``m`` algebra is
``(+)_w M_{d_w}`` with ``d_w`` the number of paths of length ``m`` ending at
``w``, which is what makes ``A_{0,0} = (+)_a C`` a genuinely non-central
subalgebra of ``A_{n,1}`` and the higher relative commutants interesting.

Operators are therefore stored as a dictionary of blocks indexed by the end
vertex.  ``Op.dense()`` materialises the full matrix when that is more
convenient.
"""

from __future__ import annotations

import sympy as sp

from .backend import get_backend

__all__ = ["PathSpace", "Op", "BlockMap", "edge_graph"]


def edge_graph(tag_from, tag_to):
    """Which graph carries an edge between two vertex sets, and in which direction.

    Returns ``(graph name, forward)`` where ``forward`` is True when
    ``tag_from`` indexes the rows of that graph.
    """
    pair = (tag_from, tag_to)
    table = {
        ("00", "01"): ("h0", True),
        ("01", "00"): ("h0", False),
        ("00", "10"): ("v0", True),
        ("10", "00"): ("v0", False),
        ("10", "11"): ("h1", True),
        ("11", "10"): ("h1", False),
        ("01", "11"): ("v1", True),
        ("11", "01"): ("v1", False),
    }
    if pair not in table:
        raise ValueError(f"no graph joins V{tag_from} to V{tag_to}")
    return table[pair]


class PathSpace:
    """Paths through a prescribed sequence of vertex sets.

    Attributes
    ----------
    types : tuple of str
        The vertex-set tags, length ``L + 1`` for paths with ``L`` edges.
    paths : list of tuple
        The ordered basis, grouped by block.
    blocks : dict
        ``end vertex -> list of paths`` ending there.
    index : dict
        ``path -> (end vertex, position within the block)``.
    beta : list
        ``beta[i]`` is the graph norm governing edge ``i``; used by the Markov
        trace and by the conditional expectations.
    P : list
        ``P[m] = beta[0] * ... * beta[m-1]``, so ``P[L]`` normalises the trace.
    """

    def __init__(self, square, types):
        self.square = square
        self.types = tuple(types)
        self.L = len(self.types) - 1
        if self.L < 0:
            raise ValueError("a path space needs at least one vertex set")

        self.steps = []  # (graph name, forward) per edge
        for i in range(self.L):
            self.steps.append(edge_graph(self.types[i], self.types[i + 1]))
        self.beta = [square.beta(g) for g, _ in self.steps]
        self.P = [sp.Integer(1)]
        for b in self.beta:
            self.P.append(sp.expand(self.P[-1] * b))

        # successors at each step
        self._succ = []
        for i, (gname, fwd) in enumerate(self.steps):
            g = square.graphs[gname]
            self._succ.append(g.row_nbrs if fwd else g.col_nbrs)

        # level_paths[m] = every path with m edges (a prefix of the full space)
        self.level_paths = [[(x,) for x in square.V[self.types[0]]]]
        for i in range(self.L):
            succ = self._succ[i]
            self.level_paths.append(
                [p + (y,) for p in self.level_paths[-1] for y in succ[p[-1]]]
            )

        self.paths_all = self.level_paths[self.L]
        self.level_blocks = []
        self.level_index = []
        for m in range(self.L + 1):
            blocks = {}
            for p in self.level_paths[m]:
                blocks.setdefault(p[-1], []).append(p)
            index = {}
            for key, plist in blocks.items():
                for i, p in enumerate(plist):
                    index[p] = (key, i)
            self.level_blocks.append(blocks)
            self.level_index.append(index)

        self.blocks = self.level_blocks[self.L]
        self.index = self.level_index[self.L]
        self.keys = list(self.blocks.keys())
        # global ordering, block by block
        self.paths = [p for key in self.keys for p in self.blocks[key]]
        self.pos = {p: i for i, p in enumerate(self.paths)}
        self.dim = len(self.paths)

    # --------------------------------------------------------------- tails
    def tails(self, m, vertex, upto=None):
        """Continuations from ``vertex`` at position ``m`` out to position ``upto``.

        ``upto`` defaults to the end of the space.
        """
        if upto is None:
            upto = self.L
        cur = [(vertex,)]
        for i in range(m, upto):
            succ = self._succ[i]
            cur = [p + (y,) for p in cur for y in succ[p[-1]]]
        return cur

    def end_weight(self, path):
        """Perron-Frobenius weight of the last vertex of ``path``."""
        return self.square.mu[self.types[len(path) - 1]][path[-1]]

    def block_dims(self):
        return {k: len(v) for k, v in self.blocks.items()}

    def __repr__(self):
        return f"<PathSpace {'-'.join(self.types)}: dim {self.dim}, {len(self.keys)} blocks>"


class Op:
    """A block-diagonal operator on a :class:`PathSpace`.

    Blocks are indexed by the end vertex of the paths; a missing key means a
    zero block.  Supports ``+ - * @`` (operator or scalar), ``.H`` for the
    adjoint, and comparison via ``==`` / :meth:`is_zero`.
    """

    __slots__ = ("space", "backend", "blocks")

    def __init__(self, space, blocks=None, backend=None):
        self.space = space
        self.backend = get_backend(backend if backend is not None else space.square.backend)
        self.blocks = {} if blocks is None else blocks

    # ------------------------------------------------------------ constructors
    @classmethod
    def zero(cls, space, backend=None):
        return cls(space, {}, backend)

    @classmethod
    def identity(cls, space, backend=None):
        be = get_backend(backend if backend is not None else space.square.backend)
        return cls(space, {k: be.eye(len(v)) for k, v in space.blocks.items()}, be)

    @classmethod
    def matrix_unit(cls, space, xi, eta, backend=None):
        """The matrix unit ``E_{xi, eta}`` of the level-``m`` algebra, ``m = len(xi)-1``.

        ``xi`` and ``eta`` must have the same length and end at the same vertex;
        they need not start at the same one.  When ``m < L`` the unit is embedded
        in the full algebra as ``sum_tau E_{xi tau, eta tau}``.
        """
        be = get_backend(backend if backend is not None else space.square.backend)
        if len(xi) != len(eta):
            raise ValueError("matrix unit needs two paths of equal length")
        if xi[-1] != eta[-1]:
            raise ValueError("matrix unit needs two paths ending at the same vertex")
        m = len(xi) - 1
        op = cls(space, {}, be)
        for tail in space.tails(m, xi[-1]):
            p, q = xi + tail[1:], eta + tail[1:]
            key, i = space.index[p]
            _, j = space.index[q]
            blk = op.blocks.get(key)
            if blk is None:
                blk = be.zeros(len(space.blocks[key]), len(space.blocks[key]))
                op.blocks[key] = blk
            blk[i, j] = be.num(1)
        return op

    @classmethod
    def from_dense(cls, space, M, backend=None):
        be = get_backend(backend if backend is not None else space.square.backend)
        op = cls(space, {}, be)
        for key, plist in space.blocks.items():
            n = len(plist)
            blk = be.zeros(n, n)
            for i, p in enumerate(plist):
                for j, q in enumerate(plist):
                    blk[i, j] = M[space.pos[p], space.pos[q]]
            if not be.mat_is_zero(blk):
                op.blocks[key] = blk
        return op

    @classmethod
    def from_entries(cls, space, entries, backend=None):
        """Build from a ``{(path, path): value}`` mapping."""
        be = get_backend(backend if backend is not None else space.square.backend)
        op = cls(space, {}, be)
        for (p, q), val in entries.items():
            key, i = space.index[p]
            key2, j = space.index[q]
            if key != key2:
                raise ValueError(f"entry ({p}, {q}) does not end at a common vertex")
            blk = op.blocks.get(key)
            if blk is None:
                blk = be.zeros(len(space.blocks[key]), len(space.blocks[key]))
                op.blocks[key] = blk
            blk[i, j] = blk[i, j] + be.num(val)
        return op

    # --------------------------------------------------------------- algebra
    def _blk(self, key):
        blk = self.blocks.get(key)
        if blk is None:
            n = len(self.space.blocks[key])
            return self.backend.zeros(n, n)
        return blk

    def __add__(self, other):
        be = self.backend
        out = {}
        for key in set(self.blocks) | set(other.blocks):
            out[key] = be.simp_mat(be.add(self._blk(key), other._blk(key)))
        return Op(self.space, out, be)

    def __sub__(self, other):
        be = self.backend
        out = {}
        for key in set(self.blocks) | set(other.blocks):
            out[key] = be.simp_mat(be.sub(self._blk(key), other._blk(key)))
        return Op(self.space, out, be)

    def __neg__(self):
        be = self.backend
        return Op(self.space, {k: be.smul(-1, v) for k, v in self.blocks.items()}, be)

    def __mul__(self, other):
        # Entries are normalised after every product.  Without this, repeated
        # multiplication of connection entries grows expressions fast enough
        # that even evaluating one numerically becomes the bottleneck.
        be = self.backend
        if isinstance(other, Op):
            out = {}
            for key in set(self.blocks) & set(other.blocks):
                out[key] = be.simp_mat(be.mul(self.blocks[key], other.blocks[key]))
            return Op(self.space, out, be)
        return Op(
            self.space,
            {k: be.simp_mat(be.smul(other, v)) for k, v in self.blocks.items()},
            be,
        )

    __matmul__ = __mul__

    def __rmul__(self, scalar):
        be = self.backend
        return Op(self.space, {k: be.smul(scalar, v) for k, v in self.blocks.items()}, be)

    @property
    def H(self):
        """Adjoint."""
        be = self.backend
        return Op(self.space, {k: be.dagger(v) for k, v in self.blocks.items()}, be)

    def comm(self, other):
        """Commutator ``[self, other]``."""
        return self * other - other * self

    def simplify(self):
        be = self.backend
        return Op(self.space, {k: be.simp_mat(v) for k, v in self.blocks.items()}, be)

    # ------------------------------------------------------------- inspection
    def is_zero(self):
        return all(self.backend.mat_is_zero(v) for v in self.blocks.values())

    def __eq__(self, other):
        if not isinstance(other, Op):
            return NotImplemented
        return (self - other).is_zero()

    def __hash__(self):
        raise TypeError("Op is not hashable")

    def entry(self, p, q):
        key, i = self.space.index[p]
        key2, j = self.space.index[q]
        if key != key2:
            return self.backend.num(0)
        return self._blk(key)[i, j]

    def dense(self):
        """The full matrix over ``space.paths``."""
        be = self.backend
        n = self.space.dim
        M = be.zeros(n, n)
        for key, blk in self.blocks.items():
            plist = self.space.blocks[key]
            for i, p in enumerate(plist):
                for j, q in enumerate(plist):
                    M[self.space.pos[p], self.space.pos[q]] = blk[i, j]
        return M

    def trace(self):
        """The Markov trace, normalised so that ``trace(1) == 1``."""
        sq = self.space.square
        be = self.backend
        total = be.num(0)
        Z0 = sp.Add(*[sq.mu["00"][a] for a in sq.V["00"]])
        norm = be.num(Z0 * self.space.P[self.space.L])
        endtag = self.space.types[-1]
        for key, blk in self.blocks.items():
            w = be.num(sq.mu[endtag][key])
            total = total + w * be.trace(blk)
        return be.simp(total / norm)

    def vector_array(self):
        """:meth:`vector` as one flat array; only for the array backends.

        The same coordinates in the same order, but without ``dim A_{n,1}``
        Python objects being created to hold them -- which matters once that is
        tens of thousands of entries per operator.
        """
        be = self.backend
        return be.xp.concatenate(
            [self._blk(key).reshape(-1) for key in self.space.keys]
        )

    def vector(self):
        """Flatten to a coordinate vector over a fixed basis of the block algebra."""
        be = self.backend
        out = []
        for key in self.space.keys:
            blk = self._blk(key)
            n = len(self.space.blocks[key])
            for i in range(n):
                for j in range(n):
                    out.append(blk[i, j])
        return out

    @classmethod
    def from_vector(cls, space, vec, backend=None):
        be = get_backend(backend if backend is not None else space.square.backend)
        op = cls(space, {}, be)
        k = 0
        for key in space.keys:
            n = len(space.blocks[key])
            blk = be.zeros(n, n)
            for i in range(n):
                for j in range(n):
                    blk[i, j] = be.num(vec[k])
                    k += 1
            op.blocks[key] = blk
        return op

    def __repr__(self):
        return f"<Op on {'-'.join(self.space.types)}, {len(self.blocks)} nonzero blocks>"


class BlockMap:
    """A block-diagonal map between two path spaces with matching block indices.

    Used for the connection unitary ``W : H_{n+1} -> K_{n+1}`` and for each of
    the elementary moves that build it.  Every move changes one interior vertex
    only, so the end vertex -- and hence the block index -- is preserved.  ``blocks[key]`` is the matrix of the
    restriction, with rows indexed by ``dst.blocks[key]`` and columns by
    ``src.blocks[key]``.
    """

    __slots__ = ("src", "dst", "backend", "blocks")

    def __init__(self, src, dst, blocks, backend=None):
        self.src = src
        self.dst = dst
        self.backend = get_backend(backend if backend is not None else src.square.backend)
        self.blocks = blocks

    @classmethod
    def identity(cls, space, backend=None):
        be = get_backend(backend if backend is not None else space.square.backend)
        return cls(space, space, {k: be.eye(len(v)) for k, v in space.blocks.items()}, be)

    def compose(self, other):
        """``self . other``, i.e. apply ``other`` first."""
        if other.dst is not self.src:
            raise ValueError("cannot compose: intermediate spaces differ")
        be = self.backend
        out = {}
        for key in self.dst.keys:
            A = self.blocks.get(key)
            B = other.blocks.get(key)
            if A is None or B is None:
                continue
            out[key] = be.simp_mat(be.mul(A, B))
        return BlockMap(other.src, self.dst, out, be)

    @property
    def H(self):
        be = self.backend
        return BlockMap(self.dst, self.src, {k: be.dagger(v) for k, v in self.blocks.items()}, be)

    def conjugate_op(self, op):
        """Transport ``op`` on ``src`` to ``dst``: returns ``W op W*``."""
        if op.space is not self.src:
            raise ValueError("operator lives on the wrong space")
        be = self.backend
        out = {}
        for key, blk in op.blocks.items():
            A = self.blocks.get(key)
            if A is None:
                continue
            out[key] = be.simp_mat(be.mul(be.mul(A, blk), be.dagger(A)))
        return Op(self.dst, out, be)

    def pull_back(self, op):
        """Transport ``op`` on ``dst`` back to ``src``: returns ``W* op W``."""
        return self.H.conjugate_op(op)

    def is_unitary(self):
        be = self.backend
        for key in self.src.keys:
            A = self.blocks.get(key)
            n = len(self.src.blocks[key])
            m = len(self.dst.blocks[key])
            if n != m:
                return False
            if A is None:
                if n:
                    return False
                continue
            if not be.mat_is_zero(be.sub(be.mul(be.dagger(A), A), be.eye(n))):
                return False
            if not be.mat_is_zero(be.sub(be.mul(A, be.dagger(A)), be.eye(n))):
                return False
        return True

    def dense(self):
        be = self.backend
        M = be.zeros(self.dst.dim, self.src.dim)
        for key, blk in self.blocks.items():
            rows = self.dst.blocks[key]
            cols = self.src.blocks[key]
            for i, p in enumerate(rows):
                for j, q in enumerate(cols):
                    M[self.dst.pos[p], self.src.pos[q]] = blk[i, j]
        return M

    def __repr__(self):
        return (
            f"<BlockMap {'-'.join(self.src.types)} -> {'-'.join(self.dst.types)}, "
            f"{len(self.blocks)} blocks>"
        )
