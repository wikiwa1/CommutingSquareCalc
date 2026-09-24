# pathalg — path algebra computations for commuting squares

Explicit, exact computation in the tower of a commuting square of finite
dimensional algebras, from a bi-unitary connection. `A_{n,1}`, `A_{n-1,1}` and
`A_{n,0}` are all represented on the single path space `H_{n+1}`, so operators
from the three algebras can be multiplied, added and compared directly.

Requires Python 3.9+ and `sympy`; `numpy` is optional but recommended (it backs
the fast numerical mode and the rank computations). `cupy` is optional on top of
that and moves the whole tower onto the GPU -- see
[Exactness and performance](#exactness-and-performance).

```
pathalg/
  square.py     the commuting square, four graphs, connection, verification
  weights.py    exact Perron-Frobenius weights
  paths.py      path spaces, block operators, block maps
  tower.py      one level: W, the three algebras, trace, e_k, E, commutants
  eigen.py      eigenvalue 1 subspace of a positive contraction, matrix free
  intersection.py  two members of a family: angle spectrum, N_s cap N_t
  backend.py    exact (sympy) and numerical (numpy) arithmetic
  examples.py   Dynkin and Hadamard squares, ready to use
demo.py         a worked example, start to finish
ah_demo.py      the Asaeda-Haagerup square of S1122.txt, same shape as demo.py
                (from arXiv:2509.00227, section 5)
s1133_demo.py   the same for the 4-star S(1,1,3,3), index 3 + sqrt(3)
intersection_demo.py  N_s cap N_s' for two members of a family (see below)
test_pathalg.py the test suite (python test_pathalg.py [--slow])
```

## Conventions

Cells are drawn as in Kawahigashi, *[On flatness of Ocneanu's connections on
the Dynkin diagrams](https://www.ms.u-tokyo.ac.jp/~yasuyuki/flat.pdf)*, §1:

```
    a ---- b        a in V00 (top left)      b in V01 (top right)
    |      |        c in V10 (bottom left)   d in V11 (bottom right)
    c ---- d
```

for the square

```
    A00  <  A01              h0 : V00 - V01   (A00 < A01)
     ^       ^               v0 : V00 - V10   (A00 < A10)
    A10  <  A11              h1 : V10 - V11   (A10 < A11)
                             v1 : V01 - V11   (A01 < A11)
```

The two halves of the connection act on paths by

```
u^{a,d}_{c,b} :  [a,b,d]  |-->  sum_c u^{a,d}_{c,b} [a,c,d]
v^{c,b}_{a,d} :  [c,d,b]  |-->  sum_a v^{c,b}_{a,d} [c,a,b]
```

so `u` at fixed corners `(a,d)` is a matrix with rows `c`, columns `b`, and `v`
at fixed corners `(c,b)` has rows `a`, columns `d`. Unitarity of every `u` block
is the unitarity axiom; unitarity of every `v` block is bi-unitarity. The
renormalisation rule ties them together:

**`v^{c,b}_{a,d} = sqrt( mu(a) mu(d) / ( mu(b) mu(c) ) ) * conj( u^{a,d}_{c,b} )`**

You supply either half and the other is derived. Two independent confirmations
that this is the right orientation and exponent:

- a trace balance, `sum_{c,b} mu(b)mu(c) n(c,b) = sum_{a,d} mu(a)mu(d) n(a,d)`,
  holds only with the weights this way round and exponent 1/2;
- Ocneanu's published connection on the Dynkin diagrams `A_3`, `A_4`, `A_5`
  passes `check_unitarity()` *and* `check_biunitarity()` unchanged
  (`test_dynkin_biunitarity`).

Perron-Frobenius weights are always computed, never supplied: `mu` on `V00` is
normalised to 1 at the first vertex and everything else is forced. Construction
fails loudly if `h0` and `v0` induce different weights on `V00`, which means the
input is not a commuting square.

## Path spaces

`H_{n+1}` starts on `V00` and then alternates `V01, V11, V01, ...`; `K_{n+1}`
alternates `V00, V10` for `n` edges and then takes one horizontal edge. So

```
H_3 = [x0 in V00, x1 in V01, x2 in V11, x3 in V01]
K_3 = [x0 in V00, x1 in V10, x2 in V00, x3 in V01]
```

Both end on `V_{n,1}` (`V01` for even `n`, `V11` for odd `n`) — note this
follows the `K_3` example rather than the "even/odd" sentence in the original
spec, which had the two cases swapped. They must agree, because the number of
paths from `a` to `y` is the `(a,y)` entry of the inclusion matrix of
`A00 < A_{n,1}` computed either way round, so the blocks match and `W` can
exist. `Level.__init__` checks this and reports both block dimension lists if
it fails.

The algebra of a path space is the path algebra of a Bratteli diagram: the
matrix unit `E_{xi,eta}` exists whenever `xi` and `eta` **end** at the same
vertex; they need not start at the same one. That is what makes
`A_{0,0} = (+)_a C` a genuinely non-central subalgebra and the relative
commutants interesting.

`W : H_{n+1} -> K_{n+1}` pushes the single horizontal edge from the front to the
back through the `n` vertical edges, applying `u`, `v`, `u`, `v`, … to
positions `(k, k+1, k+2)`. Each move is unitary and fixes the end vertex, so
`W` is a unitary that is block diagonal and trace preserving.

## Usage

```python
import sympy as sp
from pathalg import CommutingSquare, Level

square = CommutingSquare(
    V00=["*"], V01=["b0", "b1"], V10=["c0", "c1"], V11=["@"],
    h0=[("*", "b0"), ("*", "b1")],      # (V00, V01)
    v0=[("*", "c0"), ("*", "c1")],      # (V00, V10)
    h1=[("c0", "@"), ("c1", "@")],      # (V10, V11)
    v1=[("b0", "@"), ("b1", "@")],      # (V01, V11)
    u_cells={("*", "b0", "c0", "@"):  1/sp.sqrt(2),   # keys are (a, b, c, d)
             ("*", "b0", "c1", "@"):  1/sp.sqrt(2),
             ("*", "b1", "c0", "@"):  1/sp.sqrt(2),
             ("*", "b1", "c1", "@"): -1/sp.sqrt(2)},
)
square.verify()                 # inclusion matrices, weights, unitarity, bi-unitarity

lv = Level(square, n=3)         # everything on H_4
lv.verify()                     # W unitary, trace, expectations, Jones, commuting square

lv = Level(square, n=6, backend="cupy64")   # the same, on the GPU
```

`u_cells` may instead be a callable `f(square, a, b, c, d) -> value`, evaluated
once the weights are known — convenient because most closed-form connections
are written in terms of `mu`. Pass `v_cells=` to supply the other half instead.

### Elements

| what | how |
| --- | --- |
| `A_{p,1}` matrix unit | `lv.matrix_unit(xi, eta)` — paths of length `p+1` in `H` |
| `A_{p,0}` matrix unit | `lv.matrix_unit(alpha, beta, which="K")` — length `p` in `K`, pulled back by `W` |
| all matrix units | `lv.generators(p, q)` |
| small generating set | `lv.generating_set(p, q)` (use this for commutant conditions) |
| identity | `lv.one()` |

Operators support `+ - *`, scalar multiplication, `.H` (adjoint), `.comm(other)`,
`==`, `.is_zero()`, `.trace()` (Markov trace, normalised to 1), `.dense()`,
`.entry(p, q)`. Blocks are keyed by end vertex; `.blocks` is the dict.

### The rest

```python
lv.expectation(x, p, q)              # trace preserving E onto A_{p,q}
lv.jones(j)                          # e_j implementing E_{A_{j-1,1}} on A_{j,1}
lv.jones_H(k) / lv.jones_K(k)        # by path length instead
lv.relative_commutant(p, q)          # basis of A_{p,q}' cap A_{n,1}
lv.relative_commutant_in(sub, amb)   # e.g. ((0,1), (n,0)), Ocneanu compactness
lv.relative_commutant_dimension(sub, amb)   # the same, dimension only: no basis built
lv.commutant_of(ops)                 # brute force, for an arbitrary set
lv.span_dimension(ops) / lv.in_span(x, ops) / lv.generated_algebra(ops) / lv.center(b)
lv.to_K(x) / lv.to_H(x)              # move between the two pictures
lv.available_algebras()              # every (p, q) this level represents
```

Checks, each returning a truthy `CheckResult` that prints what failed:
`check_W`, `check_trace`, `check_expectations`, `check_jones`,
`check_commuting_square(level=m)`, `check_generation`; `lv.verify()` runs them
all.

## Exactness and performance

Everything is exact by default. Two places deliberately are not, both
documented in the code and both switchable:

- **Rank and dimension counting** (`span_dimension`, `in_span`,
  `generated_algebra`, `check_generation`) runs in floating point. These are
  well conditioned rank questions, and exact elimination on cyclotomic entries
  is drastically slower. Pass `Level(..., exact_rank=True)` to force exact.
  Every *identity* — products, adjoints, expectations, commutators, zero tests
  — is exact regardless.
- **The backend** switches the arithmetic of the whole level:
  `"numpy"` is `complex128` on the CPU, `"cupy"` the same on the GPU, and
  `"cupy64"` is `complex64` on the GPU. `test_backends_agree` and
  `test_gpu_matches_cpu` check they agree on traces and dimensions.

Two things dominate exact performance, and both are easy to get wrong:

1. **Keep connection entries as `exp(i pi q)`.** sympy collapses products of
   one exponential automatically. `radsimp`, `powsimp` and `expand_complex` all
   rewrite them as nested radicals like `sqrt(-4 - 2 sqrt(2))` that never
   collapse again; in one measured case that turned 20-operation cells into
   2801-operation cells and a 5 second check into a 3 minute one. `simp` at the
   default level therefore does `expand` and nothing else.
2. **Linear algebra goes through a number field.** `nullspace` maps the matrix
   into `QQ(alpha)` via `DomainMatrix(..., extension=True)`, where elimination
   cannot swell — 0.3s versus over three minutes for a 96x16 system. Zero
   testing uses the same route, batched over a whole matrix. When sympy cannot
   build a number field (nested radicals mixed with roots of unity) it returns
   the generic `EX` domain, where a true zero written two ways compares
   unequal; a "not zero" verdict from an unreliable domain is discarded and
   `simplify` decides instead.

Homogeneous systems are hugely overdetermined, so a numerically independent
subset of rows is solved exactly and the answer is then *verified* against every
original row — sound, because `ker(subset)` always contains `ker(all)`.

### Higher relative commutants

`relative_commutant_in((0, 1), (n, 0))` is the expensive one and gets its own
algorithm whenever the backend has arrays. The cheap commutant basis
`u_i = sum_sigma E_{sigma tau, sigma rho}` is *orthogonal* for the Markov inner
product, so after rescaling to unit vectors the problem is a principal angle
one between two subspaces of `A_{n,1}`: `span(u_i)` and the ambient `A_{n,0}`.
A trace preserving expectation is the orthogonal projection onto the second, so

```
A_{0,1}' cap A_{n,0} = { x in span(u) : ||E x|| = ||x|| },
```

the singular value 1 subspace of the `k x dim A_{n,0}` matrix of coefficients
of `E u_i`. Both of those dimensions are far below `dim A_{n,1}`, the matrix is
written one chunk of rows at a time so peak memory is set by the chunk, and the
answer comes from `eigh(B B*)` rather than an SVD of `B`, since the right
singular vectors are never used and are most of the work. `_last_angle_gap`
records the largest discarded cosine against the smallest kept one: the cut is
only meaningful while those stay far apart, and in practice they are 0.89
against 1.0.

The whole route is checked against the older `k x k` Gram cut-down, which is
still what the exact backend uses, in `test_intersection_routes_agree`.

### Iterative route and memory

The dense route above stores `B`, and at `n = 7` that is 28.5 GB. But nothing
needs `B` itself, only the eigenvalue 1 eigenspace of `N = F* F`, where
`F a` is the coordinates of `E(sum a_i u_i)`. Its dimension is the answer,
which is small. [pathalg/eigen.py](pathalg/eigen.py) finds it by
Chebyshev-filtered subspace iteration (Zhou–Saad): a block of vectors
somewhat larger than the answer is repeatedly multiplied by a Chebyshev
polynomial in `N`, small on `[0, a]` and large at 1, then re-orthonormalised
and Rayleigh–Ritz projected. `F` and `F*` are applied on the fly with the same
steps that build `B`:

- `F`: assemble `sum a_i u_i` from its sparse entries, conjugate it by `W`,
  and take the partial trace;
- `F*`: assemble the element of `A_amb`, pull it back through `W`, and pair it
  with each `u_i` (a sparse lookup).

Memory is two `k x block` arrays plus one chunk: the Rayleigh–Ritz rotation
and the re-orthonormalisation (CholeskyQR2) both work in place, whereas the
library QR alone would hold three blocks. Two details matter:

- the amplification per filter pass is capped near `1/sqrt(eps)`, since
  otherwise the rest of the block drops below rounding error and comes back as
  noise. The cap also keeps the block well conditioned enough for Cholesky QR;
- it stops when the wanted vectors have converged and the first unwanted Ritz
  value has settled. That value would still be climbing if an eigenvalue 1
  direction were missing.

If every vector in the block heads for 1, the block grows.

`method="auto"` (the default) chooses per call. Before allocating anything it
estimates each route's peak memory and compares it with what is free: GPU
memory under cupy, *physical* RAM otherwise, never counting the page file.
Dense is used if it fits, iterative if only that fits, and otherwise it raises
`pathalg.InsufficientMemory` with both estimates and the free amount. Growing
the iterative block is checked the same way. `relative_commutant_dimension`
never builds the basis, so counting costs no more than the eigenspace itself.
Pass `method="iterative"` to force it; at `n = 6` it is also the faster of the
two. `test_iterative_matches_dense` and `test_memory_refusal` cover both.

### Timings

Rough exact timings (this machine): `A_3` at `n = 3` all checks ≈ 20s, all
relative commutants ≈ instant. With `backend="numpy"`, `A_4` at `n = 4`
(`dim H_5 = 21`) ≈ 8s.

For the Asaeda-Haagerup square of `ah_demo.py` — the `S(1,1,2,2)` case of the
one-parameter family of connections on 4-stars in §5 of Bisch–Cáceres, *[New
hyperfinite subfactors with infinite depth](https://arxiv.org/abs/2509.00227)*,
arXiv:2509.00227 — `dim A_(0,1)' cap A_(n,0)`, on an RTX 4070 SUPER:

| n | dim H_(n+1) | k | dim | dense numpy | dense cupy | dense cupy64 | iterative cupy64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 200 | 1976 | 42 | 9.7s | 0.9s | 0.4s | |
| 6 | 584 | 9010 | 132 | 815s | 50s | 5.0s | 0.6s |
| 7 | 912 | 41096 | 429 | needs ~85 GB | needs ~85 GB | needs ~43 GB | 10.7s (1.2 GB) |
| 8 | 2664 | 187458 | 1430 | needs ~2.2 TB | needs ~2.2 TB | needs ~1.1 TB | 157s (5.6 GB) |

`k` is the dimension of the outer commutant that gets cut down. The `cupy` /
`cupy64` gap is the card: consumer GeForce parts run `float64` at 1/64 rate, so
`complex128` matmul measures 524 GFLOP/s against 12.3 TFLOP/s for `complex64`.
Single precision is safe here because every decision is that principal angle,
and 0.96 against 1.0 is nowhere near seven digits.

## Intersections of two members of a family

Every connection on the same four graphs gives a subfactor `N_s` of one and the
same `M`. In the `H` picture the tower `A_{n,1}` and its Jones projections do
not depend on the connection, which only moves `A_{n,0}^(s) = W_s^* (K) W_s`.
So `P cap Q = N_s cap N_t` makes sense, as for the Hadamard subfactors of
Bakshi–Guin–Guruprasad, [arXiv:2510.14499](https://arxiv.org/abs/2510.14499).
[pathalg/intersection.py](pathalg/intersection.py) reduces it to finite
dimensions, exactly:

- **The angle operator.** `E_{N_s}(A_{n,1}) = A_{n,0}^(s)`, so `L^2(A_{n,1})`
  reduces both Jones projections. The restriction `T_n` of
  `T = e_P e_Q e_P` has `spec(T_n)` inside `spec(T)`, the `spec(T_n)` increase
  with `n`, and their union is dense in `spec(T)`. So the spectrum is
  approximated exactly from below: `LevelPair.angle_spectrum(p)` gives the
  squared singular values of the cross-Gram matrix of the two `A_{p,0}`.
- **The intersection.** `E_{P cap Q}(A_{n,1}) = C_n = A_{n,0}^(s) cap A_{n,0}^(t)`,
  and `P cap Q` is generated by the `C_n`. The paper's
  `Ad_U(Delta) cap Ad_V(Delta)` is `C_1`. `LevelPair.intersection(p)` gives
  `C_p` with its centre, blocks and trace.
- **Factoriality.** The centre of `P cap Q` is the space of bounded harmonic
  functions of the Markov chain the trace defines on the Bratteli diagram of
  the `C_n`. `analyse` reports the Dobrushin coefficient of that chain and the
  diagram's connected components. A component reaching every level carries a
  projection central in every `C_n`.
- **Index.** If `[M : P cap Q] < oo`, then `e_P` and `e_Q` lie in a
  finite-dimensional relative commutant and the spectrum is finite. A growing
  number of distinct eigenvalues rules finite index out. That number is a
  certified lower bound whenever the gaps between them exceed the rounding
  error, which is about `1e-13` in double precision.
  `||Lambda(C_n < A_{n,1})||^2` is also reported. It equals the index at every
  level in every finite-index case checked, and growth points to infinite
  index.

Validated on the subfactor itself (`s = t`: `C_n = A_{n,0}`, `||Lambda||^2 = delta^2`),
and against the paper's Theorem B (`U = F_4`, `V = D U`, which is `F_4 D` in
this library's convention): `|H| = dim C_1`, the index `16 / |H|` at every
level, a finite spectrum, and a factor.

For S(1,1,2,2), `s = pi/3` against every `s'` tried:
- `dim C_n` is 6, 16, 47, 146, 471, the same for all `s' != s`;
- the number of distinct angles is 3, 6, 13, 30, 62, with gaps of at least
  `2.6e-5`;
- `||Lambda||^2` grows by about 2.1–2.3 per level;
- the diagram splits at every level into `p_{d2}` and `p_A + p_{a2}`, the
  minimal central projections of `A_{0,0}`.

Through `n = 5`, that is the pattern of an infinite-index intersection with
the central projection `p_{d2}`: evidence, not a proof. The largest angle
cosine below 1 depends on `s'` (0.9817 for `s' = pi/2`) and has not moved
since `n = 1` there.

Dense levels in double precision: `n = 5` takes 18 s on the GPU with
`backend="cupy"`. `LevelPair.angle_top` goes further without forming the
cross-Gram matrix, but only gives `dim C_n` and the largest angle below 1.

## Tests

```
python test_pathalg.py            # 33 tests, ~2 minutes
python test_pathalg.py --slow     # adds the heavy exact A_4 cases and AH n = 5
```

The ones with an answer from outside the library:

- Ocneanu's `A_n` connection is unitary and bi-unitary as computed here.
- The `F_N` Fourier spin model is the `Z/N` group-subfactor commuting square:
  `dim A_{0,1}' cap A_{n,0}` comes out `1, N, N^2, ...`.
- For the flat `A_m` connection, `dim A_{0,1}' cap A_{n,0}` equals the number of
  closed walks of length `2n` at the endpoint of the Dynkin diagram — `1,2,4,8`
  for `A_3`, `1,2,5,13` for `A_4`, `1,2,5` for `A_5`.
- The cheap relative-commutant basis matches a brute force linear solve on every
  algebra of every level tested.

## Caveats

- Graphs must be **simple** (paths are recorded by their vertices only), and
  connected — `pf_bipartite` raises if the Perron-Frobenius vector is not
  strictly positive.
- Symbolic *parameters* in connection entries are not really supported: zero
  testing assumes algebraic numbers. Declare any symbols real and expect the
  slow `simplify` path.
- The GPU backends are floating point only; `backend="sympy"` stays on the CPU
  and keeps the original `k x k` Gram route.
- When `|V00| = 1`, `A_{0,0}` is the scalars and `A_{0,0}' cap A_{n,1}` is all
  of `A_{n,1}` — correct, but not informative. The higher relative commutant
  you probably want is `relative_commutant_in((0,1), (n,0))`.
