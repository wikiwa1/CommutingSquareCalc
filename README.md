# pathalg — path algebra computations for commuting squares

Explicit, exact computation in the tower of a commuting square of finite
dimensional algebras, from a bi-unitary connection. `A_{n,1}`, `A_{n-1,1}` and
`A_{n,0}` are all represented on the single path space `H_{n+1}`, so operators
from the three algebras can be multiplied, added and compared directly.

Requires Python 3.9+ and `sympy`; `numpy` is optional but recommended (it backs
the fast numerical mode and the rank computations).

```
pathalg/
  square.py     the commuting square, four graphs, connection, verification
  weights.py    exact Perron-Frobenius weights
  paths.py      path spaces, block operators, block maps
  tower.py      one level: W, the three algebras, trace, e_k, E, commutants
  backend.py    exact (sympy) and numerical (numpy) arithmetic
  examples.py   Dynkin and Hadamard squares, ready to use
demo.py         a worked example, start to finish
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
- **`backend="numpy"`** switches the whole level to `complex128`. Use it for
  larger `n`; `test_backends_agree` checks the two agree on traces and
  dimensions.

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

Rough exact timings (this machine): `A_3` at `n = 3` all checks ≈ 20s, all
relative commutants ≈ instant. With `backend="numpy"`, `A_4` at `n = 4`
(`dim H_5 = 21`) ≈ 8s.

## Tests

```
python test_pathalg.py            # 23 tests, ~100s
python test_pathalg.py --slow     # adds the heavy exact A_4 cases
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
- When `|V00| = 1`, `A_{0,0}` is the scalars and `A_{0,0}' cap A_{n,1}` is all
  of `A_{n,1}` — correct, but not informative. The higher relative commutant
  you probably want is `relative_commutant_in((0,1), (n,0))`.
