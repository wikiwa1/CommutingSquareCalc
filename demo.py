"""A worked example: build a commuting square by hand and compute in its tower.

Run with ``python demo.py``.

The square used is the spin model of the 2x2 Fourier (Hadamard) matrix,

        C            <  Delta_2
        ^                  ^
        u Delta_2 u* <  M_2

written out longhand so every input the library needs is visible.  It is the
Z/2 group-subfactor commuting square, so its higher relative commutants have
dimensions 1, 2, 4, 8, ... which the last section reproduces.
"""

from __future__ import annotations

import sympy as sp

from pathalg import CommutingSquare, Level

# ---------------------------------------------------------------- 1. the square
#
#         a ---- b          V00 = {*}          V01 = {b0, b1}
#         |      |          V10 = {c0, c1}     V11 = {@}
#         c ---- d
#
# Every graph is a star, so beta_h = beta_v = sqrt(2).

H = sp.Matrix([[1, 1], [1, -1]])  # the 2x2 Hadamard matrix

square = CommutingSquare(
    V00=["*"],
    V01=["b0", "b1"],
    V10=["c0", "c1"],
    V11=["@"],
    h0=[("*", "b0"), ("*", "b1")],            # V00 - V01
    v0=[("*", "c0"), ("*", "c1")],            # V00 - V10
    h1=[("c0", "@"), ("c1", "@")],            # V10 - V11
    v1=[("b0", "@"), ("b1", "@")],            # V01 - V11
    # u^{a,d}_{c,b} for the cell (a, b, c, d); here a = '*' and d = '@' always
    u_cells={
        ("*", f"b{j}", f"c{i}", "@"): H[i, j] / sp.sqrt(2)
        for i in range(2)
        for j in range(2)
    },
)

print("=" * 72)
print("1. the commuting square and its connection")
print("=" * 72)
print(square)
print("  Perron-Frobenius weights:")
for tag in ("00", "01", "10", "11"):
    print(f"    V{tag}: {square.mu[tag]}")
print()
for result in square.verify(verbose=False):
    print(" ", result)

print("\n  the v half, derived from u by the renormalisation rule:")
for cell in square.cells():
    print(f"    cell {cell}:  u = {square.u(*cell)}    v = {square.v(*cell)}")


# ---------------------------------------------------------------- 2. one level
N = 3
level = Level(square, N)

print("\n" + "=" * 72)
print(f"2. level n = {N}: everything represented on H_{N + 1}")
print("=" * 72)
print(f"  {level}")
print(f"  H_{N + 1} runs through {' - '.join('V' + t for t in level.H.types)}")
print(f"  K_{N + 1} runs through {' - '.join('V' + t for t in level.K.types)}")
print(f"  basis of H_{N + 1}: {level.H.paths}")
print(f"  blocks (by end vertex): {level.H.block_dims()}")
print()
for result in level.verify(verbose=False):
    print(" ", result)


# ----------------------------------------------------- 3. elements of the three algebras
print("\n" + "=" * 72)
print("3. elements of A_(n,1), A_(n-1,1) and A_(n,0), all acting on H_(n+1)")
print("=" * 72)

# a matrix unit of A_{n,1}: a pair of full length paths ending at the same vertex
blk = max(level.H.blocks.values(), key=len)
xi, eta = blk[0], blk[1]
x = level.matrix_unit(xi, eta)
print(f"  x = E[{xi} , {eta}]  in A_({N},1)")
print(f"    trace {x.trace()},  blocks {[k for k in x.blocks]}")

# a matrix unit of A_{n-1,1}: paths one edge shorter, embedded automatically.
# Pairs are taken from one block, since a matrix unit needs a common end vertex.
sub = max(level.H.level_blocks[N].values(), key=len)
y = level.matrix_unit(sub[0], sub[-1])
print(f"  y = E[{sub[0]} , {sub[-1]}]  in A_({N - 1},1)")
print(f"    trace {y.trace()}")

# a matrix unit of A_{n,0}: written in the K picture, pulled back through W
ksub = max(level.K.level_blocks[N].values(), key=len)
z = level.matrix_unit(ksub[0], ksub[-1], which="K")
print(f"  z = E[{ksub[0]} , {ksub[-1]}]  in A_({N},0), pulled back by W")
print(f"    trace {z.trace()}")

print("\n  they multiply, add and take adjoints in the one representation:")
print(f"    tr(x y)        = {(x * y).trace()}")
print(f"    tr(z z*)       = {(z * z.H).trace()}")
print(f"    [y, z] is zero?  {(y * z - z * y).is_zero()}")

print("\n  conditional expectations (Markov trace preserving):")
print(f"    tr(E_(A_({N - 1},1)) z)  = {level.expectation(z, N - 1, 1).trace()}   (= tr z)")
print(f"    E_(A_({N - 1},1)) z == E_(A_({N - 1},0)) z ?  "
      f"{(level.expectation(z, N - 1, 1) - level.expectation(z, N - 1, 0)).is_zero()}"
      "   <- the commuting square condition")

print("\n  Jones projections of the tower:")
for k in range(2, N + 1):
    f = level.jones_H(k)
    print(f"    f_{k} in the level {k + 1} algebra: projection {(f * f - f).is_zero()},"
          f" trace {f.trace()}  (1/beta_v^2 = {sp.nsimplify(1 / square.beta_v ** 2)})")


# ------------------------------------------------- 4. relative commutants
print("\n" + "=" * 72)
print("4. relative commutants")
print("=" * 72)
for p, q in level.available_algebras():
    basis = level.relative_commutant(p, q)
    print(f"  dim A_({p},{q})' cap A_({N},1) = {level.span_dimension(basis)}")

print("\n  higher relative commutants (Ocneanu compactness):")
for n in range(1, 5):
    lv = Level(square, n, backend="numpy")
    dim = lv.span_dimension(lv.relative_commutant_in((0, 1), (n, 0)))
    print(f"    n = {n}:  dim A_(0,1)' cap A_(n,0) = {dim}   (expected {2 ** (n - 1)})")

print("\ndone.")
