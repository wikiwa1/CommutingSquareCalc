# Exceptional parameters for the S(1,1,2,2) family

The parameter is `s` in `ah_demo.ah_square(s)`, with the auxiliary angle

    t(s) = (s - arccos(cos(s)/delta^2))/2,
    delta^2 = (5 + sqrt(17))/2.

This is a question about the individual subfactor, not the intersection of two subfactors.

## What the sources establish

Bisch–Cáceres, *New hyperfinite subfactors with infinite depth*,
[arXiv:2509.00227v2](https://arxiv.org/pdf/2509.00227v2),
Corollary 5.3 and Theorem 5.4, use a countability argument to establish
A-infinity examples. The paragraph following Theorem 5.4, on printed
page 49, explicitly says that this argument does not determine the
behavior at different parameters. It does not supply an exceptional list.

At this index, the irreducible alternatives are the TLJ standard invariant
(principal graph A-infinity) and the Asaeda–Haagerup standard invariant or
its dual. Crucially, AH is **5-supertransitive**. See Jones,
[*Quadratic Tangles in Planar Algebras*](https://math.berkeley.edu/~vfr/quadtangle4.pdf),
Section 3.2 and Example 3.5.9 (annular excess begins with five zeros and then one).

Consequently the first distinguishing relative-commutant dimension is:

| n | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| TLJ | 1 | 2 | 5 | 14 | 42 | 132 |
| AH or its dual | 1 | 2 | 5 | 14 | 42 | 133 |

The old claim in ah_demo.py that level 5 distinguishes them was wrong.
The comments and output interpretation have been corrected. The numerical
algorithms and connection formulas were not changed.

## An exact criterion and a finiteness argument

Let D_6(s) be dim(A_(0,1)' intersect A_(6,0)^(s)). Under the classification
above, the finite-depth parameters are exactly

    E = {s in [0,2*pi) : D_6(s) = 133}.

In particular, an **exactly certified** value D_6(s)=132 is sufficient to
deduce A-infinity at this index. This deduction uses the classification;
agreement with a finite Catalan prefix is not a general all-level criterion.

The following finiteness argument is a deduction from the classification
and this explicit analytic family, not a parameter theorem quoted from
Bisch–Cáceres.

Choose a fixed basis of the parameter-independent K-picture algebra at
level 6, of dimension b. Its image in the H-picture is W(s)* x W(s),
where W(s) is the connection transport. Impose the linear equations

    [a_i, W(s)* x W(s)] = 0

for a fixed basis (a_i) of A_(0,1). Write their coefficient matrix as L(s).
Then nullity L(s) = D_6(s), and its entries are real analytic in s. Indeed,
the square roots in ah_params have strictly positive arguments for every
real s, since delta^2>1; the phase denominators never vanish. The finite
transport matrices inherit this analytic dependence.

The existence theorem supplies a parameter with D_6=132, so some
(b-132)-square minor f(s) of L(s) is nonzero there. Every parameter in E
must satisfy f(s)=0. A nonzero real analytic function has isolated zeros;
because the formulas extend analytically across both endpoints of
[0,2*pi], it has only finitely many zeros on that compact interval.
Thus E is finite (possibly empty). Computing its elements requires
locating and certifying the simultaneous rank drops, not merely invoking
the countability argument. The argument does not give their number or values.

## Numerical investigation in this session

The focused scanner ah_parameter_scan.py computes level 6 in complex128
using the iterative eigenspace routine. Its output is saved in
ah_parameter_scan_results.json.

- s=0: dimension 132, numerical evidence for TLJ. The reported discarded
  and kept principal-angle cosines were approximately 0.944157 and 1.
  The maximum residual for the kept directions was about 6.7e-12.
- s=pi/2: the run exceeded its 150-second allowance before returning a
  converged result. No classification is assigned to this parameter.

No finite-depth parameter has been identified or certified by these runs.
They are not an exhaustive search. A floating-point dimension is not an
exact rank certificate, even when the computed gap is substantial.

From the project root in PowerShell, for example:

    $env:OPENBLAS_NUM_THREADS='1'
    & ./.venv/Scripts/python.exe -B -u Code/ah_parameter_scan.py --parameters 0 pi/2 pi 3*pi/2 pi/3 --seconds 600

The time allowance is checked between iterations, not enforced as a hard
wall-clock limit. For any apparent dimension 133, independently confirm
the eigenspace and then solve or certify the corresponding rank equations
before reporting an exact exceptional parameter. Missing a rank drop on
a finite grid does not prove there are no exceptions between grid points.
