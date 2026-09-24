"""The eigenvalue-1 eigenspace of a positive contraction, without forming it.

Used by :mod:`tower` for higher relative commutants that are too large to
handle densely.  There the operator is ``N = F* F`` with ``F`` the principal
angle map of a pair of subspaces, so ``N`` is Hermitian with spectrum in
``[0, 1]``, and the answer is its eigenvalue-1 eigenspace -- whose dimension
``r`` is small even when ``N`` itself is enormous.  ``N`` is only ever applied
to a block of vectors, so memory is ``O(k r)`` rather than ``O(k^2)``.

The method is Chebyshev-filtered subspace iteration (Zhou and Saad): a block of
``m > r`` vectors is repeatedly multiplied by a Chebyshev polynomial of ``N``
that is small on ``[0, a]`` and large at 1, then re-orthonormalised and
Rayleigh-Ritz projected.  ``a`` tracks the smallest Ritz value, so everything
outside the block is damped and the wanted vectors converge geometrically at a
rate set by the gap below 1 -- the same margin the dense route reports.
"""

from __future__ import annotations

import math
import time

__all__ = ["unit_eigenspace"]


def unit_eigenspace(apply, k, backend, *, cut, block, chunk, degree=10,
                    max_outer=100, res_tol=None, seed=0, grow_check=None,
                    progress=None):
    """Orthonormal basis of the eigenvalue-1 eigenspace of ``N``.

    Parameters
    ----------
    apply : callable
        ``apply(V) -> N @ V`` for a ``(k, c)`` array ``V`` with ``c <= chunk``.
    k : int
        Dimension of the space ``N`` acts on.
    backend : ArrayBackend
        Supplies the array module, the dtype and ``to_host``.
    cut : float
        Ritz values at or above ``cut`` count as eigenvalue 1.
    block : int
        Initial number of vectors.  Must exceed the answer's dimension to
        finish without growing; if it does not, the block grows by a quarter at
        a time, calling ``grow_check(new_size)`` first -- which may raise, for
        instance when the larger block would not fit in memory.
    chunk : int
        Most columns handed to ``apply`` at once.
    progress : callable, optional
        Called once per round with a dict -- ``round``, ``block``, ``found``
        (Ritz values at 1 so far), ``residual`` (largest among those),
        ``next`` (the first unwanted Ritz value's square root, i.e. the margin)
        and ``seconds`` -- so that long runs can show what they are doing.

    Returns
    -------
    (X, info)
        ``X`` is ``(k, r)`` with orthonormal columns; ``info`` records the
        iterations, the final block size, the largest residual, and ``gap``:
        the largest discarded and smallest kept square roots of Ritz values,
        on the same scale as the dense route's principal angle cosines.
    """
    xp = backend.xp
    if res_tol is None:
        # a Ritz value's error is about residual^2 / gap, so this puts it well
        # inside the backend's tolerance for any gap of order 0.1
        res_tol = 0.1 * math.sqrt(backend.tol)
    m = int(min(k, max(1, block)))
    applies = 0

    def apply_block(V, out=None):
        nonlocal applies
        if out is None:
            out = xp.empty_like(V)
        for j in range(0, V.shape[1], chunk):
            out[:, j : j + chunk] = apply(V[:, j : j + chunk])
        applies += V.shape[1]
        return out

    def rayleigh_ritz(holder):
        # The block arrives in a one-element list and is popped, so this frame
        # holds the only reference to it.  It is then rotated onto the Ritz
        # vectors in place, together with its image, so the peak is two k x m
        # blocks: Q and N Q.
        Q = holder.pop()
        NQ = apply_block(Q)
        H = Q.conj().T @ NQ
        w, S = xp.linalg.eigh((H + H.conj().T) / 2)
        S = xp.ascontiguousarray(S[:, ::-1])  # descending
        _right_multiply_inplace(xp, Q, S)
        _right_multiply_inplace(xp, NQ, S)
        theta = backend.to_host(w).real[::-1].copy()
        wdev = xp.asarray(theta, dtype=backend.real_dtype)
        res = []
        for j in range(0, Q.shape[1], chunk):
            R = NQ[:, j : j + chunk] - Q[:, j : j + chunk] * wdev[j : j + chunk]
            res.append(backend.to_host(xp.linalg.norm(R, axis=0)).real)
        del NQ
        return Q, theta, _concat(res)

    # Ritz values at or above `near` are on their way to 1.  It is looser than
    # `cut` on purpose: it decides when the block is full, and has to fire
    # before those vectors have converged all the way -- otherwise the damping
    # bound below sits just under 1, the filter stops damping anything, and
    # the iteration stalls.  Unwanted eigenvalues are expected well below it.
    near = 1 - max(1e-3, 10 * (1 - cut))

    # How much one filter pass may amplify the wanted directions relative to
    # the rest.  Past about 1/sqrt(eps) the other vectors in the block drop
    # below rounding error, come back from the QR as noise, and never converge;
    # capping the degree keeps them representable.
    single = backend.single
    max_gain = 1e3 if single else 1e6
    # the first unwanted Ritz value has to settle to this before stopping
    settle = 1e-5 if single else 1e-7

    started = time.time()
    X, theta, res = rayleigh_ritz([_orthonormalise(xp, _random_block(xp, backend.dtype, k, m, seed))])
    prev = None  # (r, the first unwanted Ritz value) from the round before

    for outer in range(max_outer):
        m = X.shape[1]
        buf = 0 if m == k else min(m - 1, max(8, m // 16))
        top = m - buf
        r = int((theta >= cut).sum())
        if progress is not None:
            progress({
                "round": outer,
                "block": m,
                "found": r,
                "residual": float(res[:r].max()) if r else 0.0,
                "next": math.sqrt(max(float(theta[min(r, m - 1)]), 0.0)),
                "seconds": time.time() - started,
            })
        if m < k and int((theta >= near).sum()) >= top:
            # every vector in the block is heading for 1, so the answer may be
            # bigger than the block: add fresh directions and carry on
            new_m = min(k, max(m + 2 * max(buf, 8), int(1.25 * m)))
            if grow_check is not None:
                grow_check(new_m)
            extra = _random_block(xp, backend.dtype, k, new_m - m, seed + outer + 1)
            joined = xp.concatenate([X, extra], axis=1)
            del X, extra
            holder = [_orthonormalise(xp, joined)]
            del joined
            X, theta, res = rayleigh_ritz(holder)
            prev = None
            continue
        if m == k:
            break  # the block is the whole space: Rayleigh-Ritz was exact
        # Stop when the wanted vectors have converged and the first unwanted
        # Ritz value has settled below `near`.  Subspace iteration finds the
        # dominant directions first, so a missing eigenvalue-1 vector would
        # show up as that value still climbing -- which is what the settling
        # test excludes.  The other unwanted vectors need not converge at all.
        first_unwanted = float(theta[r])
        if (
            bool((res[:r] <= res_tol).all())
            and first_unwanted < near
            and prev is not None
            and prev[0] == r
            and abs(first_unwanted - prev[1]) <= settle
        ):
            break
        prev = (r, first_unwanted)
        # damp [0, a]; a is the smallest Ritz value, which approximates the
        # m-th eigenvalue from below, so everything outside the block is
        # damped.  The block is not full, so a < near < 1.
        a = max(float(theta[-1]), 0.05)
        tau = 2 / a - 1  # where 1 lands after [0, a] is mapped onto [-1, 1]
        d = max(1, min(degree, int(math.acosh(max_gain) / math.acosh(tau))))
        applies += d * X.shape[1]
        Y = _chebyshev(apply, X, d, a, chunk, xp)  # overwrites X's storage
        del X
        holder = [_orthonormalise(xp, Y)]
        del Y
        X, theta, res = rayleigh_ritz(holder)
    else:
        raise RuntimeError(
            f"eigenvalue-1 subspace did not converge in {max_outer} rounds "
            f"(block {X.shape[1]}, largest residual {float(res.max()):.2e}, "
            f"tolerance {res_tol:.1e})"
        )

    r = int((theta >= cut).sum())
    info = {
        "iterations": outer + 1,
        "block": X.shape[1],
        "applications": applies,
        "residual": float(res[:r].max()) if r else 0.0,
        "gap": (
            math.sqrt(max(theta[r], 0.0)) if r < theta.size else 0.0,
            math.sqrt(max(theta[r - 1], 0.0)) if r else 1.0,
        ),
        # square roots of every Ritz value whose vector converged, in
        # decreasing order: the top of the spectrum, not only its edge
        "ritz": [math.sqrt(max(float(x), 0.0))
                 for x, e in zip(theta, res) if e <= res_tol],
    }
    return X[:, :r], info


def _chebyshev(apply, X, degree, a, chunk, xp):
    """``T_d(t(N)) X / T_d(t(1))`` with ``t`` mapping ``[0, a]`` onto ``[-1, 1]``.

    Normalised at 1, so the wanted directions keep unit size while the rest
    shrink by ``1 / T_d(t(1))``.  The three-term recurrence runs in two blocks:
    each new iterate overwrites the one from two steps back, column chunk by
    column chunk.  ``X`` itself is overwritten.
    """
    centre = half = a / 2
    tau = (1 - centre) / half  # t(1) > 1
    Yp = X
    Yc = xp.empty_like(X)
    for j in range(0, X.shape[1], chunk):
        cols = slice(j, j + chunk)
        Yc[:, cols] = (apply(Yp[:, cols]) - centre * Yp[:, cols]) / (half * tau)
    s_prev = 1 / tau
    for _ in range(2, degree + 1):
        s = 1 / (2 * tau - s_prev)
        for j in range(0, X.shape[1], chunk):
            cols = slice(j, j + chunk)
            NY = apply(Yc[:, cols])
            Yp[:, cols] = (2 * s / half) * (NY - centre * Yc[:, cols]) - (s * s_prev) * Yp[:, cols]
        Yp, Yc = Yc, Yp
        s_prev = s
    return Yc


def _orthonormalise(xp, Y):
    """Orthonormal columns spanning those of ``Y``, overwriting ``Y`` if it can.

    CholeskyQR2: ``R`` from the Cholesky factor of the small Gram matrix
    ``Y* Y``, then ``Y <- Y R^-1`` a few rows at a time, done twice.  It works
    in place, where the library QR holds three ``k x m`` blocks at its peak (the
    input, a transposed copy, and the output) -- the difference between ``n = 8``
    fitting on a 12 GB card or not.  It needs ``Y`` to be reasonably
    conditioned, which the gain cap on the Chebyshev filter guarantees; the
    first pass leaves an orthogonality error of about ``eps cond(Y)^2`` and the
    second removes it.  If the factorisation fails anyway the library QR is
    used instead.
    """
    for _ in range(2):
        G = Y.conj().T @ Y
        try:
            L = xp.linalg.cholesky((G + G.conj().T) / 2)  # G = L L*, R = L*
        except Exception:  # noqa: BLE001 -- numpy raises, cupy may not
            L = None
        if L is None or not bool(xp.isfinite(L).all()):
            return xp.linalg.qr(Y)[0]
        _right_multiply_inplace(xp, Y, xp.linalg.inv(L.conj().T))
    return Y


def _right_multiply_inplace(xp, Y, S, budget=64e6):
    """``Y <- Y S`` for square ``S``, a band of rows at a time.

    Row ``i`` of ``Y S`` needs only row ``i`` of ``Y``, so the product can be
    written back over ``Y`` with nothing larger than one band as scratch.
    """
    k, m = Y.shape
    rows = max(1, int(budget // max(1, m * Y.itemsize)))
    for r0 in range(0, k, rows):
        Y[r0 : r0 + rows] = Y[r0 : r0 + rows] @ S


def _random_block(xp, dtype, k, m, seed):
    """A ``(k, m)`` complex Gaussian block, generated a few columns at a time."""
    rs = xp.random.RandomState(seed)
    out = xp.empty((k, m), dtype=dtype)
    for j in range(0, m, 64):
        c = min(64, m - j)
        out[:, j : j + c] = rs.standard_normal((k, c)) + 1j * rs.standard_normal((k, c))
    return out


def _concat(parts):
    import numpy as np

    return np.concatenate(parts) if parts else np.zeros(0)
