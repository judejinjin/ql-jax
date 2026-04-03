"""Matrix utilities: iterative solvers, decompositions, and matrix functions.

Fills gaps relative to QuantLib's matrixutilities/ module.
JAX already provides SVD, Cholesky, and eigendecomposition natively.
"""

from __future__ import annotations

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# BiCGStab – Biconjugate Gradient Stabilized
# ---------------------------------------------------------------------------

def bicgstab(A, b, x0=None, tol=1e-10, max_iter=1000):
    """Solve Ax = b via BiCGStab (for non-symmetric sparse-like systems).

    Parameters
    ----------
    A : array (n, n) or callable A(x) -> Ax
    b : array (n,)
    x0 : array (n,) initial guess
    tol : float
    max_iter : int

    Returns
    -------
    x : array (n,)
    """
    b = jnp.asarray(b, dtype=jnp.float64)
    n = b.shape[0]
    if x0 is None:
        x = jnp.zeros(n, dtype=jnp.float64)
    else:
        x = jnp.asarray(x0, dtype=jnp.float64)

    matvec = A if callable(A) else lambda v: A @ v

    r = b - matvec(x)
    r_hat = r.copy()
    rho = alpha = omega = 1.0
    v = jnp.zeros(n, dtype=jnp.float64)
    p = jnp.zeros(n, dtype=jnp.float64)

    for _ in range(max_iter):
        rho_new = float(jnp.dot(r_hat, r))
        if abs(rho_new) < 1e-30:
            break
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = matvec(p)
        alpha = rho_new / float(jnp.dot(r_hat, v))
        s = r - alpha * v
        if float(jnp.linalg.norm(s)) < tol:
            x = x + alpha * p
            break
        t = matvec(s)
        omega = float(jnp.dot(t, s)) / float(jnp.dot(t, t))
        x = x + alpha * p + omega * s
        r = s - omega * t
        rho = rho_new
        if float(jnp.linalg.norm(r)) < tol:
            break
    return x


# ---------------------------------------------------------------------------
# GMRES – Generalized Minimal Residual
# ---------------------------------------------------------------------------

def gmres(A, b, x0=None, tol=1e-10, max_iter=100, restart=None):
    """Solve Ax = b via restarted GMRES.

    Parameters
    ----------
    A : array (n, n) or callable A(x) -> Ax
    b : array (n,)
    x0 : initial guess
    tol : float
    max_iter : int – maximum Arnoldi iterations per restart
    restart : int – restart frequency (None = no restart)

    Returns
    -------
    x : array (n,)
    """
    b = jnp.asarray(b, dtype=jnp.float64)
    n = b.shape[0]
    if x0 is None:
        x = jnp.zeros(n, dtype=jnp.float64)
    else:
        x = jnp.asarray(x0, dtype=jnp.float64)

    matvec = A if callable(A) else lambda v: A @ v
    m = restart if restart is not None else max_iter

    for _ in range(max(1, max_iter // m)):
        r = b - matvec(x)
        beta = float(jnp.linalg.norm(r))
        if beta < tol:
            return x

        # Arnoldi process
        V = [r / beta]
        H = jnp.zeros((m + 1, m), dtype=jnp.float64)

        for j in range(min(m, n)):
            w = matvec(V[j])
            for i in range(j + 1):
                hij = float(jnp.dot(V[i], w))
                H = H.at[i, j].set(hij)
                w = w - hij * V[i]
            hjj = float(jnp.linalg.norm(w))
            H = H.at[j + 1, j].set(hjj)
            if hjj < 1e-14:
                # Lucky breakdown
                m_eff = j + 1
                break
            V.append(w / hjj)
        else:
            m_eff = min(m, n)

        # Solve least squares: min ||H_m y - beta e_1||
        e1 = jnp.zeros(m_eff + 1, dtype=jnp.float64).at[0].set(beta)
        H_sub = H[:m_eff + 1, :m_eff]
        y, *_ = jnp.linalg.lstsq(H_sub, e1, rcond=None)

        # Update solution
        Vm = jnp.stack(V[:m_eff], axis=1)
        x = x + Vm @ y

        if float(jnp.linalg.norm(b - matvec(x))) < tol:
            break

    return x


# ---------------------------------------------------------------------------
# QR decomposition (Householder)
# ---------------------------------------------------------------------------

def qr_decomposition(A):
    """QR decomposition via JAX (Householder reflections).

    Returns
    -------
    Q, R : arrays
    """
    return jnp.linalg.qr(jnp.asarray(A, dtype=jnp.float64))


# ---------------------------------------------------------------------------
# Matrix exponential
# ---------------------------------------------------------------------------

def matrix_exponential(A):
    """Matrix exponential exp(A) via JAX/scipy.

    Returns
    -------
    expA : array
    """
    from jax.scipy.linalg import expm
    return expm(jnp.asarray(A, dtype=jnp.float64))


# ---------------------------------------------------------------------------
# Moore-Penrose pseudoinverse
# ---------------------------------------------------------------------------

def pseudoinverse(A):
    """Moore-Penrose pseudoinverse."""
    return jnp.linalg.pinv(jnp.asarray(A, dtype=jnp.float64))


# ---------------------------------------------------------------------------
# Factor reduction (PCA for correlation matrices)
# ---------------------------------------------------------------------------

def factor_reduction(corr_matrix, n_factors: int):
    """Reduce a correlation matrix to n_factors using PCA.

    Parameters
    ----------
    corr_matrix : (n, n) array – correlation matrix
    n_factors : int – number of factors to keep

    Returns
    -------
    reduced : (n, n_factors) – factor loadings
    """
    C = jnp.asarray(corr_matrix, dtype=jnp.float64)
    eigvals, eigvecs = jnp.linalg.eigh(C)
    # eigh returns ascending order; take the largest n_factors
    idx = jnp.argsort(-eigvals)[:n_factors]
    return eigvecs[:, idx] * jnp.sqrt(eigvals[idx])


# ---------------------------------------------------------------------------
# Autocovariance / autocorrelation
# ---------------------------------------------------------------------------

def autocovariance(x, max_lag: int = None):
    """Compute autocovariance function for lags 0..max_lag.

    Returns
    -------
    acov : array of shape (max_lag+1,)
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    n = x.shape[0]
    if max_lag is None:
        max_lag = n - 1
    mean = jnp.mean(x)
    xc = x - mean
    result = []
    for lag in range(max_lag + 1):
        if lag >= n:
            result.append(0.0)
        else:
            result.append(float(jnp.sum(xc[:n - lag] * xc[lag:]) / n))
    return jnp.array(result)


def autocorrelation(x, max_lag: int = None):
    """Compute autocorrelation function (normalized autocovariance)."""
    acov = autocovariance(x, max_lag)
    gamma0 = acov[0]
    if gamma0 == 0:
        return jnp.zeros_like(acov)
    return acov / gamma0
