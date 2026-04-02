"""Finite-difference operators – differential operators on 1D grids."""

import jax.numpy as jnp
import jax
from dataclasses import dataclass


@dataclass(frozen=True)
class TridiagonalOperator:
    """Tridiagonal operator stored as three diagonals.

    Represents a matrix A where:
        A[i,i-1] = lower[i-1], A[i,i] = diag[i], A[i,i+1] = upper[i]

    Parameters
    ----------
    lower : array(n-1) – sub-diagonal
    diag : array(n) – main diagonal
    upper : array(n-1) – super-diagonal
    """
    lower: jnp.ndarray
    diag: jnp.ndarray
    upper: jnp.ndarray

    @property
    def size(self):
        return len(self.diag)

    def apply(self, v):
        """Apply operator: result = A @ v."""
        n = self.size
        result = self.diag * v
        result = result.at[1:].add(self.lower * v[:-1])
        result = result.at[:-1].add(self.upper * v[1:])
        return result

    def solve(self, rhs):
        """Solve A x = rhs using Thomas algorithm."""
        return tridiag_solve(self.lower, self.diag, self.upper, rhs)

    def add_scalar(self, alpha):
        """Return new operator with alpha added to diagonal."""
        return TridiagonalOperator(self.lower, self.diag + alpha, self.upper)

    def scale(self, alpha):
        """Return operator scaled by alpha."""
        return TridiagonalOperator(
            alpha * self.lower, alpha * self.diag, alpha * self.upper
        )

    def add(self, other):
        """Add two tridiagonal operators."""
        return TridiagonalOperator(
            self.lower + other.lower,
            self.diag + other.diag,
            self.upper + other.upper,
        )


def identity_operator(n):
    """Identity tridiagonal operator of size n."""
    return TridiagonalOperator(
        jnp.zeros(n - 1), jnp.ones(n), jnp.zeros(n - 1)
    )


def d_plus(x):
    """Forward difference operator D+ on grid x.

    (D+ f)[i] = (f[i+1] - f[i]) / (x[i+1] - x[i])
    """
    dx = jnp.diff(x)
    n = len(x)
    lower = jnp.zeros(n - 1)
    diag = jnp.zeros(n)
    upper = jnp.zeros(n - 1)

    # Interior: (f[i+1] - f[i]) / dx[i]
    diag = diag.at[:-1].set(-1.0 / dx)
    upper = upper.at[:].set(1.0 / dx)

    return TridiagonalOperator(lower, diag, upper)


def d_minus(x):
    """Backward difference operator D- on grid x.

    (D- f)[i] = (f[i] - f[i-1]) / (x[i] - x[i-1])
    """
    dx = jnp.diff(x)
    n = len(x)
    lower = jnp.zeros(n - 1)
    diag = jnp.zeros(n)
    upper = jnp.zeros(n - 1)

    # Interior
    diag = diag.at[1:].set(1.0 / dx)
    lower = lower.at[:].set(-1.0 / dx)

    return TridiagonalOperator(lower, diag, upper)


def d_plus_d_minus(x):
    """Second derivative operator D+D- (central second difference).

    (D+D- f)[i] = 2/(dx_m + dx_p) * [f[i+1]/dx_p - f[i]*(1/dx_p + 1/dx_m) + f[i-1]/dx_m]
    """
    n = len(x)
    dx = jnp.diff(x)

    lower = jnp.zeros(n - 1)
    diag = jnp.zeros(n)
    upper = jnp.zeros(n - 1)

    # Interior points i=1..n-2
    dx_m = dx[:-1]  # x[i] - x[i-1]
    dx_p = dx[1:]   # x[i+1] - x[i]
    h = 0.5 * (dx_m + dx_p)

    lower = lower.at[1:].set(1.0 / (dx_m * 2.0 * h))
    diag = diag.at[1:-1].set(-1.0 / (dx_p * h) - 1.0 / (dx_m * h))

    # Pad upper for alignment  
    upper_interior = 1.0 / (dx_p * 2.0 * h)
    upper = upper.at[:-1].set(upper_interior)

    return TridiagonalOperator(lower, diag, upper)


def d_zero(x):
    """Central difference operator D0.

    (D0 f)[i] = (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
    """
    n = len(x)
    dx = jnp.diff(x)

    lower = jnp.zeros(n - 1)
    diag = jnp.zeros(n)
    upper = jnp.zeros(n - 1)

    # Interior points
    dx_m = dx[:-1]
    dx_p = dx[1:]
    h2 = dx_m + dx_p

    lower = lower.at[1:].set(-1.0 / h2)
    upper = upper.at[:-1].set(1.0 / h2)

    return TridiagonalOperator(lower, diag, upper)


def tridiag_solve(lower, diag, upper, rhs):
    """Solve tridiagonal system using Thomas algorithm via jax.lax.scan.

    Parameters
    ----------
    lower : (n-1,) sub-diagonal
    diag : (n,) main diagonal
    upper : (n-1,) super-diagonal
    rhs : (n,) right-hand side

    Returns
    -------
    x : (n,) solution
    """
    n = len(diag)

    # Forward sweep
    c0 = upper[0] / diag[0]
    d0 = rhs[0] / diag[0]

    upper_scan = jnp.concatenate([upper[1:], jnp.array([0.0])])

    def forward_step(carry, inputs):
        c_prev, d_prev = carry
        l_i, diag_i, up_i, rhs_i = inputs
        denom = diag_i - l_i * c_prev
        c_new = up_i / denom
        d_new = (rhs_i - l_i * d_prev) / denom
        return (c_new, d_new), (c_new, d_new)

    (_, _), (c_arr, d_arr) = jax.lax.scan(
        forward_step, (c0, d0),
        (lower, diag[1:], upper_scan, rhs[1:])
    )

    c_prime = jnp.concatenate([jnp.array([c0]), c_arr])
    d_prime = jnp.concatenate([jnp.array([d0]), d_arr])

    # Back substitution
    def backward_step(x_next, inputs):
        c_i, d_i = inputs
        x_i = d_i - c_i * x_next
        return x_i, x_i

    x_last = d_prime[-1]
    _, x_rev = jax.lax.scan(
        backward_step, x_last,
        (c_prime[:-1], d_prime[:-1]),
        reverse=True,
    )

    return jnp.concatenate([x_rev, jnp.array([x_last])])
