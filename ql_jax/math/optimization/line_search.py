"""Line search methods: Armijo backtracking and Goldstein conditions."""

import jax
import jax.numpy as jnp


def armijo_backtracking(f, x, direction, gradient=None,
                        alpha_init=1.0, c1=1e-4, rho=0.5, max_iter=50):
    """Armijo backtracking line search.

    Find alpha such that f(x + alpha*d) <= f(x) + c1*alpha*grad.d

    Parameters
    ----------
    f : callable(x) -> float
    x : array – current point
    direction : array – search direction
    gradient : array – gradient at x (computed if None)
    alpha_init : float – initial step size
    c1 : float – Armijo parameter (sufficient decrease)
    rho : float – backtracking factor
    max_iter : int

    Returns float alpha.
    """
    if gradient is None:
        gradient = jax.grad(f)(x)

    f0 = f(x)
    slope = jnp.dot(gradient, direction)
    alpha = alpha_init

    for _ in range(max_iter):
        if f(x + alpha * direction) <= f0 + c1 * alpha * slope:
            return alpha
        alpha *= rho

    return alpha


def goldstein(f, x, direction, gradient=None,
              alpha_init=1.0, c=0.25, rho_lo=0.5, rho_hi=2.0, max_iter=50):
    """Goldstein line search.

    Find alpha satisfying:
      f(x) + (1-c)*alpha*slope <= f(x+alpha*d) <= f(x) + c*alpha*slope

    Parameters
    ----------
    f : callable(x) -> float
    x : current point
    direction : search direction
    gradient : gradient at x
    c : Goldstein parameter in (0, 0.5)
    """
    if gradient is None:
        gradient = jax.grad(f)(x)

    f0 = f(x)
    slope = jnp.dot(gradient, direction)
    alpha = alpha_init
    lo, hi = 0.0, jnp.inf

    for _ in range(max_iter):
        f_new = f(x + alpha * direction)
        if f_new > f0 + c * alpha * slope:
            hi = alpha
            alpha = 0.5 * (lo + hi)
        elif f_new < f0 + (1.0 - c) * alpha * slope:
            lo = alpha
            alpha = jnp.where(jnp.isinf(hi), rho_hi * alpha, 0.5 * (lo + hi))
        else:
            return alpha

    return alpha
