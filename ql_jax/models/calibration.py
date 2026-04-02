"""Model calibration framework.

Gradient-based calibration using JAX AD for computing Jacobians.
Supports Levenberg-Marquardt and BFGS optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from ql_jax.math.optimization.bfgs import minimize as minimize_bfgs


@dataclass(frozen=True)
class CalibrationResult:
    """Result of model calibration."""
    params: jnp.ndarray
    fun: float
    n_iter: int
    success: bool


def calibrate_least_squares(
    model_price_fn: Callable,
    market_prices: jnp.ndarray,
    market_vols: jnp.ndarray | None,
    initial_params: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    param_bounds: Sequence[tuple] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    use_vols: bool = False,
):
    """Calibrate model parameters by minimizing weighted squared errors.

    Parameters
    ----------
    model_price_fn : callable(params) -> array of model prices/vols
        Function that takes parameter vector and returns model values.
    market_prices : observed market prices
    market_vols : observed market implied vols (optional, used if use_vols=True)
    initial_params : starting parameter values
    weights : calibration weights (default: equal)
    param_bounds : list of (lower, upper) bounds per parameter
    max_iter : max iterations
    tol : convergence tolerance
    use_vols : calibrate to implied vols rather than prices

    Returns
    -------
    CalibrationResult
    """
    targets = market_vols if use_vols and market_vols is not None else market_prices
    targets = jnp.asarray(targets, dtype=jnp.float64)
    n = targets.shape[0]

    if weights is None:
        weights = jnp.ones(n, dtype=jnp.float64)
    else:
        weights = jnp.asarray(weights, dtype=jnp.float64)

    # Transform parameters to unconstrained space if bounds given
    if param_bounds is not None:
        transform, inv_transform = _make_transforms(param_bounds)
        x0 = inv_transform(jnp.asarray(initial_params, dtype=jnp.float64))
    else:
        transform = lambda x: x
        x0 = jnp.asarray(initial_params, dtype=jnp.float64)

    def objective(x):
        params = transform(x)
        model_vals = model_price_fn(params)
        residuals = model_vals - targets
        return 0.5 * jnp.sum(weights * residuals**2)

    result = minimize_bfgs(objective, x0, max_iter=max_iter, tol=tol)

    final_params = transform(result['x'])

    return CalibrationResult(
        params=final_params,
        fun=float(result['fun']),
        n_iter=int(result.get('n_iter', max_iter)),
        success=bool(result['fun'] < tol * 100),
    )


def calibrate_levenberg_marquardt(
    model_price_fn: Callable,
    market_prices: jnp.ndarray,
    initial_params: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    lambda_init: float = 1e-3,
):
    """Calibrate using Levenberg-Marquardt with AD Jacobian.

    Parameters
    ----------
    model_price_fn : callable(params) -> model_prices
    market_prices : target prices
    initial_params : starting params
    weights : calibration weights
    max_iter : max iterations
    tol : convergence tolerance
    lambda_init : initial damping parameter

    Returns
    -------
    CalibrationResult
    """
    targets = jnp.asarray(market_prices, dtype=jnp.float64)
    n = targets.shape[0]

    if weights is None:
        weights = jnp.ones(n, dtype=jnp.float64)
    else:
        weights = jnp.asarray(weights, dtype=jnp.float64)

    W = jnp.diag(jnp.sqrt(weights))

    def residuals_fn(params):
        model_vals = model_price_fn(params)
        return W @ (model_vals - targets)

    jacobian_fn = jax.jacrev(residuals_fn)

    params = jnp.asarray(initial_params, dtype=jnp.float64)
    lam = lambda_init

    for i in range(max_iter):
        r = residuals_fn(params)
        cost = 0.5 * jnp.sum(r**2)

        if cost < tol:
            return CalibrationResult(params=params, fun=float(cost), n_iter=i, success=True)

        J = jacobian_fn(params)
        JtJ = J.T @ J
        Jtr = J.T @ r

        # LM step: (J^T J + lambda * I) delta = -J^T r
        step = jnp.linalg.solve(JtJ + lam * jnp.eye(params.shape[0]), -Jtr)
        new_params = params + step
        new_r = residuals_fn(new_params)
        new_cost = 0.5 * jnp.sum(new_r**2)

        if new_cost < cost:
            params = new_params
            lam = lam * 0.5
        else:
            lam = lam * 2.0

    cost = 0.5 * jnp.sum(residuals_fn(params)**2)
    return CalibrationResult(params=params, fun=float(cost), n_iter=max_iter, success=False)


def _make_transforms(bounds):
    """Create parameter transforms for bounded optimization.

    Maps bounded [lo, hi] to unconstrained via logit/softplus.
    """
    bounds_arr = [(lo if lo is not None else -1e10,
                   hi if hi is not None else 1e10) for lo, hi in bounds]
    lo = jnp.array([b[0] for b in bounds_arr], dtype=jnp.float64)
    hi = jnp.array([b[1] for b in bounds_arr], dtype=jnp.float64)

    def transform(x):
        """Unconstrained -> constrained via sigmoid."""
        return lo + (hi - lo) * jax.nn.sigmoid(x)

    def inv_transform(params):
        """Constrained -> unconstrained via logit."""
        # Clip to avoid log(0)
        t = jnp.clip((params - lo) / (hi - lo), 1e-7, 1.0 - 1e-7)
        return jnp.log(t / (1.0 - t))

    return transform, inv_transform
