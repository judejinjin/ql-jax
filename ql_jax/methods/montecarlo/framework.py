"""Monte Carlo framework: generic MC model, path pricer interface, MC traits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp


@dataclass
class MCResult:
    """Result of a Monte Carlo simulation."""
    mean: float
    std_error: float
    n_paths: int
    raw_values: Optional[jnp.ndarray] = None


class PathPricer:
    """Abstract path pricer base class.

    Subclass and implement __call__(path) -> payoff.
    """

    def __call__(self, path):
        """Evaluate payoff for a single path.

        Parameters
        ----------
        path : array [n_steps+1] of asset prices along path

        Returns
        -------
        float : discounted payoff
        """
        raise NotImplementedError


class EuropeanCallPathPricer(PathPricer):
    """Path pricer for a European call."""

    def __init__(self, strike, discount):
        self._K = strike
        self._df = discount

    def __call__(self, path):
        return self._df * jnp.maximum(path[-1] - self._K, 0.0)


class EuropeanPutPathPricer(PathPricer):
    """Path pricer for a European put."""

    def __init__(self, strike, discount):
        self._K = strike
        self._df = discount

    def __call__(self, path):
        return self._df * jnp.maximum(self._K - path[-1], 0.0)


class ArithmeticAvgPathPricer(PathPricer):
    """Path pricer for arithmetic average Asian call."""

    def __init__(self, strike, discount):
        self._K = strike
        self._df = discount

    def __call__(self, path):
        avg = jnp.mean(path)
        return self._df * jnp.maximum(avg - self._K, 0.0)


def monte_carlo_model(
    path_generator,
    path_pricer,
    n_paths=10000,
    key=None,
):
    """Generic Monte Carlo pricing engine.

    Parameters
    ----------
    path_generator : callable(key, n_paths) -> array [n_paths, n_steps+1]
        Generates paths.
    path_pricer : PathPricer or callable(path) -> float
        Prices a single path.
    n_paths : int
    key : JAX PRNG key

    Returns
    -------
    MCResult
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    paths = path_generator(key, n_paths)  # [n_paths, n_steps+1]

    # Price each path
    payoffs = jax.vmap(path_pricer)(paths)

    return MCResult(
        mean=float(jnp.mean(payoffs)),
        std_error=float(jnp.std(payoffs) / jnp.sqrt(n_paths)),
        n_paths=n_paths,
        raw_values=payoffs,
    )


class LSMBasisSystem:
    """Basis function systems for Longstaff-Schwartz regression.

    Parameters
    ----------
    basis_type : 'monomial', 'laguerre', 'hermite'
    order : int (polynomial order)
    """

    def __init__(self, basis_type='monomial', order=3):
        self._type = basis_type
        self._order = order

    def __call__(self, x):
        """Evaluate all basis functions at x.

        Parameters
        ----------
        x : array [n_samples]

        Returns
        -------
        array [n_samples, order+1]
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        if self._type == 'monomial':
            return jnp.column_stack([x ** k for k in range(self._order + 1)])
        elif self._type == 'laguerre':
            # Laguerre polynomials L_k(x)
            L = [jnp.ones_like(x), 1.0 - x]
            for k in range(2, self._order + 1):
                L.append(((2 * k - 1 - x) * L[-1] - (k - 1) * L[-2]) / k)
            return jnp.column_stack(L[:self._order + 1])
        elif self._type == 'hermite':
            H = [jnp.ones_like(x), 2 * x]
            for k in range(2, self._order + 1):
                H.append(2 * x * H[-1] - 2 * (k - 1) * H[-2])
            return jnp.column_stack(H[:self._order + 1])
        else:
            raise ValueError(f"Unknown basis type: {self._type}")


def generic_ls_regression(paths, cashflows, discount_factors, basis=None, exercise_times=None):
    """Generic Longstaff-Schwartz regression for early exercise.

    Parameters
    ----------
    paths : array [n_paths, n_steps+1] — asset price paths
    cashflows : array [n_paths, n_steps] — immediate exercise values at each step
    discount_factors : array [n_steps] — one-period discount factors
    basis : LSMBasisSystem or None (default monomial order 3)
    exercise_times : array of step indices where exercise is allowed (None = all)

    Returns
    -------
    array [n_paths] — optimal payoffs
    """
    n_paths, n_steps_plus = paths.shape
    n_steps = n_steps_plus - 1

    if basis is None:
        basis = LSMBasisSystem('monomial', 3)

    if exercise_times is None:
        exercise_times = list(range(1, n_steps + 1))

    # Work backwards
    payoff = cashflows[:, -1]  # Terminal payoff

    for t in reversed(exercise_times[:-1]):
        # Discount payoff one step
        payoff = payoff * discount_factors[min(t, n_steps - 1)]

        # In-the-money paths
        exercise_val = cashflows[:, t - 1]
        itm = exercise_val > 0

        if jnp.sum(itm) > 0:
            # Regression
            X = basis(paths[itm, t])
            y = payoff[itm]
            beta = jnp.linalg.lstsq(X, y, rcond=None)[0]
            continuation = X @ beta

            # Exercise if immediate > continuation
            exercise_mask = exercise_val[itm] > continuation
            payoff = payoff.at[jnp.where(itm)[0][exercise_mask]].set(
                exercise_val[itm][exercise_mask]
            )

    return payoff * discount_factors[0]
