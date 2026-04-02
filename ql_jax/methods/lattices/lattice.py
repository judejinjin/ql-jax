"""Lattice framework: abstract lattice, 1D lattice, 2D lattice, two-factor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp


@dataclass
class Lattice1D:
    """One-dimensional lattice (recombining tree).

    Parameters
    ----------
    n_steps : int — number of time steps
    dt : float — time step size
    tree_type : 'binomial' or 'trinomial'
    """
    n_steps: int
    dt: float
    tree_type: str = 'trinomial'

    def rollback(self, values, step, discount):
        """Roll back one step.

        Parameters
        ----------
        values : array of option values at step+1
        step : current time step
        discount : one-period discount factor

        Returns
        -------
        array of option values at step
        """
        if self.tree_type == 'binomial':
            return discount * 0.5 * (values[:-1] + values[1:])
        else:
            # Trinomial: equal probs
            n = len(values) - 2
            return discount * (values[:-2] / 6.0 + 2.0 * values[1:-1] / 3.0 + values[2:] / 6.0)

    def full_rollback(self, terminal_values, discount_fn, adjustment_fn=None):
        """Roll back from terminal to initial.

        Parameters
        ----------
        terminal_values : array of terminal payoffs
        discount_fn : callable(step) -> discount factor
        adjustment_fn : optional callable(step, values) -> adjusted values
                       (e.g., for American exercise)

        Returns
        -------
        float : option value at root
        """
        values = terminal_values
        for step in range(self.n_steps - 1, -1, -1):
            df = discount_fn(step)
            values = self.rollback(values, step, df)
            if adjustment_fn is not None:
                values = adjustment_fn(step, values)
        return values[0] if len(values) == 1 else values[len(values) // 2]


@dataclass
class Lattice2D:
    """Two-dimensional lattice for two-factor models.

    Parameters
    ----------
    n_steps : int
    dt : float
    correlation : float — correlation between factors
    """
    n_steps: int
    dt: float
    correlation: float = 0.0

    def rollback(self, values, step, discount):
        """Roll back a 2D grid one step.

        Parameters
        ----------
        values : 2D array of option values at step+1
        step : int
        discount : float

        Returns
        -------
        2D array at step
        """
        rho = self.correlation
        n1, n2 = values.shape

        # 2D binomial probabilities with correlation
        # p(+,+) = (1 + rho)/4, p(+,-) = (1-rho)/4, etc.
        p_pp = (1 + rho) / 4.0
        p_pm = (1 - rho) / 4.0
        p_mp = (1 - rho) / 4.0
        p_mm = (1 + rho) / 4.0

        result = discount * (
            p_pp * values[1:, 1:]
            + p_pm * values[1:, :-1]
            + p_mp * values[:-1, 1:]
            + p_mm * values[:-1, :-1]
        )
        return result

    def full_rollback(self, terminal_values, discount_fn, adjustment_fn=None):
        """Roll back 2D grid from terminal to initial."""
        values = terminal_values
        for step in range(self.n_steps - 1, -1, -1):
            df = discount_fn(step)
            values = self.rollback(values, step, df)
            if adjustment_fn is not None:
                values = adjustment_fn(step, values)
        i, j = values.shape[0] // 2, values.shape[1] // 2
        return values[i, j]


class TwoFactorLattice:
    """Two-factor lattice for models like G2++.

    Combines two trinomial trees with correlation.

    Parameters
    ----------
    n_steps : int
    dt : float
    sigma1, sigma2 : volatilities
    kappa1, kappa2 : mean-reversion speeds
    rho : correlation
    """

    def __init__(self, n_steps, dt, sigma1, sigma2, kappa1, kappa2, rho):
        self.n_steps = n_steps
        self.dt = dt
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.rho = rho

    def build_grid(self):
        """Build the 2D grid of short-rate states.

        Returns
        -------
        x1_grid, x2_grid : arrays of factor values at each time step
        """
        jmax1 = int(0.184 / (self.sigma1 * jnp.sqrt(self.dt)) + 0.5) + 1
        jmax2 = int(0.184 / (self.sigma2 * jnp.sqrt(self.dt)) + 0.5) + 1

        dx1 = self.sigma1 * jnp.sqrt(3 * self.dt)
        dx2 = self.sigma2 * jnp.sqrt(3 * self.dt)

        x1_grid = jnp.arange(-jmax1, jmax1 + 1) * dx1
        x2_grid = jnp.arange(-jmax2, jmax2 + 1) * dx2

        return x1_grid, x2_grid

    def price(self, terminal_payoff_fn, discount_fn, adjustment_fn=None):
        """Price using full 2D tree rollback.

        Parameters
        ----------
        terminal_payoff_fn : callable(x1, x2) -> payoff
        discount_fn : callable(step, x1, x2) -> discount
        adjustment_fn : optional callable(step, x1, x2, value) -> adjusted

        Returns
        -------
        float : option price
        """
        x1_grid, x2_grid = self.build_grid()
        n1, n2 = len(x1_grid), len(x2_grid)

        # Terminal values (vectorized)
        X1, X2 = jnp.meshgrid(x1_grid, x2_grid, indexing='ij')
        values = terminal_payoff_fn(X1, X2)

        # Rollback
        for step in range(self.n_steps - 1, -1, -1):
            new_n1 = max(values.shape[0] - 2, 1)
            new_n2 = max(values.shape[1] - 2, 1)
            if new_n1 < 1 or new_n2 < 1:
                break
            df = discount_fn(step)
            new_values = df * (
                values[:new_n1, :new_n2] * (1 + self.rho) / 4.0
                + values[2:2+new_n1, :new_n2] * (1 - self.rho) / 4.0
                + values[:new_n1, 2:2+new_n2] * (1 - self.rho) / 4.0
                + values[2:2+new_n1, 2:2+new_n2] * (1 + self.rho) / 4.0
            )
            if adjustment_fn is not None:
                X1_mid = x1_grid[1:1+new_n1]
                X2_mid = x2_grid[1:1+new_n2]
                X1m, X2m = jnp.meshgrid(X1_mid, X2_mid, indexing='ij')
                new_values = adjustment_fn(step, X1m, X2m, new_values)
            values = new_values

        return values[values.shape[0] // 2, values.shape[1] // 2]
