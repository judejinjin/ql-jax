"""Step conditions for finite-difference solvers."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass(frozen=True)
class AmericanStepCondition:
    """American exercise constraint: V = max(V, payoff) after each step.

    Parameters
    ----------
    exercise_values : array – intrinsic values on the grid
    """
    exercise_values: jnp.ndarray

    def apply(self, V, t=None):
        """Apply early exercise constraint."""
        return jnp.maximum(V, self.exercise_values)


@dataclass(frozen=True)
class BermudanStepCondition:
    """Bermudan exercise constraint: exercise only at specified times.

    Parameters
    ----------
    exercise_values : array – intrinsic values on the grid
    exercise_times : array – times at which exercise is allowed
    tol : float – tolerance for matching exercise times
    """
    exercise_values: jnp.ndarray
    exercise_times: jnp.ndarray
    tol: float = 1e-6

    def apply(self, V, t):
        """Apply exercise constraint if t matches an exercise date."""
        is_exercise = jnp.any(jnp.abs(self.exercise_times - t) < self.tol)
        return jnp.where(is_exercise, jnp.maximum(V, self.exercise_values), V)


@dataclass(frozen=True)
class BarrierStepCondition:
    """Barrier knock-out: set V=rebate where grid crosses barrier.

    Parameters
    ----------
    grid : array – spatial grid
    barrier : float – barrier level
    rebate : float – rebate amount
    is_up : bool – if True, knock out above barrier; else below
    """
    grid: jnp.ndarray
    barrier: float
    rebate: float = 0.0
    is_up: bool = True

    def apply(self, V, t=None):
        """Zero out values beyond barrier."""
        if self.is_up:
            mask = self.grid >= self.barrier
        else:
            mask = self.grid <= self.barrier
        return jnp.where(mask, self.rebate, V)


@dataclass(frozen=True)
class CompositeStepCondition:
    """Applies multiple step conditions sequentially.

    Parameters
    ----------
    conditions : tuple of step conditions
    """
    conditions: tuple

    def apply(self, V, t=None):
        for cond in self.conditions:
            V = cond.apply(V, t)
        return V


@dataclass(frozen=True)
class CouponStepCondition:
    """Adjust V at coupon/dividend payment times.

    V(t-) = V(t+) + coupon_amount (for discrete dividends).

    Parameters
    ----------
    payment_times : array
    payment_amounts : array – dividend/coupon per unit notional
    tol : float
    """
    payment_times: jnp.ndarray
    payment_amounts: jnp.ndarray
    tol: float = 1e-6

    def apply(self, V, t):
        """Adjust for discrete payment at time t."""
        for i in range(len(self.payment_times)):
            is_payment = jnp.abs(self.payment_times[i] - t) < self.tol
            V = jnp.where(is_payment, V + self.payment_amounts[i], V)
        return V
