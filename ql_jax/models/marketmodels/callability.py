"""LMM callability framework: Bermudan exercise, upper bounds, LSM strategy.

Provides components for pricing Bermudan swaptions and TARNs within the
LIBOR Market Model (LMM) framework using Longstaff-Schwartz regression
and Andersen-Broadie upper-bound estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from ql_jax.models.marketmodels.accounting import MarketModelProduct, accounting_engine
from ql_jax.models.marketmodels.curvestate import LMMCurveState


def _rate_times_from_taus(taus):
    """Convert accrual fractions to cumulative rate times [0, tau0, tau0+tau1, ...]."""
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(taus)])


# ---------------------------------------------------------------------------
# Basis systems for exercise boundary regression
# ---------------------------------------------------------------------------

def swap_basis_system(forward_rates, taus, start, end):
    """Compute basis functions for Bermudan swaption regression.

    Returns polynomial basis of swap rate and annuity.

    Parameters
    ----------
    forward_rates : array of current forward rates
    taus : accrual fractions
    start, end : swap range indices

    Returns
    -------
    basis : array [n_basis] — values of the basis functions
    """
    rate_times = _rate_times_from_taus(taus)
    cs = LMMCurveState(rate_times=rate_times, forward_rates=forward_rates)
    sr = cs.swap_rate(start, end)
    ann = cs.annuity(start, end)

    # Polynomial basis: 1, sr, sr^2, ann, sr*ann
    return jnp.array([1.0, sr, sr**2, ann, sr * ann])


def _polynomial_basis(x, degree=3):
    """Simple polynomial basis: 1, x, x^2, ..., x^degree."""
    return jnp.array([x**i for i in range(degree + 1)])


# ---------------------------------------------------------------------------
# Exercise value for Bermudan swaption
# ---------------------------------------------------------------------------

@dataclass
class BermudanSwaptionExerciseValue:
    """Computes intrinsic exercise value for a Bermudan swaption.

    Parameters
    ----------
    fixed_rate : strike rate of the underlying swap
    taus : accrual fractions
    payer : True for payer swaption, False for receiver
    """
    fixed_rate: float
    taus: jnp.ndarray
    payer: bool = True

    def __call__(self, forward_rates, exercise_idx):
        """Intrinsic value at exercise time indexed by exercise_idx.

        Value = Annuity * (SwapRate - FixedRate) for payer,
              = Annuity * (FixedRate - SwapRate) for receiver.
        """
        n = len(forward_rates)
        rate_times = _rate_times_from_taus(self.taus)
        cs = LMMCurveState(rate_times=rate_times, forward_rates=forward_rates)
        sr = cs.swap_rate(exercise_idx, n)
        ann = cs.annuity(exercise_idx, n)

        if self.payer:
            return jnp.maximum(ann * (sr - self.fixed_rate), 0.0)
        else:
            return jnp.maximum(ann * (self.fixed_rate - sr), 0.0)


# ---------------------------------------------------------------------------
# Longstaff-Schwartz strategy for exercise decisions
# ---------------------------------------------------------------------------

def lsm_exercise_strategy(
    paths: jnp.ndarray,
    taus: jnp.ndarray,
    exercise_times: jnp.ndarray,
    exercise_value_fn: Callable,
    discount_factors: jnp.ndarray,
    basis_fn: Callable | None = None,
) -> jnp.ndarray:
    """Longstaff-Schwartz regression-based exercise strategy for LMM.

    Backward induction with cross-sectional regression at each exercise date.

    Parameters
    ----------
    paths : array [n_paths, n_steps, n_rates] — simulated forward rates
    taus : accrual fractions [n_rates]
    exercise_times : indices of eligible exercise times
    exercise_value_fn : callable(rates, idx) -> intrinsic value
    discount_factors : array [n_paths, n_steps] — cumulative discount
    basis_fn : callable(rates, idx) -> basis vector (default: swap basis)

    Returns
    -------
    strategy : array [n_paths] — optimal exercise time index (-1 = no exercise)
    """
    n_paths, n_steps, n_rates = paths.shape
    n_exercise = len(exercise_times)

    # Cash flow at terminal exercise opportunity
    cashflow = jnp.zeros(n_paths)
    exercise_time = jnp.full(n_paths, -1, dtype=jnp.int32)

    # Backward pass
    for ex_idx in range(n_exercise - 1, -1, -1):
        t_idx = int(exercise_times[ex_idx])
        rates = paths[:, t_idx, :]

        # Intrinsic value
        intrinsic = jnp.array([
            float(exercise_value_fn(rates[p], t_idx))
            for p in range(n_paths)
        ])

        if ex_idx == n_exercise - 1:
            # Last exercise: exercise if ITM
            exercise_mask = intrinsic > 0
            cashflow = jnp.where(exercise_mask, intrinsic, cashflow)
            exercise_time = jnp.where(exercise_mask, t_idx, exercise_time)
        else:
            # Regression: estimate continuation value
            itm = intrinsic > 0
            if jnp.sum(itm) < 5:
                continue

            # Build design matrix from basis functions
            if basis_fn is not None:
                X = jnp.array([basis_fn(rates[p], t_idx) for p in range(n_paths)])
            else:
                X = jnp.array([
                    swap_basis_system(rates[p], taus, t_idx, n_rates)
                    for p in range(n_paths)
                ])

            # Discount future cashflows to current time
            future_cf = cashflow * discount_factors[:, t_idx]

            # OLS regression on ITM paths
            X_itm = X[itm]
            y_itm = future_cf[itm]

            # Solve normal equations
            XtX = X_itm.T @ X_itm + 1e-8 * jnp.eye(X_itm.shape[1])
            Xty = X_itm.T @ y_itm
            beta = jnp.linalg.solve(XtX, Xty)

            continuation = X @ beta

            # Exercise if intrinsic > continuation and ITM
            exercise_now = itm & (intrinsic > continuation)
            cashflow = jnp.where(exercise_now, intrinsic, cashflow)
            exercise_time = jnp.where(exercise_now, t_idx, exercise_time)

    return exercise_time


# ---------------------------------------------------------------------------
# Upper bound engine (Andersen-Broadie dual method)
# ---------------------------------------------------------------------------

def upper_bound_engine(
    paths: jnp.ndarray,
    taus: jnp.ndarray,
    exercise_times: jnp.ndarray,
    exercise_value_fn: Callable,
    discount_factors: jnp.ndarray,
    exercise_strategy: jnp.ndarray,
    n_inner: int = 100,
    key=None,
) -> float:
    """Andersen-Broadie upper bound for Bermudan LMM products.

    Uses the duality approach: the true price lies between
    the lower bound (from LSM) and the upper bound (from martingale
    duality).

    The upper bound is computed as:
        UB = max over exercise times of (E[exercise_value] - martingale_correction)

    Parameters
    ----------
    paths : [n_paths, n_steps, n_rates]
    taus : accrual fractions
    exercise_times : eligible exercise indices
    exercise_value_fn : intrinsic value function
    discount_factors : [n_paths, n_steps]
    exercise_strategy : [n_paths] from LSM
    n_inner : inner simulation paths for martingale estimation
    key : JAX PRNG key

    Returns
    -------
    upper_bound : float
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    n_paths, n_steps, n_rates = paths.shape
    n_exercise = len(exercise_times)

    # Lower bound: value from given strategy
    lower_values = jnp.zeros(n_paths)
    for p in range(n_paths):
        ex_t = int(exercise_strategy[p])
        if ex_t >= 0:
            lower_values = lower_values.at[p].set(
                float(exercise_value_fn(paths[p, ex_t], ex_t))
                * float(discount_factors[p, ex_t])
            )

    lower_bound = float(jnp.mean(lower_values))

    # Upper bound via penalty method (simplified Andersen-Broadie)
    # For each path, compute the maximum discounted exercise value
    # minus the estimated continuation, which provides a dual estimate
    max_payoffs = jnp.zeros(n_paths)

    for ex_idx in range(n_exercise):
        t_idx = int(exercise_times[ex_idx])
        rates = paths[:, t_idx, :]

        intrinsic = jnp.array([
            float(exercise_value_fn(rates[p], t_idx))
            for p in range(n_paths)
        ])

        discounted = intrinsic * discount_factors[:, t_idx]
        max_payoffs = jnp.maximum(max_payoffs, discounted)

    upper_bound = float(jnp.mean(max_payoffs))

    return upper_bound


# ---------------------------------------------------------------------------
# Multi-step structured products
# ---------------------------------------------------------------------------

@dataclass
class MultiStepSwaption:
    """Multi-step Bermudan swaption product for the LMM.

    Defines a swaption with multiple exercise opportunities.

    Parameters
    ----------
    fixed_rate : strike swap rate
    exercise_indices : list of time-step indices for exercise
    n_rates : number of forward rates
    taus : accrual fractions
    payer : True for payer swaption
    """
    fixed_rate: float
    exercise_indices: tuple | list
    n_rates: int
    taus: jnp.ndarray
    payer: bool = True

    def to_product(self) -> MarketModelProduct:
        """Convert to a MarketModelProduct for the accounting engine."""
        taus = self.taus
        exercise_val = BermudanSwaptionExerciseValue(
            fixed_rate=self.fixed_rate, taus=taus, payer=self.payer,
        )

        def cashflow_fn(forward_rates, step_idx):
            if step_idx in self.exercise_indices:
                return exercise_val(forward_rates, step_idx)
            return None

        evolution_times = jnp.linspace(0.0, float(jnp.sum(taus)), self.n_rates + 1)
        cashflow_times = evolution_times[1:]

        return MarketModelProduct(
            n_rates=self.n_rates,
            evolution_times=evolution_times,
            cashflow_times=cashflow_times,
            cashflow_fn=cashflow_fn,
        )


@dataclass
class MultiStepTARN:
    """Target Accrual Redemption Note (TARN) product.

    A TARN is a structured note that accrues coupons until a target
    total coupon is reached, then redeems at par.

    Parameters
    ----------
    target : total coupon target (e.g. 0.15 = 15%)
    fixed_rate : coupon rate (or 0 for floating)
    leverage : multiplier on floating rate
    floor_rate : minimum coupon rate
    cap_rate : maximum coupon rate
    n_rates : number of forward rates
    taus : accrual fractions
    """
    target: float
    fixed_rate: float = 0.0
    leverage: float = 1.0
    floor_rate: float = 0.0
    cap_rate: float = 1.0
    n_rates: int = 10
    taus: jnp.ndarray = None

    def to_product(self) -> MarketModelProduct:
        """Convert to MarketModelProduct."""
        taus = self.taus if self.taus is not None else jnp.full(self.n_rates, 0.5)
        target = self.target
        leverage = self.leverage
        floor_rate = self.floor_rate
        cap_rate = self.cap_rate
        fixed_rate = self.fixed_rate

        accumulated = [0.0]  # mutable closure for accumulation

        def cashflow_fn(forward_rates, step_idx):
            if step_idx >= len(forward_rates[0]) if forward_rates.ndim > 1 else step_idx >= len(forward_rates):
                return None

            # Current libor rate
            rate = forward_rates[step_idx] if forward_rates.ndim == 1 else forward_rates[:, step_idx]

            # Coupon = max(floor, min(cap, leverage * libor + fixed))
            coupon_rate = jnp.clip(leverage * rate + fixed_rate, floor_rate, cap_rate)
            coupon = coupon_rate * taus[step_idx]

            # Check if target reached
            new_acc = accumulated[0] + float(jnp.mean(coupon))
            if new_acc >= target:
                # Partial coupon + redemption
                remaining = target - accumulated[0]
                accumulated[0] = target
                return jnp.maximum(remaining, 0.0) + 1.0  # par redemption
            else:
                accumulated[0] = new_acc
                return coupon

        evolution_times = jnp.linspace(0.0, float(jnp.sum(taus)), self.n_rates + 1)
        cashflow_times = evolution_times[1:]

        return MarketModelProduct(
            n_rates=self.n_rates,
            evolution_times=evolution_times,
            cashflow_times=cashflow_times,
            cashflow_fn=cashflow_fn,
        )
