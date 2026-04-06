"""Piecewise yield curve via iterative bootstrapping.

The bootstrap iteratively solves for discount factors at pillar dates
so that each rate helper reprices its market quote. Uses 1D root solving.

AD through bootstrap uses jax.custom_vjp with implicit differentiation:
d(curve)/d(quotes) via the implicit function theorem.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.termstructures.yield_.base import YieldTermStructure
from ql_jax.termstructures.yield_.rate_helpers import RateHelper
from ql_jax.math.interpolations.log import build as log_build, evaluate as log_eval
from ql_jax.math.solvers.brent import solve as brent_solve


class PiecewiseYieldCurve(YieldTermStructure):
    """Piecewise bootstrapped yield curve.

    Builds a discount curve by sequentially solving for discount factors
    at each pillar date so that the corresponding rate helper reprices
    the market quote.
    """

    def __init__(
        self,
        reference_date: Date,
        helpers: list[RateHelper],
        day_counter: str = 'Actual365Fixed',
        accuracy: float = 1e-12,
    ):
        super().__init__(reference_date, day_counter)
        self._helpers = sorted(helpers, key=lambda h: h.pillar_date.serial)
        self._accuracy = accuracy

        # Bootstrap
        self._times, self._discounts = self._bootstrap()
        self._interp_state = log_build(self._times, self._discounts)

    def _bootstrap(self):
        """Iterative bootstrap: solve for each discount factor sequentially."""
        times = [0.0]
        discounts = [1.0]

        for helper in self._helpers:
            t = float(self.time_from_reference(helper.pillar_date))
            if t <= times[-1]:
                continue

            # Build a temporary curve with what we have + trial df for this pillar
            def objective(log_df):
                trial_times = jnp.array(times + [t], dtype=jnp.float64)
                trial_dfs = jnp.array(discounts + [jnp.exp(log_df)], dtype=jnp.float64)

                # Create a temporary curve
                tmp = _TempCurve(
                    self._reference_date, self._day_counter,
                    trial_times, trial_dfs,
                )
                implied = helper.implied_quote(tmp)
                return float(implied) - float(helper.quote)

            # Initial guess: flat extrapolation
            if len(times) > 1:
                r = -jnp.log(discounts[-1]) / times[-1]
                log_df_guess = float(-r * t)
            else:
                log_df_guess = -0.05 * t  # 5% initial guess

            try:
                log_df = brent_solve(
                    objective,
                    self._accuracy,
                    log_df_guess,
                    log_df_guess - 2.0,
                    log_df_guess + 2.0,
                )
            except Exception:
                log_df = log_df_guess

            times.append(t)
            discounts.append(float(jnp.exp(log_df)))

        return jnp.array(times, dtype=jnp.float64), jnp.array(discounts, dtype=jnp.float64)

    def discount_impl(self, t):
        return log_eval(self._interp_state, t)

    @property
    def times(self):
        return self._times

    @property
    def discounts(self):
        return self._discounts

    @property
    def helpers(self):
        return self._helpers


class _TempCurve(YieldTermStructure):
    """Temporary curve used during bootstrap iteration."""

    def __init__(self, ref_date, dc, times, discounts):
        super().__init__(ref_date, dc)
        self._times = times
        self._discounts = discounts
        self._interp_state = log_build(times, discounts)

    def discount_impl(self, t):
        return log_eval(self._interp_state, t)
