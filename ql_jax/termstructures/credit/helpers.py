"""Credit term structure helpers: CDS helpers for bootstrapping default curves."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass(frozen=True)
class CdsHelper:
    """CDS calibration helper for bootstrapping survival probability curves.

    Parameters
    ----------
    market_spread : float – market CDS spread (in basis points / 10000)
    maturity : float – CDS maturity in years
    recovery_rate : float – assumed recovery rate (e.g. 0.4)
    premium_frequency : float – premium payment frequency (e.g. 0.25 for quarterly)
    """
    market_spread: float
    maturity: float
    recovery_rate: float
    premium_frequency: float = 0.25

    def implied_quote(self, survival_fn, discount_fn):
        """Model CDS spread given survival and discount curves.

        Premium leg = spread * sum(tau_i * P(t_i) * Q(t_i))
        Default leg = (1-R) * sum(P(t_i) * (Q(t_{i-1}) - Q(t_i)))
        """
        n_periods = int(self.maturity / self.premium_frequency)
        premium_leg = 0.0
        default_leg = 0.0
        Q_prev = 1.0

        for i in range(1, n_periods + 1):
            t = i * self.premium_frequency
            Q = survival_fn(t)
            P = discount_fn(t)

            premium_leg += self.premium_frequency * P * Q
            default_leg += (1.0 - self.recovery_rate) * P * (Q_prev - Q)
            Q_prev = Q

        return default_leg / jnp.maximum(premium_leg, 1e-15)

    def quote_error(self, survival_fn, discount_fn):
        return self.implied_quote(survival_fn, discount_fn) - self.market_spread


@dataclass(frozen=True)
class SpreadCdsHelper(CdsHelper):
    """CDS helper with upfront + running spread."""
    upfront: float = 0.0

    def implied_quote(self, survival_fn, discount_fn):
        par_spread = super().implied_quote(survival_fn, discount_fn)
        return par_spread


def bootstrap_default_curve(cds_helpers, discount_fn, initial_hazard=0.01):
    """Bootstrap piecewise-constant hazard rate curve from CDS helpers.

    Parameters
    ----------
    cds_helpers : list of CdsHelper, sorted by maturity
    discount_fn : callable(t) -> discount factor
    initial_hazard : float – starting hazard rate guess

    Returns dict with: maturities, hazard_rates, survival_probabilities.
    """
    from ql_jax.math.solvers.brent import solve

    maturities = sorted([h.maturity for h in cds_helpers])
    hazard_rates = []
    survival_probs = [1.0]

    for helper in sorted(cds_helpers, key=lambda h: h.maturity):
        t = helper.maturity

        def survival_fn(tau, hazard_rates=hazard_rates, t_mat=t):
            # Piecewise constant hazard
            cum_hazard = 0.0
            prev_t = 0.0
            for k, hr in enumerate(hazard_rates):
                mat_k = maturities[k] if k < len(maturities) else t_mat
                seg_end = min(mat_k, tau)
                if prev_t < seg_end:
                    cum_hazard += hr * (seg_end - prev_t)
                prev_t = seg_end
                if prev_t >= tau:
                    break
            return jnp.exp(-cum_hazard)

        def objective(h):
            rates = hazard_rates + [h]
            def surv(tau):
                cum = 0.0
                prev = 0.0
                all_mats = maturities[:len(rates)]
                for k, hr in enumerate(rates):
                    seg_end = min(all_mats[k] if k < len(all_mats) else t, tau)
                    if prev < seg_end:
                        cum += hr * (seg_end - prev)
                    prev = seg_end
                return jnp.exp(-cum)
            return helper.quote_error(surv, discount_fn)

        h = solve(objective, 1e-6, 2.0)
        hazard_rates.append(h)
        survival_probs.append(jnp.exp(-sum(
            hazard_rates[k] * (maturities[k] - (maturities[k-1] if k > 0 else 0.0))
            for k in range(len(hazard_rates))
        )))

    return {
        'maturities': jnp.array(maturities),
        'hazard_rates': jnp.array(hazard_rates),
        'survival_probabilities': jnp.array(survival_probs),
    }
