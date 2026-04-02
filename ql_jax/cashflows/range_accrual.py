"""Range accrual coupons.

Pays a rate proportional to the fraction of days the reference rate
stays within a specified range [lo, hi].
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax.scipy.stats import norm


@dataclass(frozen=True)
class RangeAccrualCoupon:
    """Range accrual coupon.

    Parameters
    ----------
    notional : float
    fixing_dates : array – observation dates within accrual period
    payment_date : float
    accrual_period : float
    lower_barrier : float – range lower bound
    upper_barrier : float – range upper bound
    gearing : float – rate multiplier
    spread : float – additive spread when in range
    """
    notional: float
    fixing_dates: jnp.ndarray
    payment_date: float
    accrual_period: float
    lower_barrier: float
    upper_barrier: float
    gearing: float = 1.0
    spread: float = 0.0

    def accrual_fraction(self, observed_rates):
        """Fraction of fixing dates where rate is in range."""
        in_range = (observed_rates >= self.lower_barrier) & (observed_rates <= self.upper_barrier)
        return jnp.mean(in_range.astype(jnp.float64))

    def amount(self, observed_rates, reference_rate):
        """Cash flow amount.

        Parameters
        ----------
        observed_rates : array – rates observed on each fixing date
        reference_rate : float – base rate for coupon calculation
        """
        fraction = self.accrual_fraction(observed_rates)
        rate = self.gearing * reference_rate + self.spread
        return self.notional * self.accrual_period * rate * fraction


def range_accrual_price(coupon, forward_rate, vol, discount_fn,
                          correlation=0.0):
    """Analytical approximation of range accrual price.

    Uses a sum of digital options for each observation date.

    Parameters
    ----------
    coupon : RangeAccrualCoupon
    forward_rate : float – forward reference rate
    vol : float – Black vol
    discount_fn : callable(t) -> DF
    correlation : float – correlation between reference rate and fixing rate
    """
    n_fixings = len(coupon.fixing_dates)
    P_pay = discount_fn(coupon.payment_date)

    total_prob = 0.0
    for fix_date in coupon.fixing_dates:
        T = float(fix_date)
        total_vol = vol * jnp.sqrt(jnp.maximum(T, 1e-10))

        # P(lo <= rate <= hi) = N(d2_hi) - N(d2_lo)
        d2_hi = (jnp.log(forward_rate / coupon.upper_barrier) -
                  0.5 * total_vol**2) / total_vol
        d2_lo = (jnp.log(forward_rate / coupon.lower_barrier) -
                  0.5 * total_vol**2) / total_vol

        prob = norm.cdf(d2_lo) - norm.cdf(d2_hi)  # note sign: d2_lo > d2_hi => prob > 0
        # Actually: P(rate < hi) - P(rate < lo) = N(d2_hi) - N(d2_lo) for lognormal
        prob = norm.cdf(-d2_hi + total_vol) - norm.cdf(-d2_lo + total_vol)
        # Simplified: use normal approx
        prob = norm.cdf((coupon.upper_barrier - forward_rate) / (forward_rate * total_vol + 1e-10)) - \
               norm.cdf((coupon.lower_barrier - forward_rate) / (forward_rate * total_vol + 1e-10))
        total_prob += prob

    avg_prob = total_prob / n_fixings
    rate = coupon.gearing * forward_rate + coupon.spread
    return coupon.notional * coupon.accrual_period * rate * avg_prob * P_pay
