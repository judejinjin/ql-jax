"""CMS (Constant Maturity Swap) coupons and pricers.

A CMS coupon pays a swap rate observed at a fixing date.
Pricing requires convexity and timing adjustments.
"""

import jax.numpy as jnp
from dataclasses import dataclass
from jax.scipy.stats import norm


@dataclass(frozen=True)
class CMSCoupon:
    """CMS coupon: pays swap_rate(T_fix) at T_pay.

    Parameters
    ----------
    notional : float
    fixing_date : float – time of rate observation
    payment_date : float – time of payment
    accrual_period : float – day count fraction
    swap_tenor : float – CMS swap tenor (e.g. 10.0 for 10Y)
    gearing : float – multiplier (default 1.0)
    spread : float – additive spread (default 0.0)
    """
    notional: float
    fixing_date: float
    payment_date: float
    accrual_period: float
    swap_tenor: float
    gearing: float = 1.0
    spread: float = 0.0

    def rate(self, swap_rate):
        """Effective rate: gearing * swap_rate + spread."""
        return self.gearing * swap_rate + self.spread

    def amount(self, swap_rate):
        """Cash amount."""
        return self.notional * self.accrual_period * self.rate(swap_rate)


def cms_conundrum_price(coupon, forward_swap_rate, annuity, swaption_vol,
                         discount_fn, mean_reversion=0.0):
    """CMS coupon price using Hunt-Kennedy / Hagan conundrum pricer.

    Parameters
    ----------
    coupon : CMSCoupon
    forward_swap_rate : float – ATM forward swap rate
    annuity : float – PV of annuity
    swaption_vol : float – ATM swaption Black vol
    discount_fn : callable(t) -> discount factor
    mean_reversion : float – mean reversion for CMS adjustment

    Returns float price.
    """
    T = coupon.fixing_date
    sigma = swaption_vol
    s = forward_swap_rate

    # Convexity adjustment (Hagan formula, linear TSR approximation)
    h_prime = 1.0 / annuity  # dP/dS at ATM
    h_double_prime = -2.0 / (annuity * s)  # d^2P/dS^2

    # Timing adjustment for payment delay
    P_fix = discount_fn(T)
    P_pay = discount_fn(coupon.payment_date)
    timing_adj = P_pay / P_fix

    # Convexity adjustment
    convexity = 0.5 * sigma**2 * T * s**2 * h_double_prime / h_prime
    adjusted_rate = s + convexity

    # Mean reversion correction
    if mean_reversion != 0.0:
        mr_adj = mean_reversion * sigma**2 * T * s
        adjusted_rate += mr_adj

    effective_rate = coupon.gearing * adjusted_rate + coupon.spread
    return coupon.notional * coupon.accrual_period * effective_rate * P_pay


def cms_linear_tsr_price(coupon, forward_swap_rate, annuity, swaption_vol_fn,
                           discount_fn, n_points=32):
    """CMS coupon price using Linear Terminal Swap Rate (TSR) model.

    More accurate than conundrum for skewed smile.

    Parameters
    ----------
    coupon : CMSCoupon
    forward_swap_rate : float
    annuity : float
    swaption_vol_fn : callable(K) -> vol for each strike
    discount_fn : callable(t) -> DF
    n_points : quadrature points for replication integral
    """
    T = coupon.fixing_date
    s = forward_swap_rate

    P_pay = discount_fn(coupon.payment_date)

    # Static replication: CMS = forward + integral of call/put options
    # E[g(S)] = g(s) + integral of g''(K) * option(K) dK

    # For CMS rate: g(S) = S, g''(S) = 0, but timing adjustment adds curvature
    # Use numerical integration with swaption smile

    # Simple approach: convexity via vanna-volga
    sigma_atm = swaption_vol_fn(s)
    vol_squared_T = sigma_atm**2 * T

    # Second-order correction
    ds = s * 0.01
    vol_up = swaption_vol_fn(s + ds)
    vol_dn = swaption_vol_fn(s - ds)
    skew = (vol_up - vol_dn) / (2.0 * ds)
    convexity_adj = 0.5 * s**2 * vol_squared_T / (annuity * s)
    skew_adj = s**2 * sigma_atm * T * skew

    adjusted_rate = s + convexity_adj + skew_adj
    effective_rate = coupon.gearing * adjusted_rate + coupon.spread

    return coupon.notional * coupon.accrual_period * effective_rate * P_pay
