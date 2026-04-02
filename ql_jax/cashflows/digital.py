"""Digital coupons: IBOR digital, CMS digital.

A digital coupon pays a fixed amount when the underlying rate is
above/below a strike (binary payoff).
"""

import jax.numpy as jnp
from dataclasses import dataclass
from jax.scipy.stats import norm


@dataclass(frozen=True)
class DigitalCoupon:
    """Digital coupon that pays a fixed rate when condition is met.

    Parameters
    ----------
    notional : float
    fixing_date : float
    payment_date : float
    accrual_period : float
    strike : float – digital strike
    digital_rate : float – rate paid if in-the-money
    is_call : bool – True if pays when rate > strike
    """
    notional: float
    fixing_date: float
    payment_date: float
    accrual_period: float
    strike: float
    digital_rate: float
    is_call: bool = True

    def payoff(self, rate):
        """Digital payoff: digital_rate if condition met, else 0."""
        if self.is_call:
            return jnp.where(rate > self.strike, self.digital_rate, 0.0)
        else:
            return jnp.where(rate < self.strike, self.digital_rate, 0.0)

    def amount(self, rate):
        return self.notional * self.accrual_period * self.payoff(rate)


@dataclass(frozen=True)
class DigitalIborCoupon(DigitalCoupon):
    """Digital coupon on an IBOR rate."""
    pass


@dataclass(frozen=True)
class DigitalCMSCoupon(DigitalCoupon):
    """Digital coupon on a CMS rate."""
    swap_tenor: float = 10.0


def digital_coupon_price(coupon, forward_rate, vol, discount_fn):
    """Price a digital coupon using Black formula.

    Parameters
    ----------
    coupon : DigitalCoupon
    forward_rate : float – forward rate
    vol : float – Black vol of the underlying rate
    discount_fn : callable(t) -> discount factor
    """
    T = coupon.fixing_date
    total_vol = vol * jnp.sqrt(T)
    d2 = (jnp.log(forward_rate / coupon.strike) - 0.5 * total_vol**2) / total_vol

    P_pay = discount_fn(coupon.payment_date)

    if coupon.is_call:
        prob = norm.cdf(d2)
    else:
        prob = norm.cdf(-d2)

    return coupon.notional * coupon.accrual_period * coupon.digital_rate * prob * P_pay


def digital_replication_price(coupon, forward_rate, vol, discount_fn,
                                shift=0.0001):
    """Price digital via call spread replication (more robust with skew).

    Digital ≈ (Call(K-eps) - Call(K+eps)) / (2*eps)
    """
    T = coupon.fixing_date
    P_pay = discount_fn(coupon.payment_date)

    K = coupon.strike
    K_lo = K - shift
    K_hi = K + shift

    call_lo = _black_call(forward_rate, K_lo, T, vol)
    call_hi = _black_call(forward_rate, K_hi, T, vol)

    digital_pv = (call_lo - call_hi) / (2.0 * shift)
    if not coupon.is_call:
        digital_pv = 1.0 - digital_pv

    return coupon.notional * coupon.accrual_period * coupon.digital_rate * digital_pv * P_pay


def _black_call(F, K, T, sigma):
    """Black call price (undiscounted)."""
    total_vol = sigma * jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol
    return F * norm.cdf(d1) - K * norm.cdf(d2)
