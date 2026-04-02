"""CPI coupon pricer – pricing zero-coupon and year-on-year inflation coupons."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass(frozen=True)
class CPICoupon:
    """Zero-coupon inflation coupon.

    Pays notional * (CPI(T) / CPI(base) - 1) * accrual  [or variant].

    Parameters
    ----------
    notional : float
    base_cpi : float – CPI fixing at base date
    fixing_date : float – observation date for final CPI
    payment_date : float – cash flow date
    accrual : float – day count fraction
    base_date : float – base observation date
    observation_lag : float – lag in years
    """
    notional: float
    base_cpi: float
    fixing_date: float
    payment_date: float
    accrual: float
    base_date: float = 0.0
    observation_lag: float = 3.0 / 12.0

    def index_ratio(self, cpi_at_fixing):
        """CPI(T) / CPI(base)."""
        return cpi_at_fixing / self.base_cpi

    def amount(self, cpi_at_fixing):
        """Cash flow = notional * accrual * (I(T)/I(0) - 1)."""
        return self.notional * self.accrual * (self.index_ratio(cpi_at_fixing) - 1.0)


@dataclass(frozen=True)
class YoYCoupon:
    """Year-on-year inflation coupon.

    Pays notional * accrual * (gearing * yoy_rate + spread).
    """
    notional: float
    accrual: float
    gearing: float = 1.0
    spread: float = 0.0
    floor: float = -jnp.inf
    cap: float = jnp.inf

    def amount(self, yoy_rate):
        """Compute cash flow given the YoY rate."""
        effective = self.gearing * yoy_rate + self.spread
        effective = jnp.clip(effective, self.floor, self.cap)
        return self.notional * self.accrual * effective


def cpi_coupon_forecast(base_cpi, zero_inflation_rate, T):
    """Forecast CPI at time T given base CPI and zero inflation rate.

    CPI(T) = base_cpi * (1 + z)^T
    """
    return base_cpi * (1.0 + zero_inflation_rate) ** T


def cpi_coupon_price(coupon, zero_inflation_rate, discount_fn):
    """Price a CPI coupon.

    Parameters
    ----------
    coupon : CPICoupon
    zero_inflation_rate : float – annualized zero inflation rate to fixing
    discount_fn : callable(t) -> DF

    Returns
    -------
    float – present value
    """
    T = coupon.fixing_date - coupon.base_date
    cpi_forecast = cpi_coupon_forecast(coupon.base_cpi, zero_inflation_rate, T)
    cf = coupon.amount(cpi_forecast)
    return cf * discount_fn(coupon.payment_date)


def yoy_coupon_price(coupon, yoy_rate, payment_date, discount_fn):
    """Price a YoY coupon.

    Parameters
    ----------
    coupon : YoYCoupon
    yoy_rate : float – forward YoY rate
    payment_date : float – payment date
    discount_fn : callable(t) -> DF

    Returns
    -------
    float – present value
    """
    cf = coupon.amount(yoy_rate)
    return cf * discount_fn(payment_date)


def yoy_capfloor_black(coupon, yoy_forward, vol, strike, payment_date,
                        discount_fn, fixing_time, is_cap=True):
    """Black formula for YoY cap/floor.

    Parameters
    ----------
    coupon : YoYCoupon
    yoy_forward : float – forward YoY rate
    vol : float – YoY vol
    strike : float – cap/floor strike on YoY rate
    payment_date, fixing_time : float
    discount_fn : callable
    is_cap : bool
    """
    from jax.scipy.stats import norm

    total_vol = vol * jnp.sqrt(fixing_time)
    d1 = (jnp.log(yoy_forward / strike) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol

    P = discount_fn(payment_date)

    if is_cap:
        intrinsic = yoy_forward * norm.cdf(d1) - strike * norm.cdf(d2)
    else:
        intrinsic = strike * norm.cdf(-d2) - yoy_forward * norm.cdf(-d1)

    return coupon.notional * coupon.accrual * intrinsic * P
