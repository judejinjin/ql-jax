"""Overnight indexed coupon pricer (Black model)."""

import jax.numpy as jnp
from dataclasses import dataclass
from jax.scipy.stats import norm


@dataclass(frozen=True)
class OvernightIndexedCoupon:
    """Overnight indexed coupon (compounded or averaged).

    Parameters
    ----------
    notional : float
    start_date : float
    end_date : float
    overnight_rates : array – daily overnight rates
    rate_dates : array – dates corresponding to rates
    averaging : str – 'compounding' or 'averaging'
    spread : float
    """
    notional: float
    start_date: float
    end_date: float
    overnight_rates: jnp.ndarray
    rate_dates: jnp.ndarray
    averaging: str = 'compounding'
    spread: float = 0.0

    def effective_rate(self):
        """Compute effective rate from overnight rates."""
        n = len(self.overnight_rates)
        if n == 0:
            return 0.0

        if self.averaging == 'compounding':
            # Compounded: prod(1 + r_i * dt_i) - 1
            product = 1.0
            for i in range(n):
                dt_i = (self.rate_dates[i + 1] - self.rate_dates[i]) if i < n - 1 else 1.0 / 365.0
                product *= (1.0 + self.overnight_rates[i] * dt_i)
            total_time = self.end_date - self.start_date
            return (product - 1.0) / total_time
        else:
            # Simple averaging
            return jnp.mean(self.overnight_rates)

    def amount(self):
        """Cash flow amount."""
        tau = self.end_date - self.start_date
        return self.notional * tau * (self.effective_rate() + self.spread)


def black_overnight_coupon_price(notional, start, end, forward_rate, vol,
                                   strike, discount_fn, is_cap=True):
    """Price an overnight cap/floor using Black formula.

    Parameters
    ----------
    notional : float
    start, end : float – accrual period
    forward_rate : float – forward overnight compound rate
    vol : float – Black vol
    strike : float – cap/floor strike
    discount_fn : callable(t) -> DF
    is_cap : bool
    """
    T = start
    tau = end - start
    total_vol = vol * jnp.sqrt(T)

    d1 = (jnp.log(forward_rate / strike) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol

    P_pay = discount_fn(end)

    if is_cap:
        price = forward_rate * norm.cdf(d1) - strike * norm.cdf(d2)
    else:
        price = strike * norm.cdf(-d2) - forward_rate * norm.cdf(-d1)

    return notional * tau * price * P_pay
