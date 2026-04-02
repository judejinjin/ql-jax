"""Multiple resets coupon – sub-period coupons with compounding or averaging."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass(frozen=True)
class SubPeriodsCoupon:
    """Coupon with multiple IBOR resets within a single accrual period.

    Parameters
    ----------
    notional : float
    start_date : float
    end_date : float
    sub_start_dates : array – sub-period start dates
    sub_end_dates : array – sub-period end dates
    sub_fixings : array – forward rates for each sub-period
    spreads : array or float – spread per sub-period
    gearings : array or float – gearing per sub-period
    averaging_method : str – 'compounding', 'averaging', or 'flat_compounding'
    """
    notional: float
    start_date: float
    end_date: float
    sub_start_dates: jnp.ndarray
    sub_end_dates: jnp.ndarray
    sub_fixings: jnp.ndarray
    spreads: float = 0.0
    gearings: float = 1.0
    averaging_method: str = 'compounding'

    def effective_rate(self):
        """Compute the effective compounded/averaged rate."""
        n = len(self.sub_fixings)
        sub_taus = self.sub_end_dates - self.sub_start_dates

        if self.averaging_method == 'compounding':
            # Full compounding: prod(1 + (g*r + s)*tau) - 1
            product = jnp.ones(())
            for i in range(n):
                sub_rate = self.gearings * self.sub_fixings[i] + self.spreads
                product = product * (1.0 + sub_rate * sub_taus[i])
            total_tau = self.end_date - self.start_date
            return (product - 1.0) / total_tau

        elif self.averaging_method == 'flat_compounding':
            # Flat compounding: compound without spread, add spread at end
            product = jnp.ones(())
            for i in range(n):
                product = product * (1.0 + self.gearings * self.sub_fixings[i] * sub_taus[i])
            total_tau = self.end_date - self.start_date
            return (product - 1.0) / total_tau + self.spreads

        else:  # averaging
            # Simple weighted average
            weighted_sum = jnp.sum(
                (self.gearings * self.sub_fixings + self.spreads) * sub_taus
            )
            total_tau = self.end_date - self.start_date
            return weighted_sum / total_tau

    def amount(self):
        """Cash flow amount."""
        tau = self.end_date - self.start_date
        return self.notional * tau * self.effective_rate()


def sub_periods_leg_npv(coupons, payment_dates, discount_fn):
    """NPV of a leg of sub-period coupons.

    Parameters
    ----------
    coupons : list of SubPeriodsCoupon
    payment_dates : array – payment dates
    discount_fn : callable(t) -> DF

    Returns
    -------
    float – present value
    """
    npv = 0.0
    for i, c in enumerate(coupons):
        npv += c.amount() * discount_fn(payment_dates[i])
    return npv
