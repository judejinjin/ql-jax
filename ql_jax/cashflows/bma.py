"""Average BMA (Bond Market Association) coupon – weighted average of BMA rates."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass(frozen=True)
class AverageBMACoupon:
    """Average BMA (municipal swap index) coupon.

    Pays notional * tau * weighted_avg(bma_rates).

    Parameters
    ----------
    notional : float
    start_date : float
    end_date : float
    bma_rates : array – weekly BMA reset rates
    reset_dates : array – dates at which BMA resets
    """
    notional: float
    start_date: float
    end_date: float
    bma_rates: jnp.ndarray
    reset_dates: jnp.ndarray

    def weighted_average_rate(self):
        """Compute time-weighted average of BMA rates over accrual period."""
        n = len(self.bma_rates)
        if n == 0:
            return 0.0
        if n == 1:
            return self.bma_rates[0]

        # Time-weighted average
        weights = jnp.zeros(n)
        for i in range(n):
            t_start = jnp.maximum(self.reset_dates[i], self.start_date)
            t_end = self.reset_dates[i + 1] if i < n - 1 else self.end_date
            t_end = jnp.minimum(t_end, self.end_date)
            weights = weights.at[i].set(jnp.maximum(t_end - t_start, 0.0))

        total_weight = jnp.sum(weights)
        return jnp.where(
            total_weight > 0,
            jnp.sum(self.bma_rates * weights) / total_weight,
            0.0,
        )

    def accrual(self):
        """Day count fraction."""
        return self.end_date - self.start_date

    def amount(self):
        """Cash flow amount."""
        return self.notional * self.accrual() * self.weighted_average_rate()


def bma_swap_npv(fixed_rate, bma_coupons, payment_dates, discount_fn):
    """Price a BMA swap (fixed vs BMA floating).

    Parameters
    ----------
    fixed_rate : float
    bma_coupons : list of AverageBMACoupon
    payment_dates : array – payment dates for fixed and floating legs
    discount_fn : callable(t) -> DF

    Returns
    -------
    float – NPV (positive = pay fixed)
    """
    # Floating leg
    floating_npv = 0.0
    for i, coupon in enumerate(bma_coupons):
        floating_npv += coupon.amount() * discount_fn(payment_dates[i])

    # Fixed leg (same schedule)
    fixed_npv = 0.0
    for i, coupon in enumerate(bma_coupons):
        tau = coupon.accrual()
        fixed_npv += coupon.notional * tau * fixed_rate * discount_fn(payment_dates[i])

    return floating_npv - fixed_npv
