"""Zero coupon swap.

Fixed leg pays a single zero coupon amount at maturity.
Float leg makes periodic payments.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class ZeroCouponSwap:
    """Zero coupon swap.

    Parameters
    ----------
    notional : notional
    fixed_rate : annualized fixed rate (compounded to maturity)
    maturity : swap maturity in years
    float_payment_times : float leg schedule
    float_accrual : float leg accrual fractions
    payer_fixed : True if paying fixed (zero coupon) leg
    compounding : 'simple' or 'compounded'
    """
    notional: float
    fixed_rate: float
    maturity: float
    float_payment_times: jnp.ndarray
    float_accrual: jnp.ndarray
    payer_fixed: bool = True
    compounding: str = 'compounded'

    def npv(self, forward_rates, discount_fn):
        omega = 1.0 if self.payer_fixed else -1.0
        N = self.notional

        # Fixed (zero coupon) leg: N * [(1+K)^T - 1] at maturity
        if self.compounding == 'simple':
            fixed_amount = N * self.fixed_rate * self.maturity
        else:
            fixed_amount = N * ((1.0 + self.fixed_rate)**self.maturity - 1.0)
        fixed_pv = fixed_amount * float(discount_fn(self.maturity))

        # Float leg
        float_pv = sum(
            N * float(forward_rates[i]) * float(self.float_accrual[i]) *
            float(discount_fn(float(self.float_payment_times[i])))
            for i in range(len(self.float_payment_times))
        )

        return omega * (float_pv - fixed_pv)
