"""Non-standard swap and swaption instruments.

Supports amortizing schedules, step-up rates, and varying notionals.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class NonstandardSwap:
    """Non-standard interest rate swap.

    Supports varying notionals and step-up fixed rates.

    Parameters
    ----------
    fixed_notionals : notional per fixed period
    floating_notionals : notional per floating period
    fixed_rates : fixed rate per period
    fixed_payment_times : fixed leg payment schedule
    floating_payment_times : float leg payment schedule
    fixed_accrual : accrual fractions for fixed leg
    floating_accrual : accrual fractions for floating leg
    payer : True if paying fixed
    """
    fixed_notionals: jnp.ndarray
    floating_notionals: jnp.ndarray
    fixed_rates: jnp.ndarray
    fixed_payment_times: jnp.ndarray
    floating_payment_times: jnp.ndarray
    fixed_accrual: jnp.ndarray
    floating_accrual: jnp.ndarray
    payer: bool = True

    def npv(self, forward_rates, discount_fn):
        omega = 1.0 if self.payer else -1.0
        fixed_pv = sum(
            float(self.fixed_notionals[i]) * float(self.fixed_rates[i]) *
            float(self.fixed_accrual[i]) * float(discount_fn(float(self.fixed_payment_times[i])))
            for i in range(len(self.fixed_payment_times))
        )
        float_pv = sum(
            float(self.floating_notionals[i]) * float(forward_rates[i]) *
            float(self.floating_accrual[i]) * float(discount_fn(float(self.floating_payment_times[i])))
            for i in range(len(self.floating_payment_times))
        )
        return omega * (float_pv - fixed_pv)


@dataclass(frozen=True)
class NonstandardSwaption:
    """Option on a NonstandardSwap.

    Parameters
    ----------
    underlying : NonstandardSwap
    exercise_time : option expiry
    """
    underlying: NonstandardSwap
    exercise_time: float
