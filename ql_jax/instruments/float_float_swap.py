"""Float-float swap instrument.

A swap with two floating legs potentially on different indexes.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class FloatFloatSwap:
    """Two-floating-leg swap.

    Parameters
    ----------
    notional : notional amount
    payment_times_1 : payment schedule for leg 1
    payment_times_2 : payment schedule for leg 2
    accrual_1 : accrual fractions for leg 1
    accrual_2 : accrual fractions for leg 2
    spread_1 : spread on leg 1
    spread_2 : spread on leg 2
    payer_leg1 : True if paying leg 1
    """
    notional: float
    payment_times_1: jnp.ndarray
    payment_times_2: jnp.ndarray
    accrual_1: jnp.ndarray
    accrual_2: jnp.ndarray
    spread_1: float = 0.0
    spread_2: float = 0.0
    payer_leg1: bool = True

    def npv(self, forward_rates_1, forward_rates_2, discount_fn):
        """Net present value given forward rates for each leg.

        Parameters
        ----------
        forward_rates_1 : array of forward rates for leg 1
        forward_rates_2 : array of forward rates for leg 2
        discount_fn : callable(t) -> discount factor
        """
        omega = 1.0 if self.payer_leg1 else -1.0
        N = self.notional

        pv1 = sum(
            N * (float(forward_rates_1[i]) + self.spread_1) *
            float(self.accrual_1[i]) * float(discount_fn(float(self.payment_times_1[i])))
            for i in range(len(self.payment_times_1))
        )
        pv2 = sum(
            N * (float(forward_rates_2[i]) + self.spread_2) *
            float(self.accrual_2[i]) * float(discount_fn(float(self.payment_times_2[i])))
            for i in range(len(self.payment_times_2))
        )
        return omega * (pv2 - pv1)
