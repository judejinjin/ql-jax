"""BMA swap instrument.

BMA (Bond Market Association) / SIFMA swap: floating leg pays a
municipal bond rate against a LIBOR/SOFR-based floating leg.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class BmaSwap:
    """BMA/SIFMA swap.

    Parameters
    ----------
    notional : notional amount
    bma_payment_times : BMA leg payment schedule (weekly resets)
    libor_payment_times : LIBOR leg payment schedule
    bma_accrual : BMA accrual fractions
    libor_accrual : LIBOR accrual fractions
    libor_spread : spread on LIBOR leg
    payer_bma : True if paying BMA leg
    """
    notional: float
    bma_payment_times: jnp.ndarray
    libor_payment_times: jnp.ndarray
    bma_accrual: jnp.ndarray
    libor_accrual: jnp.ndarray
    libor_spread: float = 0.0
    payer_bma: bool = True

    def npv(self, bma_rates, libor_rates, discount_fn):
        """NPV given forward BMA and LIBOR rates."""
        omega = 1.0 if self.payer_bma else -1.0
        N = self.notional
        bma_pv = sum(
            N * float(bma_rates[i]) * float(self.bma_accrual[i]) *
            float(discount_fn(float(self.bma_payment_times[i])))
            for i in range(len(self.bma_payment_times))
        )
        libor_pv = sum(
            N * (float(libor_rates[i]) + self.libor_spread) *
            float(self.libor_accrual[i]) *
            float(discount_fn(float(self.libor_payment_times[i])))
            for i in range(len(self.libor_payment_times))
        )
        return omega * (libor_pv - bma_pv)

    def bma_ratio(self, bma_rates, libor_rates):
        """BMA/LIBOR ratio (typical ~67%)."""
        avg_bma = jnp.mean(jnp.asarray(bma_rates))
        avg_libor = jnp.mean(jnp.asarray(libor_rates))
        return avg_bma / avg_libor
