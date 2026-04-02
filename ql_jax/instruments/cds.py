"""Credit Default Swap (CDS) instrument.

A CDS pays protection on a reference entity's default in exchange for
periodic premium payments.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class CreditDefaultSwap:
    """Credit Default Swap.

    Parameters
    ----------
    protection_side : 'buyer' or 'seller'
    notional : notional amount
    spread : CDS spread (annual, in decimal)
    maturity : maturity in years
    payment_frequency : premium payment frequency in years (0.25 = quarterly)
    recovery_rate : assumed recovery rate on default
    """
    protection_side: str  # 'buyer' or 'seller'
    notional: float
    spread: float
    maturity: float
    payment_frequency: float = 0.25
    recovery_rate: float = 0.4

    @property
    def n_periods(self):
        return int(round(self.maturity / self.payment_frequency))

    @property
    def payment_dates(self):
        n = self.n_periods
        return jnp.array([(i + 1) * self.payment_frequency for i in range(n)],
                          dtype=jnp.float64)


def make_cds(notional, spread, maturity, recovery_rate=0.4, frequency=0.25):
    """Create a standard CDS (protection buyer side).

    Parameters
    ----------
    notional : notional amount
    spread : annual spread
    maturity : in years
    recovery_rate : LGD = 1 - recovery_rate
    frequency : payment frequency

    Returns
    -------
    CreditDefaultSwap
    """
    return CreditDefaultSwap(
        protection_side='buyer',
        notional=notional,
        spread=spread,
        maturity=maturity,
        payment_frequency=frequency,
        recovery_rate=recovery_rate,
    )
