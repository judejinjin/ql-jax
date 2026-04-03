"""Equity total return swap."""

from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class EquityTotalReturnSwap:
    """Total return swap on equity.

    Equity leg pays total return (price + dividends).
    Financing leg pays LIBOR + spread.

    Parameters
    ----------
    notional : notional amount
    S0 : initial equity price
    payment_times : payment schedule
    accrual_fractions : accrual periods
    financing_spread : spread over LIBOR
    payer_equity : True if paying equity return
    """
    notional: float
    S0: float
    payment_times: jnp.ndarray
    accrual_fractions: jnp.ndarray
    financing_spread: float = 0.0
    payer_equity: bool = True

    def npv(self, S_current, forward_rates, dividend_yield, discount_fn):
        """NPV of the TRS.

        Parameters
        ----------
        S_current : current equity price
        forward_rates : LIBOR forward rates
        dividend_yield : annualized dividend yield
        discount_fn : discount factor function
        """
        omega = 1.0 if self.payer_equity else -1.0
        N = self.notional

        # Equity leg: total return = price return + dividends
        equity_return = (S_current / self.S0 - 1.0)
        T_final = float(self.payment_times[-1])
        equity_pv = N * equity_return * float(discount_fn(T_final))

        # Financing leg: sum of (LIBOR + spread) * tau * discount
        fin_pv = sum(
            N * (float(forward_rates[i]) + self.financing_spread) *
            float(self.accrual_fractions[i]) *
            float(discount_fn(float(self.payment_times[i])))
            for i in range(len(self.payment_times))
        )

        return omega * (fin_pv - equity_pv)
