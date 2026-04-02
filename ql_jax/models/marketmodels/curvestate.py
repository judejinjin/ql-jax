"""Market model curve states: LMM and Coterminal.

CurveState tracks the forward rates and swap rates at a given time step,
handling the mapping between different rate representations.
"""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LMMCurveState:
    """LIBOR Market Model curve state.

    Stores forward rates at current time and provides derived quantities.

    Attributes
    ----------
    rate_times : array – [T0, T1, ..., Tn] tenor times
    forward_rates : array – [f0, f1, ..., f_{n-1}] forward rates
    """
    rate_times: jnp.ndarray
    forward_rates: jnp.ndarray

    @property
    def n_rates(self):
        return len(self.forward_rates)

    def accruals(self):
        """Year fractions tau_i = T_{i+1} - T_i."""
        return jnp.diff(self.rate_times)

    def discount_ratio(self, i, j):
        """P(T_i) / P(T_j) = prod_{k=i}^{j-1} 1/(1 + tau_k * f_k)."""
        taus = self.accruals()
        ratio = 1.0
        for k in range(i, j):
            ratio /= (1.0 + taus[k] * self.forward_rates[k])
        return ratio

    def swap_rate(self, start, end):
        """Par swap rate from T_start to T_end."""
        taus = self.accruals()
        annuity = 0.0
        for k in range(start, end):
            annuity += taus[k] * self.discount_ratio(0, k + 1)
        return (self.discount_ratio(0, start) - self.discount_ratio(0, end)) / annuity

    def coterminal_swap_rates(self):
        """All coterminal swap rates: S(i, n) for i = 0, ..., n-1."""
        n = self.n_rates
        return jnp.array([self.swap_rate(i, n) for i in range(n)])

    def annuity(self, start, end):
        """Swap annuity from T_start to T_end."""
        taus = self.accruals()
        return sum(taus[k] * self.discount_ratio(0, k + 1) for k in range(start, end))

    def with_new_rates(self, new_rates):
        """Return new state with updated forward rates."""
        return LMMCurveState(rate_times=self.rate_times, forward_rates=new_rates)


@dataclass(frozen=True)
class CoterminalCurveState:
    """Coterminal swap market curve state.

    Tracks coterminal swap rates and derives forward LIBOR rates.
    """
    rate_times: jnp.ndarray
    swap_rates: jnp.ndarray

    @property
    def n_rates(self):
        return len(self.swap_rates)

    def accruals(self):
        return jnp.diff(self.rate_times)

    def forward_rates(self):
        """Derive LIBOR rates from coterminal swap rates.

        Uses bootstrap from the longest swap backward.
        """
        n = self.n_rates
        taus = self.accruals()
        fwds = jnp.zeros(n)

        # Start from the last rate: S(n-1, n) = f_{n-1}
        fwds = fwds.at[n - 1].set(self.swap_rates[n - 1])

        # Backward bootstrap
        annuity_remainder = taus[n - 1] / (1.0 + taus[n - 1] * fwds[n - 1])
        for i in range(n - 2, -1, -1):
            # S(i, n) * annuity(i, n) = 1 - P(T_n)/P(T_i)
            # annuity(i, n) = sum_{k=i}^{n-1} tau_k * P(T_{k+1})/P(T_i)
            s = self.swap_rates[i]
            # f_i = (1 + s * annuity_remainder) / tau_i - 1/tau_i (simplified)
            fwd = (1.0 / (taus[i] * (1.0 + s * annuity_remainder)) - 1.0) * (1.0 / taus[i])
            fwds = fwds.at[i].set(jnp.maximum(fwd, 0.0))
            annuity_remainder += taus[i] / (1.0 + taus[i] * fwds[i])

        return fwds
