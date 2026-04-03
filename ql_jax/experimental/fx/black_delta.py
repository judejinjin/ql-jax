"""FX experimental — Black delta calculator and GK support."""

import jax.numpy as jnp
from jax.scipy.stats import norm
from dataclasses import dataclass


@dataclass
class BlackDeltaCalculator:
    """Delta and strike conversions for FX options using Garman-Kohlhagen.

    Supports spot delta, forward delta, and various delta conventions.

    Parameters
    ----------
    option_type : 1 for call, -1 for put
    delta_type : 'spot', 'forward', 'pips_spot', 'pips_forward'
    spot : FX spot rate
    dom_df : domestic discount factor to expiry
    for_df : foreign discount factor to expiry
    vol : implied volatility
    T : time to expiry
    """
    option_type: int
    delta_type: str
    spot: float
    dom_df: float
    for_df: float
    vol: float
    T: float

    @property
    def forward(self):
        return self.spot * self.for_df / self.dom_df

    def strike_from_delta(self, delta):
        """Compute strike from delta value.

        Parameters
        ----------
        delta : delta value (e.g. 0.25 for 25-delta)

        Returns
        -------
        strike : corresponding strike
        """
        phi = self.option_type
        F = self.forward
        vol = self.vol
        T = self.T
        vol_sqrt_T = vol * jnp.sqrt(T)

        if self.delta_type == 'forward':
            # Forward delta: delta = phi * N(phi * d1)
            d1 = phi * norm.ppf(phi * delta)
            K = F * jnp.exp(-d1 * vol_sqrt_T + 0.5 * vol**2 * T)
        elif self.delta_type == 'spot':
            # Spot delta: delta = phi * for_df * N(phi * d1)
            d1 = phi * norm.ppf(phi * delta / self.for_df)
            K = F * jnp.exp(-d1 * vol_sqrt_T + 0.5 * vol**2 * T)
        elif self.delta_type == 'pips_forward':
            # Pips forward delta: delta = phi * K/F * N(phi * d2)
            # Need iterative solve
            K = self._pips_strike(delta, forward=True)
        elif self.delta_type == 'pips_spot':
            K = self._pips_strike(delta, forward=False)
        else:
            raise ValueError(f"Unknown delta type: {self.delta_type}")

        return float(K)

    def delta_from_strike(self, K):
        """Compute delta from strike.

        Parameters
        ----------
        K : strike price

        Returns
        -------
        delta : option delta
        """
        phi = self.option_type
        F = self.forward
        vol = self.vol
        T = self.T
        vol_sqrt_T = vol * jnp.sqrt(T)

        d1 = (jnp.log(F / K) + 0.5 * vol**2 * T) / vol_sqrt_T

        if self.delta_type == 'forward':
            return float(phi * norm.cdf(phi * d1))
        elif self.delta_type == 'spot':
            return float(phi * self.for_df * norm.cdf(phi * d1))
        elif self.delta_type in ('pips_forward', 'pips_spot'):
            d2 = d1 - vol_sqrt_T
            if self.delta_type == 'pips_forward':
                return float(-phi * norm.cdf(-phi * d2))
            else:
                return float(-phi * self.dom_df * norm.cdf(-phi * d2))
        else:
            raise ValueError(f"Unknown delta type: {self.delta_type}")

    def atm_strike(self, atm_type='delta_neutral'):
        """Compute ATM strike.

        Parameters
        ----------
        atm_type : 'delta_neutral', 'forward', 'spot'

        Returns
        -------
        K_atm : ATM strike
        """
        F = self.forward
        vol = self.vol
        T = self.T

        if atm_type == 'forward':
            return float(F)
        elif atm_type == 'delta_neutral':
            return float(F * jnp.exp(0.5 * vol**2 * T))
        elif atm_type == 'spot':
            return float(self.spot)
        else:
            raise ValueError(f"Unknown ATM type: {atm_type}")

    def _pips_strike(self, delta, forward=True):
        """Iterative solve for pips delta strike."""
        F = self.forward
        vol = self.vol
        T = self.T
        phi = self.option_type
        K = F  # initial guess

        for _ in range(20):
            d = self.delta_from_strike(K)
            err = d - delta
            # Newton step: d(delta)/d(K)
            vol_sqrt_T = vol * jnp.sqrt(T)
            d1 = (jnp.log(F / K) + 0.5 * vol**2 * T) / vol_sqrt_T
            dd_dK = -norm.pdf(d1) / (K * vol_sqrt_T)
            if forward:
                dd_dK = dd_dK
            else:
                dd_dK = dd_dK * self.dom_df
            if abs(float(dd_dK)) < 1e-15:
                break
            K = K - err / dd_dK
            K = jnp.maximum(K, 1e-6)

        return float(K)


def garman_kohlhagen_price(S, K, T, r_dom, r_for, vol, option_type=1):
    """Garman-Kohlhagen FX option price.

    C = S * e^{-r_for*T} * N(d1) - K * e^{-r_dom*T} * N(d2)

    Parameters
    ----------
    S : spot FX rate
    K : strike
    T : time to expiry
    r_dom : domestic risk-free rate
    r_for : foreign risk-free rate
    vol : volatility
    option_type : 1=call, -1=put
    """
    phi = option_type
    vol_sqrt_T = vol * jnp.sqrt(T)
    d1 = (jnp.log(S / K) + (r_dom - r_for + 0.5 * vol**2) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T

    price = phi * (S * jnp.exp(-r_for * T) * norm.cdf(phi * d1) -
                    K * jnp.exp(-r_dom * T) * norm.cdf(phi * d2))
    return float(jnp.maximum(price, 0.0))
