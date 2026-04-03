"""FX-style delta-quoted Black volatility surface and Heston-implied vol surface."""

from __future__ import annotations

import jax.numpy as jnp
from dataclasses import dataclass
from jax.scipy.stats import norm


@dataclass
class BlackVolSurfaceDelta:
    """Black volatility surface parameterised by delta and expiry.

    In FX markets, implied vols are quoted by delta (e.g., 25-delta put,
    ATM, 25-delta call) rather than by absolute strike. This structure
    stores the vol grid in delta/expiry space and converts to strike
    on the fly.

    Parameters
    ----------
    expiries : 1-D array of option tenors (year fractions)
    deltas : 1-D array of delta values (e.g., [0.10, 0.25, 0.50, 0.75, 0.90])
    vols : 2-D array shape (len(expiries), len(deltas))
    spot : current spot FX rate
    r_dom : domestic rate
    r_for : foreign rate
    delta_type : 'spot' or 'forward'
    """
    expiries: jnp.ndarray
    deltas: jnp.ndarray
    vols: jnp.ndarray
    spot: float
    r_dom: float = 0.0
    r_for: float = 0.0
    delta_type: str = "spot"

    def volatility(self, T: float, delta: float) -> float:
        """Interpolate volatility at given expiry and delta."""
        # Bilinear interpolation in (T, delta) space
        t_idx = jnp.searchsorted(self.expiries, T)
        t_idx = jnp.clip(t_idx, 1, len(self.expiries) - 1)
        d_idx = jnp.searchsorted(self.deltas, delta)
        d_idx = jnp.clip(d_idx, 1, len(self.deltas) - 1)

        # Weights
        wt = (T - self.expiries[t_idx - 1]) / (self.expiries[t_idx] - self.expiries[t_idx - 1] + 1e-15)
        wt = jnp.clip(wt, 0.0, 1.0)
        wd = (delta - self.deltas[d_idx - 1]) / (self.deltas[d_idx] - self.deltas[d_idx - 1] + 1e-15)
        wd = jnp.clip(wd, 0.0, 1.0)

        v00 = self.vols[t_idx - 1, d_idx - 1]
        v01 = self.vols[t_idx - 1, d_idx]
        v10 = self.vols[t_idx, d_idx - 1]
        v11 = self.vols[t_idx, d_idx]

        vol = (1 - wt) * ((1 - wd) * v00 + wd * v01) + wt * ((1 - wd) * v10 + wd * v11)
        return float(vol)

    def strike_from_delta(self, T: float, delta: float) -> float:
        """Convert delta to strike using BS formula inversion.

        For a call: delta = e^{-r_for*T} * N(d1) =>
            d1 = N^{-1}(delta * e^{r_for*T})
            K = F * exp(-d1 * sigma * sqrt(T) + 0.5 * sigma^2 * T)
        """
        vol = self.volatility(T, delta)
        F = self.spot * jnp.exp((self.r_dom - self.r_for) * T)
        sqrt_T = jnp.sqrt(T)

        if self.delta_type == "forward":
            d1 = float(norm.ppf(delta))
        else:
            d1 = float(norm.ppf(delta * jnp.exp(self.r_for * T)))

        K = F * jnp.exp(-d1 * vol * sqrt_T + 0.5 * vol**2 * T)
        return float(K)

    def volatility_strike(self, T: float, K: float) -> float:
        """Get vol for a given (T, K) by inverting the delta mapping."""
        F = self.spot * jnp.exp((self.r_dom - self.r_for) * T)
        # Approximate delta from BS
        vol_atm = self.volatility(T, 0.5)
        sqrt_T = jnp.sqrt(jnp.maximum(T, 1e-10))
        d1 = (jnp.log(F / K) + 0.5 * vol_atm**2 * T) / (vol_atm * sqrt_T)
        delta_approx = float(norm.cdf(d1))
        return self.volatility(T, delta_approx)


@dataclass
class HestonBlackVolSurface:
    """Black implied volatility surface derived from a calibrated Heston model.

    Given Heston parameters, produces implied vols for any (T, K) by
    pricing via the characteristic function and inverting to Black vol.

    Parameters
    ----------
    spot : current spot price
    r, q : rates
    v0 : initial variance
    kappa, theta, xi, rho : Heston parameters
    """
    spot: float
    r: float
    q: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def volatility(self, T: float, K: float) -> float:
        """Compute Heston-implied Black vol at (T, K)."""
        from ql_jax.engines.analytic.heston import heston_price

        price = heston_price(
            self.spot, K, T, self.r, self.q,
            self.v0, self.kappa, self.theta, self.xi, self.rho,
            option_type=1,
        )

        # Invert to find implied Black vol using Newton's method
        from ql_jax.engines.analytic.black_formula import black_price as bs_price

        sigma = 0.2  # initial guess
        F = self.spot * jnp.exp((self.r - self.q) * T)
        df = jnp.exp(-self.r * T)
        for _ in range(50):
            bs = bs_price(F, K, T, sigma, df, option_type=1)
            vega = self._bs_vega(self.spot, K, T, self.r, self.q, sigma)
            if abs(vega) < 1e-15:
                break
            sigma = sigma - float(bs - price) / vega
            sigma = max(sigma, 1e-6)
            if abs(float(bs) - float(price)) < 1e-12:
                break

        return sigma

    @staticmethod
    def _bs_vega(S, K, T, r, q, sigma):
        """Black-Scholes vega."""
        sqrt_T = jnp.sqrt(T)
        d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        return float(S * jnp.exp(-q * T) * norm.pdf(d1) * sqrt_T)

    def black_variance(self, T: float, K: float) -> float:
        """Return total Black variance sigma^2 * T."""
        sigma = self.volatility(T, K)
        return sigma * sigma * T
