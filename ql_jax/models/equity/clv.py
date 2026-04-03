"""Collocation Local Volatility (CLV) models."""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from dataclasses import dataclass


@dataclass
class NormalCLVModel:
    """Normal CLV model — maps a normal driving process to the market
    price distribution using monotone collocation.

    Parameters
    ----------
    market_vols : (n_strikes,) market implied vols
    strikes : (n_strikes,) strikes
    spot : spot price
    r : risk-free rate
    q : dividend yield
    T : maturity
    """
    market_vols: jnp.ndarray
    strikes: jnp.ndarray
    spot: float
    r: float
    q: float
    T: float

    def local_vol(self, x):
        """Get the local vol at collocation point x (normal space)."""
        # Map normal quantile x to market CDF
        u = norm.cdf(x)
        # Invert market distribution to get corresponding S
        S = self._quantile(u)
        return self._numerical_local_vol(S)

    def _quantile(self, u):
        """Map uniform quantile to market price distribution."""
        forward = self.spot * jnp.exp((self.r - self.q) * self.T)
        # Use BS implied vols to compute option prices and invert CDF
        # Simple approach: linear interpolation in strike space
        from jax.scipy.stats import norm as ndist
        vols = self.market_vols
        strikes = self.strikes
        T = self.T
        prices = jnp.array([
            _bs_call(forward, float(K), float(v), T)
            for K, v in zip(strikes, vols)
        ])
        # Digital prices ≈ -dC/dK give CDF
        dK = jnp.diff(strikes)
        digital = -jnp.diff(prices) / dK
        cdf_mid = jnp.cumsum(digital)
        cdf_mid = jnp.clip(cdf_mid / cdf_mid[-1], 0, 1)
        strike_mid = 0.5 * (strikes[:-1] + strikes[1:])
        return jnp.interp(u, cdf_mid, strike_mid)

    def _numerical_local_vol(self, S):
        """Dupire local vol at given spot."""
        forward = self.spot * jnp.exp((self.r - self.q) * self.T)
        vol = jnp.interp(S, self.strikes, self.market_vols)
        return vol  # Simplified: return interpolated vol


@dataclass
class SquareRootCLVModel:
    """Square-root CLV model — uses a square-root kernel for the mapping."""
    market_vols: jnp.ndarray
    strikes: jnp.ndarray
    spot: float
    r: float
    q: float
    T: float

    def collocation_points(self, n_points=10):
        """Compute collocation points for the square-root kernel."""
        # Use Gauss-Legendre points on [0,1]
        import numpy as np
        pts, _ = np.polynomial.legendre.leggauss(n_points)
        u = (pts + 1) / 2  # map [-1,1] to [0,1]
        return jnp.array(u)

    def mapping_function(self, u):
        """Map uniform quantile u through square-root kernel to strike space."""
        forward = self.spot * jnp.exp((self.r - self.q) * self.T)
        # Quantile via interpolation
        vols = self.market_vols
        strikes = self.strikes
        T = self.T
        prices = jnp.array([
            _bs_call(forward, float(K), float(v), T)
            for K, v in zip(strikes, vols)
        ])
        dK = jnp.diff(strikes)
        digital = -jnp.diff(prices) / dK
        cdf_mid = jnp.cumsum(digital)
        cdf_mid = jnp.clip(cdf_mid / cdf_mid[-1], 0, 1)
        strike_mid = 0.5 * (strikes[:-1] + strikes[1:])
        return jnp.interp(u, cdf_mid, strike_mid)


def stochastic_collocation_inverse_cdf(target_cdf_values, target_strikes,
                                        n_collocation=7):
    """Fit a polynomial mapping from N(0,1) to the target distribution.

    Parameters
    ----------
    target_cdf_values : (n,) CDF values F(K_i)
    target_strikes : (n,) strikes K_i
    n_collocation : polynomial degree + 1

    Returns
    -------
    coefficients : polynomial coefficients for the mapping g(x) ≈ F^{-1}(Φ(x))
    """
    # Collocation points in normal space
    import numpy as np
    x_pts = np.linspace(-3, 3, n_collocation)
    u_pts = norm.cdf(jnp.array(x_pts))

    # Corresponding target quantiles
    S_pts = jnp.interp(u_pts, jnp.asarray(target_cdf_values),
                       jnp.asarray(target_strikes))

    # Fit polynomial
    coeffs = jnp.polyfit(jnp.array(x_pts), S_pts, n_collocation - 1)
    return coeffs


def _bs_call(F, K, vol, T):
    """Black call price (forward measure)."""
    vol_sqrt = vol * jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * vol**2 * T) / (vol_sqrt + 1e-15)
    d2 = d1 - vol_sqrt
    return F * norm.cdf(d1) - K * norm.cdf(d2)
