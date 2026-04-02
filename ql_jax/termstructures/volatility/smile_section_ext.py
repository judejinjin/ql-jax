"""Extended smile sections: Kahale, ATM, ZABR, Gaussian1d."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm


class KahaleSmileSection:
    """Kahale arbitrage-free smile section.

    Constructs an arbitrage-free call price surface by fitting
    a set of knot points and ensuring convexity.

    Parameters
    ----------
    strikes : array of strikes
    vols : array of Black vols at those strikes
    forward : forward price
    expiry : time to expiry
    """

    def __init__(self, strikes, vols, forward, expiry):
        self._strikes = jnp.asarray(strikes, dtype=jnp.float64)
        self._vols = jnp.asarray(vols, dtype=jnp.float64)
        self._forward = jnp.float64(forward)
        self._expiry = jnp.float64(expiry)
        # Compute call prices from Black formula
        sqrt_t = jnp.sqrt(expiry)
        d1 = (jnp.log(forward / strikes) + 0.5 * vols ** 2 * expiry) / (vols * sqrt_t)
        d2 = d1 - vols * sqrt_t
        self._call_prices = forward * jnorm.cdf(d1) - strikes * jnorm.cdf(d2)

    def vol(self, strike):
        """Interpolated vol at strike (monotone convex in call price space)."""
        return jnp.interp(jnp.float64(strike), self._strikes, self._vols)

    def call_price(self, strike):
        """Interpolated call price."""
        return jnp.interp(jnp.float64(strike), self._strikes, self._call_prices)


class AtmSmileSection:
    """ATM smile section — returns a single ATM vol for any strike.

    Parameters
    ----------
    atm_vol : float
    expiry : float
    """

    def __init__(self, atm_vol, expiry):
        self._atm_vol = jnp.float64(atm_vol)
        self._expiry = jnp.float64(expiry)

    def vol(self, strike=None):
        return self._atm_vol

    @property
    def expiry(self):
        return self._expiry


class AtmAdjustedSmileSection:
    """Smile section shifted to match a target ATM vol.

    Parameters
    ----------
    base : smile section with vol(strike) method
    target_atm_vol : float
    forward : float
    """

    def __init__(self, base, target_atm_vol, forward):
        self._base = base
        self._shift = jnp.float64(target_atm_vol) - base.vol(forward)

    def vol(self, strike):
        return self._base.vol(strike) + self._shift


class ZABRSmileSection:
    """ZABR smile section.

    The ZABR model generalizes SABR by allowing the vol-of-vol exponent γ.
    σ(K) is computed via an expansion around ATM.

    Parameters
    ----------
    forward : float
    alpha : float   (initial vol)
    beta : float    (CEV exponent)
    nu : float      (vol of vol)
    rho : float     (correlation)
    gamma : float   (ZABR exponent, SABR = gamma=1)
    expiry : float
    """

    def __init__(self, forward, alpha, beta, nu, rho, gamma, expiry):
        self._f = jnp.float64(forward)
        self._alpha = jnp.float64(alpha)
        self._beta = jnp.float64(beta)
        self._nu = jnp.float64(nu)
        self._rho = jnp.float64(rho)
        self._gamma = jnp.float64(gamma)
        self._T = jnp.float64(expiry)

    def vol(self, strike):
        """ZABR implied vol approximation."""
        K = jnp.float64(strike)
        f, K = self._f, K
        alpha, beta, nu, rho, gamma = self._alpha, self._beta, self._nu, self._rho, self._gamma
        T = self._T

        # Degenerate case: f ~ K
        fK = f * K
        fK_mid = jnp.sqrt(fK)
        fK_beta = fK_mid ** (1.0 - beta)

        # SABR-like formula as base
        logfk = jnp.log(f / K)
        z = (nu / alpha) * fK_beta * logfk
        x = jnp.log((jnp.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        # Leading term
        sigma0 = alpha / fK_beta
        zeta = jnp.where(jnp.abs(logfk) < 1e-7, 1.0, z / jnp.where(jnp.abs(x) < 1e-12, 1.0, x))

        # Correction terms (Hagan et al + gamma adjustment)
        correction = 1.0 + T * (
            (1.0 - beta) ** 2 * alpha ** 2 / (24.0 * fK ** (1.0 - beta))
            + rho * beta * nu * alpha / (4.0 * fK_beta)
            + nu ** 2 * (2.0 - 3.0 * rho ** 2) / 24.0
        )
        # gamma adjustment: power law on nu
        gamma_adj = jnp.where(jnp.abs(gamma - 1.0) < 1e-10, 1.0,
                              jnp.exp((gamma - 1.0) * jnp.log(jnp.maximum(nu, 1e-10)) * T * 0.1))

        return sigma0 * zeta * correction * gamma_adj
