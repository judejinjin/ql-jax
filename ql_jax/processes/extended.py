"""Process extensions: GSR, Hybrid Heston-HW, Forward Measure, Joint Stochastic."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class GSRProcess:
    """GSR (Gaussian Short Rate) process.

    dx(t) = -kappa(t) * x(t) dt + sigma(t) dW
    r(t) = x(t) + alpha(t)

    where alpha(t) is chosen to fit the initial term structure.

    Parameters
    ----------
    sigma_fn : callable(t) -> sigma(t)  (or constant float)
    kappa : float (mean-reversion speed)
    """
    sigma_fn: object  # callable or float
    kappa: float

    def sigma(self, t):
        if callable(self.sigma_fn):
            return self.sigma_fn(t)
        return self.sigma_fn

    def drift(self, t, x):
        return -self.kappa * x

    def diffusion(self, t, x):
        return self.sigma(t)

    def variance(self, t0, t1):
        """Integrated variance from t0 to t1."""
        if callable(self.sigma_fn):
            from ql_jax.math.integrals import simpson_integral
            return simpson_integral(lambda s: self.sigma(s) ** 2, t0, t1, n=50)
        return self.sigma_fn ** 2 * (t1 - t0)

    def evolve(self, t, x, dt, dw):
        """Euler evolution."""
        return x + self.drift(t, x) * dt + self.diffusion(t, x) * jnp.sqrt(dt) * dw


@dataclass
class HybridHestonHullWhiteProcess:
    """Hybrid Heston + Hull-White process for equity with stochastic rates.

    dS/S = r dt + sqrt(V) dW_1
    dV   = kappa*(theta - V)dt + sigma*sqrt(V) dW_2
    dr   = (theta_r(t) - a*r)dt + sigma_r * dW_3

    Correlations: rho_SV, rho_Sr, rho_Vr.

    Parameters
    ----------
    S0 : float, initial spot
    v0, kappa, theta, sigma, rho_sv : Heston params
    a, sigma_r : Hull-White params
    rho_sr : correlation S-r
    rho_vr : correlation V-r
    r0 : initial short rate
    """
    S0: float
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho_sv: float
    a: float
    sigma_r: float
    rho_sr: float = 0.0
    rho_vr: float = 0.0
    r0: float = 0.05

    def correlation_matrix(self):
        rho12 = self.rho_sv
        rho13 = self.rho_sr
        rho23 = self.rho_vr
        return jnp.array([
            [1.0, rho12, rho13],
            [rho12, 1.0, rho23],
            [rho13, rho23, 1.0],
        ])

    def evolve(self, t, state, dt, dw3):
        """Euler step. state = (S, V, r), dw3 = (dW1, dW2, dW3) correlated."""
        S, V, r = state
        dW1, dW2, dW3 = dw3
        sqrt_dt = jnp.sqrt(dt)
        V_pos = jnp.maximum(V, 0.0)

        S_new = S * jnp.exp(
            (r - 0.5 * V_pos) * dt + jnp.sqrt(V_pos) * sqrt_dt * dW1
        )
        V_new = V + self.kappa * (self.theta - V_pos) * dt + self.sigma * jnp.sqrt(V_pos) * sqrt_dt * dW2
        r_new = r + self.a * (self.r0 - r) * dt + self.sigma_r * sqrt_dt * dW3

        return (S_new, jnp.maximum(V_new, 0.0), r_new)


@dataclass
class ForwardMeasureProcess:
    """Forward measure process wrapper.

    Transforms a base process to forward measure with a given numeraire.

    Parameters
    ----------
    base_process : process with drift(t, x) and diffusion(t, x)
    bond_vol_fn : callable(t) -> bond volatility for change of measure
    """
    base_process: object
    bond_vol_fn: object  # callable or float

    def drift(self, t, x):
        base_drift = self.base_process.drift(t, x)
        sigma = self.base_process.diffusion(t, x)
        bv = self.bond_vol_fn(t) if callable(self.bond_vol_fn) else self.bond_vol_fn
        # Forward measure drift = base drift + sigma * bond_vol (Girsanov)
        return base_drift + sigma * bv

    def diffusion(self, t, x):
        return self.base_process.diffusion(t, x)


@dataclass
class MfStateProcess:
    """Markov-functional model state process.

    dx(t) = sigma(t) * dW under the terminal measure.
    x(0) = 0.

    Parameters
    ----------
    sigma_fn : callable(t) -> float, or float constant
    """
    sigma_fn: object

    def sigma(self, t):
        if callable(self.sigma_fn):
            return self.sigma_fn(t)
        return self.sigma_fn

    def drift(self, t, x):
        return 0.0

    def diffusion(self, t, x):
        return self.sigma(t)

    def variance(self, t0, t1):
        if callable(self.sigma_fn):
            from ql_jax.math.integrals import simpson_integral
            return simpson_integral(lambda s: self.sigma(s) ** 2, t0, t1, n=50)
        return self.sigma_fn ** 2 * (t1 - t0)
