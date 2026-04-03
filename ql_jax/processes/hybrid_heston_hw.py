"""Hybrid Heston-Hull-White process.

3-factor model: Heston SV for equity + Hull-White 1-factor for rates.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class HybridHestonHullWhiteProcess:
    """Heston-Hull-White hybrid process.

    dS = (r(t) - q) S dt + sqrt(v) S dW_s
    dv = kappa*(theta - v) dt + xi*sqrt(v) dW_v
    dr = (theta_r(t) - a*r) dt + sigma_r dW_r

    Correlations: rho_sv between S and v, rho_sr between S and r.

    Parameters
    ----------
    S0 : initial spot
    v0 : initial variance
    r0 : initial short rate
    q : dividend yield
    kappa, theta, xi : Heston parameters
    rho_sv : S-v correlation
    a : HW mean reversion
    sigma_r : HW volatility
    rho_sr : S-r correlation
    """
    S0: float
    v0: float
    r0: float
    q: float = 0.0
    kappa: float = 1.0
    theta: float = 0.04
    xi: float = 0.3
    rho_sv: float = -0.7
    a: float = 0.03
    sigma_r: float = 0.01
    rho_sr: float = 0.1

    def evolve(self, dt, n_paths, key, theta_r_fn=None):
        """Single Euler step.

        Parameters
        ----------
        dt : time step
        n_paths : number of paths
        key : JAX random key
        theta_r_fn : callable(t) -> theta_r(t), defaults to a*r0

        Returns
        -------
        S, v, r : arrays of shape (n_paths,)
        """
        if theta_r_fn is None:
            theta_r_fn = lambda t: self.a * self.r0

        key1, key2, key3 = jax.random.split(key, 3)

        # Correlated Brownian motions
        z1 = jax.random.normal(key1, (n_paths,))
        z2 = jax.random.normal(key2, (n_paths,))
        z3 = jax.random.normal(key3, (n_paths,))

        # Cholesky decomposition for 3 correlated BMs
        dW_s = z1
        dW_v = self.rho_sv * z1 + jnp.sqrt(1.0 - self.rho_sv**2) * z2
        dW_r = (self.rho_sr * z1 +
                (0.0) * z2 +  # assume rho_vr = 0
                jnp.sqrt(jnp.maximum(1.0 - self.rho_sr**2, 0.0)) * z3)

        sqrt_v = jnp.sqrt(jnp.maximum(self.v0, 0.0))
        sqrt_dt = jnp.sqrt(dt)

        S = self.S0 * jnp.exp(
            (self.r0 - self.q - 0.5 * self.v0) * dt
            + sqrt_v * sqrt_dt * dW_s
        )
        v = jnp.maximum(
            self.v0 + self.kappa * (self.theta - self.v0) * dt
            + self.xi * sqrt_v * sqrt_dt * dW_v,
            0.0
        )
        r = (self.r0 + (float(theta_r_fn(0.0)) - self.a * self.r0) * dt
             + self.sigma_r * sqrt_dt * dW_r)

        return S, v, r

    def simulate(self, T, n_steps, n_paths, key, theta_r_fn=None):
        """Full path simulation.

        Returns
        -------
        S_paths : (n_steps+1, n_paths)
        v_paths : (n_steps+1, n_paths)
        r_paths : (n_steps+1, n_paths)
        """
        if theta_r_fn is None:
            theta_r_fn = lambda t: self.a * self.r0

        dt = T / n_steps
        sqrt_dt = jnp.sqrt(dt)

        S = jnp.ones(n_paths) * self.S0
        v = jnp.ones(n_paths) * self.v0
        r = jnp.ones(n_paths) * self.r0

        S_all = [S]
        v_all = [v]
        r_all = [r]

        for step in range(n_steps):
            key, k1, k2, k3 = jax.random.split(key, 4)
            z1 = jax.random.normal(k1, (n_paths,))
            z2 = jax.random.normal(k2, (n_paths,))
            z3 = jax.random.normal(k3, (n_paths,))

            dW_s = z1
            dW_v = self.rho_sv * z1 + jnp.sqrt(1.0 - self.rho_sv**2) * z2
            dW_r = (self.rho_sr * z1 +
                    jnp.sqrt(jnp.maximum(1.0 - self.rho_sr**2, 0.0)) * z3)

            sqrt_v = jnp.sqrt(jnp.maximum(v, 0.0))
            t = step * dt
            th_r = float(theta_r_fn(t))

            S = S * jnp.exp((r - self.q - 0.5 * v) * dt + sqrt_v * sqrt_dt * dW_s)
            v = jnp.maximum(v + self.kappa * (self.theta - v) * dt
                            + self.xi * sqrt_v * sqrt_dt * dW_v, 0.0)
            r = r + (th_r - self.a * r) * dt + self.sigma_r * sqrt_dt * dW_r

            S_all.append(S)
            v_all.append(v)
            r_all.append(r)

        return jnp.stack(S_all), jnp.stack(v_all), jnp.stack(r_all)
