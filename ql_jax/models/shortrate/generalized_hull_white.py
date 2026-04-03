"""Generalized Hull-White and Andersen's QE discretization."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class GeneralizedHullWhite:
    """Multi-factor generalized Hull-White short rate model.

    dr(t) = [theta(t) - sum_i a_i * x_i(t)] dt + sum_i sigma_i dW_i(t)

    where x_i(t) are the factors driven by:
    dx_i = -a_i x_i dt + sigma_i dW_i

    Parameters
    ----------
    a : (n_factors,) mean-reversion speeds
    sigma : (n_factors,) factor volatilities
    correlation : (n_factors, n_factors) factor correlations (optional)
    """
    a: jnp.ndarray
    sigma: jnp.ndarray
    correlation: jnp.ndarray = None

    @property
    def n_factors(self):
        return len(self.a)

    def discount_bond(self, r0, T, theta_integral=0.0):
        """Zero-coupon bond price P(0,T) in multi-factor model.

        For the n-factor model, the bond price is:
        P(0,T) = exp(-A(T) + B(T) * r0)

        Parameters
        ----------
        r0 : initial short rate
        T : maturity
        theta_integral : integral of theta(s) from 0 to T

        Returns
        -------
        price : bond price
        """
        a = self.a
        sig = self.sigma
        n = self.n_factors

        # B_i(T) = (1 - exp(-a_i*T)) / a_i
        B = (1.0 - jnp.exp(-a * T)) / (a + 1e-15)

        # A(T) involves the integral of B_i * sigma_i^2 etc.
        A = 0.0
        for i in range(n):
            A += sig[i]**2 / (2 * a[i]**2) * (T - B[i] - 0.5 * a[i] * B[i]**2)

        if self.correlation is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    rho_ij = float(self.correlation[i, j])
                    A += rho_ij * sig[i] * sig[j] / (a[i] * a[j]) * (
                        T - B[i] - B[j] +
                        (1 - jnp.exp(-(a[i] + a[j]) * T)) / (a[i] + a[j])
                    )

        return jnp.exp(-A + theta_integral - r0 * B.sum())

    def simulate(self, r0, T, n_steps, n_paths, theta_func=None, seed=42):
        """Simulate paths of the generalized Hull-White model.

        Parameters
        ----------
        r0 : initial short rate
        T : time horizon
        n_steps : number of time steps
        n_paths : number of paths
        theta_func : callable, theta(t) -> drift adjustment. If None, uses 0.
        seed : RNG seed

        Returns
        -------
        t_grid : (n_steps+1,) time grid
        r_paths : (n_paths, n_steps+1) short rate paths
        """
        dt = T / n_steps
        n = self.n_factors
        a = self.a
        sig = self.sigma
        key = jax.random.PRNGKey(seed)

        # Factor correlation
        if self.correlation is not None:
            L = jnp.linalg.cholesky(jnp.asarray(self.correlation, dtype=jnp.float64))
        else:
            L = jnp.eye(n)

        x = jnp.zeros((n_paths, n))  # factor values
        r_paths = jnp.zeros((n_paths, n_steps + 1))
        r_paths = r_paths.at[:, 0].set(r0)

        for step in range(n_steps):
            t = step * dt
            key, subkey = jax.random.split(key)
            Z = jax.random.normal(subkey, (n_paths, n))
            corr_Z = Z @ L.T

            # Update factors
            dx = -a[None, :] * x * dt + sig[None, :] * jnp.sqrt(dt) * corr_Z
            x = x + dx

            theta = 0.0 if theta_func is None else theta_func(t + dt)
            r = r0 + x.sum(axis=1) + theta
            r_paths = r_paths.at[:, step + 1].set(r)

        t_grid = jnp.linspace(0, T, n_steps + 1)
        return t_grid, r_paths


def andersen_qe_step(v, kappa, theta, sigma, dt, key):
    """Andersen's Quadratic Exponential (QE) scheme for CIR/Heston variance.

    Efficient and exact-in-mean discretization of:
    dv = kappa * (theta - v) dt + sigma * sqrt(v) dW

    Parameters
    ----------
    v : current variance (scalar or array)
    kappa, theta, sigma : CIR/Heston parameters
    dt : time step
    key : JAX PRNG key

    Returns
    -------
    v_new : next variance value
    """
    # Moments of conditional distribution
    m = theta + (v - theta) * jnp.exp(-kappa * dt)
    s2 = (v * sigma**2 * jnp.exp(-kappa * dt) / kappa *
          (1.0 - jnp.exp(-kappa * dt)) +
          theta * sigma**2 / (2.0 * kappa) *
          (1.0 - jnp.exp(-kappa * dt))**2)
    s2 = jnp.maximum(s2, 1e-15)

    psi = s2 / (m**2 + 1e-30)

    # Switch between exponential and quadratic approximation
    psi_c = 1.5  # critical value

    key, k1, k2 = jax.random.split(key, 3)
    U = jax.random.uniform(k1, shape=v.shape if hasattr(v, 'shape') else ())

    # Exponential scheme (psi <= psi_c)
    beta_exp = 2.0 / m - 1.0 + jnp.sqrt(2.0 / m) * jnp.sqrt(jnp.maximum(2.0 / m - 1.0, 0.0))
    # Actually use simpler formulas
    p = (psi - 1.0) / (psi + 1.0)
    beta_qe = (1.0 - p) / (m + 1e-15)
    # QE: v_new = (1/beta) * (Z_p)^2 where Z_p ~ (random with point mass at 0)
    # Simplified: if U <= p, v_new = 0; else v_new = (1/beta) * (Phi_inv((U-p)/(1-p)))^2

    from jax.scipy.stats import norm
    # For psi <= psi_c: use moment-matching
    b2 = 2.0 / psi - 1.0 + jnp.sqrt(2.0 / psi) * jnp.sqrt(jnp.maximum(2.0 / psi - 1.0, 0.0))
    a = m / (1.0 + b2)
    Z = jax.random.normal(k2, shape=v.shape if hasattr(v, 'shape') else ())
    v_exp = a * (jnp.sqrt(b2) + Z)**2

    # For psi > psi_c: use exponential
    p_exp = jnp.clip(p, 0.0, 1.0)
    beta_val = (1.0 - p_exp) / jnp.maximum(m, 1e-15)
    v_qe = jnp.where(U <= p_exp, 0.0,
                      jnp.log((1.0 - p_exp) / (1.0 - U + 1e-15)) / (beta_val + 1e-15))

    v_new = jnp.where(psi <= psi_c, v_exp, v_qe)
    return jnp.maximum(v_new, 0.0)
