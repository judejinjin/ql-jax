"""MC Heston engine – Monte Carlo pricing under Heston model."""

import jax
import jax.numpy as jnp

from ql_jax._util.types import OptionType


def mc_heston_price(S, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                     option_type: int, n_paths=100_000, n_steps=200,
                     key=None, scheme='qe'):
    """Price a vanilla option under Heston using Monte Carlo.

    Parameters
    ----------
    S, K, T, r, q : float
    v0, kappa, theta, sigma_v, rho : float – Heston params
    option_type : int
    n_paths, n_steps : int
    key : PRNGKey or None
    scheme : str – 'euler' or 'qe' (Quadratic Exponential)

    Returns
    -------
    price : float
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    dt = T / n_steps

    if scheme == 'qe':
        return _mc_heston_qe(S, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                              option_type, n_paths, n_steps, dt, key)
    else:
        return _mc_heston_euler(S, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                                 option_type, n_paths, n_steps, dt, key)


def _mc_heston_euler(S, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                      option_type, n_paths, n_steps, dt, key):
    """Euler discretization of Heston."""
    key1, key2 = jax.random.split(key)
    Z1 = jax.random.normal(key1, (n_steps, n_paths))
    Z2 = jax.random.normal(key2, (n_steps, n_paths))

    # Correlate
    W1 = Z1
    W2 = rho * Z1 + jnp.sqrt(1.0 - rho**2) * Z2

    log_S = jnp.log(jnp.float64(S)) * jnp.ones(n_paths)
    v = jnp.float64(v0) * jnp.ones(n_paths)

    for i in range(n_steps):
        v_pos = jnp.maximum(v, 0.0)  # Absorption at zero

        log_S = log_S + (r - q - 0.5 * v_pos) * dt + jnp.sqrt(v_pos * dt) * W1[i]
        v = v + kappa * (theta - v_pos) * dt + sigma_v * jnp.sqrt(v_pos * dt) * W2[i]

    S_T = jnp.exp(log_S)
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    payoff = jnp.maximum(phi * (S_T - K), 0.0)

    return jnp.exp(-r * T) * jnp.mean(payoff)


def _mc_heston_qe(S, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                   option_type, n_paths, n_steps, dt, key):
    """Quadratic-Exponential (QE) scheme for Heston variance (Andersen 2008)."""
    key1, key2, key3 = jax.random.split(key, 3)
    Z_v = jax.random.normal(key1, (n_steps, n_paths))
    Z_s = jax.random.normal(key2, (n_steps, n_paths))
    U = jax.random.uniform(key3, (n_steps, n_paths))

    log_S = jnp.log(jnp.float64(S)) * jnp.ones(n_paths)
    v = jnp.float64(v0) * jnp.ones(n_paths)

    e_dt = jnp.exp(-kappa * dt)
    psi_crit = 1.5  # QE switching threshold

    for i in range(n_steps):
        # Drift-corrected mean and variance of v(t+dt) | v(t)
        m = theta + (v - theta) * e_dt
        s2 = (v * sigma_v**2 * e_dt * (1.0 - e_dt) / kappa +
              theta * sigma_v**2 * (1.0 - e_dt)**2 / (2.0 * kappa))
        s2 = jnp.maximum(s2, 1e-16)

        psi = s2 / jnp.maximum(m**2, 1e-16)

        # QE: for psi <= psi_crit, use quadratic approximation
        b2 = 2.0 / psi - 1.0 + jnp.sqrt(2.0 / psi) * jnp.sqrt(jnp.maximum(2.0 / psi - 1.0, 0.0))
        a = m / (1.0 + b2)
        v_qe = a * (jnp.sqrt(b2) + Z_v[i])**2

        # For psi > psi_crit, use exponential approximation
        p = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - p) / jnp.maximum(m, 1e-16)
        v_exp = jnp.where(
            U[i] <= p,
            0.0,
            jnp.log((1.0 - p) / jnp.maximum(1.0 - U[i], 1e-16)) / beta,
        )

        v_new = jnp.where(psi <= psi_crit, v_qe, v_exp)
        v_new = jnp.maximum(v_new, 0.0)

        # Log-spot update
        k0 = -rho * kappa * theta * dt / sigma_v
        k1 = 0.5 * dt * (kappa * rho / sigma_v - 0.5) - rho / sigma_v
        k2 = 0.5 * dt * (kappa * rho / sigma_v - 0.5) + rho / sigma_v
        k3 = 0.5 * dt * (1.0 - rho**2)

        log_S = (log_S + (r - q) * dt + k0 + k1 * v + k2 * v_new +
                 jnp.sqrt(k3 * (v + v_new)) * Z_s[i])

        v = v_new

    S_T = jnp.exp(log_S)
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    payoff = jnp.maximum(phi * (S_T - K), 0.0)

    return jnp.exp(-r * T) * jnp.mean(payoff)
