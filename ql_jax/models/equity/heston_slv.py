"""Heston Stochastic Local Volatility (SLV) model — FDM calibration.

dS/S = (r-q) dt + L(t,S) * sqrt(v) dW_S
dv = kappa*(theta - v) dt + xi*sqrt(v) dW_v

The leverage function L(t,S) is calibrated so that the model
reproduces the local volatility surface exactly.
"""

import jax
import jax.numpy as jnp


def calibrate_leverage_function(spots, times, local_vols, v0, kappa, theta, xi, rho,
                                 n_mc_paths=10000, key=None):
    """Calibrate L(t, S) on a grid using Monte Carlo + Dupire matching.

    The leverage function is: L(t, S)^2 = sigma_local(t, S)^2 / E[v(t) | S(t) = S]

    Parameters
    ----------
    spots : array – spot grid
    times : array – time grid
    local_vols : 2D array (n_times, n_spots) – local vol surface
    v0, kappa, theta, xi, rho : Heston parameters
    n_mc_paths : MC paths for conditional expectation
    key : JAX PRNG key

    Returns 2D array (n_times, n_spots) of leverage values.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    n_times = len(times)
    n_spots = len(spots)
    leverage = jnp.ones((n_times, n_spots))

    # MC simulation of (S, v) paths
    dt = jnp.diff(jnp.concatenate([jnp.array([0.0]), times]))

    log_S = jnp.zeros(n_mc_paths)  # log(S/S0)
    v = jnp.full(n_mc_paths, v0)
    S0 = spots[len(spots) // 2]  # reference spot

    for t_idx in range(n_times):
        dt_step = dt[t_idx]
        key, sk1, sk2 = jax.random.split(key, 3)
        z1 = jax.random.normal(sk1, (n_mc_paths,))
        z2 = rho * z1 + jnp.sqrt(1.0 - rho**2) * jax.random.normal(sk2, (n_mc_paths,))

        sqrt_v = jnp.sqrt(jnp.maximum(v, 0.0))
        log_S = log_S + (-0.5 * v) * dt_step + sqrt_v * jnp.sqrt(dt_step) * z1
        v = v + kappa * (theta - v) * dt_step + xi * sqrt_v * jnp.sqrt(dt_step) * z2
        v = jnp.maximum(v, 0.0)

        S_paths = S0 * jnp.exp(log_S)

        # Bin paths and compute E[v | S in bin]
        for s_idx in range(n_spots):
            s = spots[s_idx]
            # Gaussian kernel for smoothing
            bandwidth = (spots[-1] - spots[0]) / n_spots * 2.0
            weights = jnp.exp(-0.5 * ((S_paths - s) / bandwidth)**2)
            w_sum = jnp.sum(weights) + 1e-30
            E_v_given_S = jnp.sum(weights * v) / w_sum

            sigma_local = local_vols[t_idx, s_idx]
            L_sq = sigma_local**2 / (E_v_given_S + 1e-10)
            leverage = leverage.at[t_idx, s_idx].set(jnp.sqrt(jnp.maximum(L_sq, 0.01)))

    return leverage


def heston_slv_price_mc(S, K, T, r, q, v0, kappa, theta, xi, rho,
                          leverage_fn, option_type=1, n_paths=50000,
                          n_steps=200, key=None):
    """Price European option under Heston SLV via Monte Carlo.

    Parameters
    ----------
    S : spot
    K : strike
    T : expiry
    leverage_fn : callable(t, s) -> L(t,s) or constant 1.0
    option_type : 1 for call, -1 for put
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    dt = T / n_steps
    log_S = jnp.full(n_paths, jnp.log(S))
    v = jnp.full(n_paths, v0)

    for step in range(n_steps):
        t = step * dt
        key, sk1, sk2 = jax.random.split(key, 3)
        z1 = jax.random.normal(sk1, (n_paths,))
        z2 = rho * z1 + jnp.sqrt(1.0 - rho**2) * jax.random.normal(sk2, (n_paths,))

        sqrt_v = jnp.sqrt(jnp.maximum(v, 0.0))
        S_curr = jnp.exp(log_S)

        # Leverage
        if callable(leverage_fn):
            L = jax.vmap(lambda s: leverage_fn(t, s))(S_curr)
        else:
            L = 1.0

        log_S = log_S + (r - q - 0.5 * L**2 * v) * dt + L * sqrt_v * jnp.sqrt(dt) * z1
        v = v + kappa * (theta - v) * dt + xi * sqrt_v * jnp.sqrt(dt) * z2
        v = jnp.maximum(v, 0.0)

    S_T = jnp.exp(log_S)
    discount = jnp.exp(-r * T)
    payoff = jnp.maximum(option_type * (S_T - K), 0.0)
    return discount * jnp.mean(payoff)
