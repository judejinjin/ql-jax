"""Exotic multi-asset path-dependent options: Himalaya, Everest, Pagoda."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class HimalayaOption:
    """Himalaya option — at each observation date the best-performing asset
    is removed from the basket and its return is locked in."""
    spots: jnp.ndarray  # (n_assets,) initial spot prices
    maturity: float  # years
    n_obs: int  # number of observation dates
    r: float = 0.05
    q: jnp.ndarray = None  # (n_assets,) div yields
    correlation: jnp.ndarray = None  # (n_assets, n_assets)
    vols: jnp.ndarray = None  # (n_assets,) volatilities


def mc_himalaya_price(option, n_paths=50000, seed=42):
    """Monte Carlo pricing of Himalaya option.

    Parameters
    ----------
    option : HimalayaOption
    n_paths : number of MC paths
    seed : RNG seed

    Returns
    -------
    price : option price
    """
    n_assets = len(option.spots)
    n_obs = option.n_obs
    dt = option.maturity / n_obs
    spots = jnp.asarray(option.spots, dtype=jnp.float64)
    vols = jnp.asarray(option.vols, dtype=jnp.float64)
    q = jnp.zeros(n_assets) if option.q is None else jnp.asarray(option.q)
    r = option.r

    # Cholesky for correlated normals
    if option.correlation is not None:
        L = jnp.linalg.cholesky(jnp.asarray(option.correlation, dtype=jnp.float64))
    else:
        L = jnp.eye(n_assets)

    key = jax.random.PRNGKey(seed)

    # Simulate paths
    drift = (r - q - 0.5 * vols**2) * dt
    vol_sqrt_dt = vols * jnp.sqrt(dt)

    key, subkey = jax.random.split(key)
    Z = jax.random.normal(subkey, (n_paths, n_obs, n_assets))
    corr_Z = jnp.einsum('ij,...j->...i', L, Z)

    log_returns = drift[None, None, :] + vol_sqrt_dt[None, None, :] * corr_Z
    log_spots = jnp.log(spots)[None, None, :] + jnp.cumsum(log_returns, axis=1)
    spot_paths = jnp.exp(log_spots)

    # At each observation, lock in best performer's return and remove
    total_return = jnp.zeros(n_paths)
    mask = jnp.ones((n_paths, n_assets), dtype=bool)

    for obs in range(n_obs):
        prices = spot_paths[:, obs, :]
        returns = prices / spots[None, :] - 1.0
        # Mask already-removed assets
        masked_returns = jnp.where(mask, returns, -jnp.inf)
        best_idx = jnp.argmax(masked_returns, axis=1)
        best_return = masked_returns[jnp.arange(n_paths), best_idx]
        total_return = total_return + best_return
        mask = mask.at[jnp.arange(n_paths), best_idx].set(False)

    avg_return = total_return / n_obs
    payoff = jnp.maximum(avg_return, 0.0)
    disc = jnp.exp(-r * option.maturity)
    return float(payoff.mean() * disc)


@dataclass
class EverestOption:
    """Everest option — pays based on worst performer in basket."""
    spots: jnp.ndarray
    maturity: float
    r: float = 0.05
    q: jnp.ndarray = None
    correlation: jnp.ndarray = None
    vols: jnp.ndarray = None


def mc_everest_price(option, n_paths=50000, seed=42):
    """Monte Carlo pricing of Everest option."""
    n_assets = len(option.spots)
    spots = jnp.asarray(option.spots, dtype=jnp.float64)
    vols = jnp.asarray(option.vols, dtype=jnp.float64)
    q = jnp.zeros(n_assets) if option.q is None else jnp.asarray(option.q)
    r = option.r
    T = option.maturity

    if option.correlation is not None:
        L = jnp.linalg.cholesky(jnp.asarray(option.correlation, dtype=jnp.float64))
    else:
        L = jnp.eye(n_assets)

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    Z = jax.random.normal(subkey, (n_paths, n_assets))
    corr_Z = Z @ L.T

    drift = (r - q - 0.5 * vols**2) * T
    vol_sqrt_T = vols * jnp.sqrt(T)
    log_ST = jnp.log(spots) + drift + vol_sqrt_T * corr_Z
    ST = jnp.exp(log_ST)

    returns = ST / spots - 1.0
    worst_return = returns.min(axis=1)
    payoff = jnp.maximum(1.0 + worst_return, 0.0)
    disc = jnp.exp(-r * T)
    return float(payoff.mean() * disc)


@dataclass
class PagodaOption:
    """Pagoda option — accumulates positive increments with a cap."""
    spots: jnp.ndarray
    maturity: float
    n_obs: int
    cap_per_period: float = 0.10  # max return per period
    r: float = 0.05
    q: jnp.ndarray = None
    correlation: jnp.ndarray = None
    vols: jnp.ndarray = None


def mc_pagoda_price(option, n_paths=50000, seed=42):
    """Monte Carlo pricing of Pagoda option."""
    n_assets = len(option.spots)
    n_obs = option.n_obs
    dt = option.maturity / n_obs
    spots = jnp.asarray(option.spots, dtype=jnp.float64)
    vols = jnp.asarray(option.vols, dtype=jnp.float64)
    q = jnp.zeros(n_assets) if option.q is None else jnp.asarray(option.q)
    r = option.r

    if option.correlation is not None:
        L = jnp.linalg.cholesky(jnp.asarray(option.correlation, dtype=jnp.float64))
    else:
        L = jnp.eye(n_assets)

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    Z = jax.random.normal(subkey, (n_paths, n_obs, n_assets))
    corr_Z = jnp.einsum('ij,...j->...i', L, Z)

    drift = (r - q - 0.5 * vols**2) * dt
    vol_sqrt_dt = vols * jnp.sqrt(dt)

    log_returns = drift[None, None, :] + vol_sqrt_dt[None, None, :] * corr_Z
    cum_log = jnp.cumsum(log_returns, axis=1)
    prev_cum = jnp.concatenate([jnp.zeros((n_paths, 1, n_assets)), cum_log[:, :-1, :]], axis=1)
    period_returns = jnp.exp(log_returns) - 1.0

    # Average return across assets per period, capped
    avg_returns = period_returns.mean(axis=2)
    capped = jnp.clip(avg_returns, -option.cap_per_period, option.cap_per_period)
    total = capped.sum(axis=1)
    payoff = jnp.maximum(total, 0.0)
    disc = jnp.exp(-r * option.maturity)
    return float(payoff.mean() * disc)


def mc_spread_option_price(S1, S2, K, T, r, q1, q2, vol1, vol2, rho,
                            option_type=1, n_paths=100000, seed=42):
    """Monte Carlo spread option price: payoff = max(phi*(S1-S2-K), 0).

    Parameters
    ----------
    S1, S2 : initial spot prices
    K : strike
    T : maturity
    r : risk-free rate
    q1, q2 : dividend yields
    vol1, vol2 : volatilities
    rho : correlation
    option_type : 1 for call, -1 for put
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    Z1 = jax.random.normal(k1, (n_paths,))
    Z2 = rho * Z1 + jnp.sqrt(1 - rho**2) * jax.random.normal(k2, (n_paths,))

    ST1 = S1 * jnp.exp((r - q1 - 0.5*vol1**2)*T + vol1*jnp.sqrt(T)*Z1)
    ST2 = S2 * jnp.exp((r - q2 - 0.5*vol2**2)*T + vol2*jnp.sqrt(T)*Z2)

    phi = option_type
    payoff = jnp.maximum(phi * (ST1 - ST2 - K), 0.0)
    disc = jnp.exp(-r * T)
    return float(payoff.mean() * disc)
