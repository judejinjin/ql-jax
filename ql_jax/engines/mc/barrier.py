"""Monte Carlo barrier option pricing."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax._util.types import OptionType, BarrierType
from ql_jax.methods.montecarlo.path import generate_paths_bs


def mc_barrier_bs(
    S, K, T, r, q, sigma,
    option_type: int,
    barrier_type: int,
    barrier: float,
    rebate: float = 0.0,
    n_paths: int = 100_000,
    n_steps: int = 252,
    key=None,
    antithetic: bool = True,
):
    """Monte Carlo barrier option price under Black-Scholes.

    Monitors the barrier at discrete time steps (daily by default).

    Parameters
    ----------
    S, K, T, r, q, sigma : option parameters
    option_type : OptionType.Call or Put
    barrier_type : BarrierType
    barrier : barrier level
    rebate : rebate on knock-out
    n_paths, n_steps : MC parameters
    key : jax.random.PRNGKey
    antithetic : use antithetic variates

    Returns
    -------
    price : MC estimate
    stderr : standard error
    """
    from ql_jax.processes.black_scholes import BlackScholesProcess

    if key is None:
        key = jax.random.PRNGKey(42)

    S = jnp.asarray(S, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)
    barrier = jnp.asarray(barrier, dtype=jnp.float64)

    process = BlackScholesProcess(
        spot=float(S), rate=float(r), dividend=float(q), volatility=float(sigma),
    )

    log_paths = generate_paths_bs(process, T, n_steps, n_paths, key, antithetic=antithetic)
    spot_paths = jnp.exp(log_paths)

    # Check barrier crossing — use jnp.where for JAX traceability
    is_down = (barrier_type == BarrierType.DownIn) | (barrier_type == BarrierType.DownOut)
    is_knock_in = (barrier_type == BarrierType.DownIn) | (barrier_type == BarrierType.UpIn)

    # Down barrier: crossed if any spot <= barrier
    # Up barrier: crossed if any spot >= barrier
    crossed_down = jnp.any(spot_paths <= barrier, axis=1)
    crossed_up = jnp.any(spot_paths >= barrier, axis=1)
    crossed = jnp.where(is_down, crossed_down, crossed_up)

    # Terminal payoff
    S_T = spot_paths[:, -1]
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    vanilla_payoff = jnp.maximum(phi * (S_T - K), 0.0)

    # Knock-in: pay off only if barrier was hit
    # Knock-out: pay off only if barrier was NOT hit
    payoffs_ki = jnp.where(crossed, vanilla_payoff, rebate)
    payoffs_ko = jnp.where(crossed, rebate, vanilla_payoff)
    payoffs = jnp.where(is_knock_in, payoffs_ki, payoffs_ko)

    df = jnp.exp(-r * T)
    discounted = df * payoffs

    price = jnp.mean(discounted)
    stderr = jnp.std(discounted) / jnp.sqrt(jnp.float64(discounted.shape[0]))

    return price, stderr
