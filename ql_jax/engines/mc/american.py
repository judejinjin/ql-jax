"""Monte Carlo American option pricing via Longstaff-Schwartz."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax._util.types import OptionType
from ql_jax.methods.montecarlo.path import generate_paths_bs
from ql_jax.methods.montecarlo.longstaff_schwartz import lsm_american_option


def mc_american_bs(
    S, K, T, r, q, sigma, option_type: int,
    n_paths: int = 50_000,
    n_steps: int = 50,
    key=None,
    antithetic: bool = True,
    n_basis: int = 3,
):
    """Monte Carlo American option price using Longstaff-Schwartz.

    Parameters
    ----------
    S, K, T, r, q, sigma : option parameters
    option_type : OptionType.Call or Put
    n_paths, n_steps : MC parameters
    key : jax.random.PRNGKey
    antithetic : use antithetic variates
    n_basis : number of polynomial basis functions for LSM

    Returns
    -------
    price : MC estimate
    """
    from ql_jax.processes.black_scholes import BlackScholesProcess

    if key is None:
        key = jax.random.PRNGKey(42)

    S_val = jnp.asarray(S, dtype=jnp.float64)

    process = BlackScholesProcess(
        spot=float(S_val), rate=float(r), dividend=float(q), volatility=float(sigma),
    )

    log_paths = generate_paths_bs(process, T, n_steps, n_paths, key, antithetic=antithetic)
    spot_paths = jnp.exp(log_paths)

    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    price = lsm_american_option(spot_paths, K, T, r, phi, n_basis=n_basis)

    return price
