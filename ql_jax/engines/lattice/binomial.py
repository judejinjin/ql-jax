"""Binomial tree pricing for vanilla options.

Supports CRR (Cox-Ross-Rubinstein), JR (Jarrow-Rudd), Tian, Trigeorgis,
and Leisen-Reimer tree types. Uses jax.lax.fori_loop for efficient
backward induction.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax

from ql_jax._util.types import OptionType


def binomial_price(
    S, K, T, r, q, sigma, option_type: int,
    n_steps: int = 200,
    american: bool = False,
    tree_type: str = "crr",
):
    """Price a vanilla option using a binomial tree.

    Parameters
    ----------
    S, K, T, r, q, sigma : option parameters
    option_type : OptionType.Call or OptionType.Put
    n_steps : number of time steps in the tree
    american : if True, allow early exercise at each node
    tree_type : 'crr', 'jr', 'tian', 'trigeorgis', or 'leisen_reimer'

    Returns
    -------
    price : float
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))
    dt = T / n_steps
    n = n_steps

    u, d, p = _tree_params(S, K, T, r, q, sigma, dt, n, tree_type)
    disc_factor = jnp.exp(-r * dt)

    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)

    # Terminal payoff: n+1 values
    j = jnp.arange(n + 1, dtype=jnp.float64)
    spots = S * u ** (n - j) * d ** j
    values = jnp.maximum(phi * (spots - K), 0.0)

    # Index array for masking (reused in body)
    j_arr = jnp.arange(n, dtype=jnp.float64)

    if american:
        def body_fn(i, values):
            step = n - 1 - i
            continuation = disc_factor * (p * values[:-1] + (1.0 - p) * values[1:])
            spots_step = S * u ** (step - j_arr) * d ** j_arr
            exercise = jnp.maximum(phi * (spots_step - K), 0.0)
            new_vals = jnp.maximum(continuation, exercise)
            mask = j_arr <= step
            new_vals = jnp.where(mask, new_vals, 0.0)
            return jnp.concatenate([new_vals, jnp.array([0.0])])
    else:
        def body_fn(i, values):
            step = n - 1 - i
            continuation = disc_factor * (p * values[:-1] + (1.0 - p) * values[1:])
            mask = j_arr <= step
            new_vals = jnp.where(mask, continuation, 0.0)
            return jnp.concatenate([new_vals, jnp.array([0.0])])

    values = jax.lax.fori_loop(0, n, body_fn, values)
    return values[0]


def _tree_params(S, K, T, r, q, sigma, dt, n_steps, tree_type):
    """Compute (u, d, p) for the specified tree type."""
    if tree_type == "crr":
        u = jnp.exp(sigma * jnp.sqrt(dt))
        d = 1.0 / u
        p = (jnp.exp((r - q) * dt) - d) / (u - d)
    elif tree_type == "jr":
        # Jarrow-Rudd (equal probability)
        nu = r - q - 0.5 * sigma**2
        u = jnp.exp(nu * dt + sigma * jnp.sqrt(dt))
        d = jnp.exp(nu * dt - sigma * jnp.sqrt(dt))
        p = jnp.float64(0.5)
    elif tree_type == "tian":
        v = jnp.exp(sigma**2 * dt)
        m = jnp.exp((r - q) * dt)
        u = 0.5 * m * v * (v + 1.0 + jnp.sqrt(v**2 + 2.0 * v - 3.0))
        d = 0.5 * m * v * (v + 1.0 - jnp.sqrt(v**2 + 2.0 * v - 3.0))
        p = (m - d) / (u - d)
    elif tree_type == "trigeorgis":
        nu = r - q - 0.5 * sigma**2
        dx = jnp.sqrt(sigma**2 * dt + nu**2 * dt**2)
        u = jnp.exp(dx)
        d = jnp.exp(-dx)
        p = (jnp.exp((r - q) * dt) - d) / (u - d)
    elif tree_type == "leisen_reimer":
        # Leisen-Reimer (odd n_steps for convergence)
        d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
        d2 = d1 - sigma * jnp.sqrt(T)
        n = n_steps
        # Peizer-Pratt inversion
        p2 = _pp_inversion(d2, n)
        p1 = _pp_inversion(d1, n)
        u = jnp.exp((r - q) * dt) * p1 / p2
        d = (jnp.exp((r - q) * dt) - p2 * u) / (1.0 - p2)
        p = p2
    else:
        raise ValueError(f"Unknown tree type: {tree_type}")

    return u, d, p


def _pp_inversion(z, n):
    """Peizer-Pratt inversion for Leisen-Reimer tree."""
    from jax.scipy.stats import norm
    # Approximate inversion: h(z, n) ≈ 0.5 + sign(z)*sqrt(0.25 - 0.25*exp(-((z/(n+1/3+0.1/(n+1)))^2)*(n+1/6)))
    z2 = z / (n + 1.0/3.0 + 0.1 / (n + 1.0))
    return 0.5 + jnp.sign(z) * 0.5 * jnp.sqrt(
        1.0 - jnp.exp(-(z2**2) * (n + 1.0/6.0))
    )
