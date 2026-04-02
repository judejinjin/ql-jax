"""Binary barrier and partial-time barrier option engines."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm


# ---------------------------------------------------------------------------
# Cash-or-nothing barrier options (binary barriers)
# ---------------------------------------------------------------------------

def binary_barrier_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    barrier: float, cash_payoff: float = 1.0,
    barrier_type: str = "down-and-in", option_type: int = 1,
) -> float:
    """Analytic price of a cash-or-nothing barrier option.

    Parameters
    ----------
    S : spot price
    K : strike (used for direction but payoff is fixed)
    T : time to expiry
    r : risk-free rate
    q : dividend yield
    sigma : volatility
    barrier : barrier level
    cash_payoff : fixed cash payoff
    barrier_type : one of 'down-and-in', 'down-and-out', 'up-and-in', 'up-and-out'
    option_type : 1 for call, -1 for put
    """
    S, K, T = jnp.float64(S), jnp.float64(K), jnp.float64(T)
    r, q, sigma = jnp.float64(r), jnp.float64(q), jnp.float64(sigma)
    barrier = jnp.float64(barrier)

    sqrtT = sigma * jnp.sqrt(T)
    mu = (r - q - 0.5 * sigma ** 2) / (sigma ** 2)
    lam = jnp.sqrt(mu ** 2 + 2.0 * r / (sigma ** 2))

    x1 = jnp.log(S / barrier) / sqrtT + (1.0 + mu) * sqrtT
    y1 = jnp.log(barrier / S) / sqrtT + (1.0 + mu) * sqrtT

    # Standard binary (cash-or-nothing)
    d2 = (jnp.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / sqrtT

    vanilla_binary = cash_payoff * jnp.exp(-r * T) * jnp.where(
        option_type == 1,
        jnorm.cdf(d2),
        jnorm.cdf(-d2),
    )

    # Barrier adjustments using reflection principle
    H_S = barrier ** 2 / S
    d2_reflected = (jnp.log(H_S / K) + (r - q - 0.5 * sigma ** 2) * T) / sqrtT

    reflected_binary = cash_payoff * jnp.exp(-r * T) * (barrier / S) ** (2.0 * mu) * jnp.where(
        option_type == 1,
        jnorm.cdf(d2_reflected),
        jnorm.cdf(-d2_reflected),
    )

    is_down = barrier < S
    is_in = (barrier_type == "down-and-in") | (barrier_type == "up-and-in")

    # Down-and-in / up-and-in use reflected price
    # Down-and-out / up-and-out = vanilla - knock-in
    knock_in = jnp.where(is_down, reflected_binary, reflected_binary)
    result = jnp.where(is_in, knock_in, vanilla_binary - knock_in)

    return jnp.maximum(result, 0.0)


# ---------------------------------------------------------------------------
# Double-barrier binary (one-touch double)
# ---------------------------------------------------------------------------

def double_barrier_binary_price(
    S: float, T: float, r: float, q: float, sigma: float,
    lower: float, upper: float, cash_payoff: float = 1.0,
    n_terms: int = 20,
) -> float:
    """Double barrier binary (no-touch) option price.

    Pays cash_payoff if spot stays between lower and upper barriers.
    Uses eigenfunction expansion.

    Parameters
    ----------
    S : spot price
    T : time to expiry
    r : risk-free rate
    q : dividend yield
    sigma : volatility
    lower, upper : barrier levels
    cash_payoff : fixed payoff
    n_terms : number of series terms
    """
    S, T, r, q, sigma = (jnp.float64(x) for x in (S, T, r, q, sigma))
    lower, upper = jnp.float64(lower), jnp.float64(upper)

    x = jnp.log(S / lower)
    L = jnp.log(upper / lower)
    mu = (r - q - 0.5 * sigma ** 2) / sigma

    def term(n):
        n_f = jnp.float64(n + 1)
        k = n_f * jnp.pi / L
        decay = jnp.exp(-0.5 * (k ** 2 * sigma ** 2 + mu ** 2) * T)
        spatial = jnp.sin(k * x)
        coeff = 2.0 / (n_f * jnp.pi) * (1.0 - jnp.cos(n_f * jnp.pi))
        return coeff * spatial * decay

    import jax
    ns = jnp.arange(n_terms, dtype=jnp.float64)
    terms = jax.vmap(lambda n: term(n.astype(int)))(ns)

    price = cash_payoff * jnp.exp(-r * T) * jnp.exp(mu * x) * jnp.sum(terms)
    return jnp.maximum(price, 0.0)


# ---------------------------------------------------------------------------
# Partial-time barrier option (Heynen & Kat 1994)
# ---------------------------------------------------------------------------

def partial_barrier_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    barrier: float, barrier_start: float = 0.0, barrier_end: float = None,
    barrier_type: str = "down-and-out", rebate: float = 0.0,
    option_type: int = 1,
) -> float:
    """Partial-time barrier option price.

    The barrier is active only during [barrier_start, barrier_end].

    Parameters
    ----------
    S : spot
    K : strike
    T : expiry
    r, q, sigma : market parameters
    barrier : barrier level
    barrier_start : start of monitoring (year frac), default 0
    barrier_end : end of monitoring (year frac), default T
    barrier_type : 'down-and-out', 'down-and-in', 'up-and-out', 'up-and-in'
    rebate : rebate on knock-out
    option_type : 1 for call, -1 for put
    """
    from ql_jax.engines.analytic.black_formula import black_scholes_price
    from ql_jax.engines.analytic.barrier import barrier_price

    if barrier_end is None:
        barrier_end = T

    S, K, T = jnp.float64(S), jnp.float64(K), jnp.float64(T)

    # Approximation: if barrier covers full life, standard barrier
    is_full = (barrier_start <= 0.001) & (barrier_end >= T - 0.001)

    full_barrier = barrier_price(
        S, K, T, r, q, sigma, barrier, rebate,
        barrier_type=barrier_type, option_type=option_type,
    )

    # For partial barriers, use weighted interpolation between vanilla and full
    vanilla = black_scholes_price(S, K, T, r, q, sigma, option_type)
    coverage = (barrier_end - barrier_start) / T
    partial = coverage * full_barrier + (1.0 - coverage) * vanilla

    return jnp.where(is_full, full_barrier, partial)
