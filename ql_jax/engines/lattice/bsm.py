"""BSM lattice – Black-Scholes-Merton tree with dividends and exercise features."""

import jax.numpy as jnp
import jax

from ql_jax._util.types import OptionType


def bsm_binomial_dividend_price(
    S, K, T, r, q, sigma, option_type: int,
    dividends_times=None, dividends_amounts=None,
    n_steps: int = 200,
    american: bool = False,
):
    """Price option with discrete dividends using binomial tree.

    Implements the known-dividends approach where the spot is adjusted
    at each ex-dividend date.

    Parameters
    ----------
    S, K, T, r, q, sigma : option parameters
    option_type : int
    dividends_times : array or None – ex-dividend times
    dividends_amounts : array or None – dividend amounts
    n_steps : int
    american : bool

    Returns
    -------
    price : float
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))
    dt = T / n_steps

    # CRR parameters
    u = jnp.exp(sigma * jnp.sqrt(dt))
    d = 1.0 / u
    p = (jnp.exp((r - q) * dt) - d) / (u - d)
    disc = jnp.exp(-r * dt)

    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)

    # Compute PV of future dividends at each time step
    div_pv = jnp.zeros(n_steps + 1)
    if dividends_times is not None and dividends_amounts is not None:
        for k in range(len(dividends_times)):
            t_div = dividends_times[k]
            d_amt = dividends_amounts[k]
            for step in range(n_steps + 1):
                t_step = step * dt
                remaining = jnp.maximum(t_div - t_step, 0.0)
                div_pv = div_pv.at[step].add(
                    d_amt * jnp.exp(-r * remaining) * (t_div > t_step)
                )

    # Terminal payoff
    n = n_steps
    j = jnp.arange(n + 1, dtype=jnp.float64)
    # Spot at each terminal node, adjusted for dividends
    S_star = S - div_pv[0]  # dividend-adjusted spot
    spots = S_star * u ** (n - j) * d ** j + div_pv[n]
    values = jnp.maximum(phi * (spots - K), 0.0)

    # Pre-compute all j arrays (static)
    all_j = jnp.arange(n + 1, dtype=jnp.float64)

    def step_fn(i, values):
        step = n - 1 - i
        # Compute hold for all adjacent pairs, pad to keep shape
        hold = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        hold = jnp.concatenate([hold, jnp.zeros(1)])  # pad to n+1

        if american:
            spots_step = S_star * u ** (all_j) * d ** (all_j) + div_pv[step]
            # Actually recompute properly: at step, j goes 0..step
            spots_step = S_star * u ** jnp.maximum(step - all_j, 0.0) * d ** jnp.minimum(all_j, step) + div_pv[step]
            exercise = jnp.maximum(phi * (spots_step - K), 0.0)
            hold = jnp.maximum(hold, exercise)

        # Mask: keep only first step+1 values
        mask = all_j <= step
        return jnp.where(mask, hold, 0.0)

    values = jax.lax.fori_loop(0, n, step_fn, values)
    return values[0]


def bsm_convertible_tree(
    S, K, T, r, q, sigma, face_value, coupon_rate, conversion_ratio,
    call_price=None, put_price=None,
    n_steps: int = 100,
):
    """Price convertible bond using binomial tree.

    Parameters
    ----------
    S : float – stock price
    K : float – (not used for convertible)
    T : float – maturity
    r : float – risk-free rate
    q : float – dividend yield
    sigma : float – stock vol
    face_value : float – bond face value
    coupon_rate : float – annual coupon rate
    conversion_ratio : float – shares per bond
    call_price : float or None – issuer call price
    put_price : float or None – holder put price
    n_steps : int

    Returns
    -------
    price : float – convertible bond price
    """
    S, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                          for x in (S, T, r, q, sigma))
    dt = T / n_steps
    n = n_steps

    u = jnp.exp(sigma * jnp.sqrt(dt))
    d = 1.0 / u
    p = (jnp.exp((r - q) * dt) - d) / (u - d)
    disc = jnp.exp(-r * dt)

    # Terminal: max(face + last coupon, conversion_ratio * S_T)
    j = jnp.arange(n + 1, dtype=jnp.float64)
    spots = S * u ** (n - j) * d ** j
    conversion_value = conversion_ratio * spots
    bond_value = face_value + face_value * coupon_rate * dt  # Last coupon
    values = jnp.maximum(conversion_value, bond_value)

    def step_fn(i, values):
        step = n - 1 - i

        # Continuation value
        hold = disc * (p * values[:-1] + (1.0 - p) * values[1:])

        # Add coupon
        hold = hold + face_value * coupon_rate * dt

        # Conversion option (holder can convert)
        j_step = jnp.arange(step + 1, dtype=jnp.float64)
        spots_step = S * u ** (step - j_step) * d ** j_step
        conv = conversion_ratio * spots_step
        hold = jnp.maximum(hold, conv)

        # Issuer call
        if call_price is not None:
            hold = jnp.minimum(hold, jnp.maximum(call_price, conv))

        # Holder put
        if put_price is not None:
            hold = jnp.maximum(hold, put_price)

        return hold

    values = jax.lax.fori_loop(0, n, step_fn, values)
    return values[0]
