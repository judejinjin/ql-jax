"""Analytic cliquet (ratchet) option pricing."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def cliquet_price(S, T, r, q, sigma, reset_dates, local_floor=-jnp.inf,
                   local_cap=jnp.inf, global_floor=-jnp.inf,
                   global_cap=jnp.inf):
    """Cliquet option price under Black-Scholes.

    A cliquet accumulates capped/floored periodic returns.
    Each period return is R_i = S(t_{i+1})/S(t_i) - 1,
    capped at local_cap and floored at local_floor.
    Final payoff = max(global_floor, min(global_cap, sum(R_i_capped))).

    Under BS, each period return is independent and log-normal.

    Parameters
    ----------
    S : float – spot (not needed for pricing with proper forward)
    T : float – total maturity
    r, q, sigma : float
    reset_dates : array – reset times (including 0 and T)
    local_floor, local_cap : float
    global_floor, global_cap : float

    Returns
    -------
    price : float
    """
    n = len(reset_dates) - 1  # number of periods

    # Price each caplet/floorlet on periodic returns
    total_ev = 0.0
    for i in range(n):
        dt = reset_dates[i + 1] - reset_dates[i]
        forward_return = jnp.exp((r - q) * dt) - 1.0

        # E[min(cap, max(floor, R))] = E[R] + put(floor) - call(cap)
        # where R is the period return

        # Forward vol for the period
        vol = sigma * jnp.sqrt(dt)

        ev_clipped = _capped_floored_return_ev(
            forward_return, vol, local_floor, local_cap, r, dt
        )
        total_ev += ev_clipped

    # Global cap/floor: approximate with normal
    price = jnp.clip(total_ev, global_floor, global_cap) * jnp.exp(-r * T)
    return price


def _capped_floored_return_ev(forward, vol, floor, cap, r, dt):
    """Expected value of capped/floored return under log-normal."""
    ev = forward

    # Floor: like buying a put at floor strike
    if floor > -1e10:
        d1_f = (jnp.log((1.0 + forward) / (1.0 + floor)) + 0.5 * vol**2) / vol
        d2_f = d1_f - vol
        floor_val = ((1.0 + floor) * norm.cdf(-d2_f) -
                     (1.0 + forward) * norm.cdf(-d1_f))
        ev += floor_val

    # Cap: like selling a call at cap strike
    if cap < 1e10:
        d1_c = (jnp.log((1.0 + forward) / (1.0 + cap)) + 0.5 * vol**2) / vol
        d2_c = d1_c - vol
        cap_val = ((1.0 + forward) * norm.cdf(d1_c) -
                   (1.0 + cap) * norm.cdf(d2_c))
        ev -= cap_val

    return ev


def forward_start_price(S, T1, T2, r, q, sigma, option_type='call',
                         strike_ratio=1.0):
    """Forward-starting option price.

    At T1, a new ATM (or strike_ratio * S(T1)) option is created
    expiring at T2.

    Under BS, the price is the same as a regular option scaled by
    the forward ratio.

    Parameters
    ----------
    S : float – spot
    T1 : float – forward start date
    T2 : float – expiry date
    r, q, sigma : float
    option_type : 'call' or 'put'
    strike_ratio : float – K = strike_ratio * S(T1)

    Returns
    -------
    price : float
    """
    S, T1, T2 = (jnp.asarray(x, dtype=jnp.float64) for x in (S, T1, T2))
    tau = T2 - T1  # remaining life after forward start

    K_eff = strike_ratio
    d1 = (jnp.log(1.0 / K_eff) + (r - q + 0.5 * sigma**2) * tau) / (sigma * jnp.sqrt(tau))
    d2 = d1 - sigma * jnp.sqrt(tau)

    phi = 1.0 if option_type == 'call' else -1.0

    # Price = S * exp(-q*T1) * [BS formula with S=1, K=strike_ratio]
    price = S * jnp.exp(-q * T1) * phi * (
        jnp.exp(-q * tau) * norm.cdf(phi * d1) -
        K_eff * jnp.exp(-r * tau) * norm.cdf(phi * d2)
    )
    return price
