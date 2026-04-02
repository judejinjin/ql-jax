"""Static replication of payoffs via portfolio of calls/puts."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def replicate_with_calls_puts(payoff_fn, forward, vol, T, discount,
                               lower, upper, n_strikes=50):
    """Replicate a European payoff via static strip of calls and puts.

    Uses Carr-Madan decomposition:
        V = payoff(F) + payoff'(F)(S-F) + int_0^F payoff''(K) P(K) dK
            + int_F^inf payoff''(K) C(K) dK

    Parameters
    ----------
    payoff_fn : callable(S) -> payoff
    forward : float – forward price
    vol : float – Black vol (flat assumption)
    T : float – time to expiry
    discount : float – discount factor
    lower : float – lower strike bound
    upper : float – upper strike bound
    n_strikes : int – number of strikes in the grid

    Returns
    -------
    float – price of the replicated payoff
    """
    strikes = jnp.linspace(lower, upper, n_strikes)
    dk = strikes[1] - strikes[0]

    # Compute second derivative of payoff (butterfly weights)
    eps = dk * 0.5
    payoff_pp = (payoff_fn(strikes + eps) - 2.0 * payoff_fn(strikes) +
                 payoff_fn(strikes - eps)) / eps**2

    # Black prices
    total_vol = vol * jnp.sqrt(T)
    prices = jnp.where(
        strikes < forward,
        _black_put(forward, strikes, total_vol),
        _black_call(forward, strikes, total_vol),
    )

    # Integrate
    value = payoff_fn(forward) + discount * jnp.sum(payoff_pp * prices * dk)
    return value * discount


def _black_call(F, K, total_vol):
    """Undiscounted Black call price."""
    d1 = (jnp.log(F / K) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def _black_put(F, K, total_vol):
    """Undiscounted Black put price."""
    d1 = (jnp.log(F / K) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol
    return K * norm.cdf(-d2) - F * norm.cdf(-d1)


def replicate_cms_caplet(swap_rate, annuity, vol_fn, T, discount,
                          strike, n_strikes=100):
    """Replicate a CMS caplet using swaption smile.

    Parameters
    ----------
    swap_rate : float – forward swap rate
    annuity : float – swap annuity
    vol_fn : callable(K) -> Black vol at strike K
    T : float – swaption expiry
    discount : float – DF to payment
    strike : float – CMS cap strike
    n_strikes : int

    Returns
    -------
    float – CMS caplet price
    """
    upper = swap_rate * 3.0
    strikes = jnp.linspace(strike, upper, n_strikes)
    dk = strikes[1] - strikes[0]

    # Each point: d/dK [annuity * Black_call(S, K, vol(K))]
    # Approximate numerically
    prices = jnp.zeros(n_strikes)
    for i in range(n_strikes):
        K = strikes[i]
        v = vol_fn(K)
        tv = v * jnp.sqrt(T)
        prices = prices.at[i].set(annuity * _black_call(swap_rate, K, tv))

    # CMS caplet ≈ (S - K) * annuity * DF + integral of second derivative
    integrand = jnp.zeros(n_strikes)
    for i in range(1, n_strikes - 1):
        integrand = integrand.at[i].set(
            (prices[i + 1] - 2.0 * prices[i] + prices[i - 1]) / dk**2
        )

    # Payment delay adjustment
    delay_adj = discount / (annuity * discount)  # simplified

    return delay_adj * jnp.sum(integrand * (strikes - strike) * dk)
