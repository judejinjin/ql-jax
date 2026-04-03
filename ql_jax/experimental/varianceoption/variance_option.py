"""Variance option and integral Heston variance option engine."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class VarianceOption:
    """Option on realized variance (NOT variance swap).

    Payoff: max(phi * (realized_var - strike_var), 0) * notional

    Parameters
    ----------
    strike_var : strike variance
    notional : contract notional
    maturity : years to expiry
    option_type : 1=call on variance, -1=put on variance
    """
    strike_var: float
    notional: float
    maturity: float
    option_type: int = 1


def integral_heston_variance_option(option, v0, kappa, theta, sigma_v, rho,
                                     r, n_points=64):
    """Price variance option under Heston using integral method.

    Uses the characteristic function of integrated variance under Heston.

    Parameters
    ----------
    option : VarianceOption
    v0 : initial variance
    kappa, theta, sigma_v, rho : Heston parameters
    r : risk-free rate
    n_points : quadrature points

    Returns
    -------
    price : option price
    """
    T = option.maturity
    K = option.strike_var
    phi = option.option_type
    disc = jnp.exp(-r * T)

    # Expected realized variance under Heston
    # E[V_bar] = theta + (v0 - theta) * (1 - exp(-kappa*T)) / (kappa*T)
    E_V = theta + (v0 - theta) * (1.0 - jnp.exp(-kappa * T)) / (kappa * T + 1e-15)

    # Variance of realized variance (second moment)
    s2 = sigma_v**2 / (kappa**3 * T**2 + 1e-15) * (
        (v0 - theta) * (1.0 - jnp.exp(-2 * kappa * T)) +
        kappa * theta * T / 2 * (1.0 - jnp.exp(-kappa * T))**2
    )
    s2 = jnp.maximum(s2, 1e-15)
    std_V = jnp.sqrt(s2)

    # Approximate with lognormal distribution
    mu_ln = jnp.log(E_V**2 / jnp.sqrt(E_V**2 + s2))
    sig_ln = jnp.sqrt(jnp.log(1.0 + s2 / E_V**2))

    # Black-like formula for lognormal variance
    from jax.scipy.stats import norm
    d1 = (mu_ln - jnp.log(K) + sig_ln**2) / (sig_ln + 1e-15)
    d2 = d1 - sig_ln

    if phi == 1:
        price = disc * option.notional * (
            jnp.exp(mu_ln + 0.5 * sig_ln**2) * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = disc * option.notional * (
            K * norm.cdf(-d2) - jnp.exp(mu_ln + 0.5 * sig_ln**2) * norm.cdf(-d1))

    return float(jnp.maximum(price, 0.0))


def mc_digital_price(S, K, T, r, q, vol, option_type=1, is_cash=True,
                      n_paths=100000, seed=42):
    """Monte Carlo digital/binary option price.

    Parameters
    ----------
    S : spot price
    K : strike
    T : maturity
    r : risk-free rate
    q : dividend yield
    vol : volatility
    option_type : 1=call (pays if S>K), -1=put
    is_cash : True=cash-or-nothing, False=asset-or-nothing
    n_paths : number of paths
    seed : RNG seed
    """
    key = jax.random.PRNGKey(seed)
    Z = jax.random.normal(key, (n_paths,))
    ST = S * jnp.exp((r - q - 0.5 * vol**2) * T + vol * jnp.sqrt(T) * Z)

    phi = option_type
    in_money = jnp.where(phi > 0, ST > K, ST < K).astype(jnp.float64)

    if is_cash:
        payoff = in_money  # pays 1
    else:
        payoff = in_money * ST  # pays S_T

    disc = jnp.exp(-r * T)
    return float(payoff.mean() * disc)
