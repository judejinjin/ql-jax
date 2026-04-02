"""Bates stochastic volatility + jumps model.

Extends Heston with log-normal jumps:
  dS/S = (r - q - lambda*m_bar) dt + sqrt(v) dW_S + (e^J - 1) dN
  dv = kappa*(theta - v) dt + xi*sqrt(v) dW_v
  <dW_S, dW_v> = rho dt
  J ~ N(log(1+m_bar) - delta^2/2, delta^2)
  N is Poisson with intensity lambda

Uses Fourier inversion (characteristic function) for pricing.
"""

import jax
import jax.numpy as jnp
from ql_jax._util.types import OptionType


def bates_characteristic_function(u, S, K, T, r, q, v0, kappa, theta, xi, rho,
                                   lambda_j, mu_j, delta_j):
    """Characteristic function of log(S_T) under Bates model.

    Parameters
    ----------
    u : complex – Fourier variable
    lambda_j : float – jump intensity
    mu_j : float – mean jump size (log space)
    delta_j : float – jump vol
    """
    # Heston part
    d = jnp.sqrt((rho * xi * 1j * u - kappa)**2 + xi**2 * (1j * u + u**2))
    g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

    exp_dT = jnp.exp(-d * T)
    C = (r - q) * 1j * u * T + kappa * theta / xi**2 * (
        (kappa - rho * xi * 1j * u - d) * T - 2.0 * jnp.log((1.0 - g * exp_dT) / (1.0 - g))
    )
    D = (kappa - rho * xi * 1j * u - d) / xi**2 * (1.0 - exp_dT) / (1.0 - g * exp_dT)

    # Jump part
    log_1pm = jnp.log(1.0 + mu_j)
    jump_cf = lambda_j * T * (
        jnp.exp(1j * u * (log_1pm - 0.5 * delta_j**2) - 0.5 * u**2 * delta_j**2 + 0.5 * delta_j**2 * 1j * u) - 1.0
        - 1j * u * mu_j
    )

    return jnp.exp(C + D * v0 + 1j * u * jnp.log(S) + jump_cf)


def bates_price(S, K, T, r, q, v0, kappa, theta, xi, rho,
                lambda_j, mu_j, delta_j, option_type=OptionType.Call, n_points=64):
    """European option price under Bates model via Fourier inversion.

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry
    r : risk-free rate
    q : dividend yield
    v0 : initial variance
    kappa : mean reversion speed
    theta : long-run variance
    xi : vol-of-vol
    rho : correlation
    lambda_j : jump intensity
    mu_j : mean jump size
    delta_j : jump vol
    option_type : OptionType.Call or Put
    n_points : quadrature points
    """
    log_K = jnp.log(K)
    discount = jnp.exp(-r * T)

    # Gauss-Laguerre quadrature
    du = 0.5
    integral_1 = 0.0
    integral_2 = 0.0

    for j in range(1, n_points + 1):
        u = j * du
        cf1 = bates_characteristic_function(
            u - 1j, S, K, T, r, q, v0, kappa, theta, xi, rho, lambda_j, mu_j, delta_j
        ) / (1j * u * S * jnp.exp((r - q) * T))
        cf2 = bates_characteristic_function(
            u, S, K, T, r, q, v0, kappa, theta, xi, rho, lambda_j, mu_j, delta_j
        ) / (1j * u)

        integral_1 += jnp.real(jnp.exp(-1j * u * log_K) * cf1) * du
        integral_2 += jnp.real(jnp.exp(-1j * u * log_K) * cf2) * du

    P1 = 0.5 + integral_1 / jnp.pi
    P2 = 0.5 + integral_2 / jnp.pi

    call = S * jnp.exp(-q * T) * P1 - K * discount * P2

    if option_type == OptionType.Put:
        return call - S * jnp.exp(-q * T) + K * discount
    return call


def calibrate_bates(S, r, q, market_strikes, market_maturities, market_prices,
                     option_types=None, initial_params=None, max_iter=100):
    """Calibrate Bates model to market option prices.

    Parameters
    ----------
    initial_params : (v0, kappa, theta, xi, rho, lambda_j, mu_j, delta_j)

    Returns dict with calibrated parameters.
    """
    from ql_jax.models.calibration import calibrate_least_squares

    strikes = jnp.asarray(market_strikes, dtype=jnp.float64)
    mats = jnp.asarray(market_maturities, dtype=jnp.float64)
    prices = jnp.asarray(market_prices, dtype=jnp.float64)

    if option_types is None:
        option_types = jnp.full(len(strikes), OptionType.Call, dtype=jnp.int32)

    if initial_params is None:
        initial_params = jnp.array([0.04, 2.0, 0.04, 0.5, -0.5, 0.1, 0.0, 0.1])

    def model_fn(params):
        v0, kappa, theta, xi, rho, lam, mu, delta = params
        def price_one(K, T, ot):
            return bates_price(S, K, T, r, q, v0, kappa, theta, xi, rho, lam, mu, delta, ot)
        return jax.vmap(price_one)(strikes, mats, option_types)

    return calibrate_least_squares(model_fn, initial_params, prices, max_iter=max_iter)
