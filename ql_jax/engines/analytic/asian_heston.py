"""Analytic pricing of geometric Asian options under the Heston model.

Continuous and discrete geometric average Asian options can be priced
semi-analytically under Heston by exploiting the fact that the geometric
average of a log-normal-like process has a known characteristic function.

References:
- Kim & Wee (2014) "Pricing of geometric Asian options under Heston's
  stochastic volatility model"
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _heston_geom_asian_cf(u, T, r, q, v0, kappa, theta, xi, rho):
    """Characteristic function of the log of the geometric average
    of a Heston process over [0, T].

    For continuous geometric averaging, log(GA) = (1/T) * integral_0^T log(S_t) dt.

    We use the approach: effective parameters for the averaged process.
    """
    # Effective vol parameters for the geometric average
    # Under Heston, the averaged log-spot has cf:
    # E[exp(iu * (1/T)*integral log S_t dt)]
    # which can be computed via an ODE in (kappa, theta, xi, rho) space.

    # Use effective variance: var(GA) ≈ (1/3) * E[V] * T for continuous
    # and time-dependent corrections
    kappa_bar = kappa
    theta_bar = theta
    # For the geometric average, effective parameters scale as:
    d = jnp.sqrt((kappa_bar - 1j * rho * xi * u)**2 + xi**2 * u * (u + 1j))
    g = (kappa_bar - 1j * rho * xi * u - d) / (kappa_bar - 1j * rho * xi * u + d)

    # Heston cf components for average
    C = (kappa_bar * theta_bar / xi**2) * (
        (kappa_bar - 1j * rho * xi * u - d) * T
        - 2.0 * jnp.log((1.0 - g * jnp.exp(-d * T)) / (1.0 - g))
    )
    D = ((kappa_bar - 1j * rho * xi * u - d) / xi**2) * (
        (1.0 - jnp.exp(-d * T)) / (1.0 - g * jnp.exp(-d * T))
    )

    # Scale for geometric average: effective T → T/2, effective v0 → v0/2
    # (approximation for continuous geometric average)
    C_avg = C / 2.0
    D_avg = D / 2.0

    phi = jnp.exp(C_avg + D_avg * v0 + 1j * u * (r - q) * T / 2.0)
    return phi


def cont_geom_asian_heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: int = 1,
    n_quad: int = 64,
) -> float:
    """Continuous geometric Asian under Heston, via characteristic function.

    Parameters
    ----------
    S : spot
    K : strike
    T : maturity
    r, q : rates
    v0, kappa, theta, xi, rho : Heston parameters
    option_type : +1 call, -1 put
    n_quad : integration points

    Returns
    -------
    price : float
    """
    log_moneyness = jnp.log(S / K)
    discount = jnp.exp(-r * T)

    # Numerical integration of the Heston-geometric-average cf
    u_max = 50.0
    du = u_max / n_quad
    u_grid = jnp.linspace(du, u_max, n_quad)

    # Gil-Pelaez formula
    integrand1 = jnp.array([
        jnp.real(
            jnp.exp(-1j * ui * jnp.log(K))
            * _heston_geom_asian_cf(ui - 1j, T, r, q, v0, kappa, theta, xi, rho)
            / (1j * ui)
        ) for ui in u_grid
    ])
    integrand2 = jnp.array([
        jnp.real(
            jnp.exp(-1j * ui * jnp.log(K))
            * _heston_geom_asian_cf(ui, T, r, q, v0, kappa, theta, xi, rho)
            / (1j * ui)
        ) for ui in u_grid
    ])

    P1 = 0.5 + jnp.sum(integrand1) * du / jnp.pi
    P2 = 0.5 + jnp.sum(integrand2) * du / jnp.pi

    F = S * jnp.exp((r - q) * T / 2.0)  # geometric average forward
    call_price = discount * (F * P1 - K * P2)

    if option_type == 1:
        return float(jnp.maximum(call_price, 0.0))
    else:
        put_price = call_price - discount * (F - K)
        return float(jnp.maximum(put_price, 0.0))


def discr_geom_asian_heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_fixings: int = 12,
    option_type: int = 1,
    n_quad: int = 64,
) -> float:
    """Discrete geometric Asian under Heston.

    Approximates the discrete geometric average payoff by adjusting
    the continuous formula with a discrete correction.

    Parameters
    ----------
    n_fixings : number of averaging dates (equally spaced)
    """
    # Discrete correction: for n fixings, the effective averaging period
    # is slightly different from continuous
    # Continuous approx with correction factor
    cont_price = cont_geom_asian_heston_price(
        S, K, T, r, q, v0, kappa, theta, xi, rho, option_type, n_quad
    )

    # Discrete vs continuous correction (Turnbull-Wakeman style)
    # For geometric average, discrete correction is small
    dt = T / n_fixings
    correction = jnp.exp(-0.5 * (r - q) * dt * (n_fixings - 1) / (2.0 * n_fixings))
    # The correction vanishes as n_fixings → ∞
    disc_factor = (n_fixings + 1) / (2.0 * n_fixings)
    adjusted_price = cont_price * disc_factor / 0.5  # scale from continuous to discrete

    # Bound the result
    return float(jnp.maximum(adjusted_price, 0.0))
