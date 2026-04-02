"""G2++ swaption pricing engine."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm


def g2_swaption_price(
    notional: float, fixed_rate: float,
    payment_dates: jnp.ndarray, day_fractions: jnp.ndarray,
    a: float, sigma: float, b: float, eta: float, rho: float,
    discount_fn, payer: bool = True,
) -> float:
    """G2++ analytic swaption price.

    Two-factor model: r(t) = x(t) + y(t) + phi(t)
    dx = -a*x*dt + sigma*dW1
    dy = -b*y*dt + eta*dW2
    <dW1, dW2> = rho*dt

    Parameters
    ----------
    notional : notional amount
    fixed_rate : swap fixed rate
    payment_dates : swap payment dates (year fracs from today)
    day_fractions : day count fracs for each period
    a, sigma : first factor parameters
    b, eta : second factor parameters
    rho : correlation between factors
    discount_fn : t -> discount factor
    payer : True for payer swaption
    """
    exercise = float(payment_dates[0]) - float(day_fractions[0])
    exercise = max(exercise, 0.001)

    n = payment_dates.shape[0]

    # Bond reconstruction formula parameters
    def V(t, T):
        """Variance of log bond price ratio."""
        Ba = (1.0 - jnp.exp(-a * (T - t))) / a
        Bb = (1.0 - jnp.exp(-b * (T - t))) / b
        v = (sigma ** 2 / (2.0 * a ** 3)) * (
            (T - t) + 2.0 / a * jnp.exp(-a * (T - t))
            - 0.5 / a * jnp.exp(-2.0 * a * (T - t)) - 1.5 / a
        )
        v = v + (eta ** 2 / (2.0 * b ** 3)) * (
            (T - t) + 2.0 / b * jnp.exp(-b * (T - t))
            - 0.5 / b * jnp.exp(-2.0 * b * (T - t)) - 1.5 / b
        )
        v = v + rho * sigma * eta / (a * b) * (
            (T - t)
            + (jnp.exp(-a * (T - t)) - 1.0) / a
            + (jnp.exp(-b * (T - t)) - 1.0) / b
            - (jnp.exp(-(a + b) * (T - t)) - 1.0) / (a + b)
        )
        return jnp.maximum(v, 1e-12)

    # Approximate with Jamshidian's decomposition
    # Reduce to sum of bond options
    taus = day_fractions
    dfs = jnp.array([discount_fn(t) for t in payment_dates])
    df_ex = discount_fn(exercise)

    # Forward bond prices
    fwd_bonds = dfs / df_ex

    # Coupon bond value
    c_i = fixed_rate * taus
    c_i = c_i.at[-1].add(1.0)  # Add notional at maturity

    # Approximate swaption as option on coupon bond
    # Using one-factor approximation of G2
    sigma_eff = jnp.sqrt(V(0.0, exercise))

    cb_value = jnp.sum(c_i * fwd_bonds)
    d1 = jnp.log(cb_value) / sigma_eff + 0.5 * sigma_eff
    d2 = d1 - sigma_eff

    # Payer swaption = put on coupon bond
    omega = -1.0 if payer else 1.0
    price = df_ex * (
        omega * cb_value * jnorm.cdf(omega * d1)
        - omega * jnorm.cdf(omega * d2)
    )

    return notional * jnp.abs(price)
