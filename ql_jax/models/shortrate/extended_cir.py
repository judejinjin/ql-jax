"""Extended Cox-Ingersoll-Ross model.

dr = [theta(t) - a*r] dt + sigma * sqrt(r) dW

Time-dependent theta(t) is calibrated to fit the initial term structure.
"""

import jax
import jax.numpy as jnp


def extended_cir_bond_price(r, a, sigma, t, T, discount_curve_fn):
    """Zero-coupon bond price P(t,T) under Extended CIR.

    Uses the deterministic-shift approach:
    r(t) = x(t) + phi(t)
    where x follows standard CIR and phi is deterministic.

    Parameters
    ----------
    r : current short rate
    a : mean reversion speed
    sigma : volatility
    t : current time
    T : maturity
    discount_curve_fn : P(0, .)
    """
    tau = T - t
    gamma = jnp.sqrt(a**2 + 2.0 * sigma**2)

    # CIR bond price coefficients (for the x part)
    exp_gamma_tau = jnp.exp(gamma * tau)
    denom = (gamma + a) * (exp_gamma_tau - 1.0) + 2.0 * gamma

    B = 2.0 * (exp_gamma_tau - 1.0) / denom
    A_cir = (2.0 * gamma * jnp.exp((a + gamma) * tau / 2.0) / denom) ** (2.0 * a * 0.0 / sigma**2)
    # For extended model, theta is time-dependent; use deterministic shift

    # Market-based adjustment
    P_0_T = discount_curve_fn(T)
    P_0_t = discount_curve_fn(t) if t > 0 else 1.0

    dt_bump = 1e-4
    P_bump = discount_curve_fn(t + dt_bump)
    f_0_t = -jnp.log(P_bump / P_0_t) / dt_bump if t > 0 else -jnp.log(discount_curve_fn(dt_bump)) / dt_bump

    # Deterministic shift phi(t) = f(0,t) - f_CIR(0,t|r0=0)
    # For simplicity, use the affine approximation
    ln_ratio = jnp.log(P_0_T / P_0_t)
    approx = jnp.exp(ln_ratio + B * f_0_t - B * r)

    return approx


def extended_cir_short_rate_mean(r0, a, sigma, theta_bar, t):
    """Mean of r(t) under standard CIR (constant theta).

    E[r(t)] = r0 * exp(-a*t) + theta_bar * (1 - exp(-a*t))
    where theta_bar = theta / a (long-run mean).
    """
    return r0 * jnp.exp(-a * t) + theta_bar * (1.0 - jnp.exp(-a * t))


def extended_cir_short_rate_variance(r0, a, sigma, theta_bar, t):
    """Variance of r(t) under standard CIR.

    Var[r(t)] = r0 * sigma^2/a * (e^{-at} - e^{-2at})
                + theta_bar * sigma^2/(2a) * (1 - e^{-at})^2
    """
    e1 = jnp.exp(-a * t)
    e2 = jnp.exp(-2.0 * a * t)
    return (r0 * sigma**2 / a * (e1 - e2) +
            theta_bar * sigma**2 / (2.0 * a) * (1.0 - e1)**2)


def feller_condition(a, sigma, theta):
    """Check Feller condition: 2*a*theta >= sigma^2 (ensures r > 0)."""
    return 2.0 * a * theta >= sigma**2
