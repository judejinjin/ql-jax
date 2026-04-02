"""Black-Karasinski short-rate model.

d(ln r) = [theta(t) - a * ln r] dt + sigma dW

Log-normal short rate model — rates are strictly positive.
Must be calibrated numerically (no closed-form bond price).
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def black_karasinski_tree_bond_price(r0, a, sigma, T, n_steps, discount_curve_fn):
    """Approximate zero-coupon bond price using a trinomial tree.

    Parameters
    ----------
    r0 : current short rate
    a : mean reversion speed
    sigma : volatility of log(r)
    T : maturity
    n_steps : tree steps
    discount_curve_fn : P(0, .) market curve

    Returns float – P(0, T).
    """
    dt = T / n_steps
    V = sigma**2 * (1.0 - jnp.exp(-2.0 * a * dt)) / (2.0 * a)
    dx = jnp.sqrt(3.0 * V)

    # Tree width
    j_max = int(jnp.ceil(0.184 / (a * dt)).item())
    n_nodes = 2 * j_max + 1

    # Theta calibration: match market discount
    thetas = jnp.zeros(n_steps)
    ln_r0 = jnp.log(r0)

    # Build simple pricing array
    values = jnp.ones(n_nodes)

    for step in range(n_steps - 1, -1, -1):
        t_step = step * dt
        new_values = jnp.zeros(n_nodes)

        # Calibrate theta(step) from market
        dt_bump = 1e-4
        P_t = discount_curve_fn(t_step) if t_step > 0 else 1.0
        P_t_dt = discount_curve_fn(t_step + dt)
        f_t = -jnp.log(P_t_dt / P_t) / dt if t_step > 0 else -jnp.log(P_t_dt) / dt
        theta_step = jnp.log(f_t) + a * jnp.log(f_t) + sigma**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * t_step))
        thetas = thetas.at[step].set(theta_step)

        for j in range(-j_max, j_max + 1):
            idx = j + j_max
            ln_r = ln_r0 + j * dx
            r_node = jnp.exp(ln_r)

            # Transition probabilities (trinomial)
            drift = (theta_step - a * ln_r) * dt
            eta = drift / dx

            p_up = (1.0 / 6.0) + (eta**2 + eta) / 2.0
            p_mid = 2.0 / 3.0 - eta**2
            p_down = (1.0 / 6.0) + (eta**2 - eta) / 2.0

            # Clip node indices
            j_up = jnp.clip(idx + 1, 0, n_nodes - 1).astype(int)
            j_mid = idx
            j_down = jnp.clip(idx - 1, 0, n_nodes - 1).astype(int)

            v = jnp.exp(-r_node * dt) * (
                p_up * values[j_up] + p_mid * values[j_mid] + p_down * values[j_down]
            )
            new_values = new_values.at[idx].set(v)

        values = new_values

    return values[j_max]


def black_karasinski_caplet_mc(r0, a, sigma, K, T_reset, T_pay,
                                discount_curve_fn, n_paths=50000, key=None):
    """Monte Carlo caplet price under Black-Karasinski.

    Parameters
    ----------
    r0 : initial short rate
    a : mean reversion
    sigma : vol of log(r)
    K : caplet strike
    T_reset, T_pay : reset and payment dates
    discount_curve_fn : P(0, .)
    n_paths : MC paths
    key : JAX PRNG key
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    n_steps = max(int(T_pay * 50), 10)
    dt = T_pay / n_steps
    step_reset = int(T_reset / dt)

    ln_r = jnp.full(n_paths, jnp.log(r0))
    discount = jnp.ones(n_paths)

    # Theta function from market curve
    def theta(t):
        eps = 1e-4
        P1 = discount_curve_fn(t)
        P2 = discount_curve_fn(t + eps)
        f = -jnp.log(P2 / P1) / eps
        return jnp.log(jnp.maximum(f, 1e-10))

    for step in range(n_steps):
        key, sk = jax.random.split(key)
        dw = jax.random.normal(sk, (n_paths,)) * jnp.sqrt(dt)
        t = step * dt

        drift = (theta(t) - a * ln_r) * dt
        ln_r = ln_r + drift + sigma * dw
        r = jnp.exp(ln_r)
        discount *= jnp.exp(-r * dt)

    # Caplet payoff
    tau = T_pay - T_reset
    payoff = jnp.maximum(jnp.exp(ln_r) - K, 0.0) * tau
    return jnp.mean(discount * payoff)
