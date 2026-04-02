"""GJR-GARCH(1,1) model for equity options.

h(t+1) = omega + (alpha + gamma * I_{r<0}) * r(t)^2 + beta * h(t)

where h is conditional variance, r is return, and I_{r<0} is the
leverage indicator (asymmetric volatility response to negative returns).
"""

import jax
import jax.numpy as jnp
from ql_jax._util.types import OptionType


def gjrgarch_variance_path(omega, alpha, beta, gamma, returns, h0=None):
    """Compute conditional variance path from historical returns.

    Parameters
    ----------
    omega : float – constant term
    alpha : float – ARCH coefficient
    beta : float – GARCH coefficient
    gamma : float – leverage coefficient
    returns : array – historical returns
    h0 : float – initial variance (default: unconditional)

    Returns array of conditional variances.
    """
    if h0 is None:
        h0 = omega / (1.0 - alpha - beta - 0.5 * gamma)

    n = len(returns)
    h = jnp.zeros(n + 1)
    h = h.at[0].set(h0)

    for t in range(n):
        leverage = jnp.where(returns[t] < 0, gamma, 0.0)
        h = h.at[t + 1].set(omega + (alpha + leverage) * returns[t]**2 + beta * h[t])

    return h


def gjrgarch_option_price_mc(S, K, T, r, q, v0, omega, alpha, beta, gamma,
                               lambda_param=0.0, option_type=OptionType.Call,
                               n_paths=50000, n_steps_per_day=1,
                               days_per_year=252, key=None):
    """European option price under GJR-GARCH via Monte Carlo.

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry
    r : risk-free rate
    q : dividend yield
    v0 : initial variance (annualized)
    omega, alpha, beta, gamma : GARCH parameters
    lambda_param : risk premium
    option_type : Call or Put
    n_paths : MC paths
    n_steps_per_day : steps per trading day
    days_per_year : trading days per year
    key : JAX PRNG key
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    n_days = int(T * days_per_year)
    n_steps = n_days * n_steps_per_day
    dt = 1.0 / days_per_year / n_steps_per_day

    log_S = jnp.full(n_paths, jnp.log(S))
    h = jnp.full(n_paths, v0 / days_per_year)  # daily variance

    for step in range(n_steps):
        key, sk = jax.random.split(key)
        z = jax.random.normal(sk, (n_paths,))

        # Daily return
        sigma_daily = jnp.sqrt(jnp.maximum(h, 1e-10))
        log_S = log_S + (r - q - 0.5 * h * days_per_year) * dt + sigma_daily * jnp.sqrt(dt * days_per_year) * z

        # Update GARCH variance (daily)
        if step % n_steps_per_day == 0:
            ret = sigma_daily * z * jnp.sqrt(days_per_year * dt)
            leverage = jnp.where(ret < 0, gamma, 0.0)
            h = omega + (alpha + leverage) * ret**2 + beta * h

    S_T = jnp.exp(log_S)
    discount = jnp.exp(-r * T)

    if option_type == OptionType.Call:
        payoff = jnp.maximum(S_T - K, 0.0)
    else:
        payoff = jnp.maximum(K - S_T, 0.0)

    return discount * jnp.mean(payoff)


def calibrate_gjrgarch(returns, initial_params=None):
    """Calibrate GJR-GARCH(1,1) to historical returns via MLE.

    Parameters
    ----------
    returns : array of log returns

    Returns dict with: omega, alpha, beta, gamma, log_likelihood.
    """
    returns = jnp.asarray(returns, dtype=jnp.float64)
    n = len(returns)

    def neg_log_likelihood(params):
        omega = jnp.exp(params[0])
        alpha = jnp.exp(params[1])
        beta = jnp.exp(params[2])
        gamma = jnp.exp(params[3])

        h = omega / (1.0 - alpha - beta - 0.5 * gamma + 1e-10)
        ll = 0.0
        for t in range(n):
            r_t = returns[t]
            ll += -0.5 * (jnp.log(2.0 * jnp.pi * h) + r_t**2 / h)
            leverage = jnp.where(r_t < 0, gamma, 0.0)
            h = omega + (alpha + leverage) * r_t**2 + beta * h
            h = jnp.maximum(h, 1e-10)
        return -ll

    if initial_params is None:
        x0 = jnp.array([jnp.log(1e-6), jnp.log(0.05), jnp.log(0.9), jnp.log(0.05)])
    else:
        x0 = jnp.log(jnp.array([initial_params['omega'], initial_params['alpha'],
                                  initial_params['beta'], initial_params['gamma']]))

    from ql_jax.math.optimization.bfgs import minimize
    result = minimize(neg_log_likelihood, x0, max_iterations=200)
    p = result['x']

    return {
        'omega': jnp.exp(p[0]),
        'alpha': jnp.exp(p[1]),
        'beta': jnp.exp(p[2]),
        'gamma': jnp.exp(p[3]),
        'log_likelihood': -result['fun'],
    }
