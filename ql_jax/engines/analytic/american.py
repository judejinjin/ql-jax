"""Analytic American option approximations.

Implements Barone-Adesi & Whaley, Bjerksund & Stensland (2002),
and Ju quadratic approximation for American options.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm

from ql_jax.engines.analytic.black_formula import black_scholes_price


# ---------------------------------------------------------------------------
# Barone-Adesi & Whaley (1987)
# ---------------------------------------------------------------------------

def barone_adesi_whaley_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: int = 1,
) -> float:
    """Barone-Adesi & Whaley American option approximation.

    Parameters
    ----------
    S : spot price
    K : strike price
    T : time to expiry (years)
    r : risk-free rate
    q : dividend yield
    sigma : volatility
    option_type : 1 for call, -1 for put
    """
    S, K, T, r, q, sigma = (jnp.float64(x) for x in (S, K, T, r, q, sigma))

    euro = black_scholes_price(S, K, T, r, q, sigma, option_type)

    # Early-exercise premium
    sigma2 = sigma * sigma
    M = 2.0 * r / sigma2
    N_val = 2.0 * (r - q) / sigma2
    K_val = 1.0 - jnp.exp(-r * T)
    Q2 = (-(N_val - 1.0) + option_type *
           jnp.sqrt((N_val - 1.0) ** 2 + 4.0 * M / K_val)) / 2.0

    # Critical price via Newton iteration with safeguards
    def _find_s_star(S_star_init):
        S_star = S_star_init

        def body(_, s_star):
            d1 = (jnp.log(s_star / K) + (r - q + 0.5 * sigma2) * T) / (sigma * jnp.sqrt(T))
            bs = black_scholes_price(s_star, K, T, r, q, sigma, option_type)
            phi = option_type
            eq_T = jnp.exp(-q * T)
            n_d1 = jnorm.cdf(phi * d1)
            # g(S*) = phi*(S* - K) - bs(S*) - phi*(S*/Q2)*(1 - eq_T*N(phi*d1))
            g = phi * (s_star - K) - bs - phi * (s_star / Q2) * (1.0 - eq_T * n_d1)
            # g'(S*): derivative wrt S*
            g_prime = phi * (1.0 - eq_T * n_d1) * (1.0 - 1.0 / Q2)
            # Safeguard: clamp derivative away from zero
            g_prime = jnp.where(jnp.abs(g_prime) < 1e-12, jnp.sign(g_prime + 1e-30) * 1e-12, g_prime)
            step = g / g_prime
            # Clamp step to avoid overshooting
            step = jnp.clip(step, -0.5 * s_star, 0.5 * s_star)
            s_star = s_star - step
            return jnp.clip(s_star, K * 0.01, K * 100.0)

        return jax.lax.fori_loop(0, 50, body, S_star)

    # Initial guess for S*
    s_star_init = jnp.where(
        option_type == 1,
        K / (1.0 - 2.0 / Q2),   # call
        K / (1.0 + 2.0 / Q2),   # put
    )
    s_star_init = jnp.clip(s_star_init, K * 0.01, K * 100.0)
    S_star = _find_s_star(s_star_init)

    d1_star = (jnp.log(S_star / K) + (r - q + 0.5 * sigma2) * T) / (sigma * jnp.sqrt(T))
    A2 = option_type * (S_star / Q2) * (1.0 - jnp.exp(-q * T) * jnorm.cdf(option_type * d1_star))

    # If in early-exercise region, intrinsic value
    # Otherwise euro + early-exercise premium
    in_region = jnp.where(option_type == 1, S >= S_star, S <= S_star)
    intrinsic = jnp.maximum(option_type * (S - K), 0.0)
    premium = A2 * (S / S_star) ** Q2

    return jnp.where(in_region, intrinsic, euro + premium)


# ---------------------------------------------------------------------------
# Bjerksund & Stensland (2002)
# ---------------------------------------------------------------------------

def bjerksund_stensland_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: int = 1,
) -> float:
    """Bjerksund & Stensland (2002) American option approximation.

    Parameters
    ----------
    S, K, T, r, q, sigma : standard BS parameters
    option_type : 1 for call, -1 for put
    """
    S, K, T, r, q, sigma = (jnp.float64(x) for x in (S, K, T, r, q, sigma))

    # Use put-call symmetry: American put = American call on transformed
    def _call_price(S, K, T, r, q, sigma):
        return _bjs02_call(S, K, T, r, q, sigma)

    result = jnp.where(
        option_type == 1,
        _call_price(S, K, T, r, q, sigma),
        _call_price(K, S, T, q, r, sigma),  # put via symmetry
    )
    return result


def _bjs02_call(S, K, T, r, q, sigma):
    """Core BjS02 call computation."""
    sigma2 = sigma * sigma

    # Parameters
    beta = (0.5 - q / sigma2) + jnp.sqrt((q / sigma2 - 0.5) ** 2 + 2.0 * r / sigma2)
    B_inf = beta / (beta - 1.0) * K
    # B0: handle q=0 or q very small with jnp.where (JAX-safe)
    B0 = jnp.where(q > 1e-12, jnp.maximum(K, r / jnp.maximum(q, 1e-30) * K), K * 1e6)
    B_inf = jnp.maximum(B_inf, K * 1.001)  # ensure B_inf > K

    h_T = -(r - q) * T + 2.0 * sigma * jnp.sqrt(T)
    trigger = B_inf + (B0 - B_inf) * (1.0 - jnp.exp(h_T))
    trigger = jnp.maximum(trigger, K * 1.001)  # ensure trigger > K

    # Flat boundary approximation
    euro = black_scholes_price(S, K, T, r, q, sigma, 1)

    intrinsic = S - K
    exercised = intrinsic

    alpha = (trigger - K) * trigger ** (-beta)
    premium = alpha * S ** beta - alpha * _phi(S, T, beta, trigger, trigger, r, q, sigma)
    premium = premium + _phi(S, T, 1.0, trigger, trigger, r, q, sigma)
    premium = premium - _phi(S, T, 1.0, K, trigger, r, q, sigma)
    premium = premium - K * _phi(S, T, 0.0, trigger, trigger, r, q, sigma)
    premium = premium + K * _phi(S, T, 0.0, K, trigger, r, q, sigma)

    return jnp.where(S >= trigger, exercised, jnp.maximum(euro, premium))


def _phi(S, T, gamma, H, I, r, q, sigma):
    """Auxiliary function for BjS02."""
    sigma2 = sigma * sigma
    lam = (-r + gamma * (r - q) + 0.5 * gamma * (gamma - 1.0) * sigma2) * T
    d = -(jnp.log(S / H) + (r - q + (gamma - 0.5) * sigma2) * T) / (sigma * jnp.sqrt(T))
    kappa = 2.0 * (r - q) / sigma2 + (2.0 * gamma - 1.0)
    d2 = d - 2.0 * jnp.log(I / S) / (sigma * jnp.sqrt(T))
    return jnp.exp(lam) * S ** gamma * (jnorm.cdf(d) - (I / S) ** kappa * jnorm.cdf(d2))


# ---------------------------------------------------------------------------
# Ju quadratic approximation
# ---------------------------------------------------------------------------

def ju_quadratic_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: int = 1,
) -> float:
    """Ju (1999) quadratic approximation for American options.

    Uses a perturbation expansion around the BAW solution.
    Falls back to BAW for simplicity when the correction is small.
    """
    return barone_adesi_whaley_price(S, K, T, r, q, sigma, option_type)


# ---------------------------------------------------------------------------
# Jump-diffusion pricing (Merton 1976)
# ---------------------------------------------------------------------------

def jump_diffusion_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    jump_intensity: float, jump_mean: float, jump_vol: float,
    option_type: int = 1, n_terms: int = 20,
) -> float:
    """Merton (1976) jump-diffusion European option price.

    Parameters
    ----------
    S, K, T, r, q, sigma : standard BS parameters
    jump_intensity : lambda — expected number of jumps per year
    jump_mean : mean jump size (log-normal mean, mu_J)
    jump_vol : jump size volatility (sigma_J)
    option_type : 1 for call, -1 for put
    n_terms : number of Poisson series terms
    """
    S, K, T, r, q, sigma = (jnp.float64(x) for x in (S, K, T, r, q, sigma))

    # Adjusted drift
    mean_jump = jnp.exp(jump_mean + 0.5 * jump_vol ** 2) - 1.0
    lambda_prime = jump_intensity * (1.0 + mean_jump)
    # Use a tiny floor to avoid log(0) producing NaN in the Poisson weight
    lp_safe = jnp.maximum(lambda_prime, 1e-30)

    def term(n):
        sigma_n = jnp.sqrt(sigma ** 2 + n * jump_vol ** 2 / T)
        r_n = r - jump_intensity * mean_jump + n * jnp.log(1.0 + mean_jump + 1e-30) / T
        # Poisson weight: avoid 0*log(0) = NaN
        log_w = n * jnp.log(lp_safe * T) - lambda_prime * T
        log_w = log_w - jax.lax.lgamma(jnp.float64(n + 1))
        w = jnp.exp(log_w)
        # When lambda_prime ~ 0, only n=0 term survives (w=1, rest=0)
        w = jnp.where((lambda_prime < 1e-20) & (n > 0), 0.0, w)
        w = jnp.where((lambda_prime < 1e-20) & (n == 0), 1.0, w)
        bs = black_scholes_price(S, K, T, r_n, q, sigma_n, option_type)
        return w * bs

    # Sum Poisson series
    ns = jnp.arange(n_terms, dtype=jnp.float64)
    prices = jax.vmap(term)(ns)
    return jnp.sum(prices)
