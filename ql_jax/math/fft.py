"""FFT wrapper for pricing via characteristic functions."""

from __future__ import annotations

import jax.numpy as jnp


def fft(x):
    """Fast Fourier Transform wrapper.

    Parameters
    ----------
    x : complex array

    Returns
    -------
    complex array — DFT of x
    """
    return jnp.fft.fft(x)


def ifft(x):
    """Inverse FFT wrapper."""
    return jnp.fft.ifft(x)


def fft_price_call(
    char_fn,
    S, r, T, K,
    N=4096, eta=0.25, alpha=1.5,
):
    """Price a European call via FFT (Carr-Madan method).

    Parameters
    ----------
    char_fn : callable(u) -> characteristic function of log(S_T)
              where phi(u) = E[exp(i*u*log(S_T))]
    S : spot price
    r : risk-free rate
    T : maturity
    K : strike (or array of strikes)
    N : FFT grid size (power of 2)
    eta : grid spacing in frequency domain
    alpha : dampening parameter (typically 1.0-2.0)

    Returns
    -------
    array of call prices at the FFT strike grid (use interpolation for specific K)
    """
    # Grid setup
    lam = 2 * jnp.pi / (N * eta)  # log-strike spacing
    b = N * lam / 2.0  # log-strike range

    # Frequency grid
    v = jnp.arange(N) * eta
    # Log-strike grid
    ku = -b + lam * jnp.arange(N)

    # Modified characteristic function for call pricing
    # psi(v) = exp(-r*T) * phi(v - (alpha+1)*i) / (alpha^2 + alpha - v^2 + i*(2*alpha+1)*v)
    u = v - (alpha + 1.0) * 1j
    phi_u = char_fn(u)
    denom = alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v
    psi = jnp.exp(-r * T) * phi_u / denom

    # Simpson's rule weights
    weights = jnp.ones(N)
    weights = weights.at[0].set(0.5)
    # No need for last - FFT handles it

    x = jnp.exp(1j * b * v) * psi * eta * weights
    y = fft(x)
    call_prices = jnp.exp(-alpha * ku) / jnp.pi * jnp.real(y)

    # Interpolate to desired strike(s)
    if K is not None:
        K_arr = jnp.atleast_1d(jnp.asarray(K, dtype=jnp.float64))
        log_K = jnp.log(K_arr)
        prices = jnp.interp(log_K, ku, call_prices)
        return jnp.squeeze(jnp.maximum(prices, 0.0))

    return call_prices, ku


def fft_price_heston_call(
    S, K, r, q, T,
    v0, kappa, theta, sigma, rho,
    N=4096, eta=0.25, alpha=1.5,
):
    """Price European call under Heston model via FFT.

    Parameters
    ----------
    S : spot
    K : strike
    r, q : rates
    T : maturity
    v0, kappa, theta, sigma, rho : Heston params

    Returns
    -------
    float : call price
    """
    log_S = jnp.log(S)
    fwd = log_S + (r - q) * T

    def char_fn(u):
        """Heston characteristic function of log(S_T)."""
        xi = kappa - sigma * rho * 1j * u
        d = jnp.sqrt(xi ** 2 + sigma ** 2 * (u ** 2 + 1j * u))
        g = (xi - d) / (xi + d)
        C = kappa * theta / sigma ** 2 * (
            (xi - d) * T - 2.0 * jnp.log((1 - g * jnp.exp(-d * T)) / (1 - g))
        )
        D = (xi - d) / sigma ** 2 * (1 - jnp.exp(-d * T)) / (1 - g * jnp.exp(-d * T))
        return jnp.exp(1j * u * fwd + C + D * v0)

    return fft_price_call(char_fn, S, r, T, K, N, eta, alpha)
