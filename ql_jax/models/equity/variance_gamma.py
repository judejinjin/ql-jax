"""Variance Gamma process.

The Variance Gamma process is a subordinated Brownian motion:
    X(t) = theta * G(t) + sigma * W(G(t))
where G(t) is a Gamma process with unit mean rate and variance rate nu.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class VarianceGammaProcess:
    """Variance Gamma process parameters.

    Parameters
    ----------
    s0 : float – initial spot price
    r : float – risk-free rate
    q : float – dividend yield
    sigma : float – volatility of the Brownian motion component
    nu : float – variance rate of the gamma subordinator
    theta : float – drift of the Brownian motion component
    """
    s0: float
    r: float
    q: float
    sigma: float
    nu: float
    theta: float

    @property
    def omega(self):
        """Martingale correction: log(1 - theta*nu - sigma^2*nu/2) / nu."""
        return jnp.log(1.0 - self.theta * self.nu
                       - 0.5 * self.sigma ** 2 * self.nu) / self.nu

    def characteristic_exponent(self, u):
        """Characteristic exponent Psi(u) such that E[e^{iuX(t)}] = e^{t*Psi(u)}.

        Psi(u) = -1/nu * log(1 - i*theta*nu*u + 0.5*sigma^2*nu*u^2)
        """
        iu = 1j * u
        return -1.0 / self.nu * jnp.log(
            1.0 - iu * self.theta * self.nu
            + 0.5 * self.sigma ** 2 * self.nu * u ** 2
        )

    def log_characteristic_function(self, u, t):
        """Log of the risk-neutral characteristic function of log(S(t)/S(0)).

        log phi(u) = iu*(r-q+omega)*t + t*Psi(u)
        """
        iu = 1j * u
        return iu * (self.r - self.q + self.omega) * t + t * self.characteristic_exponent(u)


@dataclass
class VarianceGammaModel:
    """Calibratable Variance Gamma model.

    Wraps a VarianceGammaProcess and supports calibration to market data.
    """
    process: VarianceGammaProcess

    @classmethod
    def from_params(cls, s0, r, q, sigma, nu, theta):
        return cls(VarianceGammaProcess(s0, r, q, sigma, nu, theta))

    def calibrate(self, strikes, market_prices, expiry, is_call=True):
        """Calibrate sigma, nu, theta to match market option prices.

        Uses Levenberg-Marquardt on model vs market prices.
        """
        from ql_jax.math.optimization.levenberg_marquardt import minimize as lm_minimize

        def residuals(params):
            sigma_, nu_, theta_ = params
            sigma_ = jnp.abs(sigma_) + 1e-6
            nu_ = jnp.abs(nu_) + 1e-6
            proc = VarianceGammaProcess(
                self.process.s0, self.process.r, self.process.q,
                sigma_, nu_, theta_,
            )
            model_prices = jax.vmap(
                lambda k: analytic_vg_price(proc, k, expiry, is_call)
            )(jnp.asarray(strikes))
            return model_prices - jnp.asarray(market_prices)

        x0 = jnp.array([self.process.sigma, self.process.nu, self.process.theta])
        try:
            x, _, _ = lm_minimize(residuals, x0)
        except Exception:
            x = x0

        self.process = VarianceGammaProcess(
            self.process.s0, self.process.r, self.process.q,
            float(jnp.abs(x[0])) + 1e-6,
            float(jnp.abs(x[1])) + 1e-6,
            float(x[2]),
        )


def analytic_vg_price(process: VarianceGammaProcess, strike: float,
                      expiry: float, is_call: bool = True) -> float:
    """Price a European option under the Variance Gamma model.

    Uses the characteristic function via Carr-Madan FFT/numerical integration.
    """
    s0 = process.s0
    r = process.r
    q = process.q
    t = expiry
    k = jnp.log(strike)

    # Numerical integration of the characteristic function
    # Using the Carr-Madan approach with dampening parameter alpha
    alpha = 1.5
    n_points = 4096
    eta = 0.05
    u_grid = jnp.arange(n_points) * eta

    def integrand(u):
        cf = jnp.exp(process.log_characteristic_function(u - (alpha + 1) * 1j, t))
        denom = alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u
        return jnp.exp(-1j * u * k) * cf / denom

    vals = jax.vmap(integrand)(u_grid)
    # Simpson weights
    weights = jnp.ones(n_points)
    weights = weights.at[0].set(0.5)
    integral = jnp.real(jnp.sum(vals * weights * eta)) / jnp.pi

    call_price = jnp.exp(-r * t) * s0 * jnp.exp((r - q) * t) * 0.0 + \
                 jnp.exp(-alpha * k) * integral

    # Ensure non-negative
    call_price = jnp.maximum(call_price, 0.0)

    if is_call:
        return float(call_price)
    else:
        # Put via put-call parity
        fwd = s0 * jnp.exp((r - q) * t)
        return float(call_price - jnp.exp(-r * t) * (fwd - strike))


def fft_vg_price(process: VarianceGammaProcess, strikes, expiry: float,
                 is_call: bool = True, n_fft: int = 4096, eta: float = 0.05):
    """Price European options for an array of strikes using FFT.

    More efficient than calling analytic_vg_price per strike.
    """
    s0 = process.s0
    r = process.r
    q = process.q
    t = expiry
    alpha = 1.5

    u_grid = jnp.arange(n_fft) * eta
    lam = 2 * jnp.pi / (n_fft * eta)
    b = n_fft * lam / 2

    def psi(u):
        cf = jnp.exp(process.log_characteristic_function(u - (alpha + 1) * 1j, t))
        denom = alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u
        return cf / denom

    # Simpson integration weights
    j = jnp.arange(n_fft)
    simpson = (3.0 + (-1.0) ** (j + 1)) / 3.0
    simpson = simpson.at[0].set(1.0 / 3.0)

    x = jnp.exp(1j * b * u_grid) * psi(u_grid) * eta * simpson

    from jax.numpy.fft import fft as jnp_fft
    fft_result = jnp_fft(x)

    k_grid = -b + lam * jnp.arange(n_fft)
    call_prices = jnp.exp(-alpha * k_grid) / jnp.pi * jnp.real(fft_result)
    call_prices = call_prices * jnp.exp(-r * t)
    call_prices = jnp.maximum(call_prices, 0.0)

    # Interpolate to requested strikes
    log_strikes = jnp.log(jnp.asarray(strikes, dtype=jnp.float64))
    prices = jnp.interp(log_strikes, k_grid, call_prices)

    if not is_call:
        fwd = s0 * jnp.exp((r - q) * t)
        prices = prices - jnp.exp(-r * t) * (fwd - jnp.asarray(strikes))

    return prices
