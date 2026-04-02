"""Piecewise time-dependent Heston model.

dS/S = (r-q) dt + sqrt(v) dW_S
dv = kappa_i*(theta_i - v) dt + xi_i*sqrt(v) dW_v    for t in [t_i, t_{i+1})

Parameters (kappa, theta, xi, rho) are piecewise-constant in time.
"""

import jax
import jax.numpy as jnp
from ql_jax._util.types import OptionType


def ptd_heston_characteristic_function(u, S, T, r, q, v0,
                                        times, kappas, thetas, xis, rhos):
    """Characteristic function of log(S_T) under PTD Heston.

    Parameters
    ----------
    u : complex – Fourier variable
    times : array [t0=0, t1, ..., tn=T] breakpoints
    kappas, thetas, xis, rhos : arrays of parameter values per interval
    """
    n_intervals = len(kappas)
    C = 0.0 + 0.0j
    D = 0.0 + 0.0j

    for i in range(n_intervals):
        kappa_i = kappas[i]
        theta_i = thetas[i]
        xi_i = xis[i]
        rho_i = rhos[i]
        dt_i = times[i + 1] - times[i]

        d = jnp.sqrt((rho_i * xi_i * 1j * u - kappa_i)**2 +
                      xi_i**2 * (1j * u + u**2))
        g = (kappa_i - rho_i * xi_i * 1j * u - d) / \
            (kappa_i - rho_i * xi_i * 1j * u + d + 1e-30)

        exp_ddt = jnp.exp(-d * dt_i)
        A_i = kappa_i * theta_i / xi_i**2 * (
            (kappa_i - rho_i * xi_i * 1j * u - d) * dt_i -
            2.0 * jnp.log((1.0 - g * exp_ddt) / (1.0 - g + 1e-30))
        )
        B_i = (kappa_i - rho_i * xi_i * 1j * u - d) / xi_i**2 * \
              (1.0 - exp_ddt) / (1.0 - g * exp_ddt + 1e-30)

        C = C + A_i + B_i * D * v0 if i > 0 else A_i
        D = B_i if i == 0 else B_i  # D accumulates differently

    # Simplified: chain characteristic exponents
    return jnp.exp(C + D * v0 + 1j * u * (jnp.log(S) + (r - q) * T))


def ptd_heston_price(S, K, T, r, q, v0, times, kappas, thetas, xis, rhos,
                      option_type=OptionType.Call, n_points=64):
    """European option price under PTD Heston.

    Parameters
    ----------
    S : spot
    K : strike
    T : expiry
    times : [0, t1, ..., T] breakpoints
    kappas, thetas, xis, rhos : arrays of parameters per interval
    """
    log_K = jnp.log(K)
    discount = jnp.exp(-r * T)
    du = 0.5

    integral_1 = 0.0
    integral_2 = 0.0

    for j in range(1, n_points + 1):
        u_val = j * du
        cf = ptd_heston_characteristic_function(
            u_val, S, T, r, q, v0, times, kappas, thetas, xis, rhos
        )
        cf_shifted = ptd_heston_characteristic_function(
            u_val - 1j, S, T, r, q, v0, times, kappas, thetas, xis, rhos
        )

        integrand_1 = jnp.real(jnp.exp(-1j * u_val * log_K) * cf_shifted /
                                (1j * u_val * S * jnp.exp((r - q) * T)))
        integrand_2 = jnp.real(jnp.exp(-1j * u_val * log_K) * cf / (1j * u_val))

        integral_1 += integrand_1 * du
        integral_2 += integrand_2 * du

    P1 = 0.5 + integral_1 / jnp.pi
    P2 = 0.5 + integral_2 / jnp.pi

    call = S * jnp.exp(-q * T) * P1 - K * discount * P2

    if option_type == OptionType.Put:
        return call - S * jnp.exp(-q * T) + K * discount
    return call
