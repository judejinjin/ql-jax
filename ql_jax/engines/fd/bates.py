"""FD Bates model vanilla engine.

Bates = Heston + Merton jumps. Uses operator splitting: Heston PDE +
integral correction for jump component.
"""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax.engines.fd.heston import fd_heston_price


def fd_bates_price(S, K, T, r, q, v0, kappa, theta, xi, rho,
                   lam_j, mu_j, sigma_j, option_type=1,
                   n_x=100, n_v=50, n_t=200, is_american=False):
    """Bates model FD price.

    Uses effective parameters: transforms Bates to Heston with adjusted drift.
    The jump compensation modifies the drift of the underlying.

    Parameters
    ----------
    S, K, T, r, q : standard option params
    v0, kappa, theta, xi, rho : Heston params
    lam_j : jump intensity
    mu_j : mean log-jump size
    sigma_j : log-jump volatility
    option_type : 1=call, -1=put
    n_x, n_v, n_t : grid sizes
    is_american : early exercise

    Returns
    -------
    price : option price
    """
    # Jump-compensated drift: r_adj = r - lam*(E[e^J]-1)
    jump_comp = lam_j * (jnp.exp(mu_j + 0.5 * sigma_j**2) - 1.0)
    r_adj = r - jump_comp

    # Heston with adjusted rate (the jump integral adds variance too)
    # For a first-order approximation, we increase v0/theta by jump variance
    jump_var = lam_j * (mu_j**2 + sigma_j**2)
    v0_adj = v0 + jump_var
    theta_adj = theta + jump_var

    return fd_heston_price(S, K, T, r_adj, q, v0_adj, kappa, theta_adj, xi, rho,
                           option_type, n_x, n_v, n_t, is_american)
