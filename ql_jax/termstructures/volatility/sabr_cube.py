"""SABR swaption volatility cube.

3D volatility surface: expiry x tenor x strike, using SABR on each
expiry-tenor point for smile interpolation.
"""

import jax.numpy as jnp
from ql_jax.math.interpolations.sabr import sabr_vol, build as sabr_build


def build_sabr_cube(expiries, tenors, strikes_grid, market_vols,
                     forward_swap_rates, beta=0.5):
    """Calibrate a SABR cube to swaption vol data.

    Parameters
    ----------
    expiries : array (n_exp,) – swaption expiries
    tenors : array (n_ten,) – swap tenors
    strikes_grid : array (n_k,) – strike offsets from ATM
    market_vols : array (n_exp, n_ten, n_k) – market Black vols
    forward_swap_rates : array (n_exp, n_ten) – forward swap rates
    beta : float – SABR beta (typical: 0.5 for rates)

    Returns dict with SABR parameters per (expiry, tenor) point.
    """
    n_exp = len(expiries)
    n_ten = len(tenors)

    alphas = jnp.zeros((n_exp, n_ten))
    nus = jnp.zeros((n_exp, n_ten))
    rhos = jnp.zeros((n_exp, n_ten))

    for i in range(n_exp):
        for j in range(n_ten):
            f = forward_swap_rates[i, j]
            t = expiries[i]
            vols = market_vols[i, j]
            ks = f + strikes_grid  # absolute strikes

            # Calibrate SABR at this point
            try:
                params = sabr_build(ks, vols, f, t, beta=beta)
                alphas = alphas.at[i, j].set(params['alpha'])
                nus = nus.at[i, j].set(params['nu'])
                rhos = rhos.at[i, j].set(params['rho'])
            except Exception:
                # Fallback: use ATM vol
                alphas = alphas.at[i, j].set(vols[len(vols) // 2])

    return {
        'expiries': jnp.array(expiries),
        'tenors': jnp.array(tenors),
        'forward_swap_rates': forward_swap_rates,
        'beta': beta,
        'alphas': alphas,
        'nus': nus,
        'rhos': rhos,
    }


def evaluate_sabr_cube(cube, expiry, tenor, strike):
    """Evaluate SABR cube at (expiry, tenor, strike).

    Bilinearly interpolates SABR parameters over expiry and tenor,
    then evaluates SABR formula at the strike.
    """
    # Find nearest expiry/tenor indices
    exp_idx = jnp.searchsorted(cube['expiries'], expiry, side='right') - 1
    exp_idx = jnp.clip(exp_idx, 0, len(cube['expiries']) - 2)
    ten_idx = jnp.searchsorted(cube['tenors'], tenor, side='right') - 1
    ten_idx = jnp.clip(ten_idx, 0, len(cube['tenors']) - 2)

    # Bilinear interpolation weights
    e0, e1 = cube['expiries'][exp_idx], cube['expiries'][exp_idx + 1]
    t0, t1 = cube['tenors'][ten_idx], cube['tenors'][ten_idx + 1]
    we = (expiry - e0) / (e1 - e0 + 1e-30)
    wt = (tenor - t0) / (t1 - t0 + 1e-30)

    def interp_param(arr):
        v00 = arr[exp_idx, ten_idx]
        v01 = arr[exp_idx, ten_idx + 1]
        v10 = arr[exp_idx + 1, ten_idx]
        v11 = arr[exp_idx + 1, ten_idx + 1]
        return (1 - we) * ((1 - wt) * v00 + wt * v01) + we * ((1 - wt) * v10 + wt * v11)

    alpha = interp_param(cube['alphas'])
    nu = interp_param(cube['nus'])
    rho = interp_param(cube['rhos'])
    f = interp_param(cube['forward_swap_rates'])

    return sabr_vol(f, strike, expiry, alpha, cube['beta'], rho, nu)
