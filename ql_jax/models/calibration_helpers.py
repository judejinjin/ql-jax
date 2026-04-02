"""Market model calibration helpers: swaption and cap helpers."""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def swaption_helper_npv(vol, expiry, swap_length, strike, annuity,
                          forward_swap_rate, is_payer=True):
    """Black swaption NPV given a Black vol.

    Parameters
    ----------
    vol : float – Black swaption vol
    expiry : float – option expiry
    swap_length : float – swap tenor
    strike : float – swaption strike
    annuity : float – present value of annuity
    forward_swap_rate : float – ATM forward swap rate
    is_payer : bool
    """
    total_vol = vol * jnp.sqrt(expiry)
    d1 = (jnp.log(forward_swap_rate / strike) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol

    sign = 1.0 if is_payer else -1.0
    return annuity * sign * (
        forward_swap_rate * norm.cdf(sign * d1) - strike * norm.cdf(sign * d2)
    )


def cap_helper_npv(vol, forward_rate, strike, accrual, discount, expiry):
    """Black caplet NPV.

    Parameters
    ----------
    vol : float – Black caplet vol
    forward_rate : float – forward LIBOR rate
    strike : float – caplet strike
    accrual : float – day count fraction
    discount : float – discount factor to payment
    expiry : float – option expiry
    """
    total_vol = vol * jnp.sqrt(expiry)
    d1 = (jnp.log(forward_rate / strike) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol

    return accrual * discount * (
        forward_rate * norm.cdf(d1) - strike * norm.cdf(d2)
    )


def calibrate_model_to_swaptions(model_price_fn, market_vols, swap_data,
                                   initial_params, bounds=None):
    """Generic calibration to swaption market.

    Parameters
    ----------
    model_price_fn : callable(params, expiry, swap_length, strike) -> price
    market_vols : array of market Black vols
    swap_data : list of dicts with keys: expiry, swap_length, strike, annuity, forward
    initial_params : initial model parameters

    Returns calibrated parameters.
    """
    def objective(params):
        errors = []
        for i, sd in enumerate(swap_data):
            model_price = model_price_fn(params, sd['expiry'], sd['swap_length'], sd['strike'])
            market_price = swaption_helper_npv(
                market_vols[i], sd['expiry'], sd['swap_length'],
                sd['strike'], sd['annuity'], sd['forward']
            )
            errors.append(model_price - market_price)
        return jnp.array(errors)

    from ql_jax.math.optimization.levenberg_marquardt import minimize
    result = minimize(objective, initial_params, max_iterations=200)
    return result['x']
