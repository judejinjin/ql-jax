"""Optionlet stripping from cap/floor market quotes.

OptionletStripper1: bootstraps optionlet vols from ATM cap vols
OptionletStripper2: adjusts with smile information
"""

import jax.numpy as jnp
from jax.scipy.stats import norm


def strip_optionlet_vols(cap_maturities, cap_vols, forward_rates, accruals,
                          discount_factors):
    """Bootstrap optionlet (caplet) vols from flat cap vols.

    Parameters
    ----------
    cap_maturities : array (n,) – cap maturities
    cap_vols : array (n,) – flat cap Black vols
    forward_rates : array (n,) – forward LIBOR rates for each caplet
    accruals : array (n,) – day count fractions
    discount_factors : array (n,) – discount factors to payment dates

    Returns array (n,) of optionlet (caplet) Black vols.
    """
    n = len(cap_maturities)
    optionlet_vols = jnp.zeros(n)

    # First caplet vol = first cap vol
    optionlet_vols = optionlet_vols.at[0].set(cap_vols[0])

    for i in range(1, n):
        # Cap price = sum of caplet prices
        # Cap_i(sigma_flat_i) = sum_{j=0}^{i} Caplet_j(sigma_j)

        # Compute cap price at flat vol
        cap_price = 0.0
        for j in range(i + 1):
            cap_price += _black_caplet(forward_rates[j], forward_rates[j],  # ATM
                                        cap_vols[i], cap_maturities[j],
                                        accruals[j], discount_factors[j])

        # Known caplet prices (already stripped)
        known_price = 0.0
        for j in range(i):
            known_price += _black_caplet(forward_rates[j], forward_rates[j],
                                          optionlet_vols[j], cap_maturities[j],
                                          accruals[j], discount_factors[j])

        # Solve for optionlet vol of caplet i
        target = cap_price - known_price

        from ql_jax.math.solvers.brent import solve

        def objective(vol, i=i, target=target):
            p = _black_caplet(forward_rates[i], forward_rates[i], vol,
                               cap_maturities[i], accruals[i], discount_factors[i])
            return p - target

        try:
            vol_i = solve(objective, 0.001, 3.0)
            optionlet_vols = optionlet_vols.at[i].set(vol_i)
        except Exception:
            optionlet_vols = optionlet_vols.at[i].set(cap_vols[i])

    return optionlet_vols


def _black_caplet(forward, strike, vol, expiry, accrual, discount):
    """Black caplet price."""
    total_vol = vol * jnp.sqrt(expiry)
    d1 = (jnp.log(forward / strike) + 0.5 * total_vol**2) / (total_vol + 1e-30)
    d2 = d1 - total_vol
    return accrual * discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def optionlet_stripper_with_smile(atm_optionlet_vols, cap_maturities,
                                    smile_sections, forward_rates):
    """Adjust ATM optionlet vols using smile sections at each maturity.

    Parameters
    ----------
    atm_optionlet_vols : array (n,) – ATM optionlet vols (from stripper1)
    cap_maturities : array (n,)
    smile_sections : list of callables smile_fn(K) -> vol at each maturity
    forward_rates : array (n,) – ATM forwards

    Returns callable(maturity, strike) -> optionlet vol.
    """
    def vol_fn(maturity, strike):
        idx = jnp.searchsorted(jnp.array(cap_maturities), maturity, side='right') - 1
        idx = jnp.clip(idx, 0, len(cap_maturities) - 1)
        atm_vol = atm_optionlet_vols[int(idx)]
        smile_vol = smile_sections[int(idx)](strike)
        atm_smile = smile_sections[int(idx)](forward_rates[int(idx)])
        # Shift: maintain ATM level from stripper1, add smile shape
        return atm_vol + (smile_vol - atm_smile)

    return vol_fn
