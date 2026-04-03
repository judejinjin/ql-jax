"""CPI cap/floor term price surface."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class CPICapFloorTermPriceSurface:
    """Market prices for CPI caps and floors across strikes and maturities.

    Parameters
    ----------
    maturities : (n_mat,) array of maturities (years)
    strikes : (n_k,) array of CPI inflation rate strikes
    cap_prices : (n_mat, n_k) cap prices
    floor_prices : (n_mat, n_k) floor prices
    base_cpi : base CPI index level
    """
    maturities: jnp.ndarray
    strikes: jnp.ndarray
    cap_prices: jnp.ndarray
    floor_prices: jnp.ndarray
    base_cpi: float

    def cap_price(self, maturity, strike):
        """Interpolate cap price at given maturity and strike."""
        return _bilinear_interp(self.maturities, self.strikes,
                                self.cap_prices, maturity, strike)

    def floor_price(self, maturity, strike):
        """Interpolate floor price at given maturity and strike."""
        return _bilinear_interp(self.maturities, self.strikes,
                                self.floor_prices, maturity, strike)

    def atm_rate(self, maturity):
        """Estimate ATM CPI inflation rate at given maturity."""
        # Find strike where cap = floor (put-call parity)
        idx = jnp.argmin(jnp.abs(self.cap_prices - self.floor_prices), axis=1)
        mat_idx = jnp.argmin(jnp.abs(self.maturities - maturity))
        return float(self.strikes[idx[mat_idx]])


@dataclass
class YoYCapFloorTermPriceSurface:
    """Market prices for Year-on-Year inflation caps and floors.

    Parameters
    ----------
    maturities : (n_mat,) maturities
    strikes : (n_k,) YoY inflation rate strikes
    cap_prices : (n_mat, n_k)
    floor_prices : (n_mat, n_k)
    """
    maturities: jnp.ndarray
    strikes: jnp.ndarray
    cap_prices: jnp.ndarray
    floor_prices: jnp.ndarray

    def cap_price(self, maturity, strike):
        return _bilinear_interp(self.maturities, self.strikes,
                                self.cap_prices, maturity, strike)

    def floor_price(self, maturity, strike):
        return _bilinear_interp(self.maturities, self.strikes,
                                self.floor_prices, maturity, strike)


def strip_yoy_optionlet_vols(yoy_surface, discount_factors, yoy_rates,
                              vol_type='normal'):
    """Strip Year-on-Year optionlet volatilities from cap/floor prices.

    Parameters
    ----------
    yoy_surface : YoYCapFloorTermPriceSurface
    discount_factors : (n_mat,) discount factors at each maturity
    yoy_rates : (n_mat,) forward YoY inflation rates
    vol_type : 'normal' or 'lognormal'

    Returns
    -------
    optionlet_vols : (n_mat, n_k) stripped optionlet volatilities
    """
    n_mat = len(yoy_surface.maturities)
    n_k = len(yoy_surface.strikes)
    df = jnp.asarray(discount_factors)
    fwd = jnp.asarray(yoy_rates)

    vols = jnp.zeros((n_mat, n_k))

    # Bootstrap: subtract previous caplets from cap price
    for j in range(n_k):
        cum_caplet = 0.0
        for i in range(n_mat):
            cap_price_val = float(yoy_surface.cap_prices[i, j])
            optionlet_price = cap_price_val - cum_caplet

            T = float(yoy_surface.maturities[i])
            K = float(yoy_surface.strikes[j])
            F = float(fwd[i])
            D = float(df[i])

            if optionlet_price > 1e-12 and D > 1e-12:
                vol = _implied_normal_vol(optionlet_price, F, K, T, D)
                vols = vols.at[i, j].set(vol)

            cum_caplet = cap_price_val

    return vols


def _implied_normal_vol(price, F, K, T, D):
    """Approximate implied normal vol from Bachelier formula."""
    from jax.scipy.stats import norm
    intrinsic = jnp.maximum(F - K, 0.0) * D
    time_value = jnp.maximum(price - intrinsic, 1e-15)
    # Bachelier: C ≈ D * sigma * sqrt(T) * phi(0) when ATM
    sigma_approx = price / (D * jnp.sqrt(T) * 0.3989)  # phi(0)
    return float(jnp.clip(sigma_approx, 1e-6, 1.0))


def _bilinear_interp(xs, ys, zs, x, y):
    """Simple bilinear interpolation on a grid."""
    val_at_x = jnp.array([float(jnp.interp(x, xs, zs[:, j]))
                          for j in range(len(ys))])
    return float(jnp.interp(y, ys, val_at_x))


@dataclass
class CappedFlooredYoYInflationCoupon:
    """Year-on-Year inflation coupon with cap and/or floor.

    Parameters
    ----------
    yoy_rate : float, forward YoY inflation rate
    gearing : multiplicative factor
    spread : additive spread
    cap : cap rate (None if uncapped)
    floor : floor rate (None if unfloored)
    """
    yoy_rate: float
    gearing: float = 1.0
    spread: float = 0.0
    cap: float = None
    floor: float = None

    def rate(self):
        """Compute the capped/floored coupon rate."""
        r = self.gearing * self.yoy_rate + self.spread
        if self.floor is not None:
            r = max(r, self.floor)
        if self.cap is not None:
            r = min(r, self.cap)
        return r
