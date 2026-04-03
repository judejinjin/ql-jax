"""Arithmetic average OIS and irregular swaptions."""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Sequence


@dataclass
class ArithmeticAverageOIS:
    """OIS swap with arithmetic (vs geometric) averaging of overnight rates.

    Parameters
    ----------
    notional : swap notional
    fixed_rate : fixed leg rate
    payment_times : (n_periods,) payment dates
    overnight_rates : (n_obs_per_period, n_periods) overnight rates
    """
    notional: float
    fixed_rate: float
    payment_times: jnp.ndarray
    overnight_rates: jnp.ndarray = None

    def npv(self, discount_factors):
        """Net present value given discount factors at payment times."""
        df = jnp.asarray(discount_factors)
        n = len(self.payment_times)
        npv = 0.0
        for i in range(n):
            if self.overnight_rates is not None:
                # Arithmetic average
                floating = float(self.overnight_rates[:, i].mean())
            else:
                floating = 0.0
            dt = float(self.payment_times[i])
            if i > 0:
                dt = float(self.payment_times[i] - self.payment_times[i - 1])
            npv += (floating - self.fixed_rate) * dt * float(df[i])
        return self.notional * npv


@dataclass
class IrregularSwap:
    """Swap with non-standard features (different notionals per period, etc.).

    Parameters
    ----------
    fixed_notionals : (n,) notionals for fixed leg periods
    float_notionals : (n,) notionals for floating leg periods
    fixed_rates : (n,) fixed rates per period
    float_spreads : (n,) floating leg spreads
    payment_times : (n,) payment times
    """
    fixed_notionals: jnp.ndarray
    float_notionals: jnp.ndarray
    fixed_rates: jnp.ndarray
    float_spreads: jnp.ndarray
    payment_times: jnp.ndarray

    def npv(self, forward_rates, discount_factors):
        """Compute NPV of irregular swap."""
        fwd = jnp.asarray(forward_rates)
        df = jnp.asarray(discount_factors)
        n = len(self.payment_times)
        npv = 0.0
        for i in range(n):
            dt = float(self.payment_times[i])
            if i > 0:
                dt = float(self.payment_times[i] - self.payment_times[i - 1])
            fixed_cf = float(self.fixed_notionals[i] * self.fixed_rates[i]) * dt
            float_cf = float(self.float_notionals[i] * (fwd[i] + self.float_spreads[i])) * dt
            npv += (float_cf - fixed_cf) * float(df[i])
        return npv


@dataclass
class IrregularSwaption:
    """Option on an irregular swap.

    Parameters
    ----------
    underlying : IrregularSwap
    exercise_time : exercise date
    is_payer : True = payer swaption
    """
    underlying: IrregularSwap
    exercise_time: float
    is_payer: bool = True


def hagan_irregular_swaption_price(swaption, vol, forward_rates,
                                    discount_factors):
    """Price irregular swaption using Hagan's method.

    Approximates as a basket of co-terminal swaptions.

    Parameters
    ----------
    swaption : IrregularSwaption
    vol : swaption volatility
    forward_rates : forward rates
    discount_factors : discount factors

    Returns
    -------
    price : swaption price
    """
    from jax.scipy.stats import norm

    swap = swaption.underlying
    fwd = jnp.asarray(forward_rates)
    df = jnp.asarray(discount_factors)
    T = swaption.exercise_time

    # Effective annuity and forward swap rate
    n = len(swap.payment_times)
    annuity = 0.0
    float_leg = 0.0
    for i in range(n):
        dt = float(swap.payment_times[i])
        if i > 0:
            dt = float(swap.payment_times[i] - swap.payment_times[i - 1])
        annuity += float(swap.float_notionals[i]) * dt * float(df[i])
        float_leg += float(swap.float_notionals[i] * (fwd[i] + swap.float_spreads[i])) * dt * float(df[i])

    fixed_leg = 0.0
    for i in range(n):
        dt = float(swap.payment_times[i])
        if i > 0:
            dt = float(swap.payment_times[i] - swap.payment_times[i - 1])
        fixed_leg += float(swap.fixed_notionals[i] * swap.fixed_rates[i]) * dt * float(df[i])

    fwd_swap_rate = float_leg / (annuity + 1e-15)
    strike = fixed_leg / (annuity + 1e-15)

    # Black formula
    vol_sqrt_T = vol * jnp.sqrt(T)
    d1 = (jnp.log(fwd_swap_rate / strike) + 0.5 * vol**2 * T) / (vol_sqrt_T + 1e-15)
    d2 = d1 - vol_sqrt_T

    phi = 1.0 if swaption.is_payer else -1.0
    price = phi * annuity * (fwd_swap_rate * norm.cdf(phi * d1) -
                              strike * norm.cdf(phi * d2))
    return float(jnp.maximum(price, 0.0))


@dataclass
class TenorMappedVol:
    """Volatility mapped from one tenor to another (basis model)."""
    base_vols: jnp.ndarray  # (n_expiry, n_tenor) vol surface
    base_tenors: jnp.ndarray  # base tenor grid
    target_tenor: float

    def vol(self, expiry_idx):
        """Get mapped vol for a given expiry."""
        return float(jnp.interp(self.target_tenor, self.base_tenors,
                                 self.base_vols[expiry_idx, :]))
