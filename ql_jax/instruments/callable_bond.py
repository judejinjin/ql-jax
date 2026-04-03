"""Callable bond instruments and engines."""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Sequence, Optional


@dataclass
class CallScheduleEntry:
    """One entry in a call/put schedule."""
    exercise_time: float  # time from valuation date (years)
    price: float = 100.0  # call/put price (clean)
    is_call: bool = True


@dataclass
class CallableFixedRateBond:
    """Fixed-rate bond with embedded call and/or put options.

    Parameters
    ----------
    face : face value
    coupon_rate : annual coupon rate
    coupon_times : times of coupon payments
    maturity : maturity in years
    call_schedule : list of callable dates/prices
    put_schedule : optional list of puttable dates/prices
    """
    face: float
    coupon_rate: float
    coupon_times: jnp.ndarray
    maturity: float
    call_schedule: Sequence[CallScheduleEntry]
    put_schedule: Optional[Sequence[CallScheduleEntry]] = None


def black_callable_bond_price(bond, yield_vol, discount_curve_times,
                              discount_curve_values):
    """Price callable bond using Black's model.

    Uses Black's formula on each call date with yield volatility.

    Parameters
    ----------
    bond : CallableFixedRateBond
    yield_vol : yield volatility (annualized)
    discount_curve_times : times for discount factors
    discount_curve_values : discount factors

    Returns
    -------
    price : callable bond dirty price
    """
    dc_t = jnp.asarray(discount_curve_times)
    dc_v = jnp.asarray(discount_curve_values)

    def _disc(t):
        return jnp.interp(t, dc_t, dc_v)

    # Price the straight bond first
    straight = 0.0
    coupon_amt = bond.face * bond.coupon_rate
    for t in bond.coupon_times:
        t = float(t)
        if t > 0:
            straight += coupon_amt * float(_disc(t))
    straight += bond.face * float(_disc(bond.maturity))

    # Subtract call option value using Black's model
    from jax.scipy.stats import norm
    call_value = 0.0
    for entry in bond.call_schedule:
        if not entry.is_call:
            continue
        T = entry.exercise_time
        if T <= 0:
            continue
        # Forward bond price at call date
        df_T = float(_disc(T))
        # Remaining coupons and principal after T
        fwd_bond = 0.0
        for t in bond.coupon_times:
            t = float(t)
            if t > T:
                fwd_bond += coupon_amt * float(_disc(t)) / df_T
        fwd_bond += bond.face * float(_disc(bond.maturity)) / df_T

        K = entry.price
        vol_T = yield_vol * jnp.sqrt(T)
        if vol_T > 1e-10:
            d1 = (jnp.log(fwd_bond / K) + 0.5 * yield_vol**2 * T) / vol_T
            d2 = d1 - vol_T
            opt = df_T * (fwd_bond * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            opt = df_T * jnp.maximum(fwd_bond - K, 0.0)
        call_value += float(opt)

    # Add put option value
    put_value = 0.0
    if bond.put_schedule:
        for entry in bond.put_schedule:
            T = entry.exercise_time
            if T <= 0:
                continue
            df_T = float(_disc(T))
            fwd_bond = 0.0
            for t in bond.coupon_times:
                t = float(t)
                if t > T:
                    fwd_bond += coupon_amt * float(_disc(t)) / df_T
            fwd_bond += bond.face * float(_disc(bond.maturity)) / df_T
            K = entry.price
            vol_T = yield_vol * jnp.sqrt(T)
            if vol_T > 1e-10:
                d1 = (jnp.log(fwd_bond / K) + 0.5 * yield_vol**2 * T) / vol_T
                d2 = d1 - vol_T
                opt = df_T * (K * norm.cdf(-d2) - fwd_bond * norm.cdf(-d1))
            else:
                opt = df_T * jnp.maximum(K - fwd_bond, 0.0)
            put_value += float(opt)

    return straight - call_value + put_value


@dataclass
class CallableBondConstantVolatility:
    """Constant yield volatility for callable bond pricing."""
    vol: float

    def volatility(self, t=None):
        return self.vol
