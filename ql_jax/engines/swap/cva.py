"""CVA (Credit Valuation Adjustment) swap engine."""

from __future__ import annotations

import jax.numpy as jnp


def cva_swap_engine(
    swap_mtm_fn,
    exposure_dates: jnp.ndarray,
    discount_fn,
    survival_fn,
    recovery: float = 0.4,
) -> float:
    """Compute unilateral CVA for an interest rate swap.

    CVA = (1 - R) * sum_i [ DF(t_i) * EE(t_i) * (Q(t_{i-1}) - Q(t_i)) ]

    Parameters
    ----------
    swap_mtm_fn : callable(t) -> expected positive exposure at time t
    exposure_dates : array of exposure monitoring dates (year fracs)
    discount_fn : t -> discount factor
    survival_fn : t -> counterparty survival probability
    recovery : recovery rate

    Returns
    -------
    float : CVA (positive = charge to counterparty)
    """
    n = exposure_dates.shape[0]
    t_prev = jnp.concatenate([jnp.zeros(1), exposure_dates[:-1]])

    cva = 0.0
    for i in range(n):
        t = float(exposure_dates[i])
        tp = float(t_prev[i])
        df = discount_fn(t)
        ee = jnp.maximum(swap_mtm_fn(t), 0.0)  # Expected positive exposure
        dQ = survival_fn(tp) - survival_fn(t)
        cva = cva + df * ee * dQ

    return (1.0 - recovery) * cva


def dva_swap_engine(
    swap_mtm_fn,
    exposure_dates: jnp.ndarray,
    discount_fn,
    own_survival_fn,
    own_recovery: float = 0.4,
) -> float:
    """Compute DVA (Debit Valuation Adjustment).

    DVA = (1 - R_own) * sum_i [ DF(t_i) * ENE(t_i) * (Q_own(t_{i-1}) - Q_own(t_i)) ]

    Parameters
    ----------
    swap_mtm_fn : callable(t) -> swap MTM (can be negative)
    exposure_dates : monitoring dates
    discount_fn : discount curve
    own_survival_fn : own default survival probability
    own_recovery : own recovery rate
    """
    n = exposure_dates.shape[0]
    t_prev = jnp.concatenate([jnp.zeros(1), exposure_dates[:-1]])

    dva = 0.0
    for i in range(n):
        t = float(exposure_dates[i])
        tp = float(t_prev[i])
        df = discount_fn(t)
        ene = jnp.maximum(-swap_mtm_fn(t), 0.0)  # Expected negative exposure
        dQ = own_survival_fn(tp) - own_survival_fn(t)
        dva = dva + df * ene * dQ

    return (1.0 - own_recovery) * dva


def bilateral_cva(
    swap_mtm_fn,
    exposure_dates: jnp.ndarray,
    discount_fn,
    cpty_survival_fn,
    own_survival_fn,
    cpty_recovery: float = 0.4,
    own_recovery: float = 0.4,
) -> float:
    """Bilateral CVA = CVA - DVA."""
    cva = cva_swap_engine(
        swap_mtm_fn, exposure_dates, discount_fn,
        cpty_survival_fn, cpty_recovery,
    )
    dva = dva_swap_engine(
        swap_mtm_fn, exposure_dates, discount_fn,
        own_survival_fn, own_recovery,
    )
    return cva - dva


# ---------------------------------------------------------------------------
# Brigo-Masetti swaption-based CVA
# ---------------------------------------------------------------------------

def _black_swaption_value(
    discount_fn, forward_rate, strike, sigma, T_exercise,
    payment_dates, accrual_fracs, omega,
):
    """Black swaption price (internal helper).

    omega = +1 for payer swaption, -1 for receiver.
    """
    from jax.scipy.stats import norm

    S = jnp.float64(forward_rate)
    K = jnp.float64(strike)
    vol = jnp.float64(sigma)
    T = jnp.float64(T_exercise)

    annuity = jnp.float64(0.0)
    for i in range(len(payment_dates)):
        annuity = annuity + accrual_fracs[i] * discount_fn(payment_dates[i])

    sqrt_T = jnp.sqrt(jnp.maximum(T, 1e-15))
    d1 = (jnp.log(S / K) + 0.5 * vol**2 * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    return annuity * omega * (S * norm.cdf(omega * d1) - K * norm.cdf(omega * d2))


def _par_swap_rate(discount_fn, start_time, payment_dates, accrual_fracs):
    """Forward par swap rate from discount function."""
    P_start = discount_fn(start_time)
    P_end = discount_fn(payment_dates[-1])

    annuity = jnp.float64(0.0)
    for i in range(len(payment_dates)):
        annuity = annuity + accrual_fracs[i] * discount_fn(payment_dates[i])

    return (P_start - P_end) / annuity


def cva_swap_brigo_masetti(
    discount_fn,
    black_vol: float,
    swap_fixed_rate: float,
    payment_times,
    accrual_fracs,
    notional: float,
    cpty_survival_fn,
    cpty_recovery: float = 0.4,
    invst_survival_fn=None,
    invst_recovery: float = 0.999,
    swap_type: int = 1,
):
    """CVA-adjusted swap NPV using the Brigo-Masetti (2005) approach.

    Models exposure via swaption pricing at each coupon date. The
    counterparty-risky swap NPV is:

      NPV = baseNPV - (1-R_cpty) * Σ swaptionlet_i * P_default(t_{i-1}, t_i)
                     + (1-R_invst) * Σ putSwaptionlet_i * P_own_default(t_{i-1}, t_i)

    Parameters
    ----------
    discount_fn : callable(t) -> discount factor P(0,t)
    black_vol : Black swaption volatility
    swap_fixed_rate : fixed coupon rate of the swap
    payment_times : array-like of payment times (year fractions)
    accrual_fracs : array-like of accrual fractions per period
    notional : swap notional
    cpty_survival_fn : callable(t) -> counterparty survival probability Q(t)
    cpty_recovery : counterparty recovery rate
    invst_survival_fn : callable(t) -> investor survival probability
        If None, investor is assumed default-free.
    invst_recovery : investor recovery rate
    swap_type : +1 for payer, -1 for receiver

    Returns
    -------
    dict with keys: 'base_npv', 'cva', 'dva', 'risky_npv', 'risky_fair_rate'
    """
    payment_times = [float(t) for t in payment_times]
    accrual_fracs = [float(tau) for tau in accrual_fracs]
    n = len(payment_times)

    # Base (risk-free) NPV
    base_fair_rate = float(_par_swap_rate(
        discount_fn, 0.0, payment_times, accrual_fracs,
    ))

    # Base annuity
    base_annuity = 0.0
    for i in range(n):
        base_annuity += accrual_fracs[i] * float(discount_fn(payment_times[i]))
    base_npv = swap_type * notional * base_annuity * (base_fair_rate - swap_fixed_rate)

    # Swaptionlet CVA/DVA sums
    cum_call = jnp.float64(0.0)  # payer swaptions (counterparty CVA)
    cum_put = jnp.float64(0.0)   # receiver swaptions (investor DVA)

    swaplet_start = 0.0
    for j in range(n):
        t_j = payment_times[j]
        # Remaining swap: from t_j to end
        remaining_dates = payment_times[j:]
        remaining_fracs = accrual_fracs[j:]

        if len(remaining_dates) < 1:
            swaplet_start = t_j
            continue

        # Forward par rate for remaining swap starting at swaplet_start
        fwd_rate = _par_swap_rate(
            discount_fn, swaplet_start, remaining_dates, remaining_fracs,
        )

        # ATM swaption: strike = base_fair_rate (the risk-free fair rate)
        T_ex = jnp.float64(swaplet_start)

        # Only price if exercise time > 0
        if swaplet_start > 1e-10:
            # Payer swaption (counterparty exposure)
            call_val = _black_swaption_value(
                discount_fn, fwd_rate, base_fair_rate, black_vol, T_ex,
                remaining_dates, remaining_fracs, jnp.float64(swap_type),
            )
            # Receiver swaption (investor exposure)
            put_val = _black_swaption_value(
                discount_fn, fwd_rate, base_fair_rate, black_vol, T_ex,
                remaining_dates, remaining_fracs, jnp.float64(-swap_type),
            )
        else:
            # At t=0, the "swaption" is just max(swap_value, 0)
            call_val = jnp.maximum(jnp.float64(swap_type) * base_npv / notional, 0.0)
            put_val = jnp.maximum(jnp.float64(-swap_type) * base_npv / notional, 0.0)

        # Default probability in period [swaplet_start, t_j]
        dp_cpty = cpty_survival_fn(swaplet_start) - cpty_survival_fn(t_j)
        cum_call = cum_call + call_val * dp_cpty

        if invst_survival_fn is not None:
            dp_invst = invst_survival_fn(swaplet_start) - invst_survival_fn(t_j)
            cum_put = cum_put + put_val * dp_invst

        swaplet_start = t_j

    cva = (1.0 - cpty_recovery) * notional * cum_call
    dva = (1.0 - invst_recovery) * notional * cum_put

    risky_npv = base_npv - cva + dva

    # Risky fair rate: rate that makes risky NPV = 0
    # base_npv = type * N * A * (S_fair - K)
    # risky_npv = type * N * A * (S_risky_fair - K)
    # => S_risky_fair = K + risky_npv / (type * N * A)
    if abs(base_annuity) > 1e-20:
        risky_fair_rate = swap_fixed_rate + risky_npv / (swap_type * notional * base_annuity)
    else:
        risky_fair_rate = base_fair_rate

    return {
        "base_npv": base_npv,
        "cva": cva,
        "dva": dva,
        "risky_npv": risky_npv,
        "base_fair_rate": base_fair_rate,
        "risky_fair_rate": risky_fair_rate,
    }
