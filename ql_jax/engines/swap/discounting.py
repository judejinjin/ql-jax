"""Discounting swap engine — price fixed-vs-floating swaps."""

from __future__ import annotations

from ql_jax.time.date import Date
from ql_jax.cashflows.analytics import npv as cf_npv, _cf_date, _cf_amount


def discounting_swap_npv(
    swap,
    discount_curve,
    forecast_curve=None,
    settlement_date: Date | None = None,
) -> float:
    """Compute NPV of a vanilla swap.

    Parameters
    ----------
    swap : VanillaSwap or OvernightIndexedSwap
    discount_curve : YieldTermStructure
        Curve for discounting cash flows.
    forecast_curve : YieldTermStructure or None
        Curve for forecasting floating rates. If None, uses discount_curve.
    settlement_date : Date or None

    Returns
    -------
    float : NPV from the perspective of the swap type (payer = pay fixed)
    """
    fc = forecast_curve or discount_curve
    settle = settlement_date or discount_curve.reference_date

    # Fixed leg NPV
    fixed_npv = cf_npv(swap.fixed_leg, discount_curve, settlement_date=settle)

    # Floating leg NPV — need to project floating rates
    float_npv = 0.0
    for cf in swap.floating_leg:
        d = _cf_date(cf)
        if d <= settle:
            continue
        if hasattr(cf, "amount_with_curve"):
            amount = cf.amount_with_curve(fc)
        else:
            amount = _cf_amount(cf)
        t = discount_curve.time_from_reference(d)
        df = discount_curve.discount(t)
        float_npv += amount * df

    # Payer: pays fixed, receives floating → NPV = float - fixed
    # Receiver: receives fixed, pays floating → NPV = fixed - float
    return swap.type_ * (float_npv - fixed_npv)


def discounting_swap_fair_rate(
    swap,
    discount_curve,
    forecast_curve=None,
    settlement_date: Date | None = None,
) -> float:
    """Compute the par (fair) fixed rate for a swap.

    The fair rate is the fixed rate that makes the swap NPV = 0.
    fair_rate = float_leg_npv / fixed_leg_annuity
    """
    fc = forecast_curve or discount_curve
    settle = settlement_date or discount_curve.reference_date

    # Fixed leg annuity (BPS-like: sum of notional * accrual * df)
    annuity = 0.0
    for cf in swap.fixed_leg:
        d = _cf_date(cf)
        if d <= settle:
            continue
        if hasattr(cf, "accrual_period") and hasattr(cf, "day_counter"):
            tau = cf.accrual_period(cf.day_counter)
        else:
            continue
        t = discount_curve.time_from_reference(d)
        df = discount_curve.discount(t)
        annuity += cf.nominal * tau * df

    # Floating leg NPV
    float_npv = 0.0
    for cf in swap.floating_leg:
        d = _cf_date(cf)
        if d <= settle:
            continue
        if hasattr(cf, "amount_with_curve"):
            amount = cf.amount_with_curve(fc)
        else:
            amount = _cf_amount(cf)
        t = discount_curve.time_from_reference(d)
        df = discount_curve.discount(t)
        float_npv += amount * df

    if abs(annuity) < 1e-20:
        return 0.0
    return float_npv / annuity


def discounting_swap_fair_spread(
    swap,
    discount_curve,
    forecast_curve=None,
    settlement_date: Date | None = None,
) -> float:
    """Compute the par (fair) floating spread for a swap.

    The fair spread is the spread that makes the swap NPV = 0.
    """
    fc = forecast_curve or discount_curve
    settle = settlement_date or discount_curve.reference_date

    # Fixed leg NPV
    fixed_npv = cf_npv(swap.fixed_leg, discount_curve, settlement_date=settle)

    # Floating leg NPV (at zero spread)
    float_npv = 0.0
    float_annuity = 0.0
    for cf in swap.floating_leg:
        d = _cf_date(cf)
        if d <= settle:
            continue
        if hasattr(cf, "amount_with_curve"):
            amount = cf.amount_with_curve(fc)
        else:
            amount = _cf_amount(cf)
        t = discount_curve.time_from_reference(d)
        df = discount_curve.discount(t)
        float_npv += amount * df

        if hasattr(cf, "accrual_period") and hasattr(cf, "day_counter"):
            tau = cf.accrual_period(cf.day_counter)
            float_annuity += cf.nominal * tau * df

    if abs(float_annuity) < 1e-20:
        return 0.0
    # NPV = type * (float_npv + spread * annuity - fixed_npv) = 0
    # spread = (fixed_npv - float_npv) / annuity
    return (fixed_npv - float_npv) / float_annuity
