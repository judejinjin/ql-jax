"""Bond pricing engines and analytics (BondFunctions)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.time.daycounter import year_fraction
from ql_jax.cashflows.analytics import (
    npv as cf_npv,
    bps as cf_bps,
    yield_rate as cf_yield,
    duration as cf_duration,
    convexity as cf_convexity,
    accrued_amount as cf_accrued,
    Duration,
)


# ---------------------------------------------------------------------------
# Discounting bond engine
# ---------------------------------------------------------------------------

def discounting_bond_npv(bond, discount_curve, settlement_date=None) -> float:
    """Price a bond by discounting its cash flows.

    Parameters
    ----------
    bond : Bond
    discount_curve : YieldTermStructure
    settlement_date : Date or None

    Returns
    -------
    float : clean price as percentage of face value
    """
    settle = settlement_date or bond.settlement_date()
    all_cfs = bond.cashflows
    raw_npv = cf_npv(all_cfs, discount_curve, settlement_date=settle)
    return raw_npv


def discounting_bond_clean_price(bond, discount_curve, settlement_date=None) -> float:
    """Clean price = dirty price - accrued interest, as percent of face."""
    settle = settlement_date or discount_curve.reference_date
    dirty = discounting_bond_npv(bond, discount_curve, settle)
    accrued = cf_accrued(bond.cashflows, settle)
    return (dirty - accrued) / bond.notional * 100.0


def discounting_bond_dirty_price(bond, discount_curve, settlement_date=None) -> float:
    """Dirty price as percent of face value."""
    settle = settlement_date or discount_curve.reference_date
    raw = discounting_bond_npv(bond, discount_curve, settle)
    return raw / bond.notional * 100.0


# ---------------------------------------------------------------------------
# BondFunctions — static analytics
# ---------------------------------------------------------------------------

class BondFunctions:
    """Collection of bond analytics functions."""

    @staticmethod
    def clean_price(bond, discount_curve, settlement_date=None) -> float:
        return discounting_bond_clean_price(bond, discount_curve, settlement_date)

    @staticmethod
    def dirty_price(bond, discount_curve, settlement_date=None) -> float:
        return discounting_bond_dirty_price(bond, discount_curve, settlement_date)

    @staticmethod
    def npv(bond, discount_curve, settlement_date=None) -> float:
        return discounting_bond_npv(bond, discount_curve, settlement_date)

    @staticmethod
    def bps(bond, discount_curve, settlement_date=None) -> float:
        settle = settlement_date or bond.settlement_date()
        return cf_bps(bond.cashflows, discount_curve, settlement_date=settle)

    @staticmethod
    def yield_rate(
        bond,
        target_clean_price: float,
        day_counter: str,
        settlement_date: Date | None = None,
        guess: float = 0.05,
    ) -> float:
        """Solve for yield given a clean price (as percentage of face)."""
        settle = settlement_date or bond.settlement_date()
        accrued = cf_accrued(bond.cashflows, settle)
        target_npv = target_clean_price / 100.0 * bond.notional + accrued
        return cf_yield(
            bond.cashflows, target_npv, day_counter, settle, guess=guess,
        )

    @staticmethod
    def duration(
        bond,
        yield_value: float,
        day_counter: str,
        settlement_date: Date | None = None,
        duration_type: int = Duration.Modified,
    ) -> float:
        settle = settlement_date or bond.settlement_date()
        return cf_duration(
            bond.cashflows, yield_value, day_counter, settle, duration_type,
        )

    @staticmethod
    def convexity(
        bond,
        yield_value: float,
        day_counter: str,
        settlement_date: Date | None = None,
    ) -> float:
        settle = settlement_date or bond.settlement_date()
        return cf_convexity(bond.cashflows, yield_value, day_counter, settle)

    @staticmethod
    def accrued_amount(bond, settlement_date: Date | None = None) -> float:
        settle = settlement_date or bond.settlement_date()
        return cf_accrued(bond.cashflows, settle)

    @staticmethod
    def maturity_date(bond) -> Date | None:
        return bond.maturity_date

    @staticmethod
    def z_spread(
        bond,
        discount_curve,
        target_clean_price: float,
        day_counter: str,
        settlement_date: Date | None = None,
        guess: float = 0.0,
        accuracy: float = 1e-10,
        max_iter: int = 100,
    ) -> float:
        """Solve for z-spread over the discount curve."""
        settle = settlement_date or bond.settlement_date()
        accrued = cf_accrued(bond.cashflows, settle)
        target_npv = target_clean_price / 100.0 * bond.notional + accrued

        from ql_jax.cashflows.analytics import _cf_amount, _cf_date

        def _npv_at_spread(z):
            total = jnp.float64(0.0)
            for cf in bond.cashflows:
                d = _cf_date(cf)
                if d <= settle:
                    continue
                amount = _cf_amount(cf)
                t = discount_curve.time_from_reference(d)
                df = discount_curve.discount(t) * jnp.exp(-z * t)
                total = total + amount * df
            return total - target_npv

        _grad = jax.grad(_npv_at_spread)
        z = jnp.float64(guess)
        for _ in range(max_iter):
            f = _npv_at_spread(z)
            if jnp.abs(f) < accuracy:
                break
            fp = _grad(z)
            if jnp.abs(fp) < 1e-20:
                break
            z = z - f / fp
        return float(z)


# ---------------------------------------------------------------------------
# Bond forward (repo) pricing
# ---------------------------------------------------------------------------

def bond_forward_spot_income(
    bond,
    settlement_date,
    delivery_date,
    income_discount_curve,
) -> float:
    """PV of bond coupons falling between settlement and delivery.

    Parameters
    ----------
    bond : Bond
    settlement_date : Date
    delivery_date : Date
    income_discount_curve : YieldTermStructure

    Returns
    -------
    float : present value of intermediate coupon income
    """
    from ql_jax.cashflows.analytics import _cf_amount, _cf_date

    income = 0.0
    for cf in bond.cashflows:
        d = _cf_date(cf)
        if d <= settlement_date:
            continue
        if d > delivery_date:
            continue
        amount = _cf_amount(cf)
        t = income_discount_curve.time_from_reference(d)
        df = income_discount_curve.discount(t)
        income = income + amount * df
    return income


def bond_forward_dirty_price(
    bond,
    delivery_date,
    discount_curve,
    income_discount_curve=None,
    bond_curve=None,
    settlement_date=None,
) -> float:
    """Dirty forward price of a bond for delivery at a future date.

    P_DirtyFwd = (P_DirtySpot - SpotIncome) / DF(delivery)

    Parameters
    ----------
    bond : Bond
    delivery_date : Date
    discount_curve : YieldTermStructure  (repo curve — used for delivery DF)
    income_discount_curve : YieldTermStructure or None
        Curve for discounting intermediate coupons. Defaults to discount_curve.
    bond_curve : YieldTermStructure or None
        Curve for pricing the underlying bond spot. Defaults to discount_curve.
    settlement_date : Date or None

    Returns
    -------
    float : dirty forward price (dollar amount, not percentage)
    """
    settle = settlement_date or discount_curve.reference_date
    inc_curve = income_discount_curve or discount_curve
    b_curve = bond_curve or discount_curve

    dirty_spot = discounting_bond_npv(bond, b_curve, settle)
    spot_income = bond_forward_spot_income(bond, settle, delivery_date, inc_curve)

    t_delivery = discount_curve.time_from_reference(delivery_date)
    df_delivery = discount_curve.discount(t_delivery)

    return (dirty_spot - spot_income) / df_delivery


def bond_forward_clean_price(
    bond,
    delivery_date,
    discount_curve,
    income_discount_curve=None,
    bond_curve=None,
    settlement_date=None,
) -> float:
    """Clean forward price = dirty forward price - accrued at delivery.

    Parameters
    ----------
    bond : Bond
    delivery_date : Date
    discount_curve : YieldTermStructure (repo curve)
    income_discount_curve : YieldTermStructure or None
    bond_curve : YieldTermStructure or None
    settlement_date : Date or None

    Returns
    -------
    float : clean forward price (dollar amount)
    """
    dirty_fwd = bond_forward_dirty_price(
        bond, delivery_date, discount_curve, income_discount_curve,
        bond_curve, settlement_date,
    )
    ai_at_delivery = cf_accrued(bond.cashflows, delivery_date)
    return dirty_fwd - ai_at_delivery


def bond_forward_npv(
    bond,
    delivery_date,
    strike,
    discount_curve,
    income_discount_curve=None,
    bond_curve=None,
    settlement_date=None,
    position: int = 1,
) -> float:
    """NPV of a bond forward contract.

    NPV = position * (DirtyForwardPrice - Strike) * DF(delivery)

    Parameters
    ----------
    bond : Bond
    delivery_date : Date
    strike : float  (dirty forward price agreed at inception)
    discount_curve : YieldTermStructure
    income_discount_curve : YieldTermStructure or None
    bond_curve : YieldTermStructure or None
    settlement_date : Date or None
    position : +1 for long, -1 for short

    Returns
    -------
    float : NPV of the forward contract
    """
    dirty_fwd = bond_forward_dirty_price(
        bond, delivery_date, discount_curve, income_discount_curve,
        bond_curve, settlement_date,
    )
    t_delivery = discount_curve.time_from_reference(delivery_date)
    df_delivery = discount_curve.discount(t_delivery)

    return position * (dirty_fwd - strike) * df_delivery
