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
