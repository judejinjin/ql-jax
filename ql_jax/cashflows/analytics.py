"""Cash-flow analytics: NPV, BPS, yield, duration, convexity."""

from __future__ import annotations

import jax.numpy as jnp
import jax

from ql_jax.time.date import Date
from ql_jax.time.daycounter import year_fraction


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cf_amount(cf) -> float:
    """Extract amount from a cash flow (SimpleCashFlow or Coupon)."""
    if hasattr(cf, "amount"):
        a = cf.amount
        return a() if callable(a) else a
    return 0.0


def _cf_date(cf) -> Date:
    """Extract payment date from a cash flow."""
    if hasattr(cf, "payment_date"):
        return cf.payment_date
    return cf.date


# ---------------------------------------------------------------------------
# NPV
# ---------------------------------------------------------------------------

def npv(
    cashflows,
    discount_curve,
    include_settlement_cf: bool = True,
    settlement_date: Date | None = None,
    npv_date: Date | None = None,
) -> float:
    """Compute net present value of cash flows.

    Parameters
    ----------
    cashflows : list of cash-flow objects
        Each must have .date or .payment_date and .amount.
    discount_curve : YieldTermStructure
        Discount curve for present-valuing cash flows.
    include_settlement_cf : bool
        Whether to include cash flows on the settlement date.
    settlement_date : Date or None
        Cash flows on or before this date are excluded. Defaults to reference date.
    npv_date : Date or None
        Date to which NPV is discounted. Defaults to reference date.

    Returns
    -------
    float  (JAX scalar)
    """
    ref = discount_curve.reference_date
    settle = settlement_date or ref
    npv_d = npv_date or ref

    total = 0.0
    for cf in cashflows:
        d = _cf_date(cf)
        if d < settle:
            continue
        if d == settle and not include_settlement_cf:
            continue
        amount = _cf_amount(cf)
        t = discount_curve.time_from_reference(d)
        df = discount_curve.discount(t)
        total = total + amount * df

    # Adjust if npv_date != reference_date
    if npv_d != ref:
        t_npv = discount_curve.time_from_reference(npv_d)
        total = total / discount_curve.discount(t_npv)

    return total


def bps(
    cashflows,
    discount_curve,
    settlement_date: Date | None = None,
    npv_date: Date | None = None,
) -> float:
    """Basis-point sensitivity: NPV change for 1bp parallel shift.

    This equals the NPV of the cash flows where each coupon pays
    0.0001 * nominal * accrual_period instead of its actual amount.
    """
    ref = discount_curve.reference_date
    settle = settlement_date or ref
    npv_d = npv_date or ref

    total = 0.0
    for cf in cashflows:
        d = _cf_date(cf)
        if d <= settle:
            continue
        if hasattr(cf, "nominal") and hasattr(cf, "accrual_period"):
            # Coupon: BPS contribution = nominal * accrual_period * df * 1bp
            tau = cf.accrual_period(cf.day_counter) if hasattr(cf, "day_counter") else 0.0
            amount = cf.nominal * tau * 1e-4
        else:
            continue  # Non-coupon flows don't contribute to BPS
        t = discount_curve.time_from_reference(d)
        df = discount_curve.discount(t)
        total = total + amount * df

    if npv_d != ref:
        t_npv = discount_curve.time_from_reference(npv_d)
        total = total / discount_curve.discount(t_npv)

    return total


# ---------------------------------------------------------------------------
# Yield (internal rate of return)
# ---------------------------------------------------------------------------

def yield_rate(
    cashflows,
    target_npv: float,
    day_counter: str,
    reference_date: Date,
    guess: float = 0.05,
    max_iter: int = 100,
    accuracy: float = 1e-10,
) -> float:
    """Solve for the flat yield that reprices the cash flows to target_npv.

    Uses Newton's method with AD for the derivative.
    """

    def _npv_at_rate(y):
        total = jnp.float64(0.0)
        for cf in cashflows:
            d = _cf_date(cf)
            if d <= reference_date:
                continue
            amount = _cf_amount(cf)
            t = year_fraction(reference_date, d, day_counter)
            df = jnp.exp(-y * t)
            total = total + amount * df
        return total - target_npv

    _npv_grad = jax.grad(_npv_at_rate)

    y = jnp.float64(guess)
    for _ in range(max_iter):
        f = _npv_at_rate(y)
        if jnp.abs(f) < accuracy:
            break
        fp = _npv_grad(y)
        if jnp.abs(fp) < 1e-20:
            break
        y = y - f / fp
    return float(y)


# ---------------------------------------------------------------------------
# Duration & convexity
# ---------------------------------------------------------------------------

class Duration:
    Simple = 0
    Macaulay = 1
    Modified = 2


def duration(
    cashflows,
    yield_value: float,
    day_counter: str,
    reference_date: Date,
    duration_type: int = Duration.Modified,
) -> float:
    """Compute duration using AD.

    Parameters
    ----------
    duration_type : int
        Duration.Simple (dollar duration / NPV),
        Duration.Macaulay (time-weighted PV / NPV),
        Duration.Modified (Macaulay / (1 + y)).
    """
    def _npv_fn(y):
        total = jnp.float64(0.0)
        for cf in cashflows:
            d = _cf_date(cf)
            if d <= reference_date:
                continue
            amount = _cf_amount(cf)
            t = year_fraction(reference_date, d, day_counter)
            df = jnp.exp(-y * t)
            total = total + amount * df
        return total

    y = jnp.float64(yield_value)
    pv = _npv_fn(y)

    if duration_type == Duration.Simple:
        # Dollar duration = -dP/dy
        dpdy = jax.grad(_npv_fn)(y)
        return float(-dpdy / pv)
    elif duration_type == Duration.Macaulay:
        # Sum(t_i * cf_i * df_i) / PV
        mac = 0.0
        for cf in cashflows:
            d = _cf_date(cf)
            if d <= reference_date:
                continue
            amount = _cf_amount(cf)
            t = year_fraction(reference_date, d, day_counter)
            df = jnp.exp(-y * t)
            mac += t * amount * df
        return float(mac / pv)
    else:  # Modified
        dpdy = jax.grad(_npv_fn)(y)
        return float(-dpdy / pv)


def convexity(
    cashflows,
    yield_value: float,
    day_counter: str,
    reference_date: Date,
) -> float:
    """Compute convexity = (1/P) * d²P/dy² using AD."""
    def _npv_fn(y):
        total = jnp.float64(0.0)
        for cf in cashflows:
            d = _cf_date(cf)
            if d <= reference_date:
                continue
            amount = _cf_amount(cf)
            t = year_fraction(reference_date, d, day_counter)
            df = jnp.exp(-y * t)
            total = total + amount * df
        return total

    y = jnp.float64(yield_value)
    pv = _npv_fn(y)
    d2pdy2 = jax.grad(jax.grad(_npv_fn))(y)
    return float(d2pdy2 / pv)


# ---------------------------------------------------------------------------
# Accrued amount
# ---------------------------------------------------------------------------

def accrued_amount(
    cashflows,
    settlement_date: Date,
) -> float:
    """Compute accrued interest at settlement date.

    Looks for the coupon whose accrual period spans the settlement date.
    """
    for cf in cashflows:
        if not hasattr(cf, "accrual_start"):
            continue
        if cf.accrual_start <= settlement_date < cf.accrual_end:
            if hasattr(cf, "day_counter"):
                full_period = year_fraction(cf.accrual_start, cf.accrual_end, cf.day_counter)
                accrued_period = year_fraction(cf.accrual_start, settlement_date, cf.day_counter)
                if full_period > 0:
                    return cf.nominal * cf.rate * accrued_period
            break
    return 0.0
