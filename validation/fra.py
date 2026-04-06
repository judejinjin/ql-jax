"""Validation: Forward Rate Agreement (FRA)
Source: ~/QuantLib/Examples/FRA/FRA.cpp

Validates FRA pricing on flat yield curves:
  - Forward rate computation
  - NPV at market rate (≈0)
  - NPV at off-market rate
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.time.date import Date
from ql_jax.instruments.fra import ForwardRateAgreement, FRAType
from ql_jax.termstructures.yield_.flat_forward import FlatForward


# === Reference values from QuantLib 1.42 ===
# eval = Jan 15 2024, 5% flat curve
# FRA: Jul 15 2024 → Jan 15 2025 (6M FRA starting in 6M)
# Note: FRA uses Actual/360 for rate/tau, curve uses Actual/365 for discounting
REFERENCE = {
    "fra_fwd_rate":       0.0499418283,   # forward rate in Actual/360 convention
    "fra_npv_strike5":   -28.2782648980,  # very small since fwd ≈ 5% in AC360
    "fra_npv_strike4":  4832.8950561172,
}


def main():
    today = Date(15, 1, 2024)
    curve = FlatForward(today, 0.05)

    fra_start = Date(15, 7, 2024)
    fra_end = Date(15, 1, 2025)

    # FRA at strike = 5%
    fra_5 = ForwardRateAgreement(
        value_date=fra_start,
        maturity_date=fra_end,
        strike=0.05,
        notional=1_000_000.0,
        type_=FRAType.Long,
        day_counter="Actual/360",
    )

    fwd_rate = float(fra_5.forward_rate(curve))
    npv_5 = float(fra_5.npv(curve))

    # FRA at strike = 4%
    fra_4 = ForwardRateAgreement(
        value_date=fra_start,
        maturity_date=fra_end,
        strike=0.04,
        notional=1_000_000.0,
        type_=FRAType.Long,
        day_counter="Actual/360",
    )
    npv_4 = float(fra_4.npv(curve))

    results = {
        "fra_fwd_rate":    fwd_rate,
        "fra_npv_strike5": npv_5,
        "fra_npv_strike4": npv_4,
    }

    # Print
    print("=" * 78)
    print("FRA Validation (5% flat curve, 6x12 FRA)")
    print("=" * 78)
    print(f"\n{'Metric':<25} {'QuantLib':>15} {'ql-jax':>15} {'Diff':>12}")
    print("-" * 67)

    n_pass = 0
    for key in REFERENCE:
        ref = REFERENCE[key]
        val = results[key]
        diff = abs(val - ref)
        tol = max(abs(ref) * 1e-4, 0.01)
        status = "✓" if diff < tol else "✗"
        if diff < tol:
            n_pass += 1
        fmt = f"{val:>15.10f}" if "rate" in key else f"{val:>15.6f}"
        ref_fmt = f"{ref:>15.10f}" if "rate" in key else f"{ref:>15.6f}"
        print(f"{key:<25} {ref_fmt} {fmt} {diff:>12.2e} {status}")

    # AD: dNPV/d(rate) — use direct computation to avoid float() inside FRA.npv
    print("\n--- FRA DV01 via JAX AD ---")

    def fra_price_fn(rate):
        """Compute FRA NPV as pure JAX function."""
        c = FlatForward(today, rate)
        t1 = c.time_from_reference(fra_start)
        t2 = c.time_from_reference(fra_end)
        df1 = c.discount(t1)
        df2 = c.discount(t2)
        from ql_jax.time.daycounter import year_fraction
        tau = year_fraction(fra_start, fra_end, "Actual/360")
        fwd = (df1 / df2 - 1.0) / tau
        return 1_000_000.0 * tau * (fwd - 0.05) * df2

    dv01 = float(jax.grad(fra_price_fn)(jnp.float64(0.05)))
    print(f"  DV01 = {dv01:,.2f}")
    # Long FRA benefits from rising rates
    assert dv01 > 0, "Long FRA DV01 should be positive"
    print(f"  ✓ Long FRA DV01 > 0")

    print(f"\nPassed: {n_pass}/{len(REFERENCE)}")

    # Assertions
    np.testing.assert_allclose(fwd_rate, REFERENCE["fra_fwd_rate"],
                                rtol=1e-6, err_msg="Forward rate mismatch")
    np.testing.assert_allclose(npv_5, REFERENCE["fra_npv_strike5"],
                                rtol=1e-3, err_msg="FRA NPV at 5% mismatch")
    np.testing.assert_allclose(npv_4, REFERENCE["fra_npv_strike4"],
                                rtol=1e-3, err_msg="FRA NPV at 4% mismatch")
    print("✓ All FRA assertions passed.")


if __name__ == "__main__":
    main()
