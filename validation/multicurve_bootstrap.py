"""Validation: Multicurve Bootstrap
Source: ~/quantlib-risk-py/benchmarks/multicurve_bootstrap.py

Bootstrap a piecewise yield curve from deposit + swap rate helpers,
verify repricing accuracy and JAX AD through the bootstrap.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.time.date import Date, _advance_date, TimeUnit
from ql_jax.termstructures.yield_.rate_helpers import (
    DepositRateHelper, SwapRateHelper,
)
from ql_jax.termstructures.yield_.piecewise import PiecewiseYieldCurve


def _make_helpers(ref_date, dep_rates, swap_rates):
    """Build rate helpers from deposit and swap rates."""
    dep_tenors_months = [1, 3, 6]
    swap_tenor_months = [12, 24, 60, 120]

    helpers = []
    for m, r in zip(dep_tenors_months, dep_rates):
        pillar = _advance_date(ref_date, m, TimeUnit.Months)
        helpers.append(DepositRateHelper(quote=r, pillar_date=pillar,
                                          start_date=ref_date, end_date=pillar))
    for m, r in zip(swap_tenor_months, swap_rates):
        pillar = _advance_date(ref_date, m, TimeUnit.Months)
        helpers.append(SwapRateHelper(quote=r, pillar_date=pillar,
                                       start_date=ref_date, tenor_months=m))
    return helpers


def main():
    print("=" * 78)
    print("Multicurve Bootstrap (Deposit + Swap helpers)")
    print("=" * 78)

    ref_date = Date(15, 1, 2024)

    # Market data: 3 deposits + 4 swaps
    dep_rates = [0.0250, 0.0280, 0.0310]
    swap_rates = [0.0330, 0.0350, 0.0380, 0.0400]

    helpers = _make_helpers(ref_date, dep_rates, swap_rates)

    print(f"  Reference date: {ref_date}")
    print(f"  Helpers: {len(dep_rates)} deposits + {len(swap_rates)} swaps")

    # Bootstrap
    t0 = time.time()
    curve = PiecewiseYieldCurve(
        reference_date=ref_date,
        helpers=helpers,
        day_counter='Actual365Fixed',
    )
    t_boot = time.time() - t0

    print(f"\n  Bootstrap time: {t_boot:.3f}s")
    print(f"  Pillar times: {[f'{t:.4f}' for t in np.array(curve.times)]}")
    print(f"  Discount factors: {[f'{d:.6f}' for d in np.array(curve.discounts)]}")

    # === Test 1: Repricing accuracy ===
    print(f"\n  --- Repricing accuracy ---")
    max_reprice_err = 0.0
    for h in helpers:
        impl = h.implied_quote(curve)
        err = abs(float(impl) - h.quote)
        max_reprice_err = max(max_reprice_err, err)
        print(f"    {h.__class__.__name__:20s} quote={h.quote:.4%} "
              f"implied={float(impl):.4%} err={err:.2e}")

    # === Test 2: Monotone discount factors ===
    dfs = np.array(curve.discounts)
    monotone = all(dfs[i] >= dfs[i+1] for i in range(len(dfs)-1))

    # === Test 3: Zero rates positive ===
    times = np.array(curve.times)
    zr_positive = True
    for t in times[1:]:  # skip t=0
        zr = float(curve.zero_rate(t))
        if zr < 0:
            zr_positive = False

    # === Test 4: JAX AD on discount function ===
    # Differentiate df(t) w.r.t. t at the curve
    def df_at_t(t):
        return curve.discount(t)

    try:
        grad_val = float(jax.grad(df_at_t)(jnp.float64(5.0)))
        # FD check
        h = 1e-6
        fd_val = (float(df_at_t(jnp.float64(5.0 + h)))
                  - float(df_at_t(jnp.float64(5.0 - h)))) / (2 * h)
        rel = abs(grad_val - fd_val) / max(abs(fd_val), 1e-10)
        ad_ok = rel < 0.01
        print(f"\n  --- JAX AD on discount function ---")
        print(f"    ∂DF/∂t at t=5Y: AD={grad_val:.6f}  FD={fd_val:.6f}  "
              f"rel_diff={rel:.2e}")
    except Exception as e:
        print(f"\n  --- JAX AD on discount function ---")
        print(f"    SKIPPED: {e}")
        ad_ok = False

    # === Results ===
    n_pass = 0
    total = 4

    ok = max_reprice_err < 1e-8
    n_pass += int(ok)
    print(f"\n  Repricing accuracy:   {'✓' if ok else '✗'} (max err={max_reprice_err:.2e})")

    ok = monotone
    n_pass += int(ok)
    print(f"  Monotone discounts:   {'✓' if ok else '✗'}")

    ok = zr_positive
    n_pass += int(ok)
    print(f"  Positive zero rates:  {'✓' if ok else '✗'}")

    n_pass += int(ad_ok)
    print(f"  JAX AD consistency:   {'✓' if ad_ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All multicurve bootstrap validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
