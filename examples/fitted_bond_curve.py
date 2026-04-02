"""Example: Yield Curve Fitting.

Fits discount curves using:
  - Nelson-Siegel parametric form
  - Svensson (extended Nelson-Siegel)
  - Exponential splines
  - Zero curve from market rates
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.time.date import Date
from ql_jax.termstructures.yield_.fitted_bond_curve import (
    NelsonSiegel, Svensson,
)
from ql_jax.termstructures.yield_.zero_curve import ZeroCurve


def main():
    print("=" * 60)
    print("QL-JAX Example: Yield Curve Fitting")
    print("=" * 60)
    print()

    # ── Market Data ──────────────────────────────────────────
    tenors = jnp.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    zero_rates = jnp.array([0.045, 0.046, 0.047, 0.048, 0.048, 0.049, 0.049, 0.050, 0.051, 0.051])

    print("  Market zero rates:")
    for t, zr in zip(tenors, zero_rates):
        print(f"    {float(t):5.2f}y: {float(zr):.3%}")
    print()

    # ── Nelson-Siegel Fit ────────────────────────────────────
    # NS: z(t) = β0 + (β1 + β2)*(1-exp(-t/τ))/(t/τ) - β2*exp(-t/τ)
    ns = NelsonSiegel(params=[0.051, -0.006, -0.010, 2.0])
    ns_rates = ns.zero_rate(tenors)
    print("  Nelson-Siegel fit (β₀=0.051, β₁=-0.006, β₂=-0.010, τ=2.0):")
    for t, mkt, ns_r in zip(tenors, zero_rates, ns_rates):
        err = float(ns_r) - float(mkt)
        print(f"    {float(t):5.2f}y: NS={float(ns_r):.4%}  Mkt={float(mkt):.4%}  err={err:+.4%}")
    print()

    # ── Svensson Fit ─────────────────────────────────────────
    sv = Svensson(params=[0.051, -0.006, -0.010, 0.005, 2.0, 5.0])
    sv_rates = sv.zero_rate(tenors)
    print("  Svensson fit:")
    for t, sv_r in zip(tenors, sv_rates):
        print(f"    {float(t):5.2f}y: {float(sv_r):.4%}")
    print()

    # ── ZeroCurve (piecewise linear) ─────────────────────────
    ref_date = Date(15, 1, 2024)
    dates = [Date(15, int(m), int(y))
             for m, y in [(4, 2024), (7, 2024), (1, 2025), (1, 2026), (1, 2027),
                          (1, 2029), (1, 2031), (1, 2034), (1, 2044), (1, 2054)]]
    curve = ZeroCurve(ref_date, dates, list(zero_rates))

    print("  ZeroCurve (piecewise linear interpolation):")
    test_tenors = [0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 25.0]
    for t in test_tenors:
        disc = float(curve.discount(t))
        zr = -jnp.log(disc) / t
        print(f"    {t:5.1f}y: discount={disc:.6f}  zero={float(zr):.4%}")

    # ── AD: Rate sensitivities ───────────────────────────────
    print("\n  AD: Sensitivity of 5Y discount factor to each input rate:")

    def disc_5y(rates_arr):
        c = ZeroCurve(ref_date, dates, list(rates_arr))
        return c.discount(5.0)

    jac = jax.jacobian(disc_5y)(zero_rates)
    for i, t in enumerate(tenors):
        print(f"    d(DF_5Y)/d(r_{float(t):.1f}y) = {float(jac[i]):.6f}")


if __name__ == "__main__":
    main()
