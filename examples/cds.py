"""Example 3: CDS Pricing — credit default swap.

Demonstrates:
  - CDS pricing with midpoint engine
  - Fair spread calculation
  - AD sensitivity to hazard rate
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.instruments.cds import make_cds
from ql_jax.engines.credit.midpoint import (
    midpoint_cds_npv,
    cds_fair_spread,
    hazard_rate_from_spread,
    survival_probability,
)


def main():
    print("=" * 60)
    print("QL-JAX Example: Credit Default Swap")
    print("=" * 60)

    # ── Market data ──────────────────────────────────────────
    notional = 10_000_000  # 10M
    spread = 0.01          # 100bp running spread
    maturity = 5.0
    recovery = 0.40
    risk_free = 0.03
    hazard = 0.02          # 200bp hazard rate (~120bp fair spread)

    discount_fn = lambda t: jnp.exp(-risk_free * t)
    survival_fn = lambda t: survival_probability(hazard, t)

    print(f"  Notional:      {notional:,.0f}")
    print(f"  Spread:        {spread:.0%} ({spread*10000:.0f}bp)")
    print(f"  Maturity:      {maturity}y")
    print(f"  Recovery:      {recovery:.0%}")
    print(f"  Risk-free:     {risk_free:.2%}")
    print(f"  Hazard rate:   {hazard:.2%}")
    print()

    # ── CDS pricing ──────────────────────────────────────────
    cds = make_cds(notional, spread, maturity, recovery)

    npv = float(midpoint_cds_npv(cds, discount_fn, survival_fn))
    fair = float(cds_fair_spread(cds, discount_fn, survival_fn))

    print(f"  NPV (buyer):   {npv:,.2f}")
    print(f"  Fair spread:   {fair:.4%} ({fair*10000:.1f}bp)")
    print(f"  Approx spread: {hazard * (1 - recovery):.4%} ({hazard*(1-recovery)*10000:.1f}bp)")
    print()

    # ── Sensitivity via AD ───────────────────────────────────
    print("-" * 60)
    print("  AD sensitivities (jax.grad)")
    print("-" * 60)

    def npv_fn(h):
        return midpoint_cds_npv(cds, discount_fn, lambda t: jnp.exp(-h * t))

    def npv_fn_r(r):
        return midpoint_cds_npv(cds, lambda t: jnp.exp(-r * t), survival_fn)

    dndh = float(jax.grad(npv_fn)(jnp.float64(hazard)))
    dndr = float(jax.grad(npv_fn_r)(jnp.float64(risk_free)))

    print(f"  dNPV/d(hazard):    {dndh:,.2f}")
    print(f"  dNPV/d(risk-free): {dndr:,.2f}")
    print()

    # ── Fair spread across maturities ────────────────────────
    print("-" * 60)
    print("  Fair spread term structure")
    print("-" * 60)

    for mat in [1, 2, 3, 5, 7, 10]:
        cds_m = make_cds(notional, spread, float(mat), recovery)
        fs = float(cds_fair_spread(cds_m, discount_fn, survival_fn))
        print(f"    {mat:2d}y: {fs:.4%} ({fs*10000:.1f}bp)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
