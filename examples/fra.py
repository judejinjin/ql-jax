"""Example: Forward Rate Agreement (FRA) Pricing.

Prices a FRA using discount factors from a flat yield curve.
"""

import jax.numpy as jnp

import jax
jax.config.update("jax_enable_x64", True)

from ql_jax.termstructures.yield_.flat_forward import FlatForward
from ql_jax.time.date import Date


def main():
    print("=" * 60)
    print("QL-JAX Example: Forward Rate Agreement")
    print("=" * 60)
    print()

    # Market data
    ref_date = Date(15, 1, 2024)  # Jan 15, 2024
    flat_rate = 0.05

    # FRA parameters
    notional = 1_000_000.0
    fixed_rate = 0.052
    start_T = 0.5   # 6M forward
    end_T = 1.0     # to 12M
    tau = end_T - start_T  # accrual period

    print(f"  Notional:    {notional/1e6:.0f}M")
    print(f"  Fixed rate:  {fixed_rate:.3%}")
    print(f"  Start:       {start_T}y, End: {end_T}y")
    print(f"  Flat curve:  {flat_rate:.2%}")
    print()

    curve = FlatForward(ref_date, flat_rate)

    # Forward rate implied by the curve
    df_start = float(curve.discount(start_T))
    df_end = float(curve.discount(end_T))
    fwd_rate = (df_start / df_end - 1.0) / tau

    print(f"  DF({start_T}y):   {df_start:.6f}")
    print(f"  DF({end_T}y):   {df_end:.6f}")
    print(f"  Forward rate:  {fwd_rate:.4%}")
    print()

    # FRA NPV (from buyer's perspective: pay fixed, receive floating)
    fra_npv = notional * tau * (fwd_rate - fixed_rate) * df_end
    print(f"  FRA NPV (pay fixed): {fra_npv:.2f}")
    if fra_npv > 0:
        print("  → Forward rate > fixed rate, FRA buyer benefits")
    else:
        print("  → Forward rate < fixed rate, FRA buyer loses")

    # Fair FRA rate (NPV = 0)
    fair_rate = fwd_rate
    print(f"  Fair FRA rate: {fair_rate:.4%}")


if __name__ == "__main__":
    main()
