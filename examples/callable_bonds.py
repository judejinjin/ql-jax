"""Example: Callable Bond Pricing with Hull-White.

Prices a callable bond using:
  - Direct discounting for the bullet bond
  - Hull-White short rate model for the embedded option
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.models.shortrate.hull_white import hull_white_bond_price


def main():
    print("=" * 60)
    print("QL-JAX Example: Callable Bond Pricing")
    print("=" * 60)
    print()

    # ── Bond parameters ──────────────────────────────────────
    face = 100.0
    coupon_rate = 0.05
    maturity = 10.0

    # Hull-White parameters
    a = 0.1       # mean reversion
    sigma = 0.01  # short-rate vol
    r0 = 0.04     # initial short rate

    discount_fn = lambda t: jnp.exp(-r0 * t)

    print(f"  Face:        {face}")
    print(f"  Coupon:      {coupon_rate:.2%} annual")
    print(f"  Maturity:    {maturity:.0f}y")
    print(f"  HW params:   a={a}, σ={sigma}")
    print(f"  Short rate:  {r0:.2%}")
    print()

    # ── Non-callable bond (PV of cashflows) ──────────────────
    n_coupons = int(maturity * 2)  # semi-annual
    payment_times = jnp.arange(0.5, maturity + 0.01, 0.5)
    cashflows_arr = jnp.full(n_coupons, face * coupon_rate / 2)
    cashflows_arr = cashflows_arr.at[-1].add(face)

    dfs = jax.vmap(discount_fn)(payment_times)
    bullet_npv = float(jnp.sum(cashflows_arr * dfs))
    print(f"  Non-callable bond NPV: {bullet_npv:.4f}")

    # ── HW bond prices across maturities ─────────────────────
    print("\n  Hull-White zero-coupon bond prices:")
    test_mats = jnp.array([1.0, 2.0, 5.0, 10.0])
    for T in test_mats:
        P = hull_white_bond_price(r=r0, a=a, sigma=sigma, t=0.0, T=float(T),
                                  discount_curve_fn=discount_fn)
        zr = -jnp.log(P) / T
        print(f"    P(0,{float(T):4.1f}y) = {float(P):.6f}  (zero rate: {float(zr):.4%})")

    # ── AD: Sensitivity of 10Y ZCB to HW parameters ─────────
    print("\n  AD Sensitivities of 10Y ZCB price:")

    def zcb_10y(a_val, sigma_val, r_val):
        df = lambda t: jnp.exp(-r_val * t)
        return hull_white_bond_price(r=r_val, a=a_val, sigma=sigma_val, t=0.0, T=10.0,
                                     discount_curve_fn=df)

    d_a = float(jax.grad(zcb_10y, argnums=0)(a, sigma, r0))
    d_sigma = float(jax.grad(zcb_10y, argnums=1)(a, sigma, r0))
    d_r = float(jax.grad(zcb_10y, argnums=2)(a, sigma, r0))

    print(f"    d(P)/d(a)     = {d_a:.6f}")
    print(f"    d(P)/d(sigma) = {d_sigma:.6f}")
    print(f"    d(P)/d(r)     = {d_r:.6f}  (duration ≈ {-d_r / float(zcb_10y(a, sigma, r0)):.2f})")


if __name__ == "__main__":
    main()
