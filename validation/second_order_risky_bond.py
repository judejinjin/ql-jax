"""Validation: Second-order (Hessian) Risky Bond
Source: sprint 3 item 3.19

3×3 Hessian of risky bond NPV, symmetry, JIT speedup.
Uses 3 core params (rate, hazard, recovery).
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def risky_bond_npv_simple(coupon_times, coupon_amounts, face, maturity,
                           discount_fn, survival_fn, recovery):
    """Simple risky bond NPV."""
    n = coupon_times.shape[0]
    df = jnp.array([discount_fn(t) for t in coupon_times])
    q = jnp.array([survival_fn(t) for t in coupon_times])
    q_prev = jnp.concatenate([jnp.ones(1), q[:-1]])
    coupon_pv = jnp.sum(coupon_amounts * df * q)
    principal_pv = face * discount_fn(maturity) * survival_fn(maturity)
    recovery_pv = recovery * face * jnp.sum(df * (q_prev - q))
    return coupon_pv + principal_pv + recovery_pv


def main():
    print("=" * 78)
    print("Second-Order (Hessian) Risky Bond")
    print("=" * 78)

    face = 100.0
    coupon_rate = 0.04
    n_coupons = 10
    coupon_times = jnp.linspace(0.5, 5.0, n_coupons)
    coupon_amounts = jnp.full(n_coupons, face * coupon_rate * 0.5)
    maturity = 5.0

    bond = None  # unused

    # Extended parameter vector: [r, hazard, recovery]
    def price_fn(params):
        rate, hz, rec = params
        df = lambda t: jnp.exp(-rate * t)
        sf = lambda t: jnp.exp(-hz * t)
        return risky_bond_npv_simple(coupon_times, coupon_amounts, face, maturity,
                                      df, sf, rec)

    base = jnp.array([0.03, 0.02, 0.40])
    n_params = 3

    # === Test 1: Hessian computation ===
    hessian_fn = jax.jit(jax.hessian(price_fn))
    _ = hessian_fn(base)  # warmup

    t0 = time.perf_counter()
    H = hessian_fn(base)
    H.block_until_ready()
    t_hessian = time.perf_counter() - t0
    H_np = np.array(H)
    print(f"\n  Hessian shape: {H_np.shape}")
    print(f"  Hessian time:  {t_hessian*1000:.1f} ms")
    print(f"\n  Hessian matrix:")
    for i in range(n_params):
        row = " ".join(f"{H_np[i,j]:>12.4f}" for j in range(n_params))
        print(f"    [{row}]")

    # === Test 2: Symmetry ===
    sym_err = np.max(np.abs(H_np - H_np.T))
    print(f"\n  Symmetry error: {sym_err:.2e}")

    # === Test 3: FD Hessian check ===
    eps = 1e-4
    H_fd = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            xpp = base.at[i].add(eps).at[j].add(eps)
            xpm = base.at[i].add(eps).at[j].add(-eps)
            xmp = base.at[i].add(-eps).at[j].add(eps)
            xmm = base.at[i].add(-eps).at[j].add(-eps)
            H_fd[i, j] = (float(price_fn(xpp)) - float(price_fn(xpm))
                          - float(price_fn(xmp)) + float(price_fn(xmm))) / (4 * eps * eps)
    max_rel = np.max(np.abs(H_np - H_fd) / np.maximum(np.abs(H_fd), 1e-6))
    print(f"  Hessian AD vs FD max rel diff: {max_rel:.2e}")

    # === Test 4: JIT speedup ===
    t0 = time.perf_counter()
    for _ in range(100):
        h = hessian_fn(base)
    h.block_until_ready()
    t_jit = (time.perf_counter() - t0) / 100

    t0 = time.perf_counter()
    h = jax.hessian(price_fn)(base)
    t_raw = time.perf_counter() - t0
    speedup = t_raw / max(t_jit, 1e-12)
    print(f"\n  Raw time:  {t_raw*1000:.1f} ms")
    print(f"  JIT time:  {t_jit*1000:.3f} ms")
    print(f"  Speedup:   {speedup:.0f}x")

    # === Validation ===
    n_pass = 0
    total = 4

    ok = H_np.shape == (3, 3) and bool(jnp.all(jnp.isfinite(H)))
    n_pass += int(ok)
    print(f"\n  Hessian computed:   {'✓' if ok else '✗'}")

    ok = sym_err < 1e-8
    n_pass += int(ok)
    print(f"  Symmetry:           {'✓' if ok else '✗'} ({sym_err:.2e})")

    ok = max_rel < 1e-3
    n_pass += int(ok)
    print(f"  AD vs FD Hessian:   {'✓' if ok else '✗'} ({max_rel:.2e})")

    ok = speedup > 5
    n_pass += int(ok)
    print(f"  JIT speedup > 5x:  {'✓' if ok else '✗'} ({speedup:.0f}x)")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All second-order risky bond validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
