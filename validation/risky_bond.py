"""Validation: Risky Bond pricing
Source: sprint 3 item 3.10

Risky bond NPV with discount + survival curves, scenario risk via vmap.
Uses direct formula since the engine requires full Bond objects.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def risky_bond_npv_simple(coupon_times, coupon_amounts, face, maturity,
                           discount_fn, survival_fn, recovery):
    """Simple risky bond NPV:
    NPV = Σ cf_i * DF(t_i) * Q(t_i)
        + R * face * Σ DF(t_i) * (Q(t_{i-1}) - Q(t_i))
        + face * DF(T) * Q(T)
    """
    n = coupon_times.shape[0]
    df = jnp.array([discount_fn(t) for t in coupon_times])
    q = jnp.array([survival_fn(t) for t in coupon_times])

    # Shift for Q(t_{i-1})
    q_prev = jnp.concatenate([jnp.ones(1), q[:-1]])

    # Coupon payments × survival
    coupon_pv = jnp.sum(coupon_amounts * df * q)

    # Principal at maturity × survival
    principal_pv = face * discount_fn(maturity) * survival_fn(maturity)

    # Recovery on default
    recovery_pv = recovery * face * jnp.sum(df * (q_prev - q))

    return coupon_pv + principal_pv + recovery_pv


def main():
    print("=" * 78)
    print("Risky Bond Validation")
    print("=" * 78)

    # Bond: 5Y, 4% coupon, semi-annual
    face = 100.0
    coupon_rate = 0.04
    n_coupons = 10
    coupon_times = jnp.linspace(0.5, 5.0, n_coupons)
    coupon_amounts = jnp.full(n_coupons, face * coupon_rate * 0.5)
    maturity = 5.0

    # Flat curves
    r = 0.03      # discount rate
    hazard = 0.02 # hazard rate
    recovery = 0.40

    disc_fn = lambda t: jnp.exp(-r * t)
    surv_fn = lambda t: jnp.exp(-hazard * t)

    print(f"  Bond: {face} face, {coupon_rate*100:.0f}% coupon, {n_coupons} periods, 5Y")
    print(f"  r={r:.0%}, hazard={hazard:.4f}, recovery={recovery:.0%}")

    # === Test 1: Risky bond NPV ===
    npv = float(risky_bond_npv_simple(coupon_times, coupon_amounts, face, maturity,
                                       disc_fn, surv_fn, recovery))
    # Risk-free bond value
    rf_npv = float(sum(ca * jnp.exp(-r * t) for ca, t in zip(coupon_amounts, coupon_times))
                   + face * jnp.exp(-r * 5.0))
    print(f"\n  Risky bond NPV:    {npv:.4f}")
    print(f"  Risk-free bond:    {rf_npv:.4f}")
    print(f"  Credit adjustment: {rf_npv - npv:.4f}")

    # === Test 2: Recovery sensitivity ===
    recoveries = [0.0, 0.20, 0.40, 0.60, 0.80]
    npvs = []
    for rec in recoveries:
        v = float(risky_bond_npv_simple(coupon_times, coupon_amounts, face, maturity,
                                         disc_fn, surv_fn, rec))
        npvs.append(v)
        print(f"  Recovery={rec:.0%}: NPV={v:.4f}")
    # Higher recovery → higher NPV
    recovery_monotone = all(npvs[i] <= npvs[i+1] + 0.01 for i in range(len(npvs)-1))

    # === Test 3: AD Greeks ===
    def price_fn(params):
        rate, hz, rec = params
        df = lambda t: jnp.exp(-rate * t)
        sf = lambda t: jnp.exp(-hz * t)
        return risky_bond_npv_simple(coupon_times, coupon_amounts, face, maturity,
                                      df, sf, rec)

    base = jnp.array([r, hazard, recovery])
    greeks = np.array(jax.grad(price_fn)(base))
    names = ['rate', 'hazard', 'recovery']
    print(f"\n  --- AD Greeks ---")
    for name, g in zip(names, greeks):
        print(f"    d(NPV)/d({name:>8s}): {g:>10.4f}")

    # FD check
    H = 1e-6
    fd = np.zeros(3)
    for i in range(3):
        xp = base.at[i].add(H)
        xm = base.at[i].add(-H)
        fd[i] = (float(price_fn(xp)) - float(price_fn(xm))) / (2 * H)
    max_rel = np.max(np.abs(greeks - fd) / np.maximum(np.abs(fd), 1e-6))
    print(f"  AD vs FD max rel diff: {max_rel:.2e}")

    # === Test 4: 100-scenario batch ===
    vmap_grad = jax.jit(jax.vmap(jax.grad(price_fn)))
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    n_scenarios = 100
    scenarios = jnp.stack([
        0.01 + 0.04 * jax.random.uniform(k1, (n_scenarios,)),
        0.005 + 0.03 * jax.random.uniform(k2, (n_scenarios,)),
        0.20 + 0.40 * jax.random.uniform(k3, (n_scenarios,)),
    ], axis=1)

    _ = vmap_grad(scenarios)
    t0 = time.perf_counter()
    batch_greeks = vmap_grad(scenarios)
    batch_greeks.block_until_ready()
    t_batch = time.perf_counter() - t0
    print(f"\n  Batch {n_scenarios} scenarios: {t_batch*1000:.1f} ms")
    print(f"  Greeks shape: {batch_greeks.shape}")

    # === Validation ===
    n_pass = 0
    total = 4

    ok = npv < rf_npv and npv > 0
    n_pass += int(ok)
    print(f"\n  0 < risky < riskfree: {'✓' if ok else '✗'}")

    n_pass += int(recovery_monotone)
    print(f"  Recovery monotone:    {'✓' if recovery_monotone else '✗'}")

    ok = max_rel < 1e-4
    n_pass += int(ok)
    print(f"  AD vs FD:             {'✓' if ok else '✗'} ({max_rel:.2e})")

    ok = batch_greeks.shape == (100, 3) and bool(jnp.all(jnp.isfinite(batch_greeks)))
    n_pass += int(ok)
    print(f"  Batch OK:             {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All risky bond validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
