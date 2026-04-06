"""Validation: ISDA CDS Benchmarks
Source: sprint 3 item 3.8

100-scenario, 20-input CDS Greeks batch via vmap + JIT.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.engines.credit.isda import isda_cds_npv


def main():
    print("=" * 78)
    print("ISDA CDS Benchmarks (100-scenario batch)")
    print("=" * 78)

    notional = 10_000_000.0
    n_periods = 20
    payment_dates = jnp.linspace(0.25, 5.0, n_periods)
    day_fractions = jnp.full(n_periods, 0.25)

    def cds_price(params):
        """CDS NPV as function of (spread, recovery, hazard, rate)."""
        spread, recovery, hazard, rate = params
        disc_fn = lambda t: jnp.exp(-rate * t)
        surv_fn = lambda t: jnp.exp(-hazard * t)
        return isda_cds_npv(
            spread, recovery, payment_dates, day_fractions,
            disc_fn, surv_fn, notional, is_buyer=True, accrual_on_default=True)

    grad_fn = jax.jit(jax.grad(cds_price))

    # === Test 1: Single Greek computation ===
    base = jnp.array([0.01, 0.40, 0.0167, 0.03])
    _ = grad_fn(base)  # warmup

    t0 = time.perf_counter()
    greeks = grad_fn(base)
    greeks.block_until_ready()
    t_single = time.perf_counter() - t0
    greeks_np = np.array(greeks)
    names = ['spread', 'recovery', 'hazard', 'rate']
    print(f"\n  Single Greek time: {t_single*1000:.2f} ms")
    for name, g in zip(names, greeks_np):
        print(f"    d(NPV)/d({name:>8s}): {g:>14,.2f}")

    # === Test 2: vmap batch of 100 scenarios ===
    key = jax.random.PRNGKey(42)
    n_scenarios = 100
    spreads = 0.005 + 0.01 * jax.random.uniform(key, (n_scenarios,))
    recs = 0.30 + 0.20 * jax.random.uniform(jax.random.PRNGKey(1), (n_scenarios,))
    hazards = 0.01 + 0.02 * jax.random.uniform(jax.random.PRNGKey(2), (n_scenarios,))
    rates = 0.02 + 0.03 * jax.random.uniform(jax.random.PRNGKey(3), (n_scenarios,))

    scenarios = jnp.stack([spreads, recs, hazards, rates], axis=1)

    vmap_grad = jax.jit(jax.vmap(jax.grad(cds_price)))
    _ = vmap_grad(scenarios)  # warmup

    t0 = time.perf_counter()
    batch_greeks = vmap_grad(scenarios)
    batch_greeks.block_until_ready()
    t_batch = time.perf_counter() - t0
    print(f"\n  Batch {n_scenarios} scenarios: {t_batch*1000:.1f} ms ({t_batch/n_scenarios*1000:.3f} ms/scenario)")
    print(f"  Greeks shape: {batch_greeks.shape}")

    # === Test 3: vmap prices ===
    vmap_price = jax.jit(jax.vmap(cds_price))
    _ = vmap_price(scenarios)  # warmup

    t0 = time.perf_counter()
    batch_prices = vmap_price(scenarios)
    batch_prices.block_until_ready()
    t_prices = time.perf_counter() - t0
    print(f"  Batch prices: {t_prices*1000:.1f} ms")
    print(f"  Price range: [{float(jnp.min(batch_prices)):,.0f}, {float(jnp.max(batch_prices)):,.0f}]")

    # === Test 4: Consistency check — single vs batch ===
    single_g = np.array(grad_fn(scenarios[0]))
    batch_g0 = np.array(batch_greeks[0])
    max_diff = np.max(np.abs(single_g - batch_g0) / np.maximum(np.abs(single_g), 1.0))
    print(f"\n  Single vs batch[0] max rel diff: {max_diff:.2e}")

    # === Validation ===
    n_pass = 0
    total = 4

    ok = batch_greeks.shape == (100, 4)
    n_pass += int(ok)
    print(f"\n  Batch shape correct:  {'✓' if ok else '✗'}")

    ok = t_batch < 5.0  # should finish in <5s
    n_pass += int(ok)
    print(f"  Batch time < 5s:      {'✓' if ok else '✗'} ({t_batch:.2f}s)")

    ok = max_diff < 1e-8
    n_pass += int(ok)
    print(f"  Single=Batch:         {'✓' if ok else '✗'} ({max_diff:.2e})")

    ok = bool(jnp.all(jnp.isfinite(batch_greeks)))
    n_pass += int(ok)
    print(f"  All finite Greeks:    {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All ISDA CDS benchmarks passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
