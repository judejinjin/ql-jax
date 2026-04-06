"""Validation: CDS 100-Scenario Greek Benchmarks (FD vs jax.grad vs jax.vmap(jax.grad))
Source: ~/quantlib-risk-py/benchmarks/cds_benchmarks.py

Validates batch CDS Greek computation across 100 MC scenarios:
  - 6 inputs: 4 CDS spreads (3M/6M/1Y/2Y), recovery rate, risk-free rate
  - FD: bump-and-reprice each of 6 inputs per scenario
  - jax.grad: one reverse-mode pass per scenario
  - jax.vmap(jax.grad): vectorized across all 100 scenarios

Market data: eval=2007-05-15, base spreads=150bp, recovery=50%, risk-free=1%
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import statistics
import random

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.instruments.cds import make_cds
from ql_jax.engines.credit.midpoint import midpoint_cds_npv


# === Market data (matches quantlib-risk-py CDS benchmark) ===
N_SCENARIOS = 100
SCENARIO_SEED = 123
BPS = 1e-4

BASE_SPREADS = [0.0150, 0.0150, 0.0150, 0.0150]
BASE_RECOVERY = 0.5
BASE_RISKFREE = 0.01
N_INPUTS = 6  # 4 spreads + recovery + risk-free

CDS_MATURITY = 2.0  # 2Y CDS (benchmark target)
CDS_NOTIONAL = 1_000_000.0
CDS_FREQUENCY = 0.25  # quarterly


def gen_scenarios():
    """Pre-generate CDS MC scenarios deterministically (matches quantlib-risk-py)."""
    rng = random.Random(SCENARIO_SEED)
    scenarios = []
    for _ in range(N_SCENARIOS):
        spreads = [max(1e-5, s + rng.gauss(0, 2e-3)) for s in BASE_SPREADS]
        rec = min(0.99, max(0.01, BASE_RECOVERY + rng.gauss(0, 0.02)))
        rfr = max(0.0001, BASE_RISKFREE + rng.gauss(0, 5e-4))
        scenarios.append(spreads + [rec, rfr])
    return scenarios


def cds_npv_from_inputs(spread_2y, recovery, riskfree):
    """Pure JAX function: inputs → CDS NPV.

    Uses flat hazard rate approximation: h ≈ spread / (1 - recovery).
    """
    hazard_rate = spread_2y / (1.0 - recovery)
    survival_fn = lambda t: jnp.exp(-hazard_rate * t)
    discount_fn = lambda t: jnp.exp(-riskfree * t)

    cds = make_cds(
        notional=CDS_NOTIONAL,
        spread=spread_2y,
        maturity=CDS_MATURITY,
        recovery_rate=recovery,
        frequency=CDS_FREQUENCY,
    )
    return midpoint_cds_npv(cds, discount_fn, survival_fn)


def cds_npv_6inputs(spread_3m, spread_6m, spread_1y, spread_2y, recovery, riskfree):
    """Full 6-input CDS NPV (only spread_2y, recovery, riskfree affect 2Y CDS price)."""
    return cds_npv_from_inputs(spread_2y, recovery, riskfree)


def main():
    print("=" * 78)
    print("CDS 100-Scenario Greek Benchmarks (FD vs jax.grad vs jax.vmap)")
    print("=" * 78)
    print(f"  Instrument : 2Y CDS, protection seller, 1M notional")
    print(f"  Inputs     : 6 (4 spreads + recovery + risk-free)")
    print(f"  Scenarios  : {N_SCENARIOS}")
    print(f"  BPS shift  : {BPS}")
    print()

    scenarios = gen_scenarios()

    # === 1. Baseline: compute NPV for all scenarios ===
    base_npvs = []
    for s in scenarios:
        npv = float(cds_npv_6inputs(*[jnp.float64(x) for x in s]))
        base_npvs.append(npv)
    print(f"  Base NPVs: min={min(base_npvs):,.2f}  max={max(base_npvs):,.2f}  mean={np.mean(base_npvs):,.2f}")

    # === 2. FD Greeks for first scenario ===
    s0 = scenarios[0]
    args = [jnp.float64(x) for x in s0]
    base = float(cds_npv_6inputs(*args))

    fd_greeks = []
    for i in range(N_INPUTS):
        bumped = list(s0)
        bumped[i] += BPS
        npv_up = float(cds_npv_6inputs(*[jnp.float64(x) for x in bumped]))
        fd_greeks.append((npv_up - base) / BPS)

    # === 3. jax.grad for first scenario ===
    grad_fn = jax.grad(cds_npv_6inputs, argnums=tuple(range(N_INPUTS)))
    ad_greeks = [float(g) for g in grad_fn(*args)]

    input_names = ["spread_3m", "spread_6m", "spread_1y", "spread_2y", "recovery", "riskfree"]
    print(f"\n  --- Greeks for scenario 0 ---")
    print(f"  {'Input':<14} {'FD':>14} {'jax.grad':>14} {'|diff|':>12}")
    print("  " + "-" * 56)

    n_pass = 0
    for i, name in enumerate(input_names):
        diff = abs(fd_greeks[i] - ad_greeks[i])
        # FD has its own error; for zero Greeks (3m/6m/1y don't affect 2Y price), both should be ~0
        tol = max(abs(fd_greeks[i]) * 0.05, 1.0) if abs(fd_greeks[i]) > 1.0 else 1.0
        ok = diff < tol
        n_pass += int(ok)
        print(f"  {name:<14} {fd_greeks[i]:>14.4f} {ad_greeks[i]:>14.4f} {diff:>12.2e} {'✓' if ok else '✗'}")

    # === 4. jax.vmap(jax.grad) across all scenarios ===
    print(f"\n  --- Batch computation: jax.vmap(jax.grad) ---")

    # Vectorize: stack scenarios into arrays
    scenario_arr = jnp.array(scenarios)  # (100, 6)

    def cds_npv_vec(inputs):
        """Single-scenario NPV from input vector."""
        return cds_npv_6inputs(inputs[0], inputs[1], inputs[2],
                                inputs[3], inputs[4], inputs[5])

    # vmap over grad
    vmap_grad = jax.vmap(jax.grad(cds_npv_vec))
    batch_greeks = vmap_grad(scenario_arr)  # (100, 6)
    batch_greeks_np = np.array(batch_greeks)

    print(f"  Shape: {batch_greeks_np.shape}")
    print(f"  Non-zero Greeks per scenario: spread_2y, recovery, riskfree")
    print(f"    d/d(spread_2y): mean={np.mean(batch_greeks_np[:, 3]):,.2f}  std={np.std(batch_greeks_np[:, 3]):,.2f}")
    print(f"    d/d(recovery):  mean={np.mean(batch_greeks_np[:, 4]):,.2f}  std={np.std(batch_greeks_np[:, 4]):,.2f}")
    print(f"    d/d(riskfree):  mean={np.mean(batch_greeks_np[:, 5]):,.2f}  std={np.std(batch_greeks_np[:, 5]):,.2f}")

    # Verify vmap matches single-scenario grad
    single_greeks = [float(g) for g in jax.grad(cds_npv_vec)(scenario_arr[0])]
    vmap_first = [float(batch_greeks[0, i]) for i in range(6)]
    max_vmap_diff = max(abs(single_greeks[i] - vmap_first[i]) for i in range(6))
    vmap_ok = max_vmap_diff < 1e-10
    n_pass += int(vmap_ok)
    print(f"\n  vmap vs single grad consistency: max |diff| = {max_vmap_diff:.2e} {'✓' if vmap_ok else '✗'}")

    # === 5. JIT'ed vmap(grad) ===
    print(f"\n  --- JIT compilation ---")
    jit_vmap_grad = jax.jit(vmap_grad)
    _ = jit_vmap_grad(scenario_arr)  # warmup / compile
    jit_batch = jit_vmap_grad(scenario_arr)
    jit_diff = float(jnp.max(jnp.abs(batch_greeks - jit_batch)))
    jit_ok = jit_diff < 1e-6
    n_pass += int(jit_ok)
    print(f"  jit(vmap(grad)) consistency: max |diff| = {jit_diff:.2e} {'✓' if jit_ok else '✗'}")

    # === 6. Timing comparison ===
    print(f"\n  --- Timing ({N_SCENARIOS} scenarios) ---")
    n_reps = 20

    # FD timing (N+1 pricings per scenario × 100 scenarios)
    def fd_batch():
        for s in scenarios:
            args = [jnp.float64(x) for x in s]
            float(cds_npv_6inputs(*args))
            for i in range(N_INPUTS):
                bumped = list(s)
                bumped[i] += BPS
                float(cds_npv_6inputs(*[jnp.float64(x) for x in bumped]))

    fd_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fd_batch()
        fd_times.append((time.perf_counter() - t0) * 1000)

    # jax.vmap(jax.grad) timing
    vmap_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        vmap_grad(scenario_arr).block_until_ready()
        vmap_times.append((time.perf_counter() - t0) * 1000)

    # jit(vmap(grad)) timing
    jit_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        jit_vmap_grad(scenario_arr).block_until_ready()
        jit_times.append((time.perf_counter() - t0) * 1000)

    fd_med = statistics.median(fd_times)
    vmap_med = statistics.median(vmap_times)
    jit_med = statistics.median(jit_times)
    print(f"    FD  (700 pricings)            : {fd_med:>10.2f} ms")
    print(f"    vmap(grad) (100 scenarios)    : {vmap_med:>10.2f} ms  ({fd_med/vmap_med:5.1f}x vs FD)")
    print(f"    jit(vmap(grad))               : {jit_med:>10.2f} ms  ({fd_med/jit_med:5.1f}x vs FD)")

    total = N_INPUTS + 2  # 6 per-input checks + vmap consistency + jit consistency
    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All CDS scenario benchmark assertions passed.")
    else:
        print(f"✗ {total - n_pass} failures.")


if __name__ == "__main__":
    main()
