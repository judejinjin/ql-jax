"""Validation: Monte-Carlo IRS scenario risk
Source: ~/quantlib-risk-py/benchmarks/monte_carlo_irs.py

17-input IRS (9 disc rates + 8 fwd rates). Build swap NPV, differentiate
via jax.grad, and run 100 scenarios via vmap.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


FIXED_RATE = 0.03
NOTIONAL = 1_000_000.0
N_INPUTS = 17
DISC_TIMES = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
FLOAT_TIMES = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
FIXED_TIMES = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])


def irs_npv(inputs):
    """IRS NPV from 17 curve inputs."""
    disc_rates = inputs[:9]
    fwd_rates = inputs[9:17]

    def disc(t):
        return jnp.exp(-jnp.interp(t, DISC_TIMES, disc_rates) * t)

    # Fixed leg
    fixed_pv = jnp.float64(0.0)
    for i in range(5):
        t = FIXED_TIMES[i]
        fixed_pv = fixed_pv + FIXED_RATE * NOTIONAL * disc(t)

    # Float leg
    float_pv = jnp.float64(0.0)
    for i in range(8):
        t = FLOAT_TIMES[i]
        float_pv = float_pv + fwd_rates[i] * 0.5 * NOTIONAL * disc(t)
    for i in range(8, 10):
        t = FLOAT_TIMES[i]
        float_pv = float_pv + disc_rates[min(i, 8)] * 0.5 * NOTIONAL * disc(t)

    return float_pv - fixed_pv


def main():
    print("=" * 78)
    print("Monte-Carlo IRS: 17-input scenario risk")
    print("=" * 78)

    base = jnp.array([0.03] * N_INPUTS, dtype=jnp.float64)
    npv = float(irs_npv(base))
    print(f"  Base NPV: {npv:,.2f}")

    # === Test 1: JAX gradient ===
    grad_fn = jax.grad(irs_npv)
    grads = np.array(grad_fn(base))
    print(f"\n  --- JAX AD gradient (17 sensitivities) ---")
    labels = [f"disc_{i+1}" for i in range(9)] + [f"fwd_{i+1}" for i in range(8)]
    for i in range(N_INPUTS):
        print(f"    {labels[i]:>8s}: {grads[i]:>12.2f}")

    # FD check
    H = 1e-6
    fd_grads = np.zeros(N_INPUTS)
    for i in range(N_INPUTS):
        xp = base.at[i].add(H)
        xm = base.at[i].add(-H)
        fd_grads[i] = (float(irs_npv(xp)) - float(irs_npv(xm))) / (2 * H)

    max_rel = np.max(np.abs(grads - fd_grads) / np.maximum(np.abs(fd_grads), 1e-10))
    print(f"  Max rel diff (AD vs FD): {max_rel:.2e}")

    # === Test 2: 100-scenario vmap ===
    key = jax.random.PRNGKey(42)
    bump = jax.random.normal(key, (100, N_INPUTS)) * 0.002  # ±20bp
    scenarios = base + bump

    t0 = time.time()
    scenario_npvs = jax.vmap(irs_npv)(scenarios)
    t_vmap_npv = time.time() - t0

    t0 = time.time()
    scenario_grads = jax.vmap(grad_fn)(scenarios)
    t_vmap_grad = time.time() - t0

    print(f"\n  --- 100 scenarios ---")
    print(f"  NPV range: [{float(jnp.min(scenario_npvs)):>10,.2f}, {float(jnp.max(scenario_npvs)):>10,.2f}]")
    print(f"  NPV std:   {float(jnp.std(scenario_npvs)):>10,.2f}")
    print(f"  vmap NPV:  {t_vmap_npv:.3f}s")
    print(f"  vmap grad: {t_vmap_grad:.3f}s")
    print(f"  Grad shape: {scenario_grads.shape}")

    # === Test 3: JIT + vmap ===
    jit_vmap_grad = jax.jit(jax.vmap(grad_fn))
    _ = jit_vmap_grad(scenarios)  # warmup
    t0 = time.time()
    _ = jit_vmap_grad(scenarios)
    t_jit = time.time() - t0

    jit_vmap_npv = jax.jit(jax.vmap(irs_npv))
    _ = jit_vmap_npv(scenarios)
    t0 = time.time()
    _ = jit_vmap_npv(scenarios)
    t_jit_npv = time.time() - t0

    print(f"\n  JIT+vmap NPV:  {t_jit_npv:.6f}s")
    print(f"  JIT+vmap grad: {t_jit:.6f}s")

    # === Validation ===
    n_pass = 0
    total = 4

    ok = max_rel < 1e-5
    n_pass += int(ok)
    print(f"\n  AD vs FD:       {'✓' if ok else '✗'} (max rel={max_rel:.2e})")

    ok = scenario_grads.shape == (100, 17)
    n_pass += int(ok)
    print(f"  Grad shape:     {'✓' if ok else '✗'}")

    ok = t_jit < t_vmap_grad
    n_pass += int(ok)
    print(f"  JIT speedup:    {'✓' if ok else '✗'}")

    ok = float(jnp.std(scenario_npvs)) > 0
    n_pass += int(ok)
    print(f"  NPV variation:  {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All MC IRS scenario risk validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
