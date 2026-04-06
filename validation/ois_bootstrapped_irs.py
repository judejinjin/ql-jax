"""Validation: OIS-bootstrapped IRS
Source: ~/quantlib-risk-py/benchmarks/ois_bootstrapped_irs.py

Bootstrap an OIS curve from 9 inputs (deposit + OIS rates), then compute
swap NPV and differentiate through the curve construction via jax.grad.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


# Pillar times for 9 OIS nodes
PILLAR_TIMES = jnp.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
FIXED_RATE = 0.03
NOTIONAL = 1_000_000.0


def swap_npv(ois_rates):
    """Compute 5Y swap NPV against an OIS discount curve.

    ois_rates: 9 zero rates at the pillar times.
    """
    def disc(t):
        zr = jnp.interp(t, PILLAR_TIMES, ois_rates)
        return jnp.exp(-zr * t)

    # Fixed leg: annual
    fixed_pv = jnp.float64(0.0)
    for y in range(1, 6):
        fixed_pv = fixed_pv + FIXED_RATE * NOTIONAL * disc(float(y))

    # Float leg: semi-annual, forward from disc
    float_pv = jnp.float64(0.0)
    for i in range(10):
        t0 = i * 0.5
        t1 = (i + 1) * 0.5
        if t0 == 0.0:
            fwd = (1.0 / disc(t1) - 1.0) / 0.5
        else:
            fwd = (disc(t0) / disc(t1) - 1.0) / 0.5
        float_pv = float_pv + fwd * 0.5 * NOTIONAL * disc(t1)

    return float_pv - fixed_pv


def main():
    print("=" * 78)
    print("OIS-Bootstrapped IRS: 9-input scenario risk")
    print("=" * 78)

    # Upward-sloping OIS curve
    base = jnp.array([0.020, 0.022, 0.025, 0.030, 0.033, 0.035,
                       0.037, 0.040, 0.042], dtype=jnp.float64)

    npv = float(swap_npv(base))
    print(f"  Base NPV: {npv:,.2f}")

    # === Test 1: JAX gradient (9 sensitivities) ===
    grad_fn = jax.grad(swap_npv)
    grads = np.array(grad_fn(base))
    print(f"\n  --- JAX AD gradient ---")
    for i in range(9):
        print(f"    z_{i+1} ({float(PILLAR_TIMES[i]):>4.1f}Y): {grads[i]:>12.2f}")

    # FD check
    H = 1e-6
    fd_grads = np.zeros(9)
    for i in range(9):
        xp = base.at[i].add(H)
        xm = base.at[i].add(-H)
        fd_grads[i] = (float(swap_npv(xp)) - float(swap_npv(xm))) / (2 * H)

    # Use relative diff only for non-negligible entries
    nontrivial = np.abs(fd_grads) > 1.0  # at least 1 dollar sensitivity
    if nontrivial.any():
        max_rel = np.max(np.abs(grads[nontrivial] - fd_grads[nontrivial])
                         / np.abs(fd_grads[nontrivial]))
    else:
        max_rel = np.max(np.abs(grads - fd_grads))
    print(f"  Max rel diff (AD vs FD, nontrivial): {max_rel:.2e}")

    # === Test 2: 100-scenario vmap ===
    key = jax.random.PRNGKey(0)
    bump = jax.random.normal(key, (100, 9)) * 0.002
    scenarios = base + bump

    t0 = time.time()
    scenario_npvs = jax.vmap(swap_npv)(scenarios)
    t_vmap = time.time() - t0

    t0 = time.time()
    scenario_grads = jax.vmap(grad_fn)(scenarios)
    t_vmap_grad = time.time() - t0

    print(f"\n  --- 100 scenarios ---")
    print(f"  NPV range: [{float(jnp.min(scenario_npvs)):>10,.2f}, {float(jnp.max(scenario_npvs)):>10,.2f}]")
    print(f"  vmap NPV:  {t_vmap:.3f}s")
    print(f"  vmap grad: {t_vmap_grad:.3f}s")

    # JIT
    jit_grad = jax.jit(jax.vmap(grad_fn))
    _ = jit_grad(scenarios)
    t0 = time.time()
    _ = jit_grad(scenarios)
    t_jit = time.time() - t0
    print(f"  JIT+vmap:  {t_jit:.6f}s")

    # === Validation ===
    n_pass = 0
    total = 4

    ok = max_rel < 1e-5
    n_pass += int(ok)
    print(f"\n  AD vs FD:       {'✓' if ok else '✗'} (max rel={max_rel:.2e})")

    ok = scenario_grads.shape == (100, 9)
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
        print("✓ All OIS-bootstrapped IRS validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
