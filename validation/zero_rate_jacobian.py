"""Validation: Zero-rate Jacobian (9x9)
Source: ~/quantlib-risk-py/benchmarks/zero_rate_jacobian.py

9 instruments (deposits + swaps) on a 9-node zero-rate curve.
Computes 9x9 Jacobian ∂NPV_i/∂z_j using jax.jacrev vs FD.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


NOTIONAL = 1_000_000.0
# 9 tenors: 3M, 6M, 1Y, 2Y, 3Y, 4Y, 5Y, 7Y, 10Y
TENORS = jnp.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
TENORS_PY = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
N_INST = 9
N_PAY_LIST = [0, 0, 0, 2, 3, 4, 5, 7, 10]  # annual periods for swaps


def instrument_npvs(zero_rates):
    """Compute NPVs of 9 instruments from 9 zero rates.

    - Instruments 0-2: deposits at 3M, 6M, 1Y (simple interest)
    - Instruments 3-8: par swaps at 2Y, 3Y, 4Y, 5Y, 7Y, 10Y
    """
    results = []

    # Discount function
    def disc(t):
        zr = jnp.interp(t, TENORS, zero_rates)
        return jnp.exp(-zr * t)

    # Deposits: receive (1+r*T) at maturity, pay 1 at start
    for i in range(3):
        T = TENORS_PY[i]
        r = zero_rates[i]
        npv = (1.0 + r * T) * disc(T) - 1.0
        results.append(npv * NOTIONAL)

    # Swaps: pay fixed K, receive float
    for i in range(3, N_INST):
        T = TENORS_PY[i]
        n_pay = N_PAY_LIST[i]
        # Par rate (makes swap worth zero at market rates)
        # K = (1 - DF(T)) / sum(DF(t_i))
        ann = jnp.float64(0.0)
        for y in range(1, n_pay + 1):
            ann = ann + disc(float(y))
        K = (1.0 - disc(T)) / ann

        # NPV of payer swap at current curve
        fixed_pv = jnp.float64(0.0)
        for y in range(1, n_pay + 1):
            fixed_pv = fixed_pv + K * disc(float(y))
        float_pv = 1.0 - disc(T)

        results.append((float_pv - fixed_pv) * NOTIONAL)

    return jnp.array(results)


def main():
    print("=" * 78)
    print("Zero-Rate Jacobian: 9 instruments × 9 zero rates")
    print("=" * 78)

    # Upward-sloping curve
    base = jnp.array([0.020, 0.022, 0.025, 0.030, 0.033, 0.035,
                       0.037, 0.040, 0.042], dtype=jnp.float64)

    npvs = instrument_npvs(base)
    print(f"\n  Base NPVs:")
    for i in range(N_INST):
        print(f"    Inst {i+1} ({float(TENORS[i]):>4.1f}Y): {float(npvs[i]):>12,.2f}")

    # jax.jacrev
    t0 = time.time()
    jac_fn = jax.jacrev(instrument_npvs)
    jac_ad = np.array(jac_fn(base))
    t_ad = time.time() - t0

    # FD
    H = 1e-7
    t0 = time.time()
    jac_fd = np.zeros((N_INST, N_INST))
    for j in range(N_INST):
        xp = base.at[j].add(H)
        xm = base.at[j].add(-H)
        jac_fd[:, j] = (np.array(instrument_npvs(xp)) - np.array(instrument_npvs(xm))) / (2 * H)
    t_fd = time.time() - t0

    # JIT
    jit_jac = jax.jit(jac_fn)
    _ = jit_jac(base)
    t0 = time.time()
    jac_jit = np.array(jit_jac(base))
    t_jit = time.time() - t0

    max_diff = np.max(np.abs(jac_ad - jac_fd))
    max_entry = np.max(np.abs(jac_ad))
    rel_diff = max_diff / max(max_entry, 1e-10)
    jit_diff = np.max(np.abs(jac_ad - jac_jit))

    print(f"\n  --- 9x9 Jacobian comparison ---")
    print(f"  Max |J_AD - J_FD|: {max_diff:.2e}")
    print(f"  Max |J_AD|:        {max_entry:.2e}")
    print(f"  Relative diff:     {rel_diff:.2e}")
    print(f"  JIT consistency:   {jit_diff:.2e}")
    print(f"\n  --- Timing ---")
    print(f"  FD:                {t_fd:.3f}s")
    print(f"  jax.jacrev:        {t_ad:.3f}s")
    print(f"  jax.jit(jacrev):   {t_jit:.6f}s")

    # Print top-left 5x5
    print(f"\n  Jacobian (AD, top-left 5x5):")
    for i in range(min(5, N_INST)):
        print(f"    [{i}]", end="")
        for j in range(min(5, N_INST)):
            print(f"  {jac_ad[i,j]:>10.1f}", end="")
        print()

    n_pass = 0
    total = 3

    ok = rel_diff < 1e-4
    n_pass += int(ok)
    print(f"\n  AD vs FD agreement: {'✓' if ok else '✗'} (rel diff={rel_diff:.2e})")

    ok = jit_diff < 1e-6
    n_pass += int(ok)
    print(f"  JIT consistency:    {'✓' if ok else '✗'}")

    ok = t_jit < t_fd
    n_pass += int(ok)
    print(f"  JIT faster than FD: {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All zero-rate Jacobian validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
