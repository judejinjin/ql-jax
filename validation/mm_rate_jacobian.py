"""Validation: Money-market rate Jacobian
Source: ~/quantlib-risk-py/benchmarks/mm_rate_jacobian.py

Par-rate sensitivities: how does the par swap rate change when
each zero rate node is bumped? Uses jax.jacrev on par_rate computation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


# 9 zero-rate nodes
TENORS = jnp.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
# Compute par rates for swap tenors 2Y, 3Y, 4Y, 5Y, 7Y, 10Y
PAR_TENORS = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
N_PAR = len(PAR_TENORS)
N_ZR = 9


def par_rates(zero_rates):
    """Compute par swap rates from zero rates."""
    def disc(t):
        zr = jnp.interp(t, TENORS, zero_rates)
        return jnp.exp(-zr * t)

    results = []
    for T in PAR_TENORS:
        n = int(T)
        ann = jnp.float64(0.0)
        for y in range(1, n + 1):
            ann = ann + disc(float(y))
        pr = (1.0 - disc(T)) / ann
        results.append(pr)

    return jnp.array(results)


def main():
    print("=" * 78)
    print("Money-Market Rate Jacobian: 6 par rates × 9 zero rates")
    print("=" * 78)

    base = jnp.array([0.020, 0.022, 0.025, 0.030, 0.033, 0.035,
                       0.037, 0.040, 0.042], dtype=jnp.float64)

    prs = par_rates(base)
    print(f"\n  Par rates at base curve:")
    for i, T in enumerate(PAR_TENORS):
        print(f"    {T:.0f}Y: {float(prs[i])*100:.4f}%")

    # jax.jacrev
    t0 = time.time()
    jac_fn = jax.jacrev(par_rates)
    jac_ad = np.array(jac_fn(base))
    t_ad = time.time() - t0

    # FD
    H = 1e-7
    t0 = time.time()
    jac_fd = np.zeros((N_PAR, N_ZR))
    for j in range(N_ZR):
        xp = base.at[j].add(H)
        xm = base.at[j].add(-H)
        jac_fd[:, j] = (np.array(par_rates(xp)) - np.array(par_rates(xm))) / (2 * H)
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

    print(f"\n  --- 6x9 Jacobian comparison ---")
    print(f"  Max |J_AD - J_FD|: {max_diff:.2e}")
    print(f"  Max |J_AD|:        {max_entry:.2e}")
    print(f"  Relative diff:     {rel_diff:.2e}")
    print(f"  JIT consistency:   {jit_diff:.2e}")
    print(f"\n  --- Timing ---")
    print(f"  FD:                {t_fd:.3f}s")
    print(f"  jax.jacrev:        {t_ad:.3f}s")
    print(f"  jax.jit(jacrev):   {t_jit:.6f}s")

    # Row sums: ∂K/∂z should approximately sum to ~1 for each par rate
    # (a parallel shift identity holds approximately)
    row_sums = np.sum(jac_ad, axis=1)
    print(f"\n  Row sums (∂K/∂z across all z nodes):")
    for i, T in enumerate(PAR_TENORS):
        print(f"    {T:.0f}Y: {row_sums[i]:>8.4f}")

    n_pass = 0
    total = 3

    ok = rel_diff < 1e-5
    n_pass += int(ok)
    print(f"\n  AD vs FD agreement: {'✓' if ok else '✗'} (rel diff={rel_diff:.2e})")

    ok = jit_diff < 1e-8
    n_pass += int(ok)
    print(f"  JIT consistency:    {'✓' if ok else '✗'}")

    ok = t_jit < t_fd
    n_pass += int(ok)
    print(f"  JIT faster than FD: {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All money-market rate Jacobian validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
