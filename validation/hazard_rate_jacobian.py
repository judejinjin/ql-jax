"""Validation: Hazard-rate Jacobian (∂NPV/∂h for CDS)
Source: ~/quantlib-risk-py/benchmarks/hazard_rate_jacobian.py

4 CDS instruments on a common hazard-rate term structure.
Computes the 4x4 Jacobian ∂NPV_i/∂h_j using jax.jacrev vs FD.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


NOTIONAL = 10_000_000.0
COUPON = 0.01       # 100bp running coupon
RECOVERY = 0.4
DISC_RATE = 0.03    # flat discount curve

# 4 hazard rate nodes at [1Y, 2Y, 3Y, 5Y]
HAZARD_TIMES = jnp.array([1.0, 2.0, 3.0, 5.0])
# 4 CDS tenors: 1Y, 2Y, 3Y, 5Y
CDS_TENORS = [1.0, 2.0, 3.0, 5.0]


def cds_npv_vector(hazard_rates):
    """Return a 4-vector of CDS NPVs as a function of 4 hazard rates."""
    results = []
    for tenor in CDS_TENORS:
        npv = _single_cds(hazard_rates, tenor)
        results.append(npv)
    return jnp.array(results)


def _single_cds(hazard_rates, tenor):
    """NPV of a single CDS with given tenor."""
    n_periods = int(tenor * 4)  # quarterly
    tau = 0.25

    def disc(t):
        return jnp.exp(-DISC_RATE * t)

    def surv(t):
        h = jnp.interp(t, HAZARD_TIMES, hazard_rates)
        return jnp.exp(-h * t)

    # Premium leg
    prem_pv = jnp.float64(0.0)
    for i in range(n_periods):
        t = (i + 1) * tau
        prem_pv = prem_pv + COUPON * tau * NOTIONAL * disc(t) * surv(t)

    # Protection leg
    prot_pv = jnp.float64(0.0)
    for i in range(n_periods):
        t0 = i * tau
        t1 = (i + 1) * tau
        t_mid = 0.5 * (t0 + t1)
        dp = surv(t0) - surv(t1)
        prot_pv = prot_pv + (1.0 - RECOVERY) * NOTIONAL * disc(t_mid) * dp

    return prot_pv - prem_pv


def main():
    print("=" * 78)
    print("Hazard-Rate Jacobian: 4 CDS × 4 hazard rates")
    print("=" * 78)

    base = jnp.array([0.01, 0.012, 0.015, 0.02], dtype=jnp.float64)

    npvs = cds_npv_vector(base)
    print(f"\n  Base NPVs:")
    for i, t in enumerate(CDS_TENORS):
        print(f"    CDS {t:.0f}Y: {float(npvs[i]):>12,.2f}")

    # jax.jacrev
    t0 = time.time()
    jac_fn = jax.jacrev(cds_npv_vector)
    jac_ad = np.array(jac_fn(base))
    t_ad = time.time() - t0

    # FD Jacobian
    H = 1e-6
    t0 = time.time()
    jac_fd = np.zeros((4, 4))
    for j in range(4):
        xp = base.at[j].add(H)
        xm = base.at[j].add(-H)
        fp = np.array(cds_npv_vector(xp))
        fm = np.array(cds_npv_vector(xm))
        jac_fd[:, j] = (fp - fm) / (2 * H)
    t_fd = time.time() - t0

    # JIT
    jit_jac = jax.jit(jac_fn)
    _ = jit_jac(base)
    t0 = time.time()
    jac_jit = np.array(jit_jac(base))
    t_jit = time.time() - t0

    # Comparison
    max_diff = np.max(np.abs(jac_ad - jac_fd))
    max_entry = np.max(np.abs(jac_ad))
    rel_diff = max_diff / max(max_entry, 1e-10)
    jit_diff = np.max(np.abs(jac_ad - jac_jit))

    print(f"\n  --- 4x4 Jacobian comparison ---")
    print(f"  Max |J_AD - J_FD|: {max_diff:.2e}")
    print(f"  Max |J_AD|:        {max_entry:.2e}")
    print(f"  Relative diff:     {rel_diff:.2e}")
    print(f"  JIT consistency:   {jit_diff:.2e}")
    print(f"\n  --- Timing ---")
    print(f"  FD:                {t_fd:.3f}s")
    print(f"  jax.jacrev:        {t_ad:.3f}s")
    print(f"  jax.jit(jacrev):   {t_jit:.6f}s")

    # Print Jacobian
    print(f"\n  Jacobian (AD):")
    print(f"  {'':>12s}", end="")
    for j in range(4):
        print(f"  h_{j+1:d}", end="          ")
    print()
    for i in range(4):
        print(f"  CDS{CDS_TENORS[i]:.0f}Y", end="  ")
        for j in range(4):
            print(f"  {jac_ad[i,j]:>12.1f}", end="")
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
        print("✓ All hazard-rate Jacobian validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
