"""Validation: CDS-spread Jacobian (reverse direction)
Source: ~/quantlib-risk-py/benchmarks/cds_spread_jacobian.py

4 CDS instruments: compute ∂NPV/∂spread for each CDS.
Since CDS NPV = f(spread, hazard_curve), and fair-spread is where NPV=0,
this validates the sensitivity of NPV to spread changes.
Uses jax.jacrev vs FD, 4x4 matrix.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


NOTIONAL = 10_000_000.0
RECOVERY = 0.4
DISC_RATE = 0.03
HAZARD_RATE = 0.015  # flat hazard rate

CDS_TENORS = [1.0, 2.0, 3.0, 5.0]


def cds_npv_vector(spreads):
    """Return 4-vector of CDS NPVs as function of 4 coupon spreads."""
    results = []
    for idx, tenor in enumerate(CDS_TENORS):
        npv = _single_cds(spreads[idx], tenor)
        results.append(npv)
    return jnp.array(results)


def _single_cds(spread, tenor):
    """NPV of a CDS with given spread and tenor."""
    n_periods = int(tenor * 4)
    tau = 0.25

    def disc(t):
        return jnp.exp(-DISC_RATE * t)

    def surv(t):
        return jnp.exp(-HAZARD_RATE * t)

    prem_pv = jnp.float64(0.0)
    for i in range(n_periods):
        t = (i + 1) * tau
        prem_pv = prem_pv + spread * tau * NOTIONAL * disc(t) * surv(t)

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
    print("CDS-Spread Jacobian: 4 CDS × 4 spreads")
    print("=" * 78)

    base = jnp.array([0.008, 0.009, 0.010, 0.012], dtype=jnp.float64)

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
    H = 1e-7
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

    # Expected: diagonal (each CDS depends only on its own spread)
    off_diag = np.copy(jac_ad)
    np.fill_diagonal(off_diag, 0.0)
    max_off = np.max(np.abs(off_diag))
    diag_dom = max_off / max(max_entry, 1e-10)
    print(f"  Off-diagonal:      {max_off:.2e} (should be ~0)")
    print(f"  Diagonal dominance:{diag_dom:.2e}")

    # Print Jacobian
    print(f"\n  Jacobian (AD):")
    print(f"  {'':>12s}", end="")
    for j in range(4):
        print(f"  s_{j+1:d}", end="          ")
    print()
    for i in range(4):
        print(f"  CDS{CDS_TENORS[i]:.0f}Y", end="  ")
        for j in range(4):
            print(f"  {jac_ad[i,j]:>12.1f}", end="")
        print()

    n_pass = 0
    total = 4

    ok = rel_diff < 1e-5
    n_pass += int(ok)
    print(f"\n  AD vs FD agreement:  {'✓' if ok else '✗'} (rel diff={rel_diff:.2e})")

    ok = jit_diff < 1e-6
    n_pass += int(ok)
    print(f"  JIT consistency:     {'✓' if ok else '✗'}")

    ok = diag_dom < 1e-10  # should be exactly diagonal
    n_pass += int(ok)
    print(f"  Diagonal structure:  {'✓' if ok else '✗'}")

    ok = t_jit < t_fd
    n_pass += int(ok)
    print(f"  JIT faster than FD:  {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All CDS-spread Jacobian validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
