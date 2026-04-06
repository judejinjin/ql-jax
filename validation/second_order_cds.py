"""Validation: CDS 6x6 Hessian (second-order sensitivities)
Source: ~/quantlib-risk-py/benchmarks/second_order_cds.py

Computes the 6x6 Hessian of a 5Y CDS NPV w.r.t. 6 inputs:
  [spread, recovery, r1, r2, r3, r4] (spread + recovery + 4 hazard rates)
Uses jax.hessian vs FD-over-FD.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


# CDS parameters
NOTIONAL = 10_000_000.0
COUPON = 0.01  # 100bp running spread
N_INPUTS = 6


def cds_npv(inputs):
    """CDS NPV from 6 inputs: [spread, recovery, r1, r2, r3, r4].

    spread: CDS coupon spread
    recovery: recovery rate
    r1..r4: flat discount rate nodes at 1Y, 2Y, 3Y, 5Y
    """
    spread = inputs[0]
    recovery = inputs[1]
    disc_rates = inputs[2:]  # 4 zero rates at [1, 2, 3, 5]

    # Payment dates: quarterly for 5Y = 20 periods
    n_periods = 20
    tau = 0.25
    payment_dates = jnp.arange(1, n_periods + 1) * tau  # 0.25, 0.5, ..., 5.0

    # Discount factors via interpolation
    disc_nodes = jnp.array([1.0, 2.0, 3.0, 5.0])

    def disc(t):
        zr = jnp.interp(t, disc_nodes, disc_rates)
        return jnp.exp(-zr * t)

    # Survival probabilities: constant hazard implied by spread/(1-R)
    hazard = spread / (1.0 - recovery)

    def surv(t):
        return jnp.exp(-hazard * t)

    # Premium leg: sum of spread * tau * DF * Surv
    prem_pv = jnp.float64(0.0)
    for i in range(n_periods):
        t = payment_dates[i]
        prem_pv = prem_pv + COUPON * tau * NOTIONAL * disc(t) * surv(t)

    # Protection (default) leg: sum over midpoints
    prot_pv = jnp.float64(0.0)
    for i in range(n_periods):
        t0 = payment_dates[i - 1] if i > 0 else 0.0
        t1 = payment_dates[i]
        t_mid = 0.5 * (t0 + t1)
        dp = surv(t0) - surv(t1)
        prot_pv = prot_pv + (1.0 - recovery) * NOTIONAL * disc(t_mid) * dp

    # buyer pays premium, receives protection
    return prot_pv - prem_pv


def main():
    print("=" * 78)
    print("CDS 6x6 Hessian (jax.hessian vs FD)")
    print("=" * 78)
    print(f"  Instrument : 5Y CDS, coupon={COUPON*1e4:.0f}bp, 10M notional")
    print(f"  Inputs     : {N_INPUTS} (spread, recovery, 4 disc rates)")
    print()

    base = jnp.array([0.01, 0.4, 0.03, 0.035, 0.04, 0.045], dtype=jnp.float64)

    npv = float(cds_npv(base))
    print(f"  Base NPV: {npv:,.2f}")

    # jax.hessian
    t0 = time.time()
    hess_fn = jax.hessian(cds_npv)
    hess_ad = np.array(hess_fn(base))
    t_ad = time.time() - t0

    # FD Hessian
    H = 1e-5
    t0 = time.time()
    hess_fd = np.zeros((N_INPUTS, N_INPUTS))
    f0 = float(cds_npv(base))
    for i in range(N_INPUTS):
        for j in range(i, N_INPUTS):
            xpp = base.at[i].add(H).at[j].add(H)
            xp0 = base.at[i].add(H)
            x0p = base.at[j].add(H)
            hess_fd[i, j] = (float(cds_npv(xpp)) - float(cds_npv(xp0))
                              - float(cds_npv(x0p)) + f0) / (H * H)
            hess_fd[j, i] = hess_fd[i, j]
    t_fd = time.time() - t0

    # JIT
    t0 = time.time()
    jit_hess = jax.jit(hess_fn)
    _ = jit_hess(base)
    t0 = time.time()
    hess_jit = np.array(jit_hess(base))
    t_jit = time.time() - t0

    # Comparison
    max_diff = np.max(np.abs(hess_ad - hess_fd))
    max_entry = np.max(np.abs(hess_ad))
    rel_diff = max_diff / max(max_entry, 1e-10)
    max_asym = np.max(np.abs(hess_ad - hess_ad.T))
    jit_diff = np.max(np.abs(hess_ad - hess_jit))

    print(f"\n  --- 6x6 Hessian comparison ---")
    print(f"  Max |H_AD - H_FD|: {max_diff:.2e}")
    print(f"  Max |H_AD|:        {max_entry:.2e}")
    print(f"  Relative diff:     {rel_diff:.2e}")
    print(f"  Hessian symmetry:  {max_asym:.2e}")
    print(f"  JIT consistency:   {jit_diff:.2e}")
    print(f"\n  --- Timing ---")
    print(f"  FD-over-FD:        {t_fd:.3f}s")
    print(f"  jax.hessian:       {t_ad:.3f}s")
    print(f"  jax.jit(hessian):  {t_jit:.6f}s")

    n_pass = 0
    total = 4

    ok = rel_diff < 1e-3
    n_pass += int(ok)
    print(f"\n  AD vs FD agreement: {'✓' if ok else '✗'} (rel diff={rel_diff:.2e})")

    ok = max_asym < 1e-6
    n_pass += int(ok)
    print(f"  Hessian symmetry:   {'✓' if ok else '✗'} (max asym={max_asym:.2e})")

    ok = jit_diff < 1e-6
    n_pass += int(ok)
    print(f"  JIT consistency:    {'✓' if ok else '✗'}")

    ok = t_jit < t_fd
    n_pass += int(ok)
    print(f"  JIT faster than FD: {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All CDS Hessian validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
