"""Validation: IRS 17x17 Hessian (second-order sensitivities)
Source: ~/quantlib-risk-py/benchmarks/second_order_irs.py

Computes the 17x17 Hessian of a 5Y payer IRS NPV w.r.t. 17 curve inputs
(9 deposit/swap rates + 8 fra rates for the float leg forecast).
Uses jax.hessian vs FD-over-FD.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.engines.analytic.black_formula import black_scholes_price


# === Simplified IRS model: NPV as function of 17 curve rates ===
# 9 discount rates at [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5] years
# 8 forward rates for float leg at [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4] years
# Total: 17 inputs

FIXED_RATE = 0.03
NOTIONAL = 1_000_000.0
N_FIXED = 5  # annual fixed coupons
N_FLOAT = 10  # semi-annual float coupons
FIXED_TIMES = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
FLOAT_TIMES = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
N_INPUTS = 17


def irs_npv(inputs):
    """Compute IRS NPV from 17 curve inputs.

    inputs[0:9] = discount zero rates at [0.5,1,...,5]
    inputs[9:17] = forward rates for 8 float periods
    """
    disc_rates = inputs[:9]  # zero rates at 0.5,1,...,4.5 (9 values)
    fwd_rates = inputs[9:]   # forward rates for 8 semi-annual periods

    # Discount factors from zero rates
    disc_times = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

    def disc(t):
        """Interpolate discount factor."""
        zr = jnp.interp(t, disc_times, disc_rates)
        return jnp.exp(-zr * t)

    # Fixed leg PV (annual)
    fixed_pv = jnp.float64(0.0)
    for i in range(N_FIXED):
        t = FIXED_TIMES[i]
        fixed_pv = fixed_pv + FIXED_RATE * NOTIONAL * disc(t)
    fixed_pv = fixed_pv + 0.0  # no notional exchange

    # Float leg PV (semi-annual)
    float_pv = jnp.float64(0.0)
    for i in range(min(len(fwd_rates), N_FLOAT)):
        t = FLOAT_TIMES[i]
        tau = 0.5
        fwd = fwd_rates[i] if i < len(fwd_rates) else disc_rates[min(i, 8)]
        float_pv = float_pv + fwd * tau * NOTIONAL * disc(t)

    # For remaining float periods (if fwd_rates < N_FLOAT)
    for i in range(len(fwd_rates), N_FLOAT):
        t = FLOAT_TIMES[i]
        tau = 0.5
        float_pv = float_pv + disc_rates[min(i, 8)] * tau * NOTIONAL * disc(t)

    return float_pv - fixed_pv  # payer swap


def main():
    print("=" * 78)
    print("IRS 17x17 Hessian (jax.hessian vs FD)")
    print("=" * 78)
    print(f"  Instrument : 5Y Payer IRS, fixed={FIXED_RATE*100:.0f}%, 1M notional")
    print(f"  Inputs     : {N_INPUTS} (9 disc rates + 8 fwd rates)")
    print()

    # Base inputs: flat 3% for everything
    base = jnp.array([0.03] * N_INPUTS, dtype=jnp.float64)

    # === 1. Baseline NPV ===
    npv = float(irs_npv(base))
    print(f"  Base NPV: {npv:,.2f}")

    # === 2. jax.hessian ===
    t0 = time.time()
    hess_fn = jax.hessian(irs_npv)
    hess_ad = np.array(hess_fn(base))
    t_ad = time.time() - t0

    # === 3. FD-over-FD Hessian ===
    H = 1e-5
    t0 = time.time()
    hess_fd = np.zeros((N_INPUTS, N_INPUTS))
    f0 = float(irs_npv(base))
    for i in range(N_INPUTS):
        for j in range(i, N_INPUTS):
            xpp = base.at[i].add(H).at[j].add(H)
            xp0 = base.at[i].add(H)
            x0p = base.at[j].add(H)
            hess_fd[i, j] = (float(irs_npv(xpp)) - float(irs_npv(xp0))
                              - float(irs_npv(x0p)) + f0) / (H * H)
            hess_fd[j, i] = hess_fd[i, j]
    t_fd = time.time() - t0

    # === 4. jax.jit(jax.hessian) ===
    t0 = time.time()
    jit_hess = jax.jit(hess_fn)
    _ = jit_hess(base)  # warmup
    t0 = time.time()
    hess_jit = np.array(jit_hess(base))
    t_jit = time.time() - t0

    # === 5. Comparison ===
    max_diff = np.max(np.abs(hess_ad - hess_fd))
    max_entry = np.max(np.abs(hess_ad))
    rel_diff = max_diff / max(max_entry, 1e-10)

    print(f"\n  --- 17x17 Hessian comparison ---")
    print(f"  Max |H_AD - H_FD|: {max_diff:.2e}")
    print(f"  Max |H_AD|:        {max_entry:.2e}")
    print(f"  Relative diff:     {rel_diff:.2e}")

    # Symmetry
    max_asym = np.max(np.abs(hess_ad - hess_ad.T))
    print(f"  Hessian symmetry:  {max_asym:.2e}")

    # JIT consistency
    jit_diff = np.max(np.abs(hess_ad - hess_jit))
    print(f"  JIT consistency:   {jit_diff:.2e}")

    # Timing
    print(f"\n  --- Timing ---")
    print(f"  FD-over-FD:        {t_fd:.3f}s")
    print(f"  jax.hessian:       {t_ad:.3f}s")
    print(f"  jax.jit(hessian):  {t_jit:.6f}s")
    if t_jit > 0:
        print(f"  Speedup (JIT/FD):  {t_fd/t_jit:.0f}x")

    # === Validation ===
    n_pass = 0
    total = 4

    ok = rel_diff < 1e-3
    n_pass += int(ok)
    print(f"\n  AD vs FD agreement: {'✓' if ok else '✗'} (rel diff={rel_diff:.2e})")

    ok = max_asym < 1e-8
    n_pass += int(ok)
    print(f"  Hessian symmetry:   {'✓' if ok else '✗'} (max asym={max_asym:.2e})")

    ok = jit_diff < 1e-10
    n_pass += int(ok)
    print(f"  JIT consistency:    {'✓' if ok else '✗'}")

    ok = t_jit < t_fd
    n_pass += int(ok)
    print(f"  JIT faster than FD: {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All IRS Hessian validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
