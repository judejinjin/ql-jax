"""Validation: IR Cap 18x18 Hessian (second-order sensitivities)
Source: ~/quantlib-risk-py/benchmarks/second_order_ir_cap.py

Computes the 18x18 Hessian of a 5Y IR cap NPV w.r.t. 18 inputs:
  [17 curve rates + 1 flat vol]
Uses jax.hessian vs FD-over-FD.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


NOTIONAL = 1_000_000.0
STRIKE = 0.03
N_INPUTS = 18  # 17 curve + 1 vol


def cap_npv(inputs):
    """Cap NPV from 18 inputs: [9 disc rates, 8 fwd rates, 1 vol].

    Uses Black caplet formula for each semi-annual caplet.
    """
    disc_rates = inputs[:9]   # zero rates at [0.5,1,...,4.5]
    fwd_rates = inputs[9:17]  # forward rates for 8 periods
    vol = inputs[17]          # flat Black vol

    disc_times = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

    def disc(t):
        zr = jnp.interp(t, disc_times, disc_rates)
        return jnp.exp(-zr * t)

    # 10 semi-annual caplets at 0.5, 1.0, ..., 5.0
    n_caplets = 10
    tau = 0.5
    cap_pv = jnp.float64(0.0)

    for i in range(n_caplets):
        T_reset = (i + 1) * tau  # reset at 0.5, 1.0, ..., 5.0
        T_pay = T_reset          # pay on same date (simplified)

        # Forward rate for this period
        if i < len(fwd_rates):
            f = fwd_rates[i]
        else:
            f = disc_rates[jnp.minimum(i, 8)]

        # Black caplet formula
        d1 = (jnp.log(f / STRIKE) + 0.5 * vol**2 * T_reset) / (vol * jnp.sqrt(T_reset))
        d2 = d1 - vol * jnp.sqrt(T_reset)

        # Normal CDF via erf
        nd1 = 0.5 * (1.0 + jax.lax.erf(d1 / jnp.sqrt(2.0)))
        nd2 = 0.5 * (1.0 + jax.lax.erf(d2 / jnp.sqrt(2.0)))

        caplet = tau * NOTIONAL * disc(T_pay) * (f * nd1 - STRIKE * nd2)
        cap_pv = cap_pv + caplet

    return cap_pv


def main():
    print("=" * 78)
    print("IR Cap 18x18 Hessian (jax.hessian vs FD)")
    print("=" * 78)
    print(f"  Instrument : 5Y IR Cap, strike={STRIKE*100:.0f}%, 1M notional")
    print(f"  Inputs     : {N_INPUTS} (9 disc + 8 fwd + 1 vol)")
    print()

    # Base: flat 3% curve, 20% vol
    base = jnp.array([0.03]*9 + [0.03]*8 + [0.20], dtype=jnp.float64)

    npv = float(cap_npv(base))
    print(f"  Base NPV: {npv:,.2f}")

    # jax.hessian
    t0 = time.time()
    hess_fn = jax.hessian(cap_npv)
    hess_ad = np.array(hess_fn(base))
    t_ad = time.time() - t0

    # FD Hessian
    H = 1e-5
    t0 = time.time()
    hess_fd = np.zeros((N_INPUTS, N_INPUTS))
    f0 = float(cap_npv(base))
    for i in range(N_INPUTS):
        for j in range(i, N_INPUTS):
            xpp = base.at[i].add(H).at[j].add(H)
            xp0 = base.at[i].add(H)
            x0p = base.at[j].add(H)
            hess_fd[i, j] = (float(cap_npv(xpp)) - float(cap_npv(xp0))
                              - float(cap_npv(x0p)) + f0) / (H * H)
            hess_fd[j, i] = hess_fd[i, j]
    t_fd = time.time() - t0

    # JIT
    jit_hess = jax.jit(hess_fn)
    _ = jit_hess(base)  # warmup
    t0 = time.time()
    hess_jit = np.array(jit_hess(base))
    t_jit = time.time() - t0

    # Comparison
    max_diff = np.max(np.abs(hess_ad - hess_fd))
    max_entry = np.max(np.abs(hess_ad))
    rel_diff = max_diff / max(max_entry, 1e-10)
    max_asym = np.max(np.abs(hess_ad - hess_ad.T))
    jit_diff = np.max(np.abs(hess_ad - hess_jit))

    print(f"\n  --- 18x18 Hessian comparison ---")
    print(f"  Max |H_AD - H_FD|: {max_diff:.2e}")
    print(f"  Max |H_AD|:        {max_entry:.2e}")
    print(f"  Relative diff:     {rel_diff:.2e}")
    print(f"  Hessian symmetry:  {max_asym:.2e}")
    print(f"  JIT consistency:   {jit_diff:.2e}")
    print(f"\n  --- Timing ---")
    print(f"  FD-over-FD:        {t_fd:.3f}s")
    print(f"  jax.hessian:       {t_ad:.3f}s")
    print(f"  jax.jit(hessian):  {t_jit:.6f}s")
    if t_jit > 0:
        print(f"  Speedup (JIT/FD):  {t_fd/t_jit:.0f}x")

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
        print("✓ All IR Cap Hessian validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
