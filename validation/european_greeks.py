"""Validation: European Option Greek Benchmarks (FD vs jax.grad vs jax.jit(jax.grad))
Source: ~/quantlib-risk-py/benchmarks/european_option_benchmarks.py

Validates first-order Greeks for a European call via three methods:
  - Finite Differences (bump-and-reprice each input by 1 bp)
  - jax.grad (reverse-mode AD in one pass)
  - jax.jit(jax.grad) (JIT-compiled AD)

Market data: S=7, K=8, σ=0.10, r=0.05, q=0.05, T≈1.0055y, Call
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import statistics

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.engines.analytic.black_formula import black_scholes_price


# === Market data (matches quantlib-risk-py benchmarks) ===
S0 = 7.0
K = 8.0
T = 367.0 / 365.0   # May 15 1998 → May 17 1999
r0 = 0.05
q0 = 0.05
sigma0 = 0.10
BPS = 1e-4  # finite-difference bump

# === Reference Greeks from QuantLib 1.42 Analytic BSM ===
REF_NPV = 0.030334420670438
REF_GREEKS = {
    "Delta":   0.095099866793071,   # dV/dS
    "DivRho": -0.669346733675069,   # dV/dq
    "Vega":    1.171477274010069,   # dV/dσ (QuantLib vega = dV/dσ * 100 for %, but here raw)
    "Rho":     0.638846097000957,   # dV/dr
}

INPUT_NAMES = ["Delta (dV/dS)", "DivRho (dV/dq)", "Vega (dV/dσ)", "Rho (dV/dr)"]


def bsm_price(S, q, sigma, r_rate):
    """Wrap black_scholes_price for differentiation."""
    return black_scholes_price(S, K, T, r_rate, q, sigma, 1)  # Call = 1


def main():
    print("=" * 78)
    print("European Option Greek Benchmarks (FD vs jax.grad vs jax.jit(jax.grad))")
    print("=" * 78)
    print(f"  Instrument : European Call  S={S0}, K={K}, σ={sigma0}, "
          f"r={r0}, q={q0}, T={T:.4f}")
    print(f"  Engine     : Analytic BSM (black_scholes_price)")
    print(f"  BPS shift  : {BPS}")
    print()

    # === 1. Baseline NPV ===
    args = (jnp.float64(S0), jnp.float64(q0), jnp.float64(sigma0), jnp.float64(r0))
    npv = float(bsm_price(*args))
    print(f"  NPV = {npv:.15f}  (ref = {REF_NPV:.15f}  diff = {abs(npv - REF_NPV):.2e})")
    print()

    # === 2. Finite Differences ===
    fd_greeks = []
    base_args = [S0, q0, sigma0, r0]
    for i in range(4):
        bumped = list(base_args)
        bumped[i] += BPS
        npv_up = float(bsm_price(*[jnp.float64(x) for x in bumped]))
        fd_greeks.append((npv_up - npv) / BPS)

    # === 3. jax.grad (all 4 Greeks in one grad call) ===
    grad_fn = jax.grad(bsm_price, argnums=(0, 1, 2, 3))
    ad_greeks_raw = grad_fn(*args)
    ad_greeks = [float(g) for g in ad_greeks_raw]

    # === 4. jax.jit(jax.grad) ===
    jit_grad_fn = jax.jit(grad_fn)
    # warmup
    _ = jit_grad_fn(*args)
    jit_greeks_raw = jit_grad_fn(*args)
    jit_greeks = [float(g) for g in jit_greeks_raw]

    # === Print Greek comparison ===
    ref_vals = [REF_GREEKS["Delta"], REF_GREEKS["DivRho"],
                REF_GREEKS["Vega"], REF_GREEKS["Rho"]]

    print(f"  {'Input':<20} {'QuantLib':>14} {'FD':>14} {'jax.grad':>14} {'jit(grad)':>14} {'|FD-ref|':>10} {'|AD-ref|':>10}")
    print("  " + "-" * 100)

    n_pass = 0
    for i, name in enumerate(INPUT_NAMES):
        ref = ref_vals[i]
        fd = fd_greeks[i]
        ad = ad_greeks[i]
        jit = jit_greeks[i]
        fd_err = abs(fd - ref)
        ad_err = abs(ad - ref)
        jit_err = abs(jit - ref)
        # FD should be within ~BPS of exact; AD should be machine-precision
        fd_ok = fd_err < max(abs(ref) * 1e-2, 1e-4)  # FD: ~1% tolerance
        ad_ok = ad_err < max(abs(ref) * 1e-8, 1e-10)  # AD: near machine eps
        jit_ok = jit_err < max(abs(ref) * 1e-8, 1e-10)
        status = "✓" if (fd_ok and ad_ok and jit_ok) else "✗"
        n_pass += int(fd_ok and ad_ok and jit_ok)
        print(f"  {name:<20} {ref:>14.10f} {fd:>14.10f} {ad:>14.10f} {jit:>14.10f} {fd_err:>10.2e} {ad_err:>10.2e} {status}")

    # === 5. Timing comparison ===
    print()
    print("  --- Timing (median of 100 calls, ms) ---")
    n_reps = 100

    # FD timing
    def fd_call():
        v = float(bsm_price(*args))
        for i in range(4):
            bumped = list(base_args)
            bumped[i] += BPS
            float(bsm_price(*[jnp.float64(x) for x in bumped]))

    fd_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fd_call()
        fd_times.append((time.perf_counter() - t0) * 1000)

    # jax.grad timing
    ad_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        grad_fn(*args)
        ad_times.append((time.perf_counter() - t0) * 1000)

    # jax.jit(jax.grad) timing
    jit_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        jit_grad_fn(*args)
        jit_times.append((time.perf_counter() - t0) * 1000)

    fd_med = statistics.median(fd_times)
    ad_med = statistics.median(ad_times)
    jit_med = statistics.median(jit_times)
    print(f"    FD  (5 pricings)     : {fd_med:8.4f} ms")
    print(f"    jax.grad             : {ad_med:8.4f} ms  ({fd_med/ad_med:5.1f}x vs FD)")
    print(f"    jax.jit(jax.grad)    : {jit_med:8.4f} ms  ({fd_med/jit_med:5.1f}x vs FD)")

    # === 6. AD vs AD consistency ===
    print()
    print("  --- AD vs JIT consistency ---")
    max_diff = max(abs(ad_greeks[i] - jit_greeks[i]) for i in range(4))
    print(f"    max |grad - jit(grad)| = {max_diff:.2e}")
    ad_jit_ok = max_diff < 1e-12
    n_pass += int(ad_jit_ok)
    print(f"    {'✓' if ad_jit_ok else '✗'} Match to machine precision")

    print(f"\nPassed: {n_pass}/{len(INPUT_NAMES) + 1}")
    if n_pass == len(INPUT_NAMES) + 1:
        print("✓ All European Greek benchmarks passed.")
    else:
        print(f"✗ {len(INPUT_NAMES) + 1 - n_pass} failures.")


if __name__ == "__main__":
    main()
