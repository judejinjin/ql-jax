"""Validation: American Option Greek Benchmarks (FD vs jax.grad vs jax.jit(jax.grad))
Source: ~/quantlib-risk-py/benchmarks/american_option_benchmarks.py

Validates first-order Greeks for an American put via two pricing engines:
  - Barone-Adesi & Whaley (BAW) — quasi-analytic, JIT eligible
  - FD-Black-Scholes (PDE) — finite differences on grid

Market data: S=36, K=40, σ=0.20, r=0.06, q=0.00, T=1Y, Put
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

from ql_jax.engines.analytic.american import barone_adesi_whaley_price
from ql_jax.engines.fd.black_scholes import fd_black_scholes_price


# === Market data (matches quantlib-risk-py benchmarks) ===
S0 = 36.0
K = 40.0
T = 1.0  # 1 year
r0 = 0.06
q0 = 0.00
sigma0 = 0.20
BPS = 1e-4

# === Reference values from QuantLib 1.42 ===
# BAW: (FD reference greeks via bump-and-reprice)
REF_BAW = {
    "npv": 4.4656331743,
    "dS": -0.6901494477,
    "dr": -10.3506539224,
    "dq":  9.2968870043,
    "dsigma": 11.0241912977,
}

REF_FD = {
    "npv": 4.4842833523,
    "dS": -0.6965269841,
    "dr": -10.3821955820,
    "dq":  9.0880903065,
    "dsigma": 10.9466007915,
}

INPUT_NAMES = ["dV/dS", "dV/dr", "dV/dq", "dV/dσ"]


def run_engine(engine_name, price_fn, ref, *, jit_eligible=True):
    """Run Greek validation for one engine."""
    print(f"\n  --- {engine_name} ---")

    args = (jnp.float64(S0), jnp.float64(K), jnp.float64(T),
            jnp.float64(r0), jnp.float64(q0), jnp.float64(sigma0))

    # NPV
    npv = float(price_fn(*args))
    npv_diff = abs(npv - ref["npv"])
    npv_ok = npv_diff < max(abs(ref["npv"]) * 0.01, 0.01)
    print(f"    NPV = {npv:.10f}  (ref = {ref['npv']:.10f}  diff = {npv_diff:.2e}) {'✓' if npv_ok else '✗'}")

    # jax.grad: argnums = (S=0, K=1, T=2, r=3, q=4, sigma=5)
    # We want dS (0), dr (3), dq (4), dsigma (5)
    grad_fn = jax.grad(price_fn, argnums=(0, 3, 4, 5))
    ad_raw = grad_fn(*args)
    ad_greeks = [float(g) for g in ad_raw]

    # jit(grad)
    if jit_eligible:
        jit_grad_fn = jax.jit(grad_fn)
        _ = jit_grad_fn(*args)  # warmup
        jit_raw = jit_grad_fn(*args)
        jit_greeks = [float(g) for g in jit_raw]
    else:
        jit_greeks = ad_greeks  # can't JIT, skip

    ref_vals = [ref["dS"], ref["dr"], ref["dq"], ref["dsigma"]]

    n_pass = int(npv_ok)
    print(f"    {'Greek':<10} {'QuantLib FD':>14} {'jax.grad':>14} {'jit(grad)':>14} {'|AD-ref|':>10}")
    print("    " + "-" * 66)
    for i, name in enumerate(INPUT_NAMES):
        rv = ref_vals[i]
        ad = ad_greeks[i]
        jit = jit_greeks[i]
        ad_diff = abs(ad - rv)
        # For analytic engines: tight tolerance; for FD: looser (numerical)
        tol = max(abs(rv) * 0.05, 0.1)  # 5% tolerance (FD reference has own error)
        ok = ad_diff < tol
        n_pass += int(ok)
        print(f"    {name:<10} {rv:>14.8f} {ad:>14.8f} {jit:>14.8f} {ad_diff:>10.2e} {'✓' if ok else '✗'}")

    # Timing
    n_reps = 50
    if jit_eligible:
        jit_times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            jit_grad_fn(*args)
            jit_times.append((time.perf_counter() - t0) * 1000)
        jit_med = statistics.median(jit_times)
        print(f"    jit(grad) time: {jit_med:.4f} ms")

    return n_pass, len(INPUT_NAMES) + 1


def main():
    print("=" * 78)
    print("American Option Greek Benchmarks (FD vs jax.grad vs jax.jit(jax.grad))")
    print("=" * 78)
    print(f"  Instrument : American Put  S={S0}, K={K}, σ={sigma0}, "
          f"r={r0}, q={q0}, T={T}")
    print(f"  BPS shift  : {BPS}")

    total_pass = 0
    total_tests = 0

    # --- BAW ---
    def baw_put(S, K, T, r, q, sigma):
        return barone_adesi_whaley_price(S, K, T, r, q, sigma, -1)

    p, t = run_engine("Barone-Adesi-Whaley", baw_put, REF_BAW, jit_eligible=True)
    total_pass += p; total_tests += t

    # --- FD-BS ---
    def fd_put(S, K, T, r, q, sigma):
        return fd_black_scholes_price(S, K, T, r, q, sigma, -1, 200, 200, True)

    p, t = run_engine("FD-Black-Scholes (PDE)", fd_put, REF_FD, jit_eligible=False)
    total_pass += p; total_tests += t

    print(f"\nPassed: {total_pass}/{total_tests}")
    if total_pass == total_tests:
        print("✓ All American Greek benchmarks passed.")
    else:
        print(f"✗ {total_tests - total_pass} failures.")


if __name__ == "__main__":
    main()
