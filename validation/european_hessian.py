"""Validation: European Option Second-Order Sensitivities (jax.hessian)
Source: ~/quantlib-risk-py/benchmarks/second_order_european.py

Computes the full 4×4 Hessian matrix of a European call option
(gamma, vanna, volga, etc.) using:
  - jax.hessian (exact second derivatives)
  - FD-over-FD (bump-and-reprice on bump-and-reprice)
  - Validated against analytic BSM gamma, vanna, volga

Market data: S=7, K=8, σ=0.10, r=0.05, q=0.05, T=1Y
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import statistics
import math

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.engines.analytic.black_formula import black_scholes_price


# === Market data ===
S0 = 7.0
K = 8.0
T = 367.0 / 365.0
r0 = 0.05
q0 = 0.05
sigma0 = 0.10
H = 1e-5  # FD bump for Hessian

INPUT_NAMES = ["S (spot)", "q (div yield)", "σ (vol)", "r (rate)"]


# === Analytic second-order Greeks ===
def analytic_greeks_2nd(S, K_, r, q, sigma, T_):
    """Analytic gamma, vanna, volga for European call."""
    sqrtT = math.sqrt(T_)
    d1 = (math.log(S / K_) + (r - q + 0.5 * sigma**2) * T_) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf_d1 = math.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    gamma = math.exp(-q * T_) * pdf_d1 / (S * sigma * sqrtT)
    vanna = -math.exp(-q * T_) * pdf_d1 * d2 / sigma
    volga = S * math.exp(-q * T_) * pdf_d1 * sqrtT * d1 * d2 / sigma
    return gamma, vanna, volga


def bsm_price(S, q, sigma, r_rate):
    """Wrap for differentiation: inputs = (S, q, σ, r)."""
    return black_scholes_price(S, K, T, r_rate, q, sigma, 1)


def main():
    print("=" * 78)
    print("European Option 4×4 Hessian (jax.hessian vs FD vs Analytic)")
    print("=" * 78)
    print(f"  Instrument : European Call  S={S0}, K={K}, σ={sigma0}, "
          f"r={r0}, q={q0}, T={T:.4f}")
    print(f"  Engine     : Analytic BSM (black_scholes_price)")
    print()

    args = (jnp.float64(S0), jnp.float64(q0), jnp.float64(sigma0), jnp.float64(r0))

    # === 1. jax.hessian ===
    hessian_fn = jax.hessian(bsm_price, argnums=(0, 1, 2, 3))
    hess_raw = hessian_fn(*args)
    # hess_raw is a nested tuple: hess_raw[i][j] = d²f/dx_i dx_j
    N = 4
    hess_ad = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            hess_ad[i, j] = float(hess_raw[i][j])

    # === 2. FD-over-FD Hessian ===
    base_args = [S0, q0, sigma0, r0]

    def eval_price(vals):
        return float(bsm_price(*[jnp.float64(x) for x in vals]))

    hess_fd = np.zeros((N, N))
    f0 = eval_price(base_args)
    for i in range(N):
        for j in range(i, N):
            # (f(x+h_i+h_j) - f(x+h_i) - f(x+h_j) + f(x)) / (h_i * h_j)
            xpp = list(base_args); xpp[i] += H; xpp[j] += H
            xp0 = list(base_args); xp0[i] += H
            x0p = list(base_args); x0p[j] += H
            hess_fd[i, j] = (eval_price(xpp) - eval_price(xp0) - eval_price(x0p) + f0) / (H * H)
            hess_fd[j, i] = hess_fd[i, j]

    # === 3. jax.jit(jax.hessian) ===
    jit_hessian_fn = jax.jit(hessian_fn)
    _ = jit_hessian_fn(*args)  # warmup
    hess_jit_raw = jit_hessian_fn(*args)
    hess_jit = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            hess_jit[i, j] = float(hess_jit_raw[i][j])

    # === Print full Hessian ===
    print("  --- jax.hessian (4×4) ---")
    print(f"  {'':>14}", end="")
    for name in INPUT_NAMES:
        print(f" {name:>14}", end="")
    print()
    for i in range(N):
        print(f"  {INPUT_NAMES[i]:>14}", end="")
        for j in range(N):
            print(f" {hess_ad[i, j]:>14.8f}", end="")
        print()

    print()
    print("  --- FD-over-FD Hessian (4×4) ---")
    print(f"  {'':>14}", end="")
    for name in INPUT_NAMES:
        print(f" {name:>14}", end="")
    print()
    for i in range(N):
        print(f"  {INPUT_NAMES[i]:>14}", end="")
        for j in range(N):
            print(f" {hess_fd[i, j]:>14.8f}", end="")
        print()

    # === 4. Validate key entries against analytic ===
    gamma_a, vanna_a, volga_a = analytic_greeks_2nd(S0, K, r0, q0, sigma0, T)

    print()
    print("  --- Analytic cross-check ---")
    print(f"  {'Greek':<10} {'Analytic':>14} {'jax.hessian':>14} {'FD':>14} {'|AD-analytic|':>14}")
    print("  " + "-" * 70)

    n_pass = 0

    # Gamma = d²V/dS² → hess[0][0]
    gamma_ad = hess_ad[0, 0]
    gamma_fd = hess_fd[0, 0]
    g_diff = abs(gamma_ad - gamma_a)
    g_ok = g_diff < max(abs(gamma_a) * 1e-6, 1e-10)
    n_pass += int(g_ok)
    print(f"  {'Gamma':<10} {gamma_a:>14.10f} {gamma_ad:>14.10f} {gamma_fd:>14.10f} {g_diff:>14.2e} {'✓' if g_ok else '✗'}")

    # Vanna = d²V/dSdσ → hess[0][2]
    vanna_ad = hess_ad[0, 2]
    vanna_fd = hess_fd[0, 2]
    v_diff = abs(vanna_ad - vanna_a)
    v_ok = v_diff < max(abs(vanna_a) * 1e-6, 1e-10)
    n_pass += int(v_ok)
    print(f"  {'Vanna':<10} {vanna_a:>14.10f} {vanna_ad:>14.10f} {vanna_fd:>14.10f} {v_diff:>14.2e} {'✓' if v_ok else '✗'}")

    # Volga = d²V/dσ² → hess[2][2]
    volga_ad = hess_ad[2, 2]
    volga_fd = hess_fd[2, 2]
    vl_diff = abs(volga_ad - volga_a)
    vl_ok = vl_diff < max(abs(volga_a) * 1e-6, 1e-10)
    n_pass += int(vl_ok)
    print(f"  {'Volga':<10} {volga_a:>14.10f} {volga_ad:>14.10f} {volga_fd:>14.10f} {vl_diff:>14.2e} {'✓' if vl_ok else '✗'}")

    # Symmetry check: H[i,j] == H[j,i]
    max_asym = np.max(np.abs(hess_ad - hess_ad.T))
    sym_ok = max_asym < 1e-12
    n_pass += int(sym_ok)
    print(f"\n  Hessian symmetry: max |H[i,j]-H[j,i]| = {max_asym:.2e} {'✓' if sym_ok else '✗'}")

    # JIT consistency
    jit_diff = np.max(np.abs(hess_ad - hess_jit))
    jit_ok = jit_diff < 1e-12
    n_pass += int(jit_ok)
    print(f"  JIT consistency: max |H-H_jit| = {jit_diff:.2e} {'✓' if jit_ok else '✗'}")

    # === 5. Timing ===
    print()
    print("  --- Timing ---")
    n_reps = 100

    # FD-over-FD
    def fd_hess():
        f0 = eval_price(base_args)
        for i in range(N):
            for j in range(i, N):
                xpp = list(base_args); xpp[i] += H; xpp[j] += H
                xp0 = list(base_args); xp0[i] += H
                x0p = list(base_args); x0p[j] += H
                eval_price(xpp); eval_price(xp0); eval_price(x0p)

    fd_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fd_hess()
        fd_times.append((time.perf_counter() - t0) * 1000)

    # jax.hessian
    hess_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        hessian_fn(*args)
        hess_times.append((time.perf_counter() - t0) * 1000)

    # jit(hessian)
    jit_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        jit_hessian_fn(*args)
        jit_times.append((time.perf_counter() - t0) * 1000)

    fd_med = statistics.median(fd_times)
    hess_med = statistics.median(hess_times)
    jit_med = statistics.median(jit_times)
    print(f"    FD-over-FD (31 pricings)     : {fd_med:8.4f} ms")
    print(f"    jax.hessian                  : {hess_med:8.4f} ms  ({fd_med/hess_med:5.1f}x vs FD)")
    print(f"    jax.jit(jax.hessian)         : {jit_med:8.4f} ms  ({fd_med/jit_med:5.1f}x vs FD)")

    total = 5  # gamma + vanna + volga + symmetry + JIT
    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All European Hessian benchmarks passed.")
    else:
        print(f"✗ {total - n_pass} failures.")


if __name__ == "__main__":
    main()
