"""Validation: American Option Full (all engines + all binomial tree types)
Source: ~/QuantLib-SWIG/Python/examples/american-option.py

Validates American put pricing across all available engines:
  - Barone-Adesi-Whaley (analytic approximation)
  - Finite Differences (Crank-Nicolson)
  - 5 binomial tree types: JR, CRR, Tian, Trigeorgis, Leisen-Reimer

Market data: S=36, K=40, r=0.06, q=0.0, σ=0.20, T=1.0, Put
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.engines.analytic.american import barone_adesi_whaley_price
from ql_jax.engines.fd.black_scholes import fd_black_scholes_price
from ql_jax.engines.lattice.binomial import binomial_price


# === Market data ===
S = 36.0
K = 40.0
r = 0.06
q = 0.0
sigma = 0.20
T = 1.0
OPTION_TYPE = -1  # Put
N_STEPS = 801

# === Reference values from QuantLib 1.42 ===
REFERENCE = {
    "BAW":          4.4596276138,  # Barone-Adesi-Whaley
    "FD-BS":        4.4845123129,  # FD Black-Scholes (200 time, 800 space)
    "JR":           4.4865524409,
    "CRR":          4.4864152079,
    "Tian":         4.4864131375,
    "Trigeorgis":   4.4864610652,
    "LeisenReimer": 4.4860756173,
}


def main():
    print("=" * 78)
    print("American Option Full Validation (all engines + tree types)")
    print("=" * 78)
    print(f"  Instrument : American Put  S={S}, K={K}, σ={sigma}, "
          f"r={r}, q={q}, T={T}")
    print(f"  Tree steps : {N_STEPS}")
    print()

    results = {}
    n_pass = 0

    # === BAW ===
    baw = float(barone_adesi_whaley_price(S, K, T, r, q, sigma, OPTION_TYPE))
    results["BAW"] = baw

    # === FD ===
    fd = float(fd_black_scholes_price(S, K, T, r, q, sigma, OPTION_TYPE,
                                       american=True,
                                       n_time=200, n_space=800))
    results["FD-BS"] = fd

    # === Binomial trees ===
    tree_map = {
        "JR": "jr",
        "CRR": "crr",
        "Tian": "tian",
        "Trigeorgis": "trigeorgis",
        "LeisenReimer": "leisen_reimer",
    }
    for label, tree_type in tree_map.items():
        price = float(binomial_price(S, K, T, r, q, sigma, OPTION_TYPE,
                                     n_steps=N_STEPS, tree_type=tree_type,
                                     american=True))
        results[label] = price

    # Print comparison
    print(f"{'Engine':<18} {'QuantLib':>14} {'ql-jax':>14} {'Diff':>12}")
    print("-" * 60)
    for key in REFERENCE:
        ref = REFERENCE[key]
        val = results[key]
        diff = abs(val - ref)
        tol = max(abs(ref) * 1e-3, 0.005)  # 0.1% or 0.5 cents
        status = "✓" if diff < tol else "✗"
        if diff < tol:
            n_pass += 1
        print(f"{key:<18} {ref:>14.8f} {val:>14.8f} {diff:>12.2e} {status}")

    # === Cross-engine consistency ===
    prices = list(results.values())
    spread = max(prices) - min(prices)
    print(f"\n  Price range across all engines: {min(prices):.8f} – {max(prices):.8f} (spread={spread:.6f})")
    consist_ok = spread < 0.05
    n_pass += int(consist_ok)
    print(f"  Cross-engine consistency (spread < 0.05): {'✓' if consist_ok else '✗'}")

    total = len(REFERENCE) + 1
    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All American option validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
