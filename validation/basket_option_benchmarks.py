"""Validation: Basket Option Benchmarks
Source: ~/quantlib-risk-py/benchmarks/basket_option_benchmarks.py

2-asset basket options via Stulz, moment-matching, and MC.
5 Greeks via jax.grad over 100 scenarios.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.engines.analytic.basket import (
    stulz_basket_price,
    moment_matching_basket_price,
    mc_european_basket,
)


def main():
    print("=" * 78)
    print("Basket Option Benchmarks")
    print("=" * 78)

    # 2-asset basket call
    S1, S2 = 100.0, 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q1, q2 = 0.02, 0.02
    sigma1, sigma2 = 0.20, 0.25
    rho = 0.5

    print(f"  S1={S1}, S2={S2}, K={K}, T={T}, r={r*100:.0f}%")
    print(f"  q1={q1}, q2={q2}, σ1={sigma1}, σ2={sigma2}, ρ={rho}")

    # === Test 1: Stulz 2-asset ===
    stulz = float(stulz_basket_price(S1, S2, K, T, r, q1, q2, sigma1, sigma2, rho))
    print(f"\n  Stulz (call on max):  {stulz:.4f}")

    # === Test 2: Moment-matching ===
    spots = jnp.array([S1, S2])
    divs = jnp.array([q1, q2])
    sigmas = jnp.array([sigma1, sigma2])
    corr = jnp.array([[1.0, rho], [rho, 1.0]])
    weights = jnp.array([0.5, 0.5])

    mm = float(moment_matching_basket_price(spots, K, T, r, divs, sigmas, corr, weights))
    print(f"  Moment-matching:      {mm:.4f}")

    # === Test 3: MC ===
    mc = float(mc_european_basket(spots, K, T, r, divs, sigmas, corr, weights,
                                   n_paths=200_000, seed=42))
    print(f"  MC (200k paths):      {mc:.4f}")

    # === Test 4: MC vs analytical agreement ===
    mc_mm_diff = abs(mc - mm) / mm
    print(f"  MC vs MM rel diff:    {mc_mm_diff:.4f}")

    # === Test 5: JAX AD Greeks via moment-matching ===
    # Greeks: dV/dS1, dV/dS2, dV/dsigma1, dV/dr, dV/drho
    def price_from_params(params):
        """params = [S1, S2, sigma1, sigma2, r, rho]"""
        s1, s2, sig1, sig2, rate, rh = params
        sp = jnp.array([s1, s2])
        d = jnp.array([q1, q2])
        sg = jnp.array([sig1, sig2])
        cr = jnp.array([[1.0, rh], [rh, 1.0]])
        return moment_matching_basket_price(sp, K, T, rate, d, sg, cr, weights)

    base_params = jnp.array([S1, S2, sigma1, sigma2, r, rho])
    grad_fn = jax.grad(price_from_params)
    greeks = np.array(grad_fn(base_params))

    greek_names = ['∂V/∂S1', '∂V/∂S2', '∂V/∂σ1', '∂V/∂σ2', '∂V/∂r', '∂V/∂ρ']
    print(f"\n  --- JAX AD Greeks (moment-matching) ---")
    for name, val in zip(greek_names, greeks):
        print(f"    {name:>8s}: {val:>10.6f}")

    # FD check
    H = 1e-5
    fd_greeks = np.zeros(6)
    for i in range(6):
        xp = base_params.at[i].add(H)
        xm = base_params.at[i].add(-H)
        fd_greeks[i] = (float(price_from_params(xp)) - float(price_from_params(xm))) / (2 * H)

    max_greek_diff = np.max(np.abs(greeks - fd_greeks) / np.maximum(np.abs(fd_greeks), 1e-10))
    print(f"  Max Greek rel diff (AD vs FD): {max_greek_diff:.2e}")

    # === Test 6: 100-scenario vmap ===
    key = jax.random.PRNGKey(0)
    scenarios = base_params + 0.01 * jax.random.normal(key, (100, 6))
    t0 = time.time()
    scenario_greeks = jax.vmap(grad_fn)(scenarios)
    t_vmap = time.time() - t0
    print(f"\n  vmap 100 scenarios: {t_vmap:.3f}s")
    print(f"  Greek shape: {scenario_greeks.shape}")

    # JIT + vmap
    jit_vmap_grad = jax.jit(jax.vmap(grad_fn))
    _ = jit_vmap_grad(scenarios)  # warmup
    t0 = time.time()
    _ = jit_vmap_grad(scenarios)
    t_jit = time.time() - t0
    print(f"  JIT+vmap 100 scenarios: {t_jit:.6f}s")

    # === Validation ===
    n_pass = 0
    total = 5

    ok = stulz > 0
    n_pass += int(ok)
    print(f"\n  Stulz positive:         {'✓' if ok else '✗'}")

    ok = mm > 0
    n_pass += int(ok)
    print(f"  Moment-matching pos:    {'✓' if ok else '✗'}")

    ok = mc_mm_diff < 0.05  # MC within 5% of analytical
    n_pass += int(ok)
    print(f"  MC vs MM (<5%):         {'✓' if ok else '✗'} ({mc_mm_diff*100:.1f}%)")

    ok = max_greek_diff < 1e-4
    n_pass += int(ok)
    print(f"  AD Greeks vs FD:        {'✓' if ok else '✗'} (max rel={max_greek_diff:.2e})")

    ok = t_jit < t_vmap
    n_pass += int(ok)
    print(f"  JIT speedup:            {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All basket option benchmarks passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
