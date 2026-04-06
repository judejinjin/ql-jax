"""Validation: Swing Option Benchmarks
Source: sprint 3 item 3.4

Swing option Greeks via AD, vmap scenario batch, JIT speedup.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.engines.fd.swing import fd_swing_price


def main():
    print("=" * 78)
    print("Swing Option Benchmarks")
    print("=" * 78)

    S0 = 50.0
    K = 50.0
    T = 1.0
    r = 0.06
    q = 0.02
    sigma = 0.30

    # === Test 1: JIT compilation speedup ===
    @jax.jit
    def jit_price(s, vol):
        return fd_swing_price(s, K, T, r, q, vol, n_exercises=1, n_x=80, n_t=80)

    # Warmup
    _ = jit_price(S0, sigma)

    t0 = time.perf_counter()
    for _ in range(10):
        p = jit_price(S0, sigma)
    p.block_until_ready()
    t_jit = (time.perf_counter() - t0) / 10

    t0 = time.perf_counter()
    p = fd_swing_price(S0, K, T, r, q, sigma, n_exercises=1, n_x=80, n_t=80)
    t_raw = time.perf_counter() - t0

    speedup = t_raw / max(t_jit, 1e-12)
    print(f"\n  Raw time:  {t_raw*1000:.1f} ms")
    print(f"  JIT time:  {t_jit*1000:.3f} ms")
    print(f"  Speedup:   {speedup:.0f}x")

    # === Test 2: Price monotone in spot ===
    spots = jnp.array([35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0])
    prices = [float(jit_price(s, sigma)) for s in spots]
    print(f"\n  --- Prices across spots ---")
    for s, p in zip(spots, prices):
        print(f"    S={float(s):.1f}: price={p:.4f}")
    monotone_prices = all(prices[i] <= prices[i+1] + 0.01 for i in range(len(prices)-1))

    # === Test 3: Price monotone in vol ===
    vols = jnp.array([0.10, 0.20, 0.30, 0.40, 0.50])
    prices_vol = [float(jit_price(S0, v)) for v in vols]
    print(f"\n  --- Prices across vols ---")
    for v, p in zip(vols, prices_vol):
        print(f"    σ={float(v):.2f}: price={p:.4f}")
    monotone_vols = all(prices_vol[i] <= prices_vol[i+1] + 0.01 for i in range(len(prices_vol)-1))

    # === Test 4: AD delta via grad ===
    delta_fn = jax.grad(lambda s: fd_swing_price(s, K, T, r, q, sigma, n_exercises=1,
                                                   n_x=50, n_t=50))
    delta = float(delta_fn(S0))
    print(f"\n  Delta at S={S0}: {delta:.6f}")

    # === Validation ===
    n_pass = 0
    total = 4

    ok = speedup > 2
    n_pass += int(ok)
    print(f"\n  JIT speedup > 2x:   {'✓' if ok else '✗'} ({speedup:.0f}x)")

    n_pass += int(monotone_prices)
    print(f"  Monotone in spot:   {'✓' if monotone_prices else '✗'}")

    n_pass += int(monotone_vols)
    print(f"  Monotone in vol:    {'✓' if monotone_vols else '✗'}")

    ok = delta > 0
    n_pass += int(ok)
    print(f"  Positive delta:     {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All swing benchmarks passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
