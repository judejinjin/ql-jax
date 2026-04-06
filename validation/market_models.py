"""Validation: Market Models (LMM)
Source: ~/QuantLib/Examples/MarketModels/

LIBOR Market Model: simulate forward rates, price caplets, compute Greeks.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.models.marketmodels.lmm import LMMConfig, simulate_lmm_paths, lmm_caplet_price


def main():
    print("=" * 78)
    print("Market Models (LIBOR Market Model)")
    print("=" * 78)

    n_rates = 5
    tau = jnp.full(n_rates, 0.5)
    tenors = jnp.arange(0.5, 0.5 * (n_rates + 1), 0.5)
    fwds = jnp.full(n_rates, 0.05)
    vols = jnp.full(n_rates, 0.20)
    corr = jnp.eye(n_rates)

    config = LMMConfig(
        n_rates=n_rates,
        tenors=tenors,
        tau=tau,
        initial_forwards=fwds,
        volatilities=vols,
        correlation=corr,
    )

    print(f"  n_rates={n_rates}, tau=0.5, fwd=5%, vol=20%")

    # === Test 1: Path simulation ===
    key = jax.random.PRNGKey(42)
    paths = simulate_lmm_paths(config, n_paths=10000, n_steps_per_period=10, key=key)
    print(f"\n  Paths shape: {paths.shape}")
    print(f"  Initial forwards: {[f'{float(v):.4f}' for v in paths[0, :, 0]]}")
    final_fwds = np.mean(np.array(paths[:, :, -1]), axis=0)
    print(f"  Mean final forwards: {[f'{v:.4f}' for v in final_fwds]}")

    # Forwards should remain ~5% on average (martingale property under right measure)
    fwd_drift_ok = np.max(np.abs(final_fwds - 0.05)) < 0.01

    # === Test 2: Caplet pricing ===
    # Skip i=0 — the forward at step 0 is just the initial value, so ATM payoff is 0
    K = 0.05
    prices = []
    indices = list(range(1, min(n_rates, 5)))
    for i in indices:
        price, stderr = lmm_caplet_price(config, i, K, n_paths=50000, key=key)
        prices.append(float(price))
        T = float(tenors[i])
        print(f"  Caplet i={i} T={T:.1f}Y: price={float(price):.6f} ± {float(stderr):.6f}")

    # Black reference for ATM caplet
    from jax.scipy.stats import norm
    for idx, i in enumerate(indices):
        T = float(tenors[i])
        vol = 0.20
        F = 0.05
        d1 = 0.5 * vol * np.sqrt(T)
        d2 = -d1
        df = np.exp(-0.05 * (T + 0.5))
        black_caplet = 0.5 * df * (F * float(norm.cdf(d1)) - K * float(norm.cdf(d2)))
        diff = abs(prices[idx] - black_caplet) / max(black_caplet, 1e-10)
        print(f"  Black ref i={i} T={T:.1f}Y: {black_caplet:.6f}  diff={diff:.2%}")

    # === Test 3: Positive prices ===
    all_positive = all(p > 0 for p in prices)

    # === Validation ===
    n_pass = 0
    total = 3

    ok = paths.shape[0] == 10000 and paths.shape[1] == n_rates
    n_pass += int(ok)
    print(f"\n  Path shape correct: {'✓' if ok else '✗'}")

    n_pass += int(all_positive)
    print(f"  Caplet prices pos:  {'✓' if all_positive else '✗'}")

    n_pass += int(fwd_drift_ok)
    print(f"  Forward drift ok:   {'✓' if fwd_drift_ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All market model validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
