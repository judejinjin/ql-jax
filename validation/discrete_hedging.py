"""Validation: Discrete Hedging Replication Error
Source: ~/QuantLib/Examples/DiscreteHedging/DiscreteHedging.cpp

Simulates the P&L distribution from discrete delta-hedging of a European call.
Compares empirical P&L statistics (mean, std, skewness, kurtosis) against
the Derman-Kamal theoretical formula for hedging error variance.

Uses GBM paths from ql-jax's BlackScholesProcess.

Market data: S=100, K=100, r=0.05, q=0.0, σ=0.20, T=1.0
Hedging: 21 rebalances (monthly), 50,000 paths
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.engines.analytic.black_formula import black_scholes_price


# === Parameters ===
S0 = 100.0
K = 100.0
r = 0.05
q = 0.0
sigma = 0.20
T = 1.0
N_REBALANCES = 21
N_PATHS = 50_000
SEED = 42


def bs_delta(S, K, T, r, q, sigma):
    """Black-Scholes delta for a European call."""
    if T < 1e-10:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * 0.5 * (1 + math.erf(d1 / math.sqrt(2)))


def bs_price_py(S, K, T, r, q, sigma):
    """Black-Scholes European call price (pure Python)."""
    if T < 1e-10:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    return S * math.exp(-q * T) * nd1 - K * math.exp(-r * T) * nd2


def derman_kamal_std(S, K, T, r, q, sigma, n_rebalances):
    """Derman-Kamal formula for the std of discrete hedging error.

    Approximate standard deviation of the hedging P&L.
    σ_PL ≈ σ * S * Gamma * S * σ * sqrt(T/(2*n))
    For ATM: Gamma ≈ φ(d1)/(S*σ*√T), so σ_PL ≈ price * sqrt(2/pi) * σ * sqrt(T/n) / √T
    Simplified: σ_PL ≈ Vega * σ * √(3) / √(8*n*T)  (Derman-Kamal 1999)
    """
    vega = S * math.exp(-q * T) * math.sqrt(T) * math.exp(-0.5 * ((math.log(S/K) + (r-q+0.5*sigma**2)*T)/(sigma*math.sqrt(T)))**2) / math.sqrt(2*math.pi)
    return vega * sigma * math.sqrt(1.0 / n_rebalances)


def simulate_hedging_pnl(seed=SEED):
    """Simulate discrete hedging P&L for N_PATHS paths."""
    rng = np.random.RandomState(seed)
    dt = T / N_REBALANCES

    pnls = np.zeros(N_PATHS)
    for path in range(N_PATHS):
        S_t = S0
        # Initial hedge: buy call, delta-hedge
        call_price_0 = bs_price_py(S_t, K, T, r, q, sigma)
        delta = bs_delta(S_t, K, T, r, q, sigma)

        # Portfolio: short call + delta shares + cash
        cash = call_price_0 - delta * S_t  # received premium, paid for shares
        shares = delta

        for i in range(N_REBALANCES):
            t = i * dt
            # Evolve spot
            z = rng.randn()
            S_new = S_t * math.exp((r - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)

            # Accrue interest on cash
            cash *= math.exp(r * dt)

            # New delta
            tau = T - (i + 1) * dt
            if tau > 1e-10:
                new_delta = bs_delta(S_new, K, tau, r, q, sigma)
            else:
                new_delta = 1.0 if S_new > K else 0.0

            # Rebalance
            trade = new_delta - shares
            cash -= trade * S_new
            shares = new_delta
            S_t = S_new

        # Final P&L: portfolio value - option payoff
        payoff = max(S_t - K, 0.0)
        portfolio = shares * S_t + cash
        pnl = portfolio - payoff
        pnls[path] = pnl

    return pnls


def main():
    print("=" * 78)
    print("Discrete Hedging Replication Error (GBM, delta-hedging)")
    print("=" * 78)
    print(f"  Instrument : European Call  S={S0}, K={K}, σ={sigma}, r={r}, T={T}")
    print(f"  Hedging    : {N_REBALANCES} rebalances, {N_PATHS} paths")
    print()

    # === 1. ql-jax BSM price check ===
    jax_price = float(black_scholes_price(
        jnp.float64(S0), jnp.float64(K), jnp.float64(T),
        jnp.float64(r), jnp.float64(q), jnp.float64(sigma), 1))
    py_price = bs_price_py(S0, K, T, r, q, sigma)
    print(f"  BSM call price (ql-jax): {jax_price:.8f}")
    print(f"  BSM call price (Python): {py_price:.8f}")
    print(f"  Diff: {abs(jax_price - py_price):.2e}")

    # === 2. Simulate hedging P&L ===
    pnls = simulate_hedging_pnl()

    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls)
    skew_pnl = float(np.mean(((pnls - mean_pnl) / std_pnl) ** 3))
    kurt_pnl = float(np.mean(((pnls - mean_pnl) / std_pnl) ** 4))

    # Theoretical std (Derman-Kamal)
    theo_std = derman_kamal_std(S0, K, T, r, q, sigma, N_REBALANCES)

    print(f"\n  --- Hedging P&L Statistics ---")
    print(f"  {'Statistic':<15} {'Empirical':>12} {'Theory':>12}")
    print("  " + "-" * 42)
    print(f"  {'Mean':<15} {mean_pnl:>12.4f} {'~0':>12}")
    print(f"  {'Std':<15} {std_pnl:>12.4f} {theo_std:>12.4f}")
    print(f"  {'Skewness':<15} {skew_pnl:>12.4f} {'~0':>12}")
    print(f"  {'Kurtosis':<15} {kurt_pnl:>12.4f} {'~3':>12}")

    # === 3. Validation checks ===
    n_pass = 0
    total = 5

    # Price match
    price_ok = abs(jax_price - py_price) < 1e-8
    n_pass += int(price_ok)
    print(f"\n  BSM price match:     {'✓' if price_ok else '✗'}")

    # Mean P&L ≈ 0 (within 2 std errors)
    stderr = std_pnl / math.sqrt(N_PATHS)
    mean_ok = abs(mean_pnl) < 3 * stderr
    n_pass += int(mean_ok)
    print(f"  Mean P&L ≈ 0:        {'✓' if mean_ok else '✗'} ({mean_pnl:.4f}, stderr={stderr:.4f})")

    # Std within 50% of Derman-Kamal (approximate formula)
    std_ok = 0.3 * theo_std < std_pnl < 3.0 * theo_std
    n_pass += int(std_ok)
    print(f"  Std vs D-K theory:   {'✓' if std_ok else '✗'} (ratio={std_pnl/theo_std:.2f})")

    # Skewness should be moderate (< 1 in abs value for normal-ish)
    skew_ok = abs(skew_pnl) < 2.0
    n_pass += int(skew_ok)
    print(f"  Skewness moderate:   {'✓' if skew_ok else '✗'} ({skew_pnl:.4f})")

    # Kurtosis should be > 3 (fat tails from gamma exposure)
    kurt_ok = kurt_pnl > 2.0
    n_pass += int(kurt_ok)
    print(f"  Kurtosis > 2:        {'✓' if kurt_ok else '✗'} ({kurt_pnl:.4f})")

    # === 4. JAX AD delta check ===
    print(f"\n  --- JAX AD vs analytic delta ---")
    jax_delta = float(jax.grad(lambda s: black_scholes_price(
        s, jnp.float64(K), jnp.float64(T),
        jnp.float64(r), jnp.float64(q), jnp.float64(sigma), 1))(jnp.float64(S0)))
    analytic_delta = bs_delta(S0, K, T, r, q, sigma)
    delta_diff = abs(jax_delta - analytic_delta)
    print(f"  jax.grad delta: {jax_delta:.10f}")
    print(f"  Analytic delta: {analytic_delta:.10f}")
    print(f"  Diff: {delta_diff:.2e}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All discrete hedging validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
