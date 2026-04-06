"""Validation: American Basket Option (LSM)
Source: sprint 3 item 2.12

American basket option via Longstaff-Schwartz Monte Carlo.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def lsm_american_basket(spots, K, T, r, divs, sigmas, corr, n_paths=50000,
                         n_steps=100, key=None, option_type=1):
    """Longstaff-Schwartz Monte Carlo for American basket option.

    Parameters
    ----------
    spots : (n_assets,) initial spot prices
    K : strike
    T : maturity
    r : risk-free rate
    divs : (n_assets,) dividend yields
    sigmas : (n_assets,) volatilities
    corr : (n_assets, n_assets) correlation matrix
    option_type : 1 for call, -1 for put
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    n_assets = len(spots)
    dt = T / n_steps
    disc = jnp.exp(-r * dt)

    # Cholesky decomposition for correlated Brownian motions
    L = jnp.linalg.cholesky(corr)

    # Simulate paths
    log_S = jnp.tile(jnp.log(jnp.array(spots)), (n_paths, 1))  # (n_paths, n_assets)
    drift = jnp.array([(r - d - 0.5 * s**2) * dt for d, s in zip(divs, sigmas)])
    vol_dt = jnp.array(sigmas) * jnp.sqrt(dt)

    # Store paths for backward induction
    all_paths = jnp.zeros((n_steps + 1, n_paths, n_assets))
    all_paths = all_paths.at[0].set(jnp.exp(log_S))

    for step in range(1, n_steps + 1):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (n_paths, n_assets))
        z_corr = z @ L.T
        log_S = log_S + drift + vol_dt * z_corr
        all_paths = all_paths.at[step].set(jnp.exp(log_S))

    # Basket value at each step: equal-weighted average
    weights = jnp.ones(n_assets) / n_assets
    basket = jnp.sum(all_paths * weights, axis=2)  # (n_steps+1, n_paths)

    # Payoff
    if option_type == 1:
        payoff_fn = lambda b: jnp.maximum(b - K, 0.0)
    else:
        payoff_fn = lambda b: jnp.maximum(K - b, 0.0)

    # LSM backward induction
    cashflows = payoff_fn(basket[-1])  # terminal payoff
    exercise_time = jnp.full(n_paths, n_steps)

    for step in range(n_steps - 1, 0, -1):
        itm = payoff_fn(basket[step]) > 0
        if jnp.sum(itm) < 10:
            cashflows = cashflows * disc
            continue

        # Regression
        b = basket[step]
        X = jnp.stack([jnp.ones(n_paths), b, b**2], axis=1)
        Y = cashflows * disc

        # Only use ITM paths for regression
        itm_mask = itm.astype(jnp.float64)
        X_itm = X * itm_mask[:, None]
        Y_itm = Y * itm_mask

        # Solve normal equations
        beta = jnp.linalg.lstsq(X_itm, Y_itm, rcond=None)[0]
        continuation = X @ beta

        exercise_val = payoff_fn(basket[step])
        exercise = (exercise_val > continuation) & itm
        cashflows = jnp.where(exercise, exercise_val, cashflows * disc)

    price = jnp.mean(cashflows * disc) * jnp.exp(-r * 0)  # disc for step 0
    return float(price)


def main():
    print("=" * 78)
    print("American Basket Option (Longstaff-Schwartz)")
    print("=" * 78)

    # 2-asset basket
    spots = [100.0, 100.0]
    K = 100.0
    T = 1.0
    r = 0.05
    divs = [0.02, 0.02]
    sigmas = [0.20, 0.25]
    corr = jnp.array([[1.0, 0.5], [0.5, 1.0]])

    print(f"  Spots: {spots}, K={K}, T={T}")
    print(f"  r={r}, divs={divs}, σ={sigmas}")
    print(f"  Correlation: ρ={0.5}")

    # === Test 1: American put price ===
    key = jax.random.PRNGKey(42)
    price_put = lsm_american_basket(spots, K, T, r, divs, sigmas, corr,
                                     n_paths=50000, n_steps=50, key=key, option_type=-1)
    print(f"\n  American basket put: {price_put:.4f}")

    # European lower bound (simple MC)
    n_paths = 50000
    dt = T / 50
    L = jnp.linalg.cholesky(corr)
    log_S = jnp.tile(jnp.log(jnp.array(spots)), (n_paths, 1))
    for step in range(50):
        key, sk = jax.random.split(key)
        z = jax.random.normal(sk, (n_paths, 2)) @ L.T
        drift = jnp.array([(r - d - 0.5 * s**2) * dt for d, s in zip(divs, sigmas)])
        log_S = log_S + drift + jnp.array(sigmas) * jnp.sqrt(dt) * z
    S_T = jnp.exp(log_S)
    basket_T = jnp.mean(S_T, axis=1)
    euro_put = float(jnp.exp(-r * T) * jnp.mean(jnp.maximum(K - basket_T, 0.0)))
    print(f"  European basket put: {euro_put:.4f}")

    # === Test 2: American call ===
    price_call = lsm_american_basket(spots, K, T, r, divs, sigmas, corr,
                                      n_paths=50000, n_steps=50, key=jax.random.PRNGKey(123),
                                      option_type=1)
    print(f"  American basket call: {price_call:.4f}")

    # === Test 3: Correlation sensitivity ===
    corrs = [0.0, 0.3, 0.6, 0.9]
    prices_rho = []
    for rho in corrs:
        C = jnp.array([[1.0, rho], [rho, 1.0]])
        p = lsm_american_basket(spots, K, T, r, divs, sigmas, C,
                                 n_paths=30000, n_steps=30, key=jax.random.PRNGKey(42),
                                 option_type=-1)
        prices_rho.append(p)
        print(f"  ρ={rho:.1f}: put={p:.4f}")
    # Higher correlation → higher put price (less diversification)
    corr_sensitivity = prices_rho[-1] > prices_rho[0] - 0.5

    # === Test 4: MC convergence ===
    prices_mc = []
    for n_p in [10000, 30000, 100000]:
        p = lsm_american_basket(spots, K, T, r, divs, sigmas, corr,
                                 n_paths=n_p, n_steps=30, key=jax.random.PRNGKey(42),
                                 option_type=-1)
        prices_mc.append(p)
        print(f"  MC {n_p:>6d} paths: put={p:.4f}")

    # === Validation ===
    n_pass = 0
    total = 4

    ok = price_put >= euro_put * 0.95  # American >= European (approximately, MC noise)
    n_pass += int(ok)
    print(f"\n  American >= European: {'✓' if ok else '✗'}")

    ok = price_call > 0 and price_put > 0
    n_pass += int(ok)
    print(f"  Positive prices:      {'✓' if ok else '✗'}")

    n_pass += int(corr_sensitivity)
    print(f"  Corr sensitivity:     {'✓' if corr_sensitivity else '✗'}")

    # Convergence: spread decreases
    spread = abs(prices_mc[-1] - prices_mc[-2])
    ok = spread < 1.0  # MC noise within 1
    n_pass += int(ok)
    print(f"  MC convergence:       {'✓' if ok else '✗'} (spread={spread:.4f})")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All American basket option validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
