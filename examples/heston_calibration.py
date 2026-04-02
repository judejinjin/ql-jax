"""Example: Heston Model Calibration.

Demonstrates:
  - Pricing under the Heston stochastic volatility model
  - Calibrating Heston parameters to a volatility surface
  - Computing model-implied volatilities
  - AD sensitivities of model prices to parameters
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.models.equity.heston import heston_model_prices, calibrate_heston


def main():
    S = 100.0
    r = 0.05
    q = 0.02

    print("=" * 60)
    print("QL-JAX Example: Heston Model Calibration")
    print("=" * 60)
    print(f"  Spot={S}, r={r:.2%}, q={q:.2%}")
    print()

    # ── True parameters ──────────────────────────────────────
    true_params = dict(
        v0=0.04,       # initial variance
        kappa=1.5,     # mean-reversion speed
        theta=0.04,    # long-run variance
        xi=0.3,        # vol-of-vol
        rho=-0.7,      # correlation
    )
    print("  True Heston parameters:")
    for k, v in true_params.items():
        print(f"    {k:>6s} = {v}")
    print()

    # ── Generate synthetic market prices ─────────────────────
    strike_vals = jnp.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])
    maturity_vals = jnp.array([0.25, 0.5, 1.0, 2.0])
    true_tuple = (true_params['v0'], true_params['kappa'], true_params['theta'],
                  true_params['xi'], true_params['rho'])

    # Build flat arrays for all (strike, maturity) combos
    all_strikes = []
    all_mats = []
    for T in maturity_vals:
        for K in strike_vals:
            all_strikes.append(float(K))
            all_mats.append(float(T))
    all_strikes = jnp.array(all_strikes)
    all_mats = jnp.array(all_mats)
    option_types = jnp.ones_like(all_strikes, dtype=jnp.int32)  # all calls

    market_prices = heston_model_prices(
        true_tuple, S, r, q, all_strikes, all_mats, option_types
    )

    print("  Strike × Maturity grid of model prices:")
    print(f"  {'Strike':>8s}", end="")
    for T in maturity_vals:
        print(f"  {float(T):.2f}y", end="")
    print()
    n_K = len(strike_vals)
    for i, K in enumerate(strike_vals):
        print(f"  {float(K):8.0f}", end="")
        for j in range(len(maturity_vals)):
            idx = j * n_K + i
            print(f"  {float(market_prices[idx]):7.4f}", end="")
        print()
    print()

    # ── Calibrate from perturbed initial guess ───────────────
    print("  Calibrating Heston model...")
    guess_tuple = (0.06, 2.0, 0.06, 0.4, -0.5)

    result = calibrate_heston(
        S=S, r=r, q=q,
        market_strikes=all_strikes,
        market_maturities=all_mats,
        market_prices=market_prices,
        option_types=option_types,
        initial_params=guess_tuple,
        max_iter=200,
    )

    param_names = ['v0', 'kappa', 'theta', 'xi', 'rho']
    cal_params = result.params if hasattr(result, 'params') else result
    print("  Calibrated parameters:")
    for k, name in enumerate(param_names):
        true_v = true_tuple[k]
        cal_v = float(cal_params[k])
        print(f"    {name:>6s} = {cal_v:.4f}  (true: {true_v:.4f})")
    print()

    # ── AD: Sensitivities of ATM 1Y price to parameters ─────
    print("  AD Sensitivities (ATM 1Y call):")

    def atm_price(params_tuple):
        prices = heston_model_prices(
            params_tuple, S, r, q,
            jnp.array([100.0]), jnp.array([1.0]), jnp.array([1]),
        )
        return prices[0]

    grad_fn = jax.grad(atm_price)
    grads = grad_fn(jnp.array(true_tuple))
    for name, g in zip(param_names, grads):
        print(f"    d(price)/d({name:>6s}) = {float(g):.6f}")


if __name__ == "__main__":
    main()
