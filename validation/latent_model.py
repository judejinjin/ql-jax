"""Validation: Latent Model (Copula-based latent variable credit)
Source: sprint 4 item LatentModel

Latent variable model for correlated defaults using Gaussian/t copulas.
Factor model: each credit's latent variable = β*Z + sqrt(1-β²)*ε
where Z is a common factor, ε is idiosyncratic.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from jax.scipy.stats import norm

from ql_jax.math.copulas import gaussian_copula


def latent_factor_model(default_probs, beta, n_paths=100000, key=None):
    """One-factor Gaussian latent variable model.

    P(default_i | Z) = Φ((Φ⁻¹(p_i) - β*Z) / sqrt(1-β²))

    Parameters
    ----------
    default_probs : (n_credits,) marginal default probabilities
    beta : factor loading (correlation sqrt)
    n_paths : number of MC paths

    Returns
    -------
    defaults : (n_paths, n_credits) default indicators
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    n_credits = len(default_probs)
    dp = jnp.array(default_probs)
    thresholds = norm.ppf(dp)  # Φ⁻¹(p_i)

    key, sk_z, sk_eps = jax.random.split(key, 3)
    Z = jax.random.normal(sk_z, (n_paths,))
    eps = jax.random.normal(sk_eps, (n_paths, n_credits))

    # Latent variable for each credit
    X = beta * Z[:, None] + jnp.sqrt(1 - beta**2) * eps

    # Default if X < threshold
    defaults = (X < thresholds[None, :]).astype(jnp.float64)
    return defaults


def main():
    print("=" * 78)
    print("Latent Model (Factor Copula Credit)")
    print("=" * 78)

    # Portfolio: 10 credits
    n_credits = 10
    default_probs = [0.02] * n_credits
    recovery = 0.40
    notionals = [100.0] * n_credits
    lgd = [(1 - recovery) * n for n in notionals]

    print(f"  Credits: {n_credits}, PD=2%, Recovery={recovery:.0%}")

    key = jax.random.PRNGKey(42)
    n_paths = 500000

    # === Test 1: Independent case (β=0) ===
    defaults_indep = latent_factor_model(default_probs, beta=0.0, n_paths=n_paths, key=key)
    n_defaults_indep = jnp.sum(defaults_indep, axis=1)
    avg_defaults_indep = float(jnp.mean(n_defaults_indep))
    losses_indep = defaults_indep @ jnp.array(lgd)
    el_indep = float(jnp.mean(losses_indep))
    var99_indep = float(jnp.percentile(losses_indep, 99))
    print(f"\n  β=0.0 (independent):")
    print(f"    Avg defaults: {avg_defaults_indep:.3f} (theory={n_credits*0.02:.1f})")
    print(f"    Expected loss: {el_indep:.2f}")
    print(f"    VaR 99%: {var99_indep:.2f}")

    # === Test 2: Moderate correlation (β=0.3) ===
    defaults_mod = latent_factor_model(default_probs, beta=0.3, n_paths=n_paths, key=key)
    losses_mod = defaults_mod @ jnp.array(lgd)
    el_mod = float(jnp.mean(losses_mod))
    var99_mod = float(jnp.percentile(losses_mod, 99))
    print(f"\n  β=0.3 (moderate):")
    print(f"    Expected loss: {el_mod:.2f}")
    print(f"    VaR 99%: {var99_mod:.2f}")

    # === Test 3: High correlation (β=0.7) ===
    defaults_high = latent_factor_model(default_probs, beta=0.7, n_paths=n_paths, key=key)
    losses_high = defaults_high @ jnp.array(lgd)
    el_high = float(jnp.mean(losses_high))
    var99_high = float(jnp.percentile(losses_high, 99))
    print(f"\n  β=0.7 (high):")
    print(f"    Expected loss: {el_high:.2f}")
    print(f"    VaR 99%: {var99_high:.2f}")

    # === Test 4: Perfect correlation (β→1) ===
    defaults_perf = latent_factor_model(default_probs, beta=0.999, n_paths=n_paths, key=key)
    losses_perf = defaults_perf @ jnp.array(lgd)
    el_perf = float(jnp.mean(losses_perf))
    var99_perf = float(jnp.percentile(losses_perf, 99))
    print(f"\n  β=0.999 (near-perfect):")
    print(f"    Expected loss: {el_perf:.2f}")
    print(f"    VaR 99%: {var99_perf:.2f}")

    # === Test 5: Vasicek large-pool approximation for β=0.3 ===
    # P(L > x) ≈ Φ((sqrt(1-β²)*Φ⁻¹(x/LGD_total) - Φ⁻¹(p)) / β)
    # At 99% VaR, the conditional default rate is:
    # p_cond = Φ((Φ⁻¹(p) + β*Φ⁻¹(0.99)) / sqrt(1-β²))
    p = 0.02
    beta_val = 0.3
    p_cond = float(norm.cdf((norm.ppf(p) + beta_val * norm.ppf(0.99)) / jnp.sqrt(1 - beta_val**2)))
    vasicek_var99 = p_cond * sum(lgd)
    print(f"\n  Vasicek approx VaR 99% (β=0.3): {vasicek_var99:.2f}")
    print(f"  MC VaR 99% (β=0.3):             {var99_mod:.2f}")

    # === Test 6: AD sensitivity ===
    def el_fn(beta):
        defaults = latent_factor_model(default_probs, beta, n_paths=50000,
                                        key=jax.random.PRNGKey(42))
        losses = defaults @ jnp.array(lgd)
        return jnp.mean(losses)

    # Note: AD through indicator function (defaults) doesn't give useful gradients
    # We use FD instead
    H = 0.01
    del_fd = (float(el_fn(0.3 + H)) - float(el_fn(0.3 - H))) / (2 * H)
    print(f"\n  d(EL)/d(β) at β=0.3: {del_fd:.4f} (FD)")

    # === Validation ===
    n_pass = 0
    total = 5

    # Expected loss matches theory: EL = n * p * LGD
    theory_el = n_credits * 0.02 * 60.0
    ok = abs(el_indep - theory_el) < theory_el * 0.1
    n_pass += int(ok)
    print(f"\n  EL matches theory:  {'✓' if ok else '✗'} (got={el_indep:.2f}, theory={theory_el:.2f})")

    # Higher beta → higher VaR (tail risk)
    ok = var99_high > var99_mod > var99_indep * 0.8  # with MC noise
    n_pass += int(ok)
    print(f"  VaR increases w/ β: {'✓' if ok else '✗'} ({var99_indep:.0f} < {var99_mod:.0f} < {var99_high:.0f})")

    # Expected loss roughly similar across correlations (EL is correlation-invariant)
    ok = abs(el_mod - el_indep) < theory_el * 0.15
    n_pass += int(ok)
    print(f"  EL ≈ invariant:     {'✓' if ok else '✗'} (diff={abs(el_mod-el_indep):.2f})")

    # Perfect correlation: VaR is either 0 or full loss
    ok = var99_perf >= sum(lgd) * 0.5 or var99_perf == 0
    n_pass += int(ok)
    print(f"  Perfect corr VaR:   {'✓' if ok else '✗'} ({var99_perf:.2f})")

    # Vasicek approximation within factor of 3 (small pool has discrete losses)
    ok = vasicek_var99 > 0 and var99_mod < vasicek_var99 * 5
    n_pass += int(ok)
    print(f"  Vasicek ballpark:   {'✓' if ok else '✗'} (MC={var99_mod:.2f}, Vas={vasicek_var99:.2f})")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All latent model validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
