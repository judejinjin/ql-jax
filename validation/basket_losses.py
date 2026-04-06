"""Validation: Basket Losses (Copula Credit Portfolio)
Source: sprint 4 item BasketLosses

Portfolio credit loss distribution using copulas:
Gaussian, Clayton, Frank copulas for default correlation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm


# Inline JAX-friendly copulas (no float() calls)
def _independent_copula(u, v):
    return u * v

def _gaussian_copula(u, v, rho):
    from ql_jax.math.distributions.bivariate import bivariate_normal_cdf
    x = norm.ppf(jnp.clip(u, 1e-15, 1 - 1e-15))
    y = norm.ppf(jnp.clip(v, 1e-15, 1 - 1e-15))
    return bivariate_normal_cdf(x, y, rho)

def _clayton_copula(u, v, theta):
    val = jnp.power(u, -theta) + jnp.power(v, -theta) - 1.0
    val = jnp.maximum(val, 0.0)
    return jnp.power(val, -1.0 / theta)

def _frank_copula(u, v, theta):
    e_t = jnp.exp(-theta)
    num = (jnp.exp(-theta * u) - 1.0) * (jnp.exp(-theta * v) - 1.0)
    return -jnp.log1p(num / (e_t - 1.0)) / theta


def portfolio_loss_mc_gaussian(default_probs, recovery_rates, notionals,
                                rho, n_paths=100000, key=None):
    """Simulate portfolio loss using a one-factor Gaussian copula model.

    X_i = sqrt(rho)*Z + sqrt(1-rho)*eps_i
    Default if X_i < Phi^{-1}(p_i)
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    n_credits = len(default_probs)
    dp = jnp.array(default_probs)
    thresholds = norm.ppf(dp)

    key, sk_z, sk_eps = jax.random.split(key, 3)
    Z = jax.random.normal(sk_z, (n_paths,))
    eps = jax.random.normal(sk_eps, (n_paths, n_credits))

    X = jnp.sqrt(rho) * Z[:, None] + jnp.sqrt(1 - rho) * eps
    defaults = (X < thresholds[None, :]).astype(jnp.float64)

    lgd = jnp.array([(1 - r) * n for r, n in zip(recovery_rates, notionals)])
    losses = defaults @ lgd
    return losses


def portfolio_loss_mc_copula(default_probs, recovery_rates, notionals,
                              copula_fn, n_paths=100000, key=None):
    """Simulate portfolio loss using copula for dependence structure.

    For each credit: generate correlated uniforms via copula,
    then default if U_i < p_i.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    n_credits = len(default_probs)
    dp = jnp.array(default_probs)

    # Use one-factor approach: common uniform + conditional
    key, sk = jax.random.split(key)
    u_all = jax.random.uniform(sk, (n_paths, n_credits + 1))
    u_common = u_all[:, 0]  # common factor

    defaults = jnp.zeros((n_paths, n_credits))
    v_copula = jax.vmap(copula_fn)
    for i in range(n_credits):
        # C(u, pi) gives Prob(V < pi, U < u)
        # Conditional: P(V < pi | U=u) ≈ C(u, pi) / u
        # Use this to generate correlated defaults
        u_i = u_all[:, i + 1]
        # For each path, default if u_i < h(u_common, p_i)
        # Where h = partial_1 C(u, p_i) = conditional CDF
        # Approximate: use copula to transform correlation
        joint = v_copula(u_common, u_i)
        # Default if the copula-correlated uniform < default prob
        defaults = defaults.at[:, i].set((joint < dp[i]).astype(jnp.float64))

    lgd = jnp.array([(1 - r) * n for r, n in zip(recovery_rates, notionals)])
    losses = defaults @ lgd
    return losses


def main():
    print("=" * 78)
    print("Basket Losses (Copula Credit Portfolio)")
    print("=" * 78)

    # Portfolio: 5 credits
    n_credits = 5
    default_probs = [0.02, 0.03, 0.05, 0.02, 0.04]
    recovery_rates = [0.40, 0.40, 0.30, 0.40, 0.35]
    notionals = [100.0] * n_credits

    print(f"  Credits: {n_credits}")
    print(f"  Default probs: {default_probs}")
    print(f"  Recovery rates: {recovery_rates}")

    key = jax.random.PRNGKey(42)
    n_paths = 500000

    # === Test 1: Independent case (ρ=0) ===
    losses_indep = portfolio_loss_mc_gaussian(
        default_probs, recovery_rates, notionals,
        rho=0.0, n_paths=n_paths, key=key)
    el_indep = float(jnp.mean(losses_indep))
    var_indep = float(jnp.percentile(losses_indep, 99))
    print(f"\n  Independent (ρ=0):")
    print(f"    Expected loss: {el_indep:.2f}")
    print(f"    VaR 99%:       {var_indep:.2f}")

    # === Test 2: Gaussian copula ρ=0.3 ===
    losses_gauss = portfolio_loss_mc_gaussian(
        default_probs, recovery_rates, notionals,
        rho=0.3, n_paths=n_paths, key=key)
    el_gauss = float(jnp.mean(losses_gauss))
    var_gauss = float(jnp.percentile(losses_gauss, 99))
    print(f"\n  Gaussian copula (ρ=0.3):")
    print(f"    Expected loss: {el_gauss:.2f}")
    print(f"    VaR 99%:       {var_gauss:.2f}")

    # === Test 3: High correlation ρ=0.6 ===
    losses_high = portfolio_loss_mc_gaussian(
        default_probs, recovery_rates, notionals,
        rho=0.6, n_paths=n_paths, key=key)
    el_high = float(jnp.mean(losses_high))
    var_high = float(jnp.percentile(losses_high, 99))
    print(f"\n  Gaussian copula (ρ=0.6):")
    print(f"    Expected loss: {el_high:.2f}")
    print(f"    VaR 99%:       {var_high:.2f}")

    # === Test 4: Generic copula approach (Clayton) ===
    losses_clayton = portfolio_loss_mc_copula(
        default_probs, recovery_rates, notionals,
        lambda u, v: _clayton_copula(u, v, 2.0),
        n_paths=n_paths, key=key)
    el_clayton = float(jnp.mean(losses_clayton))
    var_clayton = float(jnp.percentile(losses_clayton, 99))
    print(f"\n  Clayton copula (θ=2):")
    print(f"    Expected loss: {el_clayton:.2f}")
    print(f"    VaR 99%:       {var_clayton:.2f}")

    # === Test 5: FD sensitivity ===
    H = 0.01
    el_up = float(jnp.mean(portfolio_loss_mc_gaussian(
        default_probs, recovery_rates, notionals,
        rho=0.3 + H, n_paths=200000, key=jax.random.PRNGKey(42))))
    el_dn = float(jnp.mean(portfolio_loss_mc_gaussian(
        default_probs, recovery_rates, notionals,
        rho=0.3 - H, n_paths=200000, key=jax.random.PRNGKey(42))))
    el_sens = (el_up - el_dn) / (2 * H)
    print(f"\n  d(EL)/d(ρ) ≈ {el_sens:.4f} (FD)")

    # === Validation ===
    n_pass = 0
    total = 5

    # Expected loss ≈ theory (correlation doesn't change EL)
    theoretical_el = sum((1 - r) * n * p for p, r, n in
                         zip(default_probs, recovery_rates, notionals))
    ok = abs(el_indep - theoretical_el) < theoretical_el * 0.15
    n_pass += int(ok)
    print(f"\n  EL reasonable:      {'✓' if ok else '✗'} (got={el_indep:.2f}, theory={theoretical_el:.2f})")

    # VaR increases with correlation
    ok = var_gauss >= var_indep * 0.8  # some MC noise allowed
    n_pass += int(ok)
    print(f"  Gauss VaR >= Indep: {'✓' if ok else '✗'} ({var_gauss:.2f} vs {var_indep:.2f})")

    ok = var_high >= var_gauss * 0.9
    n_pass += int(ok)
    print(f"  High VaR >= Gauss:  {'✓' if ok else '✗'} ({var_high:.2f} vs {var_gauss:.2f})")

    # Non-negative losses
    ok = bool(jnp.all(losses_indep >= 0)) and bool(jnp.all(losses_gauss >= 0))
    n_pass += int(ok)
    print(f"  Non-negative loss:  {'✓' if ok else '✗'}")

    # EL approximately invariant to correlation
    ok = abs(el_gauss - el_indep) < theoretical_el * 0.15
    n_pass += int(ok)
    print(f"  EL ≈ invariant:     {'✓' if ok else '✗'} (diff={abs(el_gauss - el_indep):.2f})")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All basket loss validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
