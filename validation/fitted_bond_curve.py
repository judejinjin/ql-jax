"""Validation: Fitted Bond Discount Curve
Source: ~/QuantLib/Examples/FittedBondCurve/FittedBondCurve.cpp

Validates parametric curve fitting methods:
  - Nelson-Siegel: 4-parameter model of zero rates
  - Svensson: 6-parameter extension of Nelson-Siegel
  - Tests zero rate recovery at various maturities
  - Tests discount factor consistency: D(t) = exp(-z(t)*t)
  - JAX AD sensitivity: dRate/dParam via jax.grad
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.termstructures.yield_.fitted_bond_curve import NelsonSiegel, Svensson


# === Nelson-Siegel reference parameters and expected zero rates ===
# Standard NS params: b0=0.06, b1=-0.03, b2=0.02, tau=1.5
NS_PARAMS = [0.06, -0.03, 0.02, 1.5]

# Expected zero rates at various maturities (computed analytically)
# z(t) = b0 + (b1+b2)*(1-exp(-t/tau))/(t/tau) - b2*exp(-t/tau)
def ns_zero_rate_ref(t, b0, b1, b2, tau):
    x = t / tau
    factor = (1 - np.exp(-x)) / x
    return b0 + b1 * factor + b2 * (factor - np.exp(-x))


NS_MATURITIES = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
NS_RATES = {t: ns_zero_rate_ref(t, *NS_PARAMS) for t in NS_MATURITIES}

# === Svensson reference parameters ===
SV_PARAMS = [0.06, -0.03, 0.02, 0.01, 1.5, 5.0]

def sv_zero_rate_ref(t, b0, b1, b2, b3, tau1, tau2):
    x1 = t / tau1
    x2 = t / tau2
    f1 = (1 - np.exp(-x1)) / x1
    f2 = (1 - np.exp(-x2)) / x2
    return b0 + b1 * f1 + b2 * (f1 - np.exp(-x1)) + b3 * (f2 - np.exp(-x2))

SV_RATES = {t: sv_zero_rate_ref(t, *SV_PARAMS) for t in NS_MATURITIES}


def main():
    print("=" * 78)
    print("Fitted Bond Curve Validation (Nelson-Siegel + Svensson)")
    print("=" * 78)

    n_pass = 0
    total = 0

    # === 1. Nelson-Siegel zero rates ===
    ns = NelsonSiegel(NS_PARAMS)
    print(f"\n  --- Nelson-Siegel (b0={NS_PARAMS[0]}, b1={NS_PARAMS[1]}, "
          f"b2={NS_PARAMS[2]}, tau={NS_PARAMS[3]}) ---")
    print(f"  {'Maturity':>10} {'Reference':>12} {'ql-jax':>12} {'Diff':>12}")
    print("  " + "-" * 48)

    for t in NS_MATURITIES:
        total += 1
        ref = NS_RATES[t]
        val = float(ns.zero_rate(jnp.float64(t)))
        diff = abs(val - ref)
        ok = diff < 1e-12
        n_pass += int(ok)
        print(f"  {t:>10.2f} {ref:>12.8f} {val:>12.8f} {diff:>12.2e} {'✓' if ok else '✗'}")

    # === 2. Nelson-Siegel discount factor consistency ===
    print(f"\n  --- NS Discount factor consistency: D(t) = exp(-z(t)*t) ---")
    for t in [1.0, 5.0, 10.0]:
        total += 1
        z = float(ns.zero_rate(jnp.float64(t)))
        df = float(ns.discount(jnp.float64(t)))
        expected_df = np.exp(-z * t)
        diff = abs(df - expected_df)
        ok = diff < 1e-14
        n_pass += int(ok)
        print(f"  t={t}: D(t)={df:.10f}, exp(-z*t)={expected_df:.10f}, diff={diff:.2e} {'✓' if ok else '✗'}")

    # === 3. Svensson zero rates ===
    sv = Svensson(SV_PARAMS)
    print(f"\n  --- Svensson (b0={SV_PARAMS[0]}, b1={SV_PARAMS[1]}, "
          f"b2={SV_PARAMS[2]}, b3={SV_PARAMS[3]}, tau1={SV_PARAMS[4]}, tau2={SV_PARAMS[5]}) ---")
    print(f"  {'Maturity':>10} {'Reference':>12} {'ql-jax':>12} {'Diff':>12}")
    print("  " + "-" * 48)

    for t in NS_MATURITIES:
        total += 1
        ref = SV_RATES[t]
        val = float(sv.zero_rate(jnp.float64(t)))
        diff = abs(val - ref)
        ok = diff < 1e-12
        n_pass += int(ok)
        print(f"  {t:>10.2f} {ref:>12.8f} {val:>12.8f} {diff:>12.2e} {'✓' if ok else '✗'}")

    # === 4. NS monotonicity: long rate → b0 ===
    total += 1
    long_rate = float(ns.zero_rate(jnp.float64(100.0)))
    ok = abs(long_rate - NS_PARAMS[0]) < 0.001  # NS converges slowly
    n_pass += int(ok)
    print(f"\n  NS long rate z(100) ≈ b0={NS_PARAMS[0]}: {long_rate:.8f} (diff={abs(long_rate-NS_PARAMS[0]):.6f}) {'✓' if ok else '✗'}")

    # === 5. SV monotonicity ===
    total += 1
    long_rate_sv = float(sv.zero_rate(jnp.float64(100.0)))
    ok = abs(long_rate_sv - SV_PARAMS[0]) < 0.001
    n_pass += int(ok)
    print(f"  SV long rate z(100) ≈ b0={SV_PARAMS[0]}: {long_rate_sv:.8f} (diff={abs(long_rate_sv-SV_PARAMS[0]):.6f}) {'✓' if ok else '✗'}")

    # === 6. JAX AD: dz/db0, dz/db1, etc. ===
    print(f"\n  --- JAX AD: dz(5Y)/d(params) ---")

    def ns_rate_fn(params):
        model = NelsonSiegel(params)
        return model.zero_rate(jnp.float64(5.0))

    params_jax = jnp.array(NS_PARAMS, dtype=jnp.float64)
    grad_ns = jax.grad(ns_rate_fn)(params_jax)

    # FD check
    h = 1e-7
    base = float(ns_rate_fn(params_jax))
    fd_grads = []
    for i in range(4):
        p_up = params_jax.at[i].add(h)
        fd_grads.append((float(ns_rate_fn(p_up)) - base) / h)

    param_names = ["b0", "b1", "b2", "tau"]
    print(f"  {'Param':<6} {'jax.grad':>12} {'FD':>12} {'|diff|':>12}")
    print("  " + "-" * 44)
    for i, name in enumerate(param_names):
        total += 1
        ad = float(grad_ns[i])
        fd = fd_grads[i]
        diff = abs(ad - fd)
        ok = diff < max(abs(fd) * 1e-4, 1e-8)
        n_pass += int(ok)
        print(f"  {name:<6} {ad:>12.8f} {fd:>12.8f} {diff:>12.2e} {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All fitted bond curve validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
