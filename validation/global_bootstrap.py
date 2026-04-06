"""Validation: Global Bootstrap
Source: ~/quantlib-risk-py/benchmarks/global_bootstrap.py

Globally calibrate a yield curve by optimizing zero rates to reprice
all rate helpers simultaneously (BFGS). Vs sequential bootstrap.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.time.date import Date, _advance_date, TimeUnit
from ql_jax.termstructures.yield_.rate_helpers import (
    DepositRateHelper, SwapRateHelper,
)
from ql_jax.termstructures.yield_.piecewise import PiecewiseYieldCurve
from ql_jax.math.optimization.bfgs import minimize as bfgs_minimize


def main():
    print("=" * 78)
    print("Global Bootstrap (BFGS optimisation)")
    print("=" * 78)

    ref_date = Date(15, 1, 2024)

    # Helpers
    dep_tenors = [3, 6]  # months
    dep_rates = [0.0280, 0.0310]
    swap_tenors = [12, 24, 60]  # months
    swap_rates = [0.0330, 0.0350, 0.0380]

    helpers = []
    for m, r in zip(dep_tenors, dep_rates):
        pillar = _advance_date(ref_date, m, TimeUnit.Months)
        helpers.append(DepositRateHelper(quote=r, pillar_date=pillar,
                                          start_date=ref_date, end_date=pillar))
    for m, r in zip(swap_tenors, swap_rates):
        pillar = _advance_date(ref_date, m, TimeUnit.Months)
        helpers.append(SwapRateHelper(quote=r, pillar_date=pillar,
                                       start_date=ref_date, tenor_months=m))

    n_helpers = len(helpers)
    print(f"  Helpers: {len(dep_rates)} deposits + {len(swap_rates)} swaps")

    # Pillar times
    from ql_jax.termstructures.yield_.base import YieldTermStructure
    dummy_ts = YieldTermStructure(ref_date, 'Actual365Fixed')
    pillar_times = jnp.array(sorted(set(
        float(dummy_ts.time_from_reference(h.pillar_date)) for h in helpers
    )))
    print(f"  Pillar times: {[f'{t:.4f}' for t in pillar_times]}")

    # === Sequential bootstrap (reference) ===
    t0 = time.time()
    seq_curve = PiecewiseYieldCurve(ref_date, helpers)
    t_seq = time.time() - t0

    seq_dfs = np.array([float(seq_curve.discount(t)) for t in pillar_times])
    print(f"\n  Sequential bootstrap: {t_seq:.3f}s")
    for i, t in enumerate(pillar_times):
        print(f"    t={float(t):.4f}: DF={seq_dfs[i]:.8f}")

    # === Global bootstrap via BFGS ===
    # Build a JAX-compatible objective without float() calls
    from ql_jax.time.daycounter import year_fraction

    # Pre-compute helper parameters (all static - no tracing needed)
    helper_params = []
    for h in helpers:
        t_end = float(dummy_ts.time_from_reference(h.pillar_date))
        if isinstance(h, DepositRateHelper):
            t_start = float(dummy_ts.time_from_reference(h.start_date))
            tau = year_fraction(h.start_date, h.end_date, h.day_counter)
            helper_params.append(('dep', t_start, t_end, tau, h.quote))
        elif isinstance(h, SwapRateHelper):
            t_start = float(dummy_ts.time_from_reference(h.start_date))
            n_periods = h.tenor_months // h.fixed_leg_frequency_months
            pay_times = []
            taus = []
            prev = h.start_date
            for i in range(1, n_periods + 1):
                payment = _advance_date(h.start_date,
                                        i * h.fixed_leg_frequency_months,
                                        TimeUnit.Months)
                tau = year_fraction(prev, payment, h.fixed_day_counter)
                pay_times.append(float(dummy_ts.time_from_reference(payment)))
                taus.append(tau)
                prev = payment
            end = _advance_date(h.start_date, h.tenor_months, TimeUnit.Months)
            t_end_swap = float(dummy_ts.time_from_reference(end))
            helper_params.append(('swap', t_start, t_end_swap,
                                  pay_times, taus, h.quote))

    def objective(zero_rates):
        """Sum of squared repricing errors (JAX-compatible)."""
        dfs = jnp.exp(-zero_rates * pillar_times)

        def disc(t):
            return jnp.exp(-jnp.interp(t, pillar_times, zero_rates) * t)

        total = jnp.float64(0.0)
        for hp in helper_params:
            if hp[0] == 'dep':
                _, t_start, t_end, tau, quote = hp
                impl = (disc(t_start) / disc(t_end) - 1.0) / tau
                total = total + (impl - quote)**2
            else:
                _, t_start, t_end_swap, pay_times, taus, quote = hp
                annuity = jnp.float64(0.0)
                for pt, tau in zip(pay_times, taus):
                    annuity = annuity + tau * disc(pt)
                impl = (disc(t_start) - disc(t_end_swap)) / annuity
                total = total + (impl - quote)**2
        return total

    x0 = jnp.full(len(pillar_times), 0.03)
    t0 = time.time()
    result = bfgs_minimize(objective, x0, max_iter=200, tol=1e-14)
    t_glob = time.time() - t0

    opt_rates = result['x']
    glob_dfs = np.array(jnp.exp(-opt_rates * pillar_times))

    print(f"\n  Global bootstrap (BFGS): {t_glob:.3f}s, {result['n_iter']} iterations")
    for i, t in enumerate(pillar_times):
        print(f"    t={float(t):.4f}: DF={glob_dfs[i]:.8f}  zr={float(opt_rates[i])*100:.4f}%")

    # === Test 1: Global repricing accuracy ===
    max_glob_err = 0.0
    print(f"\n  --- Global repricing ---")

    def disc_glob(t):
        return jnp.exp(-jnp.interp(t, pillar_times, opt_rates) * t)

    for hp in helper_params:
        if hp[0] == 'dep':
            _, t_start, t_end, tau, quote = hp
            impl = float((disc_glob(t_start) / disc_glob(t_end) - 1.0) / tau)
        else:
            _, t_start, t_end_swap, pay_times, taus_list, quote = hp
            annuity = sum(tau * float(disc_glob(pt)) for pt, tau in zip(pay_times, taus_list))
            impl = float((disc_glob(t_start) - disc_glob(t_end_swap)) / annuity)
        err = abs(impl - quote)
        max_glob_err = max(max_glob_err, err)
        print(f"    quote={quote:.4%} implied={impl:.4%} err={err:.2e}")

    # === Test 2: Agreement with sequential ===
    df_diff = np.max(np.abs(glob_dfs - seq_dfs))

    # === Test 3: Convergence ===
    final_obj = float(objective(opt_rates))

    # === Test 4: JAX AD through global bootstrap ===
    try:
        grad_vec = jax.grad(objective)(opt_rates)
        grad_norm = float(jnp.linalg.norm(grad_vec))
        ad_ok = grad_norm < 1e-4  # gradient should be ~0 at optimum
    except Exception as e:
        print(f"\n  JAX AD: SKIPPED ({e})")
        ad_ok = False

    n_pass = 0
    total = 4

    ok = max_glob_err < 1e-6
    n_pass += int(ok)
    print(f"\n  Global repricing:     {'✓' if ok else '✗'} (max err={max_glob_err:.2e})")

    ok = df_diff < 1e-3
    n_pass += int(ok)
    print(f"  Seq vs Global DFs:    {'✓' if ok else '✗'} (max diff={df_diff:.2e})")

    ok = final_obj < 1e-12
    n_pass += int(ok)
    print(f"  Converged:            {'✓' if ok else '✗'} (obj={final_obj:.2e})")

    n_pass += int(ad_ok)
    print(f"  JAX AD at optimum:    {'✓' if ad_ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All global bootstrap validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
