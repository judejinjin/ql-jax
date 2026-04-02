"""Example: CVA for Interest-Rate Swaps.

Computes Credit Valuation Adjustment for a vanilla IRS using:
  - Monte Carlo simulation of rate paths (Hull-White model)
  - Exposure profiles (EE, EPE, ENE)
  - CVA calculation with deterministic default intensity
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def main():
    print("=" * 60)
    print("QL-JAX Example: CVA for Interest-Rate Swaps")
    print("=" * 60)
    print()

    # ── Swap Parameters ──────────────────────────────────────
    notional = 10_000_000.0
    fixed_rate = 0.05
    maturity = 5.0
    payment_freq = 0.5  # semi-annual
    payment_times = jnp.arange(payment_freq, maturity + 0.01, payment_freq)
    n_payments = len(payment_times)

    # Hull-White parameters
    a = 0.1
    sigma = 0.01
    r0 = 0.05

    # Counterparty credit
    hazard_rate = 0.02   # 2% annual default intensity
    recovery = 0.40

    print(f"  Swap: {notional/1e6:.0f}M, {fixed_rate:.2%} fixed, {maturity:.0f}y")
    print(f"  HW params: a={a}, σ={sigma}, r₀={r0:.2%}")
    print(f"  Default intensity: {hazard_rate:.2%}, Recovery: {recovery:.0%}")
    print()

    # ── Monte Carlo: HW short-rate paths ─────────────────────
    n_paths = 10_000
    n_steps = int(maturity * 12)  # monthly
    dt = maturity / n_steps
    key = jax.random.PRNGKey(42)

    dW = jax.random.normal(key, (n_paths, n_steps)) * jnp.sqrt(dt)
    r_paths = jnp.zeros((n_paths, n_steps + 1))
    r_paths = r_paths.at[:, 0].set(r0)

    # Simple Euler discretization of HW
    theta_t = r0  # simplified: constant theta
    def step(carry, dw):
        r = carry
        r_new = r + a * (theta_t - r) * dt + sigma * dw
        return r_new, r_new

    for i in range(n_paths):
        _, path = jax.lax.scan(step, r_paths[i, 0], dW[i])
        r_paths = r_paths.at[i, 1:].set(path)

    time_grid = jnp.linspace(0, maturity, n_steps + 1)

    # ── Exposure Profile ─────────────────────────────────────
    # At each payment time, compute swap MtM
    print("  Exposure Profile:")
    print(f"  {'Time':>6s} {'EE':>12s} {'EPE':>12s} {'ENE':>12s}")

    exposure_times = jnp.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    expected_exposures = []

    for t_eval in exposure_times:
        # Find closest time step
        idx = int(t_eval / dt)
        r_t = r_paths[:, idx]

        # Compute remaining swap value (simplified: sum of future cashflows)
        remaining = payment_times[payment_times > t_eval]
        if len(remaining) == 0:
            ee = 0.0
            expected_exposures.append(0.0)
            continue

        # Discount future net cashflows: floating leg ≈ r_t, fixed pays fixed_rate
        tau = payment_freq
        swap_values = jnp.zeros(n_paths)
        for t_pay in remaining:
            dt_to_pay = t_pay - t_eval
            df = jnp.exp(-r_t * dt_to_pay)
            # Net cashflow: receive float (≈ r_t * tau * notional) - pay fixed
            net_cf = notional * tau * (r_t - fixed_rate)
            swap_values = swap_values + net_cf * df

        ee = float(jnp.mean(swap_values))
        epe = float(jnp.mean(jnp.maximum(swap_values, 0.0)))
        ene = float(jnp.mean(jnp.minimum(swap_values, 0.0)))
        expected_exposures.append(epe)

        print(f"  {float(t_eval):6.1f} {ee:12.0f} {epe:12.0f} {ene:12.0f}")

    print()

    # ── CVA Calculation ──────────────────────────────────────
    # CVA = (1 - R) * Σ_i EE(t_i) * DF(t_i) * PD(t_{i-1}, t_i)
    lgd = 1.0 - recovery
    cva = 0.0
    prev_t = 0.0

    for i, t in enumerate(exposure_times):
        df = jnp.exp(-r0 * t)
        surv_prev = jnp.exp(-hazard_rate * prev_t)
        surv_curr = jnp.exp(-hazard_rate * t)
        pd = surv_prev - surv_curr  # marginal default probability
        cva += lgd * expected_exposures[i] * float(df) * float(pd)
        prev_t = t

    print(f"  CVA: {cva:.2f}")
    print(f"  CVA as % of notional: {cva / notional * 100:.4f}%")
    print(f"  CVA in bps (annualized): {cva / notional / maturity * 10000:.2f} bps")


if __name__ == "__main__":
    main()
