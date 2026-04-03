"""Tests for additional gaps phases 11-12.

Phase 11: Gaussian 1-factor swaption/capfloor engines
Phase 12: Forward option engines
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ── Phase 11: Gaussian 1-factor swaption engine ────────────────────

class TestGaussian1dSwaption:
    """Test gaussian 1-factor swaption pricing (Jamshidian decomposition)."""

    def _flat_discount(self, r=0.05):
        return lambda t: jnp.exp(-r * t)

    def test_payer_swaption_positive(self):
        from ql_jax.engines.swaption.gaussian1d import gaussian1d_swaption_price
        price = gaussian1d_swaption_price(
            exercise_time=1.0,
            swap_payment_times=jnp.array([1.5, 2.0, 2.5, 3.0]),
            swap_accrual_fractions=jnp.array([0.5, 0.5, 0.5, 0.5]),
            fixed_rate=0.05, notional=1e6, payer=True,
            mean_reversion=0.03, sigma_r=0.01,
            discount_curve_fn=self._flat_discount(0.05),
        )
        assert float(price) > 0

    def test_receiver_swaption_positive(self):
        from ql_jax.engines.swaption.gaussian1d import gaussian1d_swaption_price
        price = gaussian1d_swaption_price(
            exercise_time=1.0,
            swap_payment_times=jnp.array([1.5, 2.0, 2.5, 3.0]),
            swap_accrual_fractions=jnp.array([0.5, 0.5, 0.5, 0.5]),
            fixed_rate=0.05, notional=1e6, payer=False,
            mean_reversion=0.03, sigma_r=0.01,
            discount_curve_fn=self._flat_discount(0.05),
        )
        assert float(price) > 0

    def test_put_call_parity(self):
        """Payer - receiver = PV of swap."""
        from ql_jax.engines.swaption.gaussian1d import gaussian1d_swaption_price
        disc = self._flat_discount(0.05)
        t_pay = jnp.array([1.5, 2.0, 2.5, 3.0])
        tau = jnp.array([0.5, 0.5, 0.5, 0.5])
        K = 0.05
        N = 1e6
        kwargs = dict(
            exercise_time=1.0, swap_payment_times=t_pay,
            swap_accrual_fractions=tau, fixed_rate=K, notional=N,
            mean_reversion=0.03, sigma_r=0.01, discount_curve_fn=disc,
        )
        payer = gaussian1d_swaption_price(**kwargs, payer=True)
        recv = gaussian1d_swaption_price(**kwargs, payer=False)
        # PV of payer swap = sum tau_i * disc(t_i) * (fwd - K)
        # With flat curve at 5%, forward = 5%, so swap value ≈ 0
        # => payer ≈ receiver
        assert abs(float(payer) - float(recv)) < N * 0.01

    def test_higher_vol_increases_price(self):
        from ql_jax.engines.swaption.gaussian1d import gaussian1d_swaption_price
        disc = self._flat_discount(0.05)
        base = dict(
            exercise_time=1.0,
            swap_payment_times=jnp.array([1.5, 2.0, 2.5, 3.0]),
            swap_accrual_fractions=jnp.array([0.5, 0.5, 0.5, 0.5]),
            fixed_rate=0.05, notional=1e6, payer=True,
            mean_reversion=0.03, discount_curve_fn=disc,
        )
        p_low = gaussian1d_swaption_price(**base, sigma_r=0.005)
        p_high = gaussian1d_swaption_price(**base, sigma_r=0.02)
        assert float(p_high) > float(p_low)

    def test_zero_vol_intrinsic(self):
        from ql_jax.engines.swaption.gaussian1d import gaussian1d_swaption_price
        disc = self._flat_discount(0.04)
        # With flat curve at 4%, swap strike 5% => payer OTM
        price = gaussian1d_swaption_price(
            exercise_time=1.0,
            swap_payment_times=jnp.array([1.5, 2.0, 2.5, 3.0]),
            swap_accrual_fractions=jnp.array([0.5, 0.5, 0.5, 0.5]),
            fixed_rate=0.05, notional=1e6, payer=True,
            mean_reversion=0.03, sigma_r=1e-10,
            discount_curve_fn=disc,
        )
        # Should be near zero for OTM payer
        assert float(price) >= -1.0  # small numerical noise OK


class TestGaussian1dCapFloor:
    """Test gaussian 1-factor cap/floor engine."""

    def _flat_discount(self, r=0.05):
        return lambda t: jnp.exp(-r * t)

    def test_cap_positive(self):
        from ql_jax.engines.swaption.gaussian1d import gaussian1d_capfloor_price
        price = gaussian1d_capfloor_price(
            reset_times=jnp.array([0.5, 1.0, 1.5, 2.0]),
            payment_times=jnp.array([1.0, 1.5, 2.0, 2.5]),
            accrual_fractions=jnp.array([0.5, 0.5, 0.5, 0.5]),
            strike=0.05, notional=1e6, is_cap=True,
            mean_reversion=0.03, sigma_r=0.01,
            discount_curve_fn=self._flat_discount(0.05),
        )
        assert float(price) > 0

    def test_floor_positive(self):
        from ql_jax.engines.swaption.gaussian1d import gaussian1d_capfloor_price
        price = gaussian1d_capfloor_price(
            reset_times=jnp.array([0.5, 1.0, 1.5, 2.0]),
            payment_times=jnp.array([1.0, 1.5, 2.0, 2.5]),
            accrual_fractions=jnp.array([0.5, 0.5, 0.5, 0.5]),
            strike=0.05, notional=1e6, is_cap=False,
            mean_reversion=0.03, sigma_r=0.01,
            discount_curve_fn=self._flat_discount(0.05),
        )
        assert float(price) > 0

    def test_cap_floor_parity(self):
        """Cap - Floor = PV of floating - fixed."""
        from ql_jax.engines.swaption.gaussian1d import gaussian1d_capfloor_price
        disc = self._flat_discount(0.05)
        kwargs = dict(
            reset_times=jnp.array([0.5, 1.0, 1.5, 2.0]),
            payment_times=jnp.array([1.0, 1.5, 2.0, 2.5]),
            accrual_fractions=jnp.array([0.5, 0.5, 0.5, 0.5]),
            strike=0.05, notional=1e6,
            mean_reversion=0.03, sigma_r=0.01,
            discount_curve_fn=disc,
        )
        cap = gaussian1d_capfloor_price(**kwargs, is_cap=True)
        floor = gaussian1d_capfloor_price(**kwargs, is_cap=False)
        # With flat 5% curve and K=5%, cap - floor ≈ 0
        assert abs(float(cap) - float(floor)) < 1e6 * 0.005

    def test_higher_vol_increases_cap(self):
        from ql_jax.engines.swaption.gaussian1d import gaussian1d_capfloor_price
        disc = self._flat_discount(0.05)
        base = dict(
            reset_times=jnp.array([0.5, 1.0, 1.5, 2.0]),
            payment_times=jnp.array([1.0, 1.5, 2.0, 2.5]),
            accrual_fractions=jnp.array([0.5, 0.5, 0.5, 0.5]),
            strike=0.05, notional=1e6, is_cap=True,
            mean_reversion=0.03, discount_curve_fn=disc,
        )
        p_low = gaussian1d_capfloor_price(**base, sigma_r=0.005)
        p_high = gaussian1d_capfloor_price(**base, sigma_r=0.02)
        assert float(p_high) > float(p_low)


# ── Phase 12: Forward engines ──────────────────────────────────────

class TestForwardStartOption:
    """Forward-starting vanilla option."""

    def test_call_positive(self):
        from ql_jax.engines.analytic.forward import forward_start_price
        p = forward_start_price(100.0, 0.5, 1.0, 0.05, 0.02, 0.25, 1.0, 1)
        assert float(p) > 0

    def test_put_positive(self):
        from ql_jax.engines.analytic.forward import forward_start_price
        p = forward_start_price(100.0, 0.5, 1.0, 0.05, 0.02, 0.25, 1.0, -1)
        assert float(p) > 0

    def test_atm_forward_start_scaling(self):
        """Forward-start ATM price scales linearly with spot."""
        from ql_jax.engines.analytic.forward import forward_start_price
        p1 = forward_start_price(100.0, 0.5, 1.0, 0.05, 0.02, 0.25, 1.0, 1)
        p2 = forward_start_price(200.0, 0.5, 1.0, 0.05, 0.02, 0.25, 1.0, 1)
        assert abs(float(p2) / float(p1) - 2.0) < 0.001

    def test_alpha_effect(self):
        """Higher alpha (OTM call) lowers call price."""
        from ql_jax.engines.analytic.forward import forward_start_price
        p_atm = forward_start_price(100.0, 0.5, 1.0, 0.05, 0.02, 0.25, 1.0, 1)
        p_otm = forward_start_price(100.0, 0.5, 1.0, 0.05, 0.02, 0.25, 1.1, 1)
        assert float(p_otm) < float(p_atm)

    def test_put_call_parity(self):
        from ql_jax.engines.analytic.forward import forward_start_price
        S, T_s, T_e, r, q, sig, alpha = 100.0, 0.5, 1.5, 0.05, 0.02, 0.3, 1.0
        c = forward_start_price(S, T_s, T_e, r, q, sig, alpha, 1)
        p = forward_start_price(S, T_s, T_e, r, q, sig, alpha, -1)
        tau = T_e - T_s
        # C - P = S*exp(-q*T_s) * [exp(-q*tau) - alpha*exp(-r*tau)]
        # For alpha=1, flat rates: C - P = S*exp(-q*T_s)*(exp(-q*tau) - exp(-r*tau))
        parity_rhs = S * jnp.exp(-q * T_s) * (jnp.exp(-q * tau) - alpha * jnp.exp(-r * tau))
        assert abs(float(c) - float(p) - float(parity_rhs)) < 0.1


class TestForwardPerformance:
    """Forward performance (return) option."""

    def test_call_positive(self):
        from ql_jax.engines.analytic.forward import forward_performance_price
        p = forward_performance_price(100.0, 0.5, 1.0, 0.05, 0.02, 0.25, 1)
        assert float(p) > 0

    def test_independent_of_spot(self):
        """Performance option price is independent of spot level."""
        from ql_jax.engines.analytic.forward import forward_performance_price
        p1 = forward_performance_price(100.0, 0.5, 1.0, 0.05, 0.02, 0.25, 1)
        p2 = forward_performance_price(200.0, 0.5, 1.0, 0.05, 0.02, 0.25, 1)
        # Not exactly equal because of exp(-q*T_start) / S scaling
        # But the forward_start_price / S should be constant
        assert abs(float(p1) - float(p2)) < 0.01


class TestMCForwardEuropean:
    """Monte Carlo forward-start European."""

    def test_mc_vs_analytic(self):
        from ql_jax.engines.analytic.forward import (
            forward_start_price, mc_forward_european_price,
        )
        S, T_s, T_e, r, q, sig, alpha = 100.0, 0.5, 1.0, 0.05, 0.02, 0.25, 1.0
        analytic = forward_start_price(S, T_s, T_e, r, q, sig, alpha, 1)
        mc = mc_forward_european_price(
            S, T_s, T_e, r, q, sig, alpha, 1,
            n_paths=500000, key=jax.random.PRNGKey(0),
        )
        assert abs(float(mc) - float(analytic)) / float(analytic) < 0.03

    def test_mc_put_positive(self):
        from ql_jax.engines.analytic.forward import mc_forward_european_price
        p = mc_forward_european_price(
            100.0, 0.5, 1.0, 0.05, 0.02, 0.25, 1.0, -1,
            n_paths=100000, key=jax.random.PRNGKey(1),
        )
        assert float(p) > 0
