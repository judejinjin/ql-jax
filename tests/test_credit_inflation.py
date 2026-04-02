"""Tests for Phase 7: Credit, Inflation & Market Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ── CDS Tests ──────────────────────────────────────────────────────────────

class TestCDS:
    """Tests for CDS instruments and midpoint engine."""

    def test_make_cds(self):
        from ql_jax.instruments.cds import make_cds

        cds = make_cds(10_000_000, 0.01, 5.0)
        assert cds.protection_side == 'buyer'
        assert cds.notional == 10_000_000
        assert cds.spread == 0.01  # 100bp
        assert cds.maturity == 5.0
        assert cds.recovery_rate == 0.4
        assert cds.n_periods == 20  # quarterly
        assert cds.payment_dates.shape[0] == 20

    def test_payment_dates(self):
        from ql_jax.instruments.cds import make_cds

        cds = make_cds(1e6, 0.01, 2.0, frequency=0.5)
        dates = cds.payment_dates
        assert dates.shape[0] == 4
        assert jnp.allclose(dates, jnp.array([0.5, 1.0, 1.5, 2.0]))

    def test_fair_spread_at_par(self):
        """A CDS priced at its fair spread should have NPV ≈ 0."""
        from ql_jax.instruments.cds import make_cds
        from ql_jax.engines.credit.midpoint import (
            cds_fair_spread, midpoint_cds_npv, hazard_rate_from_spread,
            survival_probability,
        )

        r = 0.05
        h = 0.02  # 200bp hazard rate
        discount_fn = lambda t: jnp.exp(-r * t)
        survival_fn = lambda t: survival_probability(h, t)

        cds = make_cds(10_000_000, 0.01, 5.0)  # initial spread doesn't matter
        fair = cds_fair_spread(cds, discount_fn, survival_fn)

        # Now create CDS at fair spread and check NPV ≈ 0
        cds_fair = make_cds(10_000_000, float(fair), 5.0)
        npv = midpoint_cds_npv(cds_fair, discount_fn, survival_fn)
        assert abs(float(npv)) < 100  # < 100 on 10M notional

    def test_protection_buyer_positive_npv_for_risky_name(self):
        """Protection buyer benefits when spread is below fair."""
        from ql_jax.instruments.cds import make_cds
        from ql_jax.engines.credit.midpoint import (
            midpoint_cds_npv, hazard_rate_from_spread, survival_probability,
        )

        r = 0.05
        h = 0.03  # risky name
        discount_fn = lambda t: jnp.exp(-r * t)
        survival_fn = lambda t: survival_probability(h, t)

        # Buyer pays only 50bp but name has ~300bp hazard rate
        cds = make_cds(1_000_000, 0.005, 5.0)
        npv = midpoint_cds_npv(cds, discount_fn, survival_fn)
        assert float(npv) > 0  # buyer benefits

    def test_seller_negative_of_buyer(self):
        """CDS seller NPV is negative of buyer NPV."""
        from ql_jax.instruments.cds import CreditDefaultSwap
        from ql_jax.engines.credit.midpoint import (
            midpoint_cds_npv, survival_probability,
        )

        r = 0.05
        h = 0.02
        discount_fn = lambda t: jnp.exp(-r * t)
        survival_fn = lambda t: survival_probability(h, t)

        buyer = CreditDefaultSwap('buyer', 1e6, 0.01, 5.0)
        seller = CreditDefaultSwap('seller', 1e6, 0.01, 5.0)

        npv_buy = midpoint_cds_npv(buyer, discount_fn, survival_fn)
        npv_sell = midpoint_cds_npv(seller, discount_fn, survival_fn)
        assert jnp.allclose(npv_buy, -npv_sell, atol=1e-8)

    def test_hazard_rate_from_spread(self):
        from ql_jax.engines.credit.midpoint import hazard_rate_from_spread

        h = hazard_rate_from_spread(0.01, 0.4)  # 100bp, 40% recovery
        assert jnp.allclose(h, 0.01 / 0.6)

    def test_survival_probability(self):
        from ql_jax.engines.credit.midpoint import survival_probability

        q = survival_probability(0.02, 5.0)
        assert jnp.allclose(q, jnp.exp(-0.1))

    def test_cds_npv_increases_with_hazard_rate(self):
        """Higher hazard rate means more value for protection buyer."""
        from ql_jax.instruments.cds import make_cds
        from ql_jax.engines.credit.midpoint import (
            midpoint_cds_npv, survival_probability,
        )

        r = 0.05
        discount_fn = lambda t: jnp.exp(-r * t)
        cds = make_cds(1_000_000, 0.01, 5.0)

        npv_low = midpoint_cds_npv(cds, discount_fn, lambda t: survival_probability(0.01, t))
        npv_high = midpoint_cds_npv(cds, discount_fn, lambda t: survival_probability(0.05, t))
        assert float(npv_high) > float(npv_low)

    def test_cds_ad_sensitivity(self):
        """AD through CDS pricing w.r.t. hazard rate."""
        from ql_jax.instruments.cds import make_cds
        from ql_jax.engines.credit.midpoint import (
            midpoint_cds_npv, survival_probability,
        )

        cds = make_cds(1_000_000, 0.01, 5.0)
        r = 0.05
        discount_fn = lambda t: jnp.exp(-r * t)

        def npv_fn(h):
            survival_fn = lambda t: jnp.exp(-h * t)
            return midpoint_cds_npv(cds, discount_fn, survival_fn)

        grad_fn = jax.grad(npv_fn)
        dndh = grad_fn(0.02)
        # d(protection_buyer_npv)/d(hazard_rate) > 0: higher h → more protection value
        assert float(dndh) > 0

    def test_fair_spread_approximation(self):
        """Fair spread ≈ h * (1 - R) for flat curves."""
        from ql_jax.instruments.cds import make_cds
        from ql_jax.engines.credit.midpoint import (
            cds_fair_spread, survival_probability,
        )

        r = 0.05
        R = 0.4
        h = 0.02
        discount_fn = lambda t: jnp.exp(-r * t)
        survival_fn = lambda t: survival_probability(h, t)

        cds = make_cds(1e6, 0.01, 5.0, recovery_rate=R)
        fair = cds_fair_spread(cds, discount_fn, survival_fn)

        # Approximate: fair_spread ≈ h * (1 - R) = 0.02 * 0.6 = 0.012
        assert abs(float(fair) - h * (1 - R)) < 0.002  # within 20bp


# ── Inflation Tests ─────────────────────────────────────────────────────────

class TestInflation:
    """Tests for inflation instruments and engines."""

    def test_yoy_inflation_cap(self):
        """YoY inflation cap price is non-negative."""
        from ql_jax.engines.inflation.capfloor import (
            InflationCapFloor, black_yoy_capfloor_price,
        )

        dates = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cap = InflationCapFloor(
            cap_or_floor='cap',
            strike=0.02,
            notional=1_000_000,
            payment_dates=dates,
        )
        discount_fn = lambda t: jnp.exp(-0.03 * t)
        fwd_rates = jnp.array([0.025, 0.025, 0.025, 0.025, 0.025])
        vols = jnp.array([0.01, 0.01, 0.01, 0.01, 0.01])

        price = black_yoy_capfloor_price(cap, discount_fn, fwd_rates, vols)
        assert float(price) > 0

    def test_yoy_floor_nonnegative(self):
        """YoY inflation floor price is non-negative."""
        from ql_jax.engines.inflation.capfloor import (
            InflationCapFloor, black_yoy_capfloor_price,
        )

        dates = jnp.array([1.0, 2.0, 3.0])
        floor = InflationCapFloor(
            cap_or_floor='floor',
            strike=0.03,
            notional=1_000_000,
            payment_dates=dates,
        )
        discount_fn = lambda t: jnp.exp(-0.03 * t)
        fwd_rates = jnp.array([0.02, 0.02, 0.02])
        vols = jnp.array([0.01, 0.01, 0.01])

        price = black_yoy_capfloor_price(floor, discount_fn, fwd_rates, vols)
        assert float(price) > 0

    def test_cap_floor_parity(self):
        """Cap - Floor = swap (approximately, for matching strikes)."""
        from ql_jax.engines.inflation.capfloor import (
            InflationCapFloor, black_yoy_capfloor_price,
        )

        dates = jnp.array([1.0, 2.0, 3.0])
        N = 1_000_000
        K = 0.025
        discount_fn = lambda t: jnp.exp(-0.03 * t)
        fwd_rates = jnp.array([0.03, 0.03, 0.03])
        vols = jnp.array([0.01, 0.01, 0.01])

        cap = InflationCapFloor('cap', K, N, dates)
        floor = InflationCapFloor('floor', K, N, dates)

        cap_price = black_yoy_capfloor_price(cap, discount_fn, fwd_rates, vols)
        floor_price = black_yoy_capfloor_price(floor, discount_fn, fwd_rates, vols)

        # Cap - Floor = sum of df * N * (F - K)
        swap_value = 0.0
        for i in range(3):
            T = float(dates[i])
            swap_value += discount_fn(T) * N * (fwd_rates[i] - K)

        assert jnp.allclose(cap_price - floor_price, swap_value, atol=1.0)

    def test_zc_inflation_swap_at_fair_rate(self):
        """ZC inflation swap NPV ≈ 0 when fixed rate matches expected inflation."""
        from ql_jax.engines.inflation.capfloor import zero_coupon_inflation_swap_npv

        base_cpi = 100.0
        inflation_rate = 0.025
        maturity = 10.0
        discount_fn = lambda t: jnp.exp(-0.04 * t)
        forward_cpi_fn = lambda t: base_cpi * (1.0 + inflation_rate)**t

        npv = zero_coupon_inflation_swap_npv(
            notional=1_000_000,
            fixed_rate=inflation_rate,
            maturity=maturity,
            discount_fn=discount_fn,
            forward_cpi_fn=forward_cpi_fn,
            base_cpi=base_cpi,
        )
        assert abs(float(npv)) < 0.01

    def test_zc_inflation_swap_payer_benefits_from_higher_inflation(self):
        """Payer (receive inflation) NPV increases with higher realized inflation."""
        from ql_jax.engines.inflation.capfloor import zero_coupon_inflation_swap_npv

        base_cpi = 100.0
        fixed_rate = 0.02
        maturity = 5.0
        discount_fn = lambda t: jnp.exp(-0.04 * t)

        npv_low = zero_coupon_inflation_swap_npv(
            1e6, fixed_rate, maturity, discount_fn,
            lambda t: base_cpi * (1.02)**t, base_cpi,
        )
        npv_high = zero_coupon_inflation_swap_npv(
            1e6, fixed_rate, maturity, discount_fn,
            lambda t: base_cpi * (1.04)**t, base_cpi,
        )
        assert float(npv_high) > float(npv_low)


# ── LMM Tests ──────────────────────────────────────────────────────────────

class TestLMM:
    """Tests for LIBOR Market Model."""

    @staticmethod
    def _simple_config(n_rates=5):
        from ql_jax.models.marketmodels.lmm import LMMConfig

        tau = jnp.ones(n_rates) * 0.5
        tenors = jnp.arange(n_rates + 1) * 0.5
        initial_forwards = jnp.ones(n_rates) * 0.05
        volatilities = jnp.ones(n_rates) * 0.20
        # Identity correlation
        correlation = jnp.eye(n_rates)
        return LMMConfig(
            n_rates=n_rates,
            tenors=tenors,
            tau=tau,
            initial_forwards=initial_forwards,
            volatilities=volatilities,
            correlation=correlation,
        )

    def test_lmm_config(self):
        config = self._simple_config()
        assert config.n_rates == 5
        assert config.initial_forwards.shape == (5,)

    def test_simulate_paths_shape(self):
        from ql_jax.models.marketmodels.lmm import simulate_lmm_paths

        config = self._simple_config(n_rates=3)
        n_paths = 100
        paths = simulate_lmm_paths(config, n_paths, n_steps_per_period=5,
                                    key=jax.random.PRNGKey(0))
        # (n_paths, n_rates, n_total_steps+1)
        assert paths.shape == (100, 3, 3 * 5 + 1)

    def test_forward_rates_positive(self):
        """Simulated forward rates should be positive (log-Euler)."""
        from ql_jax.models.marketmodels.lmm import simulate_lmm_paths

        config = self._simple_config(n_rates=3)
        paths = simulate_lmm_paths(config, 500, n_steps_per_period=5,
                                    key=jax.random.PRNGKey(1))
        assert jnp.all(paths > 0)

    def test_forward_rates_initial_condition(self):
        """Paths at t=0 should match initial_forwards."""
        from ql_jax.models.marketmodels.lmm import simulate_lmm_paths

        config = self._simple_config(n_rates=3)
        paths = simulate_lmm_paths(config, 100, n_steps_per_period=5,
                                    key=jax.random.PRNGKey(2))
        # At step 0, all paths should have initial forwards
        initial = paths[:, :, 0]  # (n_paths, n_rates)
        expected = jnp.broadcast_to(config.initial_forwards, (100, 3))
        assert jnp.allclose(initial, expected)

    def test_swap_rate(self):
        """Par swap rate from flat forward curve."""
        from ql_jax.models.marketmodels.lmm import lmm_swap_rate

        forwards = jnp.ones(5) * 0.05
        tau = jnp.ones(5) * 0.5

        swap_rate = lmm_swap_rate(forwards, tau)
        # For flat forwards, swap rate ≈ forward rate
        assert abs(float(swap_rate) - 0.05) < 0.005

    def test_swap_rate_non_flat(self):
        """Swap rate should be a weighted average of forwards."""
        from ql_jax.models.marketmodels.lmm import lmm_swap_rate

        forwards = jnp.array([0.04, 0.045, 0.05, 0.055, 0.06])
        tau = jnp.ones(5) * 0.5
        swap_rate = lmm_swap_rate(forwards, tau)

        # Should be between min and max forward
        assert 0.04 < float(swap_rate) < 0.06


# ── Forward Instruments Tests ──────────────────────────────────────────────

class TestForwardInstruments:
    """Tests for FX forward and other instruments."""

    def test_fx_forward_price_covered_interest_parity(self):
        from ql_jax.instruments.forward import fx_forward_price

        spot = 1.10  # EUR/USD
        r_d = 0.05   # USD rate
        r_f = 0.03   # EUR rate
        T = 1.0

        fwd = fx_forward_price(spot, r_d, r_f, T)
        expected = spot * jnp.exp((r_d - r_f) * T)
        assert jnp.allclose(fwd, expected)

    def test_fx_forward_price_at_zero_maturity(self):
        from ql_jax.instruments.forward import fx_forward_price

        spot = 1.10
        fwd = fx_forward_price(spot, 0.05, 0.03, 0.0)
        assert jnp.allclose(fwd, spot)

    def test_fx_forward_npv_at_inception(self):
        """At inception with fair forward, NPV ≈ 0."""
        from ql_jax.instruments.forward import FXForward, fx_forward_npv, fx_forward_price

        spot = 1.10
        r_d = 0.05
        r_f = 0.03
        T = 1.0
        N_foreign = 1_000_000

        fair_rate = fx_forward_price(spot, r_d, r_f, T)
        N_domestic = float(N_foreign * fair_rate)

        fwd = FXForward(
            domestic_notional=N_domestic,
            foreign_notional=N_foreign,
            maturity=T,
            spot_fx=spot,
        )

        npv = fx_forward_npv(
            fwd,
            lambda t: jnp.exp(-r_d * t),
            lambda t: jnp.exp(-r_f * t),
        )
        assert abs(float(npv)) < 0.01

    def test_fx_forward_npv_off_market(self):
        """Off-market FX forward should have non-zero NPV."""
        from ql_jax.instruments.forward import FXForward, fx_forward_npv

        r_d = 0.05
        r_f = 0.03
        T = 1.0

        fwd = FXForward(
            domestic_notional=1_050_000,  # slightly off
            foreign_notional=1_000_000,
            maturity=T,
            spot_fx=1.10,
        )

        npv = fx_forward_npv(
            fwd,
            lambda t: jnp.exp(-r_d * t),
            lambda t: jnp.exp(-r_f * t),
        )
        assert float(npv) != 0.0

    def test_composite_instrument(self):
        from ql_jax.instruments.forward import CompositeInstrument

        inst = CompositeInstrument(
            instruments=(100.0, 200.0, 300.0),
            weights=(1.0, -0.5, 0.25),
        )
        # Simple pricer is identity
        npv = inst.npv(lambda x: x)
        assert npv == 100.0 - 100.0 + 75.0


# ── Integration: AD through credit/inflation ───────────────────────────────

class TestADIntegration:
    """Test AD works through credit and inflation pricing."""

    def test_inflation_cap_ad(self):
        """AD through inflation cap pricing w.r.t. forward rate."""
        from ql_jax.engines.inflation.capfloor import (
            InflationCapFloor, black_yoy_capfloor_price,
        )

        dates = jnp.array([1.0, 2.0, 3.0])
        cap = InflationCapFloor('cap', 0.02, 1e6, dates)
        discount_fn = lambda t: jnp.exp(-0.03 * t)
        vols = jnp.array([0.01, 0.01, 0.01])

        def price_fn(fwd_rate):
            fwd_rates = jnp.ones(3) * fwd_rate
            return black_yoy_capfloor_price(cap, discount_fn, fwd_rates, vols)

        grad_fn = jax.grad(price_fn)
        delta = grad_fn(0.025)
        # Cap price increases with forward inflation rate
        assert float(delta) > 0

    def test_zc_swap_ad(self):
        """AD through ZC inflation swap w.r.t. inflation rate."""
        from ql_jax.engines.inflation.capfloor import zero_coupon_inflation_swap_npv

        def npv_fn(infl_rate):
            return zero_coupon_inflation_swap_npv(
                notional=1e6,
                fixed_rate=0.02,
                maturity=5.0,
                discount_fn=lambda t: jnp.exp(-0.04 * t),
                forward_cpi_fn=lambda t: 100.0 * (1.0 + infl_rate)**t,
                base_cpi=100.0,
            )

        grad_fn = jax.grad(npv_fn)
        delta = grad_fn(0.025)
        # Payer benefits from higher inflation
        assert float(delta) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
