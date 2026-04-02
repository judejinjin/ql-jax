"""Tests for credit instruments and engines."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestCDSAnalytics:
    def test_cds_fair_spread(self):
        from ql_jax.engines.credit.analytics import cds_fair_spread
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        survival_fn = lambda t: jnp.exp(-0.02 * t)
        payment_dates = jnp.array([0.25, 0.5, 0.75, 1.0])
        day_fractions = jnp.array([0.25, 0.25, 0.25, 0.25])
        spread = float(cds_fair_spread(
            recovery=0.4, payment_dates=payment_dates,
            day_fractions=day_fractions,
            discount_fn=discount_fn, survival_fn=survival_fn
        ))
        assert spread > 0.0
        assert spread < 0.05

    def test_hazard_from_spread(self):
        from ql_jax.engines.credit.analytics import hazard_rate_from_spread
        h = float(hazard_rate_from_spread(spread=0.01, recovery=0.4))
        assert h > 0.0
        assert abs(h - 0.01 / 0.6) < 0.01


class TestCDSProtection:
    def test_protection_leg(self):
        from ql_jax.engines.credit.analytics import cds_protection_leg
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        survival_fn = lambda t: jnp.exp(-0.02 * t)
        payment_dates = jnp.array([0.25, 0.5, 0.75, 1.0])
        prot = float(cds_protection_leg(
            recovery=0.4, payment_dates=payment_dates,
            discount_fn=discount_fn, survival_fn=survival_fn
        ))
        assert prot > 0.0

    def test_cds_npv(self):
        from ql_jax.engines.credit.analytics import cds_npv
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        survival_fn = lambda t: jnp.exp(-0.02 * t)
        payment_dates = jnp.array([0.25, 0.5, 0.75, 1.0])
        day_fractions = jnp.array([0.25, 0.25, 0.25, 0.25])
        npv = float(cds_npv(
            spread=0.01, recovery=0.4, payment_dates=payment_dates,
            day_fractions=day_fractions,
            discount_fn=discount_fn, survival_fn=survival_fn
        ))
        assert abs(npv) < 1.0  # bounded for notional=1
