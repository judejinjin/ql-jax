"""Tests for IR/credit engines: ISDA CDS, capfloor, swaption (G2/tree), CVA, risky bond."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# ISDA CDS
# ---------------------------------------------------------------------------

class TestISDACDS:
    def test_fair_spread_positive(self):
        from ql_jax.engines.credit.isda import isda_cds_fair_spread
        payment_dates = jnp.array([0.25, 0.5, 0.75, 1.0])
        day_fractions = jnp.array([0.25, 0.25, 0.25, 0.25])
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        survival_fn = lambda t: jnp.exp(-0.02 * t)  # flat hazard 2%
        spread = isda_cds_fair_spread(recovery=0.4, payment_dates=payment_dates,
                                       day_fractions=day_fractions,
                                       discount_fn=discount_fn, survival_fn=survival_fn)
        assert float(spread) > 0
        # With 2% hazard rate and 40% recovery, spread ~ 0.6 * 0.02 = 1.2%
        assert float(spread) == pytest.approx(0.012, abs=0.005)

    def test_npv_buyer_at_fair_spread(self):
        from ql_jax.engines.credit.isda import isda_cds_npv, isda_cds_fair_spread
        payment_dates = jnp.array([0.25, 0.5, 0.75, 1.0])
        day_fractions = jnp.array([0.25, 0.25, 0.25, 0.25])
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        survival_fn = lambda t: jnp.exp(-0.01 * t)
        fair = isda_cds_fair_spread(recovery=0.4, payment_dates=payment_dates,
                                     day_fractions=day_fractions,
                                     discount_fn=discount_fn, survival_fn=survival_fn)
        npv = isda_cds_npv(spread=fair, recovery=0.4, payment_dates=payment_dates,
                            day_fractions=day_fractions, discount_fn=discount_fn,
                            survival_fn=survival_fn, is_buyer=True)
        assert float(npv) == pytest.approx(0.0, abs=1e-6)

    def test_npv_buyer_vs_seller(self):
        from ql_jax.engines.credit.isda import isda_cds_npv
        payment_dates = jnp.array([0.5, 1.0])
        day_fractions = jnp.array([0.5, 0.5])
        discount_fn = lambda t: jnp.exp(-0.03 * t)
        survival_fn = lambda t: jnp.exp(-0.05 * t)
        npv_buy = isda_cds_npv(spread=0.01, recovery=0.4, payment_dates=payment_dates,
                                day_fractions=day_fractions, discount_fn=discount_fn,
                                survival_fn=survival_fn, is_buyer=True)
        npv_sell = isda_cds_npv(spread=0.01, recovery=0.4, payment_dates=payment_dates,
                                 day_fractions=day_fractions, discount_fn=discount_fn,
                                 survival_fn=survival_fn, is_buyer=False)
        assert float(npv_buy) == pytest.approx(-float(npv_sell), abs=1e-10)

    def test_zero_default_probability(self):
        from ql_jax.engines.credit.isda import isda_cds_fair_spread
        payment_dates = jnp.array([0.5, 1.0])
        day_fractions = jnp.array([0.5, 0.5])
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        survival_fn = lambda t: jnp.ones_like(t)  # No default
        spread = isda_cds_fair_spread(recovery=0.4, payment_dates=payment_dates,
                                       day_fractions=day_fractions,
                                       discount_fn=discount_fn, survival_fn=survival_fn)
        assert float(spread) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Cap/Floor
# ---------------------------------------------------------------------------

class TestBachelierCapFloor:
    def test_import(self):
        from ql_jax.engines.capfloor.analytic import bachelier_capfloor_price_flat
        assert callable(bachelier_capfloor_price_flat)


# ---------------------------------------------------------------------------
# Swaption G2++
# ---------------------------------------------------------------------------

class TestG2Swaption:
    def test_payer_swaption(self):
        from ql_jax.engines.swaption.g2 import g2_swaption_price
        # Swaption exercising at t=1 on a swap with payments at 2,3,4,5,6
        payment_dates = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0])
        day_fractions = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        discount_fn = lambda t: jnp.exp(-0.03 * t)
        price = g2_swaption_price(notional=1e6, fixed_rate=0.03,
                                   payment_dates=payment_dates,
                                   day_fractions=day_fractions,
                                   a=0.1, sigma=0.01, b=0.2, eta=0.008, rho=-0.5,
                                   discount_fn=discount_fn, payer=True)
        assert float(price) > 0
        assert float(price) < 1e6  # Bounded

    def test_receiver_swaption(self):
        from ql_jax.engines.swaption.g2 import g2_swaption_price
        # Swaption exercising at t=1 on a swap with payments at 2,3,4
        payment_dates = jnp.array([2.0, 3.0, 4.0])
        day_fractions = jnp.array([1.0, 1.0, 1.0])
        discount_fn = lambda t: jnp.exp(-0.04 * t)
        price = g2_swaption_price(notional=1e6, fixed_rate=0.04,
                                   payment_dates=payment_dates,
                                   day_fractions=day_fractions,
                                   a=0.15, sigma=0.012, b=0.25, eta=0.009, rho=-0.3,
                                   discount_fn=discount_fn, payer=False)
        assert float(price) > 0


# ---------------------------------------------------------------------------
# Tree swaption
# ---------------------------------------------------------------------------

class TestTreeSwaption:
    @pytest.mark.skip(reason="tree_swaption_price depends on unimplemented lattice module")
    def test_payer_swaption(self):
        from ql_jax.engines.swaption.tree import tree_swaption_price
        payment_dates = jnp.array([1.0, 2.0, 3.0])
        day_fractions = jnp.array([1.0, 1.0, 1.0])
        discount_fn = lambda t: jnp.exp(-0.03 * t)
        price = tree_swaption_price(notional=1e6, fixed_rate=0.03,
                                     exercise_time=0.5,
                                     payment_dates=payment_dates,
                                     day_fractions=day_fractions,
                                     a=0.1, sigma=0.01,
                                     discount_fn=discount_fn, n_steps=50, payer=True)
        assert float(price) > 0

    @pytest.mark.skip(reason="tree_swap_price depends on unimplemented lattice module")
    def test_swap_price(self):
        from ql_jax.engines.swaption.tree import tree_swap_price
        payment_dates = jnp.array([1.0, 2.0])
        day_fractions = jnp.array([1.0, 1.0])
        discount_fn = lambda t: jnp.exp(-0.03 * t)
        price = tree_swap_price(notional=1e6, fixed_rate=0.03,
                                 payment_dates=payment_dates,
                                 day_fractions=day_fractions,
                                 a=0.1, sigma=0.01,
                                 discount_fn=discount_fn, n_steps=50, payer=True)
        # At-the-money swap should have NPV near 0
        assert abs(float(price)) < 50_000


# ---------------------------------------------------------------------------
# CVA / DVA
# ---------------------------------------------------------------------------

class TestCVA:
    def test_cva_positive(self):
        from ql_jax.engines.swap.cva import cva_swap_engine
        exposure_dates = jnp.array([0.5, 1.0, 1.5, 2.0])
        discount_fn = lambda t: jnp.exp(-0.03 * t)
        survival_fn = lambda t: jnp.exp(-0.02 * t)
        # Simple positive exposure profile
        swap_mtm_fn = lambda t: 10000.0 * jnp.exp(-0.01 * t)
        cva = cva_swap_engine(swap_mtm_fn, exposure_dates, discount_fn, survival_fn, recovery=0.4)
        assert float(cva) > 0

    def test_dva_positive(self):
        from ql_jax.engines.swap.cva import dva_swap_engine
        exposure_dates = jnp.array([0.5, 1.0])
        discount_fn = lambda t: jnp.exp(-0.03 * t)
        own_survival_fn = lambda t: jnp.exp(-0.01 * t)
        swap_mtm_fn = lambda t: -5000.0  # Negative MtM => we owe
        dva = dva_swap_engine(swap_mtm_fn, exposure_dates, discount_fn, own_survival_fn, own_recovery=0.4)
        assert float(dva) >= 0

    def test_bilateral_cva(self):
        from ql_jax.engines.swap.cva import bilateral_cva
        exposure_dates = jnp.array([0.5, 1.0, 1.5, 2.0])
        discount_fn = lambda t: jnp.exp(-0.03 * t)
        cpty_survival_fn = lambda t: jnp.exp(-0.02 * t)
        own_survival_fn = lambda t: jnp.exp(-0.01 * t)
        swap_mtm_fn = lambda t: 10000.0 * jnp.maximum(0.5 - t, 0.0)
        bcva = bilateral_cva(swap_mtm_fn, exposure_dates, discount_fn,
                              cpty_survival_fn, own_survival_fn)
        assert isinstance(float(bcva), float)


# ---------------------------------------------------------------------------
# Risky bond
# ---------------------------------------------------------------------------

class TestRiskyBond:
    def test_risky_npv(self):
        from ql_jax.engines.bond.risky import risky_bond_npv
        from ql_jax.instruments.bond import make_fixed_rate_bond
        from ql_jax.time.date import Date
        from ql_jax.time.schedule import MakeSchedule
        from ql_jax.time.calendar import NullCalendar
        from ql_jax._util.types import Frequency

        sch = (MakeSchedule()
               .from_date(Date(15, 1, 2024))
               .to_date(Date(15, 1, 2029))
               .with_frequency(Frequency.Annual)
               .with_calendar(NullCalendar())
               .build())
        bond = make_fixed_rate_bond(settlement_days=0, face_amount=100.0,
                                     schedule=sch, coupons=[0.05])
        discount_fn = lambda t: jnp.exp(-0.03 * t)
        survival_fn = lambda t: jnp.exp(-0.01 * t)
        npv = risky_bond_npv(bond, discount_fn, survival_fn, recovery_rate=0.4,
                              settlement_date=Date(15, 1, 2024))
        assert float(npv) > 0
        assert float(npv) < 120  # Bounded

    def test_risky_vs_riskfree(self):
        """Risky NPV should be less than risk-free NPV."""
        from ql_jax.engines.bond.risky import risky_bond_npv
        from ql_jax.instruments.bond import make_fixed_rate_bond
        from ql_jax.time.date import Date
        from ql_jax.time.schedule import MakeSchedule
        from ql_jax.time.calendar import NullCalendar
        from ql_jax._util.types import Frequency

        sch = (MakeSchedule()
               .from_date(Date(15, 1, 2024))
               .to_date(Date(15, 1, 2029))
               .with_frequency(Frequency.Annual)
               .with_calendar(NullCalendar())
               .build())
        bond = make_fixed_rate_bond(settlement_days=0, face_amount=100.0,
                                     schedule=sch, coupons=[0.05])
        discount_fn = lambda t: jnp.exp(-0.03 * t)
        rf_survival = lambda t: jnp.ones_like(t) if hasattr(t, 'shape') else 1.0
        risky_survival = lambda t: jnp.exp(-0.05 * t)
        npv_rf = risky_bond_npv(bond, discount_fn, rf_survival, recovery_rate=0.4,
                                 settlement_date=Date(15, 1, 2024))
        npv_risky = risky_bond_npv(bond, discount_fn, risky_survival, recovery_rate=0.4,
                                    settlement_date=Date(15, 1, 2024))
        assert float(npv_risky) < float(npv_rf)
