"""Tests for additional gaps phases 21-35."""

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pytest


# ═══════════════════════════════════════════════════════════════════
# Phase 21: Credit Experimental
# ═══════════════════════════════════════════════════════════════════

class TestCDO:

    def test_tranche_loss_below_attachment(self):
        from ql_jax.experimental.credit.cdo import SyntheticCdo, CdoTranche
        cdo = SyntheticCdo(
            portfolio_notional=100.0,
            tranches=[CdoTranche(0.03, 0.07, 1.0)],
            recovery_rate=0.4,
        )
        loss = cdo.tranche_loss(0.02, 0)  # below attachment
        assert abs(float(loss)) < 1e-10

    def test_tranche_loss_above_exhaustion(self):
        from ql_jax.experimental.credit.cdo import SyntheticCdo, CdoTranche
        cdo = SyntheticCdo(
            portfolio_notional=100.0,
            tranches=[CdoTranche(0.03, 0.07, 1.0)],
            recovery_rate=0.4,
        )
        loss = cdo.tranche_loss(0.10, 0)  # above exhaustion
        assert abs(float(loss) - 1.0) < 1e-10

    def test_tranche_loss_in_range(self):
        from ql_jax.experimental.credit.cdo import SyntheticCdo, CdoTranche
        cdo = SyntheticCdo(
            portfolio_notional=100.0,
            tranches=[CdoTranche(0.03, 0.07, 1.0)],
            recovery_rate=0.4,
        )
        loss = cdo.tranche_loss(0.05, 0)  # midpoint
        assert 0.0 < float(loss) < 1.0
        assert abs(float(loss) - 0.5) < 1e-10

    def test_expected_tranche_loss_positive(self):
        from ql_jax.experimental.credit.cdo import expected_tranche_loss
        dp = jnp.array([0.02] * 100)  # 100 names, 2% default prob
        el = expected_tranche_loss(dp, 0.4, 0.03, 0.07, 0.3, n_scenarios=5000)
        assert el >= 0.0


class TestCDSOption:

    def test_black_cds_option_positive(self):
        from ql_jax.experimental.credit.cds_option import black_cds_option_price
        p = black_cds_option_price(0.01, 0.008, 0.4, 1.0, 4.0, option_type=1)
        assert float(p) > 0

    def test_payer_receiver_symmetry(self):
        from ql_jax.experimental.credit.cds_option import black_cds_option_price
        payer = black_cds_option_price(0.01, 0.01, 0.4, 1.0, 4.0, option_type=1)
        receiver = black_cds_option_price(0.01, 0.01, 0.4, 1.0, 4.0, option_type=-1)
        # ATM: payer ≈ receiver
        assert abs(float(payer) - float(receiver)) / float(payer) < 0.01


class TestLossModels:

    def test_gaussian_lhp_equity_tranche(self):
        from ql_jax.experimental.credit.loss_models import gaussian_lhp_loss
        el = gaussian_lhp_loss(0.03, 0.4, 0.2, 0.0, 0.03)
        assert 0.0 <= el <= 1.0

    def test_lhp_higher_corr_less_equity_loss(self):
        from ql_jax.experimental.credit.loss_models import gaussian_lhp_loss
        el_low = gaussian_lhp_loss(0.03, 0.4, 0.1, 0.0, 0.03)
        el_high = gaussian_lhp_loss(0.03, 0.4, 0.5, 0.0, 0.03)
        # Higher correlation spreads loss more evenly, reducing equity tranche loss
        assert el_high < el_low

    def test_nth_to_default(self):
        from ql_jax.experimental.credit.loss_models import (
            CreditBasket, NthToDefault)
        basket = CreditBasket(
            default_probs=jnp.array([0.03, 0.03, 0.03, 0.03, 0.03]),
            recovery_rates=jnp.array([0.4, 0.4, 0.4, 0.4, 0.4]),
            notionals=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        )
        ntd = NthToDefault(basket, nth=1, premium_rate=0.01, maturity=5.0)
        val = ntd.default_leg_value(0.3, discount_factor=0.95)
        assert val >= 0


class TestCorrelation:

    def test_base_correlation_structure(self):
        from ql_jax.experimental.credit.correlation import BaseCorrelationStructure
        bcs = BaseCorrelationStructure(
            detachment_points=jnp.array([0.03, 0.07, 0.10, 0.15, 0.30]),
            base_correlations=jnp.array([0.15, 0.25, 0.35, 0.45, 0.60]),
        )
        c = bcs.correlation(0.05)
        assert 0.15 < c < 0.25

    def test_flat_correlation(self):
        from ql_jax.experimental.credit.correlation import FlatCorrelation
        fc = FlatCorrelation(rho=0.3)
        assert fc.correlation(1.0) == 0.3

    def test_factor_spreaded_hazard(self):
        from ql_jax.experimental.credit.correlation import factor_spreaded_hazard_rate
        h = factor_spreaded_hazard_rate(0.02, 1.5)
        assert abs(h - 0.03) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# Phase 22: Callable Bonds
# ═══════════════════════════════════════════════════════════════════

class TestCallableBond:

    def test_callable_bond_less_than_straight(self):
        from ql_jax.instruments.callable_bond import (
            CallableFixedRateBond, CallScheduleEntry, black_callable_bond_price)
        bond = CallableFixedRateBond(
            face=100.0,
            coupon_rate=0.05,
            coupon_times=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            maturity=5.0,
            call_schedule=[CallScheduleEntry(2.0, 100.0, True),
                           CallScheduleEntry(3.0, 100.0, True)],
        )
        dc_t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        dc_v = jnp.exp(-0.04 * dc_t)  # 4% flat curve
        callable_price = black_callable_bond_price(bond, 0.10, dc_t, dc_v)
        # Straight bond value
        straight = sum(100*0.05*float(jnp.exp(-0.04*t)) for t in [1,2,3,4,5]) + 100*float(jnp.exp(-0.04*5))
        assert callable_price < straight

    def test_callable_bond_positive(self):
        from ql_jax.instruments.callable_bond import (
            CallableFixedRateBond, CallScheduleEntry, black_callable_bond_price)
        bond = CallableFixedRateBond(
            face=100.0,
            coupon_rate=0.06,
            coupon_times=jnp.array([0.5, 1.0, 1.5, 2.0]),
            maturity=2.0,
            call_schedule=[CallScheduleEntry(1.0, 100.0, True)],
        )
        dc_t = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
        dc_v = jnp.exp(-0.03 * dc_t)
        p = black_callable_bond_price(bond, 0.08, dc_t, dc_v)
        assert p > 0


# ═══════════════════════════════════════════════════════════════════
# Phase 23: Inflation Experimental
# ═══════════════════════════════════════════════════════════════════

class TestInflationExperimental:

    def test_cpi_surface_interpolation(self):
        from ql_jax.experimental.inflation.cpi_term_price_surface import (
            CPICapFloorTermPriceSurface)
        mats = jnp.array([1.0, 2.0, 5.0])
        strikes = jnp.array([0.01, 0.02, 0.03, 0.04])
        caps = jnp.ones((3, 4)) * 0.5  # dummy prices
        floors = jnp.ones((3, 4)) * 0.3
        surf = CPICapFloorTermPriceSurface(mats, strikes, caps, floors, 100.0)
        p = surf.cap_price(1.5, 0.025)
        assert p > 0

    def test_yoy_surface_creation(self):
        from ql_jax.experimental.inflation.cpi_term_price_surface import (
            YoYCapFloorTermPriceSurface)
        mats = jnp.array([1.0, 3.0, 5.0])
        strikes = jnp.array([0.01, 0.02, 0.03])
        caps = jnp.ones((3, 3)) * 0.4
        floors = jnp.ones((3, 3)) * 0.2
        surf = YoYCapFloorTermPriceSurface(mats, strikes, caps, floors)
        p = surf.floor_price(2.0, 0.015)
        assert p > 0

    def test_capped_floored_yoy_coupon(self):
        from ql_jax.experimental.inflation.cpi_term_price_surface import (
            CappedFlooredYoYInflationCoupon)
        c = CappedFlooredYoYInflationCoupon(yoy_rate=0.05, cap=0.03, floor=0.01)
        assert c.rate() == 0.03  # capped
        c2 = CappedFlooredYoYInflationCoupon(yoy_rate=-0.01, cap=0.03, floor=0.01)
        assert c2.rate() == 0.01  # floored


# ═══════════════════════════════════════════════════════════════════
# Phase 24: Exotic Options
# ═══════════════════════════════════════════════════════════════════

class TestExoticOptions:

    def test_himalaya_positive(self):
        from ql_jax.experimental.exotics.himalaya import (
            HimalayaOption, mc_himalaya_price)
        opt = HimalayaOption(
            spots=jnp.array([100.0, 100.0, 100.0]),
            maturity=1.0, n_obs=3, r=0.05,
            vols=jnp.array([0.2, 0.25, 0.3]),
            correlation=jnp.array([[1.0, 0.5, 0.3],
                                    [0.5, 1.0, 0.4],
                                    [0.3, 0.4, 1.0]]),
        )
        p = mc_himalaya_price(opt, n_paths=10000, seed=42)
        assert p > 0

    def test_everest_positive(self):
        from ql_jax.experimental.exotics.himalaya import (
            EverestOption, mc_everest_price)
        opt = EverestOption(
            spots=jnp.array([100.0, 100.0]),
            maturity=1.0, r=0.05,
            vols=jnp.array([0.2, 0.25]),
            correlation=jnp.array([[1.0, 0.5], [0.5, 1.0]]),
        )
        p = mc_everest_price(opt, n_paths=10000, seed=42)
        assert p > 0

    def test_pagoda_positive(self):
        from ql_jax.experimental.exotics.himalaya import (
            PagodaOption, mc_pagoda_price)
        opt = PagodaOption(
            spots=jnp.array([100.0, 100.0]),
            maturity=1.0, n_obs=4, cap_per_period=0.10,
            r=0.05,
            vols=jnp.array([0.2, 0.25]),
            correlation=jnp.array([[1.0, 0.5], [0.5, 1.0]]),
        )
        p = mc_pagoda_price(opt, n_paths=10000, seed=42)
        assert p >= 0

    def test_spread_option_positive(self):
        from ql_jax.experimental.exotics.himalaya import mc_spread_option_price
        p = mc_spread_option_price(100.0, 100.0, 0.0, 1.0, 0.05, 0.0, 0.0,
                                    0.2, 0.25, 0.5, option_type=1,
                                    n_paths=50000)
        assert p >= 0


# ═══════════════════════════════════════════════════════════════════
# Phase 25: CLV Models
# ═══════════════════════════════════════════════════════════════════

class TestCLV:

    def test_normal_clv_creation(self):
        from ql_jax.models.equity.clv import NormalCLVModel
        model = NormalCLVModel(
            market_vols=jnp.array([0.20, 0.22, 0.25, 0.28, 0.30]),
            strikes=jnp.array([80.0, 90.0, 100.0, 110.0, 120.0]),
            spot=100.0, r=0.05, q=0.0, T=1.0,
        )
        assert model.spot == 100.0

    def test_sqrt_clv_collocation(self):
        from ql_jax.models.equity.clv import SquareRootCLVModel
        model = SquareRootCLVModel(
            market_vols=jnp.array([0.20, 0.22, 0.25, 0.28, 0.30]),
            strikes=jnp.array([80.0, 90.0, 100.0, 110.0, 120.0]),
            spot=100.0, r=0.05, q=0.0, T=1.0,
        )
        pts = model.collocation_points(5)
        assert len(pts) == 5

    def test_stochastic_collocation(self):
        from ql_jax.models.equity.clv import stochastic_collocation_inverse_cdf
        cdf = jnp.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
        strikes = jnp.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
        coeffs = stochastic_collocation_inverse_cdf(cdf, strikes, 5)
        assert len(coeffs) == 5


# ═══════════════════════════════════════════════════════════════════
# Phase 26: Commodities
# ═══════════════════════════════════════════════════════════════════

class TestCommodities:

    def test_commodity_curve(self):
        from ql_jax.experimental.commodities.energy import CommodityCurve
        curve = CommodityCurve(
            times=jnp.array([0.25, 0.5, 1.0, 2.0]),
            prices=jnp.array([50.0, 52.0, 55.0, 58.0]),
        )
        assert abs(curve.forward_price(0.75) - 53.5) < 0.1

    def test_energy_future_value(self):
        from ql_jax.experimental.commodities.energy import (
            CommodityType, CommodityIndex, CommodityCurve, EnergyFuture)
        ct = CommodityType("NatGas", "MMBtu")
        idx = CommodityIndex("HH", ct)
        curve = CommodityCurve(jnp.array([0.5, 1.0]), jnp.array([3.5, 3.8]))
        fut = EnergyFuture(idx, 0.5, 1.0, quantity=1000.0)
        val = fut.fair_value(curve)
        assert val > 0

    def test_energy_swap_npv(self):
        from ql_jax.experimental.commodities.energy import (
            CommodityType, CommodityIndex, CommodityCurve, EnergySwap)
        ct = CommodityType("Power", "MWh")
        idx = CommodityIndex("ERCOT", ct)
        curve = CommodityCurve(jnp.array([0.25, 0.5, 0.75, 1.0]),
                               jnp.array([40.0, 42.0, 41.0, 43.0]))
        swap = EnergySwap(idx, fixed_price=41.0,
                          observation_times=jnp.array([0.25, 0.5, 0.75, 1.0]))
        df = jnp.exp(-0.03 * jnp.array([0.25, 0.5, 0.75, 1.0]))
        npv = swap.fair_value(curve, df)
        assert isinstance(npv, float)

    def test_geman_roncoroni_process(self):
        from ql_jax.experimental.commodities.energy import geman_roncoroni_process
        t, paths = geman_roncoroni_process(
            S0=50.0, mu=50.0, sigma=5.0, alpha=2.0,
            lam_up=1.0, lam_down=0.5, eta_up=5.0, eta_down=10.0,
            T=1.0, n_steps=100, n_paths=1000, seed=42)
        assert paths.shape == (1000, 101)
        assert float(paths[:, 0].mean()) == 50.0


# ═══════════════════════════════════════════════════════════════════
# Phase 27: Cat Bonds
# ═══════════════════════════════════════════════════════════════════

class TestCatBonds:

    def test_cat_bond_beta_risk(self):
        from ql_jax.experimental.catbonds.cat_bond import (
            CatBond, BetaRisk, mc_cat_bond_price)
        risk = BetaRisk(alpha=2.0, beta=10.0, intensity=0.5)
        bond = CatBond(notional=100.0, coupon_rate=0.08, maturity=3.0,
                       attachment=0.1, exhaustion=0.3, risk=risk)
        p = mc_cat_bond_price(bond, r=0.03, n_paths=10000)
        assert 0.5 < p < 2.0  # reasonable range

    def test_cat_bond_event_set(self):
        from ql_jax.experimental.catbonds.cat_bond import (
            CatBond, EventSetRisk, mc_cat_bond_price)
        risk = EventSetRisk(
            event_probs=jnp.array([0.02, 0.01, 0.005]),
            event_losses=jnp.array([0.05, 0.15, 0.40]),
        )
        bond = CatBond(notional=100.0, coupon_rate=0.06, maturity=5.0,
                       attachment=0.10, exhaustion=0.50, risk=risk)
        p = mc_cat_bond_price(bond, r=0.03, n_paths=10000)
        assert p > 0

    def test_cat_bond_no_catastrophe_par(self):
        """With zero event intensity, bond should price near par + coupon."""
        from ql_jax.experimental.catbonds.cat_bond import (
            CatBond, BetaRisk, mc_cat_bond_price)
        risk = BetaRisk(alpha=2.0, beta=10.0, intensity=0.0)  # no events
        bond = CatBond(notional=100.0, coupon_rate=0.05, maturity=2.0,
                       attachment=0.1, exhaustion=0.3, risk=risk)
        p = mc_cat_bond_price(bond, r=0.03, n_paths=1000)
        # Should be about exp(-0.03*2) * (1 + 0.05*2) ≈ 0.942 * 1.1 ≈ 1.036
        assert abs(p - 1.036) < 0.05


# ═══════════════════════════════════════════════════════════════════
# Phase 28: Average OIS and Irregular Swaptions
# ═══════════════════════════════════════════════════════════════════

class TestAverageOIS:

    def test_arithmetic_ois_npv(self):
        from ql_jax.experimental.averageois.average_ois import ArithmeticAverageOIS
        ois = ArithmeticAverageOIS(
            notional=1e6,
            fixed_rate=0.03,
            payment_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            overnight_rates=jnp.ones((5, 4)) * 0.035,
        )
        df = jnp.exp(-0.03 * jnp.array([0.25, 0.5, 0.75, 1.0]))
        npv = ois.npv(df)
        assert npv > 0  # floating > fixed

    def test_irregular_swap_npv(self):
        from ql_jax.experimental.averageois.average_ois import IrregularSwap
        swap = IrregularSwap(
            fixed_notionals=jnp.array([1e6, 1e6, 0.5e6]),
            float_notionals=jnp.array([1e6, 1e6, 0.5e6]),
            fixed_rates=jnp.array([0.04, 0.04, 0.04]),
            float_spreads=jnp.zeros(3),
            payment_times=jnp.array([1.0, 2.0, 3.0]),
        )
        fwd = jnp.array([0.045, 0.05, 0.055])
        df = jnp.exp(-0.04 * jnp.array([1.0, 2.0, 3.0]))
        npv = swap.npv(fwd, df)
        assert npv > 0  # forwards above fixed rate


# ═══════════════════════════════════════════════════════════════════
# Phase 29: Experimental FD
# ═══════════════════════════════════════════════════════════════════

class TestExperimentalFD:

    def test_ou_operator(self):
        from ql_jax.experimental.finitedifferences.energy_fd import (
            build_extended_ou_operator)
        grid = jnp.linspace(-0.5, 0.5, 50)
        op = build_extended_ou_operator(2.0, 0.0, 0.1, 0.05, grid)
        assert op.diag.shape == (50,)

    def test_ou_vanilla_price_positive(self):
        from ql_jax.experimental.finitedifferences.energy_fd import fd_ou_vanilla_price
        # Use parameters where there's significant volatility
        p = fd_ou_vanilla_price(50.0, 45.0, 1.0, 0.05, 0.5, 50.0, 10.0,
                                option_type=1, n_x=200, n_t=200)
        assert p > 0

    def test_glued_mesher(self):
        from ql_jax.experimental.finitedifferences.energy_fd import Glued1dMesher
        m = Glued1dMesher.create(0.0, 5.0, 10.0, 20, 30)
        locs = m.locations()
        assert len(locs) == 49  # 20 + 30 - 1
        assert float(locs[0]) == 0.0
        assert abs(float(locs[-1]) - 10.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# Phase 30: Metaheuristic Optimization
# ═══════════════════════════════════════════════════════════════════

class TestMetaheuristic:

    def test_pso_rosenbrock(self):
        from ql_jax.math.optimization.metaheuristic import particle_swarm_optimization
        def rosenbrock(x):
            return float((1 - x[0])**2 + 100*(x[1] - x[0]**2)**2)
        bounds = jnp.array([[-5.0, 5.0], [-5.0, 5.0]])
        best_x, best_val = particle_swarm_optimization(
            rosenbrock, bounds, n_particles=20, n_iterations=100, seed=42)
        assert best_val < 1.0  # should find near global min

    def test_hybrid_sa(self):
        from ql_jax.math.optimization.metaheuristic import hybrid_simulated_annealing
        def sphere(x):
            return float(jnp.sum(x**2))
        bounds = jnp.array([[-5.0, 5.0], [-5.0, 5.0]])
        best_x, best_val = hybrid_simulated_annealing(
            sphere, jnp.array([3.0, 3.0]), bounds,
            n_iterations=100, seed=42)
        assert best_val < 1.0

    def test_multidim_quadrature(self):
        from ql_jax.math.optimization.metaheuristic import multidim_quadrature
        # Integrate x^2 + y^2 over [0,1]x[0,1] = 1/3 + 1/3 = 2/3
        def f(x):
            return float(x[0]**2 + x[1]**2)
        bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
        result = multidim_quadrature(f, bounds, n_points=5)
        assert abs(result - 2.0/3) < 0.01


# ═══════════════════════════════════════════════════════════════════
# Phase 31: Short Rate Model Extensions
# ═══════════════════════════════════════════════════════════════════

class TestShortRateExtensions:

    def test_generalized_hw_bond_price(self):
        from ql_jax.models.shortrate.generalized_hull_white import GeneralizedHullWhite
        model = GeneralizedHullWhite(
            a=jnp.array([0.1, 0.3]),
            sigma=jnp.array([0.01, 0.005]),
        )
        P = model.discount_bond(0.05, 5.0)
        assert 0 < float(P) < 1

    def test_ghw_simulate(self):
        from ql_jax.models.shortrate.generalized_hull_white import GeneralizedHullWhite
        model = GeneralizedHullWhite(
            a=jnp.array([0.1]),
            sigma=jnp.array([0.01]),
        )
        t, paths = model.simulate(0.05, 1.0, 50, 100, seed=42)
        assert paths.shape == (100, 51)
        assert abs(float(paths[:, 0].mean()) - 0.05) < 1e-12

    def test_andersen_qe_positive(self):
        from ql_jax.models.shortrate.generalized_hull_white import andersen_qe_step
        key = jax.random.PRNGKey(42)
        v = jnp.array(0.04)
        v_new = andersen_qe_step(v, 1.5, 0.04, 0.3, 0.01, key)
        assert float(v_new) >= 0


# ═══════════════════════════════════════════════════════════════════
# Phase 32: Missing Cashflows
# ═══════════════════════════════════════════════════════════════════

class TestCashflows:

    def test_stripped_cap_floored_coupon(self):
        from ql_jax.cashflows.coupon_pricers import StrippedCapFlooredCoupon
        c = StrippedCapFlooredCoupon(rate=0.05, notional=1e6, accrual=0.25,
                                     cap=0.04)
        # Pure coupon (without cap effect)
        assert c.pure_coupon_amount() == 0.05 * 1e6 * 0.25
        # Caplet value should be positive
        val = c.caplet_amount(vol=0.005, df=0.98, T=0.5)
        assert val >= 0

    def test_quanto_coupon(self):
        from ql_jax.cashflows.coupon_pricers import quanto_coupon_amount
        amt = quanto_coupon_amount(0.05, 1.2, 0.1, 0.005, -0.3,
                                    1e6, 0.25, 0.98)
        assert amt > 0

    def test_linear_tsr_cms(self):
        from ql_jax.cashflows.coupon_pricers import linear_tsr_cms_rate
        cms = linear_tsr_cms_rate(0.03, 4.5, 0.20, 5.0)
        assert cms > 0.03  # convexity adjustment is positive

    def test_dividend_schedule(self):
        from ql_jax.cashflows.coupon_pricers import DividendSchedule
        ds = DividendSchedule(
            ex_div_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            amounts=jnp.array([1.0, 1.0, 1.0, 1.0]),
        )
        pv = ds.pv(jnp.exp(-0.05 * ds.ex_div_times))
        assert 3.5 < pv < 4.0


# ═══════════════════════════════════════════════════════════════════
# Phase 33: Optionlet Extensions
# ═══════════════════════════════════════════════════════════════════

class TestOptionletExtensions:

    def test_optionlet_stripper2(self):
        from ql_jax.termstructures.volatility.optionlet_extended import (
            OptionletStripper2, StrippedOptionlet)
        base = StrippedOptionlet(
            expiries=jnp.array([1.0, 2.0, 3.0, 5.0]),
            strikes=jnp.array([0.02, 0.03, 0.04]),
            optionlet_vols=jnp.ones((4, 3)) * 0.20,
        )
        stripper = OptionletStripper2(base, atm_cap_vols=jnp.array([0.22, 0.23, 0.24, 0.25]))
        adjusted = stripper.adjust()
        # ATM column (mid) should be scaled to match atm_cap_vols
        assert adjusted.optionlet_vols.shape == (4, 3)
        assert float(adjusted.optionlet_vols[0, 1]) > 0.20  # adjusted upward

    def test_stripped_optionlet_adapter(self):
        from ql_jax.termstructures.volatility.optionlet_extended import (
            StrippedOptionlet, StrippedOptionletAdapter)
        stripped = StrippedOptionlet(
            expiries=jnp.array([1.0, 2.0, 3.0]),
            strikes=jnp.array([0.02, 0.03, 0.04]),
            optionlet_vols=jnp.array([[0.20, 0.22, 0.25],
                                       [0.21, 0.23, 0.26],
                                       [0.22, 0.24, 0.27]]),
        )
        adapter = StrippedOptionletAdapter(stripped)
        v = adapter.vol(1.5)
        assert 0.20 < v < 0.25


# ═══════════════════════════════════════════════════════════════════
# Phase 34: Variance Option and MC Engines
# ═══════════════════════════════════════════════════════════════════

class TestVarianceOption:

    def test_variance_option_positive(self):
        from ql_jax.experimental.varianceoption.variance_option import (
            VarianceOption, integral_heston_variance_option)
        opt = VarianceOption(strike_var=0.04, notional=1e6, maturity=1.0)
        p = integral_heston_variance_option(
            opt, v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.05)
        assert p > 0

    def test_mc_digital_call(self):
        from ql_jax.experimental.varianceoption.variance_option import mc_digital_price
        p = mc_digital_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2,
                              option_type=1, n_paths=50000)
        # Digital call ≈ N(d2) * disc ≈ 0.56 * 0.95 ≈ 0.53
        assert 0.3 < p < 0.8

    def test_mc_digital_put(self):
        from ql_jax.experimental.varianceoption.variance_option import mc_digital_price
        p = mc_digital_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2,
                              option_type=-1, n_paths=50000)
        assert 0.2 < p < 0.7


# ═══════════════════════════════════════════════════════════════════
# Phase 35: FX Experimental
# ═══════════════════════════════════════════════════════════════════

class TestFXExperimental:

    def test_black_delta_strike(self):
        from ql_jax.experimental.fx.black_delta import BlackDeltaCalculator
        calc = BlackDeltaCalculator(
            option_type=1, delta_type='forward',
            spot=1.20, dom_df=0.95, for_df=0.97,
            vol=0.10, T=1.0)
        K = calc.strike_from_delta(0.25)
        assert K > calc.forward  # OTM call strike > forward

    def test_black_delta_atm(self):
        from ql_jax.experimental.fx.black_delta import BlackDeltaCalculator
        calc = BlackDeltaCalculator(
            option_type=1, delta_type='forward',
            spot=1.20, dom_df=0.95, for_df=0.97,
            vol=0.10, T=1.0)
        K_atm = calc.atm_strike('delta_neutral')
        assert K_atm > 0

    def test_garman_kohlhagen(self):
        from ql_jax.experimental.fx.black_delta import garman_kohlhagen_price
        p = garman_kohlhagen_price(1.20, 1.20, 1.0, 0.03, 0.01, 0.10)
        assert p > 0

    def test_gk_put_call_parity(self):
        from ql_jax.experimental.fx.black_delta import garman_kohlhagen_price
        S, K, T, r_d, r_f, vol = 1.20, 1.20, 1.0, 0.03, 0.01, 0.10
        call = garman_kohlhagen_price(S, K, T, r_d, r_f, vol, 1)
        put = garman_kohlhagen_price(S, K, T, r_d, r_f, vol, -1)
        # C - P = S*exp(-rf*T) - K*exp(-rd*T)
        parity = S * jnp.exp(-r_f * T) - K * jnp.exp(-r_d * T)
        assert abs(call - put - float(parity)) < 0.001
