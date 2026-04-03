"""Tests for additional gaps phases 13-20.

Phase 13: Extended FD operators
Phase 14: Extended meshers (already exist — verify)
Phase 15: Step conditions, RND calculators
Phase 16: FD vanilla engines (Bates, CEV, SABR, Hull-White)
Phase 17: Missing instruments
Phase 18: Missing processes
Phase 19-20: Term structures (yield, vol)
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ── Phase 13: Extended FD Operators ─────────────────────────────────

class TestExtendedOperators:

    def test_cev_operator_construction(self):
        from ql_jax.methods.finitedifferences.extended_operators import fdm_cev_operator
        x = jnp.linspace(50.0, 150.0, 50)
        L = fdm_cev_operator(x, 0.05, 0.02, 0.3, 0.5)
        assert L.diag.shape == (50,)
        assert L.lower.shape == (49,)

    def test_cev_operator_apply(self):
        from ql_jax.methods.finitedifferences.extended_operators import fdm_cev_operator
        x = jnp.linspace(50.0, 150.0, 50)
        L = fdm_cev_operator(x, 0.05, 0.02, 0.3, 0.5)
        v = jnp.maximum(x - 100.0, 0.0)
        Lv = L.apply(v)
        assert Lv.shape == (50,)
        assert jnp.isfinite(Lv).all()

    def test_hw_operator_construction(self):
        from ql_jax.methods.finitedifferences.extended_operators import fdm_hull_white_operator
        r = jnp.linspace(-0.02, 0.12, 50)
        L = fdm_hull_white_operator(r, 0.03, 0.01, 0.05)
        assert L.diag.shape == (50,)

    def test_hw_operator_discount_effect(self):
        """HW operator diagonal should reflect short-rate discounting."""
        from ql_jax.methods.finitedifferences.extended_operators import fdm_hull_white_operator
        r = jnp.linspace(-0.02, 0.12, 50)
        L = fdm_hull_white_operator(r, 0.03, 0.01, 0.05)
        # Diagonal should be negative (discounting) at positive rates
        # due to -r term
        mid = L.diag[25]
        assert float(mid) < 0  # rate 0.05 region

    def test_sabr_operator_construction(self):
        from ql_jax.methods.finitedifferences.extended_operators import fdm_sabr_operator
        f = jnp.linspace(0.01, 0.10, 50)
        L = fdm_sabr_operator(f, 0.3, 0.5, -0.3, 0.4)
        assert L.diag.shape == (50,)

    def test_bates_drift_adjustment(self):
        from ql_jax.methods.finitedifferences.extended_operators import fdm_bates_operator
        x = jnp.linspace(3.5, 5.5, 50)
        v = jnp.linspace(0.01, 0.1, 20)
        drift = fdm_bates_operator(x, v, 0.05, 0.02, 1.0, 0.04, 0.3, -0.7,
                                    0.5, -0.1, 0.2)
        # With jumps, drift is adjusted: drift_x - jump_comp
        # jump_comp = 0.5 * (exp(-0.1 + 0.5*0.04) - 1) ≈ 0.5 * (exp(-0.08) - 1) < 0
        # so drift should be > r-q-0.5 (less negative) due to negative jump comp
        base_drift = 0.05 - 0.02 - 0.5  # = -0.47
        assert float(drift) > base_drift  # jump compensation makes it less negative


# ── Phase 14-15: Meshers (existing) & RND ──────────────────────────

class TestExistingMeshers:

    def test_predefined_mesher(self):
        from ql_jax.methods.finitedifferences.meshers import Predefined1dMesher
        grid = jnp.array([1.0, 2.0, 3.0, 5.0, 10.0])
        m = Predefined1dMesher(grid)
        assert m.size == 5
        assert abs(m.low - 1.0) < 1e-10
        assert abs(m.high - 10.0) < 1e-10

    def test_exponential_jump_mesher(self):
        from ql_jax.methods.finitedifferences.meshers import ExponentialJump1dMesher
        m = ExponentialJump1dMesher(0.0, 10.0, 100, 5.0, beta=2.0)
        locs = m.locations()
        assert locs.shape == (100,)
        assert float(locs[0]) >= 0.0
        assert float(locs[-1]) <= 10.0

    def test_fdm_mesher_composite(self):
        from ql_jax.methods.finitedifferences.meshers import (
            Uniform1dMesher, FdmMesherComposite,
        )
        m1 = Uniform1dMesher(0.0, 1.0, 10)
        m2 = Uniform1dMesher(0.0, 0.5, 5)
        comp = FdmMesherComposite((m1, m2))
        assert comp.ndim == 2
        assert comp.sizes() == (10, 5)
        assert comp.total_size() == 50


class TestRND:

    def test_bsm_rnd_integrates_to_one(self):
        from ql_jax.methods.finitedifferences.rnd import bsm_rnd
        x = jnp.linspace(5.0, 500.0, 2000)
        p = bsm_rnd(100.0, 0.05, 0.02, 0.25, 1.0, x)
        integral = jnp.trapezoid(p, x)
        assert abs(float(integral) - 1.0) < 0.01

    def test_bsm_rnd_positive(self):
        from ql_jax.methods.finitedifferences.rnd import bsm_rnd
        x = jnp.linspace(5.0, 500.0, 1000)
        p = bsm_rnd(100.0, 0.05, 0.02, 0.25, 1.0, x)
        assert (p >= -1e-10).all()

    def test_bsm_rnd_mean(self):
        """Mean under RND should be the forward price."""
        from ql_jax.methods.finitedifferences.rnd import bsm_rnd
        S, r, q, T = 100.0, 0.05, 0.02, 1.0
        x = jnp.linspace(10.0, 400.0, 2000)
        p = bsm_rnd(S, r, q, 0.25, T, x)
        mean = jnp.trapezoid(x * p, x) / jnp.trapezoid(p, x)
        forward = S * jnp.exp((r - q) * T)
        # RND mean under P measure = exp(-rT)*forward? No, RND mean = forward * exp(-rT)...
        # Actually bsm_rnd includes the exp(-rT) factor, so E[x] = forward
        # is when we discount: integral of x*p(x)dx = exp(-rT)*forward
        # But our RND is the risk-neutral density (Q-measure), so E^Q[x] = forward
        assert abs(float(mean) - float(forward)) / float(forward) < 0.02


# ── Phase 16: FD Vanilla Engines ───────────────────────────────────

class TestFdBates:

    def test_bates_call_positive(self):
        from ql_jax.engines.fd.bates import fd_bates_price
        p = fd_bates_price(100.0, 100.0, 1.0, 0.05, 0.02,
                           0.04, 1.0, 0.04, 0.3, -0.7,
                           0.5, -0.1, 0.2, option_type=1,
                           n_x=60, n_v=20, n_t=50)
        assert float(p) > 0

    def test_bates_put_positive(self):
        from ql_jax.engines.fd.bates import fd_bates_price
        p = fd_bates_price(100.0, 100.0, 1.0, 0.05, 0.02,
                           0.04, 1.0, 0.04, 0.3, -0.7,
                           0.5, -0.1, 0.2, option_type=-1,
                           n_x=60, n_v=20, n_t=50)
        assert float(p) > 0

    def test_bates_zero_jumps_matches_heston(self):
        from ql_jax.engines.fd.bates import fd_bates_price
        from ql_jax.engines.fd.heston import fd_heston_price
        common = dict(S=100.0, K=100.0, T=0.5, r=0.05, q=0.0,
                      v0=0.04, kappa=1.0, theta=0.04, rho=-0.7,
                      n_x=60, n_v=20, n_t=50)
        bates = fd_bates_price(**common, xi=0.3, lam_j=0.0, mu_j=0.0, sigma_j=0.0, option_type=1)
        heston = fd_heston_price(**common, sigma_v=0.3, option_type=1)
        assert abs(float(bates) - float(heston)) / float(heston) < 0.05


class TestFdCEV:

    def test_cev_call_positive(self):
        from ql_jax.engines.fd.cev import fd_cev_price
        p = fd_cev_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.3, 0.5,
                         option_type=1, n_x=100, n_t=100)
        assert float(p) > 0

    def test_cev_put_positive(self):
        from ql_jax.engines.fd.cev import fd_cev_price
        p = fd_cev_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.3, 0.5,
                         option_type=-1, n_x=100, n_t=100)
        assert float(p) > 0

    def test_cev_beta1_near_bs(self):
        """CEV with beta=1 should approach BS."""
        from ql_jax.engines.fd.cev import fd_cev_price
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        sigma = 0.25
        p_cev = fd_cev_price(100.0, 100.0, 0.5, 0.05, 0.0, sigma, 1.0,
                             option_type=1, n_x=150, n_t=200)
        p_bs = float(black_scholes_price(100.0, 100.0, 0.5, 0.05, 0.0, sigma, 1))
        assert abs(p_cev - p_bs) / p_bs < 0.03

    def test_cev_american_geq_european(self):
        from ql_jax.engines.fd.cev import fd_cev_price
        eu = fd_cev_price(100.0, 105.0, 1.0, 0.05, 0.0, 0.3, 0.5,
                          option_type=-1, n_x=100, n_t=100, is_american=False)
        am = fd_cev_price(100.0, 105.0, 1.0, 0.05, 0.0, 0.3, 0.5,
                          option_type=-1, n_x=100, n_t=100, is_american=True)
        assert am >= eu - 0.1


class TestFdSABR:

    def test_sabr_call_positive(self):
        from ql_jax.engines.fd.sabr import fd_sabr_price
        p = fd_sabr_price(0.05, 0.05, 1.0, 0.03, 0.3, 0.5, -0.3, 0.4,
                          option_type=1, n_x=100, n_t=100)
        assert float(p) > 0

    def test_sabr_put_positive(self):
        from ql_jax.engines.fd.sabr import fd_sabr_price
        p = fd_sabr_price(0.05, 0.05, 1.0, 0.03, 0.3, 0.5, -0.3, 0.4,
                          option_type=-1, n_x=100, n_t=100)
        assert float(p) > 0


class TestFdHullWhite:

    def test_hw_bond_call_positive(self):
        from ql_jax.engines.fd.hull_white import fd_hull_white_bond_option
        disc = lambda t: jnp.exp(-0.05 * t)
        p = fd_hull_white_bond_option(
            face=100.0, K=90.0, T_option=1.0, T_bond=5.0, r0=0.05,
            a=0.1, sigma=0.01, discount_curve_fn=disc,
            option_type=1, n_r=80, n_t=80,
        )
        assert float(p) > 0

    def test_hw_bond_put_positive(self):
        from ql_jax.engines.fd.hull_white import fd_hull_white_bond_option
        disc = lambda t: jnp.exp(-0.05 * t)
        p = fd_hull_white_bond_option(
            face=100.0, K=100.0, T_option=1.0, T_bond=5.0, r0=0.05,
            a=0.1, sigma=0.01, discount_curve_fn=disc,
            option_type=-1, n_r=80, n_t=80,
        )
        assert float(p) > 0

    def test_hw_higher_vol_increases_option(self):
        from ql_jax.engines.fd.hull_white import fd_hull_white_bond_option
        disc = lambda t: jnp.exp(-0.05 * t)
        kwargs = dict(face=100.0, K=95.0, T_option=1.0, T_bond=5.0, r0=0.05,
                      a=0.1, discount_curve_fn=disc, option_type=1, n_r=80, n_t=80)
        p_low = fd_hull_white_bond_option(**kwargs, sigma=0.005)
        p_high = fd_hull_white_bond_option(**kwargs, sigma=0.02)
        assert p_high > p_low


# ── Phase 17: Missing Instruments ──────────────────────────────────

class TestFloatFloatSwap:

    def test_float_float_swap_npv(self):
        from ql_jax.instruments.float_float_swap import FloatFloatSwap
        disc = lambda t: jnp.exp(-0.05 * t)
        swap = FloatFloatSwap(
            notional=1e6,
            payment_times_1=jnp.array([0.5, 1.0]),
            payment_times_2=jnp.array([0.5, 1.0]),
            accrual_1=jnp.array([0.5, 0.5]),
            accrual_2=jnp.array([0.5, 0.5]),
            spread_1=0.0, spread_2=0.001,
        )
        fwd1 = jnp.array([0.05, 0.05])
        fwd2 = jnp.array([0.05, 0.05])
        npv = swap.npv(fwd1, fwd2, disc)
        assert isinstance(float(npv), float)


class TestNonstandardSwap:

    def test_nonstandard_swap_construction(self):
        from ql_jax.instruments.nonstandard_swap import NonstandardSwap
        swap = NonstandardSwap(
            fixed_notionals=jnp.array([1e6, 9e5, 8e5]),
            floating_notionals=jnp.array([1e6, 9e5, 8e5]),
            fixed_rates=jnp.array([0.05, 0.055, 0.06]),
            fixed_payment_times=jnp.array([1.0, 2.0, 3.0]),
            floating_payment_times=jnp.array([1.0, 2.0, 3.0]),
            fixed_accrual=jnp.array([1.0, 1.0, 1.0]),
            floating_accrual=jnp.array([1.0, 1.0, 1.0]),
        )
        disc = lambda t: jnp.exp(-0.05 * t)
        fwd = jnp.array([0.05, 0.05, 0.05])
        npv = swap.npv(fwd, disc)
        assert isinstance(float(npv), float)


class TestBmaSwap:

    def test_bma_swap_construction(self):
        from ql_jax.instruments.bma_swap import BmaSwap
        swap = BmaSwap(
            notional=1e6,
            bma_payment_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            libor_payment_times=jnp.array([0.5, 1.0]),
            bma_accrual=jnp.array([0.25] * 4),
            libor_accrual=jnp.array([0.5, 0.5]),
        )
        disc = lambda t: jnp.exp(-0.05 * t)
        bma_rates = jnp.array([0.035] * 4)
        libor_rates = jnp.array([0.05, 0.05])
        npv = swap.npv(bma_rates, libor_rates, disc)
        assert isinstance(float(npv), float)

    def test_bma_ratio(self):
        from ql_jax.instruments.bma_swap import BmaSwap
        swap = BmaSwap(
            notional=1e6,
            bma_payment_times=jnp.array([0.5, 1.0]),
            libor_payment_times=jnp.array([0.5, 1.0]),
            bma_accrual=jnp.array([0.5, 0.5]),
            libor_accrual=jnp.array([0.5, 0.5]),
        )
        ratio = swap.bma_ratio(jnp.array([0.035, 0.035]), jnp.array([0.05, 0.05]))
        assert abs(float(ratio) - 0.7) < 0.01


class TestCallabilitySchedule:

    def test_callability_schedule_creation(self):
        from ql_jax.instruments.callability import (
            CallabilitySchedule, CallabilityType,
        )
        sched = CallabilitySchedule.from_dates_prices(
            dates=[2.0, 3.0, 4.0], prices=[100.0, 100.0, 100.0],
            type_=CallabilityType.CALL,
        )
        assert len(sched.callabilities) == 3
        assert float(sched.dates()[0]) == 2.0

    def test_next_callability(self):
        from ql_jax.instruments.callability import (
            CallabilitySchedule, CallabilityType,
        )
        sched = CallabilitySchedule.from_dates_prices(
            [1.0, 2.0, 3.0], [100.0, 100.0, 100.0],
        )
        c = sched.next_callability(1.5)
        assert c is not None
        assert abs(c.date - 2.0) < 1e-10


class TestEquityTRS:

    def test_equity_trs_npv(self):
        from ql_jax.instruments.equity_trs import EquityTotalReturnSwap
        trs = EquityTotalReturnSwap(
            notional=1e6, S0=100.0,
            payment_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            accrual_fractions=jnp.array([0.25] * 4),
            financing_spread=0.01,
        )
        disc = lambda t: jnp.exp(-0.05 * t)
        npv = trs.npv(110.0, jnp.array([0.05] * 4), 0.02, disc)
        assert isinstance(float(npv), float)


class TestZeroCouponSwap:

    def test_zc_swap_npv(self):
        from ql_jax.instruments.zero_coupon_swap import ZeroCouponSwap
        swap = ZeroCouponSwap(
            notional=1e6, fixed_rate=0.05, maturity=5.0,
            float_payment_times=jnp.arange(1.0, 6.0),
            float_accrual=jnp.ones(5),
        )
        disc = lambda t: jnp.exp(-0.05 * t)
        fwd = jnp.array([0.05] * 5)
        npv = swap.npv(fwd, disc)
        assert isinstance(float(npv), float)


class TestSwingOption:

    def test_swing_construction(self):
        from ql_jax.instruments.swing import VanillaSwingOption
        opt = VanillaSwingOption(
            exercise_dates=jnp.arange(0.25, 1.25, 0.25),
            strike=50.0, min_exercises=1, max_exercises=3,
            notional_per_exercise=1000.0,
        )
        assert opt.n_dates == 4
        assert opt.max_exercises == 3


# ── Phase 18: Missing Processes ────────────────────────────────────

class TestHybridHestonHW:

    def test_single_step(self):
        from ql_jax.processes.hybrid_heston_hw import HybridHestonHullWhiteProcess
        proc = HybridHestonHullWhiteProcess(S0=100.0, v0=0.04, r0=0.05)
        S, v, r = proc.evolve(1.0 / 252.0, 1000, jax.random.PRNGKey(0))
        assert S.shape == (1000,)
        assert (S > 0).all()
        assert (v >= 0).all()

    def test_simulate(self):
        from ql_jax.processes.hybrid_heston_hw import HybridHestonHullWhiteProcess
        proc = HybridHestonHullWhiteProcess(S0=100.0, v0=0.04, r0=0.05)
        S, v, r = proc.simulate(1.0, 50, 500, jax.random.PRNGKey(1))
        assert S.shape == (51, 500)
        assert jnp.isfinite(S).all()
        # Mean should be near forward
        mean_S = jnp.mean(S[-1])
        assert abs(float(mean_S) - 100.0 * jnp.exp(0.05)) / 100.0 < 0.15


class TestExtendedOU:

    def test_ou_mean_variance(self):
        from ql_jax.processes.extended_ou import ExtendedOrnsteinUhlenbeckProcess
        proc = ExtendedOrnsteinUhlenbeckProcess(x0=1.0, speed=2.0, level=0.0, sigma=0.5)
        assert abs(float(proc.mean(10.0)) - 0.0) < 0.01
        var = proc.variance(10.0)
        # Long-run var = sigma^2/(2*speed) = 0.25/4 = 0.0625
        assert abs(float(var) - 0.0625) < 0.01

    def test_ou_simulate(self):
        from ql_jax.processes.extended_ou import ExtendedOrnsteinUhlenbeckProcess
        proc = ExtendedOrnsteinUhlenbeckProcess(x0=1.0, speed=2.0, level=0.0, sigma=0.5)
        paths = proc.simulate(5.0, 100, 1000, jax.random.PRNGKey(2))
        assert paths.shape == (101, 1000)
        # Mean at T=5 should be close to level=0
        assert abs(float(jnp.mean(paths[-1]))) < 0.1

    def test_ou_with_jumps(self):
        from ql_jax.processes.extended_ou import ExtOUWithJumpsProcess
        proc = ExtOUWithJumpsProcess(x0=0.0, speed=1.0, level=0.0, sigma=0.1,
                                      lam=2.0, mu_j=0.0, sigma_j=0.05)
        paths = proc.simulate(1.0, 50, 500, jax.random.PRNGKey(3))
        assert paths.shape == (51, 500)
        assert jnp.isfinite(paths).all()


class TestForwardMeasureProcess:

    def test_simulate(self):
        from ql_jax.processes.forward_measure import ForwardMeasureProcess
        proc = ForwardMeasureProcess(x0=0.05, sigma=0.01, T_forward=1.0)
        paths = proc.simulate(1.0, 50, 1000, jax.random.PRNGKey(4))
        assert paths.shape == (51, 1000)
        assert jnp.isfinite(paths).all()


# ── Phase 19-20: Term Structures ───────────────────────────────────

class TestGaussian1dVol:

    def test_swaption_vol_positive(self):
        from ql_jax.termstructures.volatility.gaussian1d_vol import (
            gaussian1d_swaption_vol,
        )
        disc = lambda t: jnp.exp(-0.05 * t)
        vol = gaussian1d_swaption_vol(1.0, 5.0, 0.03, 0.01, disc)
        assert float(vol) > 0

    def test_smile_section(self):
        from ql_jax.termstructures.volatility.gaussian1d_vol import (
            gaussian1d_smile_section,
        )
        disc = lambda t: jnp.exp(-0.05 * t)
        bvol = gaussian1d_smile_section(0.05, 0.05, 1.0, 5.0, 0.03, 0.01, disc)
        assert bvol > 0


class TestCmsMarket:

    def test_cms_rate(self):
        from ql_jax.termstructures.volatility.cms_market import CmsMarket
        mkt = CmsMarket(
            tenors=jnp.array([2.0, 5.0, 10.0]),
            forward_rates=jnp.array([0.03, 0.04, 0.045]),
            vol_atm=jnp.array([0.15, 0.12, 0.10]),
        )
        # CMS rate should be > forward (convexity adjustment)
        cms = mkt.cms_rate(1)
        assert cms > 0.04

    def test_calibrate(self):
        from ql_jax.termstructures.volatility.cms_market import (
            CmsMarket, calibrate_cms_model,
        )
        mkt = CmsMarket(
            tenors=jnp.array([5.0, 10.0]),
            forward_rates=jnp.array([0.04, 0.045]),
            vol_atm=jnp.array([0.12, 0.10]),
        )
        target = [0.041, 0.046]
        mrs = calibrate_cms_model(mkt, target)
        assert len(mrs) == 2
        assert (mrs >= 0).all()


class TestKahaleSmile:

    def test_atm_smile_section(self):
        from ql_jax.termstructures.volatility.kahale_smile import atm_smile_section
        vol_fn = atm_smile_section(100.0, 1.0, 0.25)
        assert vol_fn(90.0) == 0.25
        assert vol_fn(110.0) == 0.25
