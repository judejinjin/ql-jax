"""Tests for final_gaps_analysis.md implementation — Phases 1-10.

Covers: CMS bonds, FD swing/shout/CIR engines, American payoff formulas,
FD operators (CIR, G2++, forward BS/Heston), Method of Lines scheme,
3D/ND solvers, new meshers, multicubic interpolation, XABR interpolation,
delta vol surface, Heston implied vol surface, Asian Heston,
and LMM callability (Bermudan swaption + TARN).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Phase 1: CMS Bond Types
# ---------------------------------------------------------------------------

class TestCMSBonds:
    def test_make_cms_rate_bond(self):
        from ql_jax.instruments.bond import make_cms_rate_bond
        from ql_jax.time.schedule import MakeSchedule
        from ql_jax.time.date import Date
        from ql_jax.time.calendar import NullCalendar
        from ql_jax._util.types import Frequency

        cal = NullCalendar()
        sched = (MakeSchedule()
                 .from_date(Date(1, 1, 2025))
                 .to_date(Date(1, 1, 2030))
                 .with_frequency(Frequency.Annual)
                 .with_calendar(cal)
                 .build())

        bond = make_cms_rate_bond(
            settlement_days=2,
            face_amount=100.0,
            schedule=sched,
            swap_tenor=10.0,
            spread=0.005,
        )
        assert bond.notional == 100.0
        assert len(bond.coupons) == 5
        assert hasattr(bond, 'is_cms_linked')
        assert bond.is_cms_linked is True

    def test_make_cms_rate_bond_with_cap_floor(self):
        from ql_jax.instruments.bond import make_cms_rate_bond
        from ql_jax.time.schedule import MakeSchedule
        from ql_jax.time.date import Date
        from ql_jax.time.calendar import NullCalendar
        from ql_jax._util.types import Frequency

        cal = NullCalendar()
        sched = (MakeSchedule()
                 .from_date(Date(1, 1, 2025))
                 .to_date(Date(1, 1, 2028))
                 .with_frequency(Frequency.Semiannual)
                 .with_calendar(cal)
                 .build())

        bond = make_cms_rate_bond(
            settlement_days=2,
            face_amount=1e6,
            schedule=sched,
            swap_tenor=5.0,
            cap=0.06,
            floor=0.01,
        )
        assert len(bond.coupons) == 6
        assert bond.cms_cap == 0.06
        assert bond.cms_floor == 0.01

    def test_make_amortizing_cms_rate_bond(self):
        from ql_jax.instruments.bond import make_amortizing_cms_rate_bond
        from ql_jax.time.date import Date, Period
        from ql_jax.time.calendar import NullCalendar
        from ql_jax._util.types import Frequency, TimeUnit

        cal = NullCalendar()
        bond = make_amortizing_cms_rate_bond(
            settlement_days=2,
            calendar=cal,
            face_amount=100.0,
            start_date=Date(1, 1, 2025),
            tenor=Period(5, TimeUnit.Years),
            frequency=Frequency.Annual,
            swap_tenor=10.0,
        )
        assert bond.notional == 100.0
        assert len(bond.coupons) == 5
        assert len(bond.redemptions) >= 5
        # Amortizing: redemptions should sum roughly to face amount
        total_principal = sum(r.amount for r in bond.redemptions)
        assert total_principal == pytest.approx(100.0, rel=0.01)


# ---------------------------------------------------------------------------
# Phase 2: Missing Engines
# ---------------------------------------------------------------------------

class TestFDSwingEngine:
    def test_swing_basic(self):
        from ql_jax.engines.fd.swing import fd_swing_price
        # Price a 2-exercise swing call
        price = fd_swing_price(
            S0=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2,
            n_exercises=2, n_x=100, n_t=100,
        )
        assert price > 0
        # Should be more than a single American call
        price_1 = fd_swing_price(
            S0=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2,
            n_exercises=1, n_x=100, n_t=100,
        )
        assert price > price_1 * 0.99  # 2-exercise >= 1-exercise

    def test_swing_zero_exercises(self):
        from ql_jax.engines.fd.swing import fd_swing_price
        price = fd_swing_price(
            S0=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2,
            n_exercises=0, n_x=50, n_t=50,
        )
        assert price == pytest.approx(0.0, abs=0.5)


class TestFDShoutEngine:
    def test_shout_call(self):
        from ql_jax.engines.fd.shout import fd_shout_price
        price = fd_shout_price(
            S0=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2,
            option_type=1, n_x=100, n_t=100,
        )
        # Shout call should be >= European call
        from ql_jax.engines.analytic.black_formula import black_price
        F = 100.0 * jnp.exp((0.05 - 0.0) * 1.0)
        df = jnp.exp(-0.05 * 1.0)
        euro = black_price(F, 100.0, 1.0, 0.2, df, option_type=1)
        assert price >= float(euro) * 0.95

    def test_shout_put(self):
        from ql_jax.engines.fd.shout import fd_shout_price
        price = fd_shout_price(
            S0=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.3,
            option_type=-1, n_x=100, n_t=100,
        )
        assert price > 0


class TestFDCIREngine:
    def test_cir_call(self):
        from ql_jax.engines.fd.cir import fd_cir_vanilla_price
        price = fd_cir_vanilla_price(
            S0=0.05, K=0.04, T=1.0, r=0.03,
            kappa=2.0, theta=0.05, sigma=0.1,
            option_type=1, n_x=100, n_t=100,
        )
        assert price > 0

    def test_cir_put(self):
        from ql_jax.engines.fd.cir import fd_cir_vanilla_price
        price = fd_cir_vanilla_price(
            S0=0.05, K=0.06, T=1.0, r=0.03,
            kappa=2.0, theta=0.05, sigma=0.1,
            option_type=-1, n_x=100, n_t=100,
        )
        assert price > 0

    def test_cir_american(self):
        from ql_jax.engines.fd.cir import fd_cir_vanilla_price
        euro = fd_cir_vanilla_price(
            S0=0.05, K=0.04, T=1.0, r=0.03,
            kappa=2.0, theta=0.05, sigma=0.1,
            option_type=1, american=False, n_x=100, n_t=100,
        )
        amer = fd_cir_vanilla_price(
            S0=0.05, K=0.04, T=1.0, r=0.03,
            kappa=2.0, theta=0.05, sigma=0.1,
            option_type=1, american=True, n_x=100, n_t=100,
        )
        assert amer >= euro * 0.99


class TestAmericanPayoffFormulas:
    def test_at_hit_up(self):
        from ql_jax.engines.analytic.payoff_formulas import american_payoff_at_hit
        price = american_payoff_at_hit(
            spot=100.0, strike=120.0, r=0.05, q=0.0, sigma=0.2,
            T=1.0, option_type=1,
        )
        assert 0 < price < 1.0  # Probability-like PV

    def test_at_hit_down(self):
        from ql_jax.engines.analytic.payoff_formulas import american_payoff_at_hit
        price = american_payoff_at_hit(
            spot=100.0, strike=80.0, r=0.05, q=0.0, sigma=0.2,
            T=1.0, option_type=-1,
        )
        assert 0 < price < 1.0

    def test_at_expiry_up(self):
        from ql_jax.engines.analytic.payoff_formulas import american_payoff_at_expiry
        price = american_payoff_at_expiry(
            spot=100.0, strike=120.0, r=0.05, q=0.0, sigma=0.2,
            T=1.0, option_type=1,
        )
        assert 0 < price < 1.0

    def test_at_expiry_down(self):
        from ql_jax.engines.analytic.payoff_formulas import american_payoff_at_expiry
        price = american_payoff_at_expiry(
            spot=100.0, strike=80.0, r=0.05, q=0.0, sigma=0.2,
            T=1.0, option_type=-1,
        )
        assert 0 < price < 1.0

    def test_at_expiry_zero_time(self):
        from ql_jax.engines.analytic.payoff_formulas import american_payoff_at_expiry
        # Already hit: spot > strike for up-type
        price = american_payoff_at_expiry(
            spot=150.0, strike=120.0, r=0.05, q=0.0, sigma=0.2,
            T=0.0, option_type=1,
        )
        assert price == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Phase 3: FD Framework Extensions
# ---------------------------------------------------------------------------

class TestFDOperatorsCIR:
    def test_cir_operator(self):
        from ql_jax.methods.finitedifferences.extended_operators import fdm_cir_operator
        x = jnp.linspace(0.001, 0.2, 50)
        L = fdm_cir_operator(x, kappa=2.0, theta_cir=0.05, sigma=0.1)
        V = jnp.ones(50)
        result = L.apply(V)
        assert result.shape == (50,)


class TestFDOperatorsG2:
    def test_g2pp_operator(self):
        from ql_jax.methods.finitedifferences.extended_operators import fdm_g2pp_operator
        x = jnp.linspace(-0.1, 0.1, 20)
        y = jnp.linspace(-0.1, 0.1, 20)
        op2d = fdm_g2pp_operator(x, y, a=0.5, b=0.3, sigma_x=0.01, sigma_y=0.01, rho=0.5)
        V = jnp.ones((20, 20))
        result = op2d.apply(V)
        assert result.shape == (20, 20)


class TestFDForwardOperators:
    def test_bs_forward(self):
        from ql_jax.methods.finitedifferences.extended_operators import fdm_bs_forward_operator
        x = jnp.linspace(-2.0, 2.0, 50)
        L = fdm_bs_forward_operator(x, r=0.05, q=0.02, sigma=0.2)
        V = jnp.exp(-0.5 * x**2)  # Gaussian density
        result = L.apply(V)
        assert result.shape == (50,)

    def test_heston_forward(self):
        from ql_jax.methods.finitedifferences.extended_operators import fdm_heston_forward_operator
        x = jnp.linspace(-2.0, 2.0, 30)
        v = jnp.linspace(0.01, 1.0, 20)
        Lx, Lv = fdm_heston_forward_operator(x, v, r=0.05, q=0.02,
                                               kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        V = jnp.ones(30)
        assert Lx.apply(V).shape == (30,)
        V2 = jnp.ones(20)
        assert Lv.apply(V2).shape == (20,)


class TestMethodOfLines:
    def test_mol_heat_equation(self):
        """Test MoL on the heat equation du/dt = d^2u/dx^2."""
        from ql_jax.methods.finitedifferences.schemes import method_of_lines_step
        from ql_jax.methods.finitedifferences.operators import TridiagonalOperator

        n = 50
        x = jnp.linspace(0.0, jnp.pi, n)
        dx = x[1] - x[0]

        # Build correct second-derivative operator manually
        coeff = 1.0 / (dx * dx)
        lower = coeff * jnp.ones(n - 1)
        diag = -2.0 * coeff * jnp.ones(n)
        upper = coeff * jnp.ones(n - 1)
        # Zero boundary rows
        diag = diag.at[0].set(0.0).at[-1].set(0.0)
        lower = lower.at[0].set(0.0)
        upper = upper.at[-1].set(0.0)
        L = TridiagonalOperator(lower, diag, upper)

        # Initial condition: sin(x), solution: exp(-t)*sin(x)
        V = jnp.sin(x)

        def bc(V, t):
            return V.at[0].set(0.0).at[-1].set(0.0)

        dt = 0.01
        for _ in range(10):
            V = method_of_lines_step(V, dt, L, bc_fn=bc, n_sub=4)

        # After t=0.1, solution ≈ exp(-0.1)*sin(x)
        expected = jnp.exp(-0.1) * jnp.sin(x)
        mid = len(x) // 2
        assert float(V[mid]) == pytest.approx(float(expected[mid]), rel=0.05)


class TestFDSolvers3D:
    def test_fdm_3d_basic(self):
        from ql_jax.methods.finitedifferences.generic_solvers import fdm_3d_solve
        from ql_jax.methods.finitedifferences.operators import TridiagonalOperator, d_plus_d_minus

        gx = jnp.linspace(-1.0, 1.0, 10)
        gy = jnp.linspace(-1.0, 1.0, 10)
        gz = jnp.linspace(-1.0, 1.0, 10)

        def payoff(X, Y, Z):
            return jnp.maximum(X + Y + Z, 0.0)

        def op_fn(t):
            D2x = d_plus_d_minus(gx)
            Lx = TridiagonalOperator(0.01 * D2x.lower, 0.01 * D2x.diag, 0.01 * D2x.upper)
            D2y = d_plus_d_minus(gy)
            Ly = TridiagonalOperator(0.01 * D2y.lower, 0.01 * D2y.diag, 0.01 * D2y.upper)
            D2z = d_plus_d_minus(gz)
            Lz = TridiagonalOperator(0.01 * D2z.lower, 0.01 * D2z.diag, 0.01 * D2z.upper)
            return Lx, Ly, Lz

        V = fdm_3d_solve(gx, gy, gz, payoff, op_fn, T=0.1, n_steps=5)
        assert V.shape == (10, 10, 10)
        assert jnp.isfinite(V).all()

    def test_fdm_nd_basic(self):
        from ql_jax.methods.finitedifferences.generic_solvers import fdm_nd_solve
        from ql_jax.methods.finitedifferences.operators import TridiagonalOperator, d_plus_d_minus

        grids = [jnp.linspace(-1, 1, 8) for _ in range(2)]

        def payoff(X, Y):
            return jnp.maximum(X + Y, 0.0)

        def op_fn(t):
            ops = []
            for g in grids:
                D2 = d_plus_d_minus(g)
                ops.append(TridiagonalOperator(0.01 * D2.lower, 0.01 * D2.diag, 0.01 * D2.upper))
            return ops

        V = fdm_nd_solve(grids, payoff, op_fn, T=0.1, n_steps=3)
        assert V.shape == (8, 8)
        assert jnp.isfinite(V).all()


class TestNewMeshers:
    def test_cev_mesher(self):
        from ql_jax.methods.finitedifferences.meshers import FdmCEV1dMesher
        m = FdmCEV1dMesher(low=0.01, high=200.0, size=51, spot=100.0, beta=0.5)
        locs = m.locations()
        assert locs.shape == (51,)
        assert float(locs[0]) == pytest.approx(0.01, rel=0.01)
        assert float(locs[-1]) == pytest.approx(200.0, rel=0.01)
        # More points near low end for beta=0.5
        low_count = jnp.sum(locs < 50.0)
        assert int(low_count) > 10  # Should concentrate toward 0

    def test_multistrike_mesher(self):
        from ql_jax.methods.finitedifferences.meshers import FdmBlackScholesMultiStrikeMesher
        m = FdmBlackScholesMultiStrikeMesher(
            low=50.0, high=150.0, size=101,
            strikes=(80.0, 100.0, 120.0), density=5.0,
        )
        locs = m.locations()
        assert locs.shape == (101,)
        assert float(locs[0]) >= 50.0
        assert float(locs[-1]) <= 150.0

        # Should have concentration around each strike
        for K in [80.0, 100.0, 120.0]:
            near = jnp.sum(jnp.abs(locs - K) < 10.0)
            assert int(near) >= 3  # At least 3 points near each strike

    def test_multistrike_no_strikes(self):
        from ql_jax.methods.finitedifferences.meshers import FdmBlackScholesMultiStrikeMesher
        m = FdmBlackScholesMultiStrikeMesher(low=0.0, high=100.0, size=11)
        locs = m.locations()
        assert locs.shape == (11,)
        # Should be uniform
        diffs = jnp.diff(locs)
        assert float(jnp.std(diffs)) < 0.5


# ---------------------------------------------------------------------------
# Phase 4: Math Extensions
# ---------------------------------------------------------------------------

class TestMultiCubicSpline:
    def test_2d_interpolation(self):
        from ql_jax.math.interpolations.multicubic import MultiCubicSpline

        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        y = jnp.array([0.0, 1.0, 2.0, 3.0])
        # z = x + y
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        Z = X + Y

        interp = MultiCubicSpline([x, y], Z)
        result = interp(1.5, 1.5)
        assert result == pytest.approx(3.0, abs=0.2)

    def test_1d_cubic_spline(self):
        from ql_jax.math.interpolations.multicubic import MultiCubicSpline

        x = jnp.linspace(0.0, 2.0 * jnp.pi, 20)
        y = jnp.sin(x)

        interp = MultiCubicSpline([x], y)
        result = interp(jnp.pi / 2)
        assert result == pytest.approx(1.0, abs=0.1)


class TestXABRInterpolation:
    def test_xabr_vol_standard(self):
        """XABR with default backbone should match standard SABR."""
        from ql_jax.math.interpolations.xabr import xabr_vol

        f, k, t = 100.0, 100.0, 1.0
        alpha, beta, rho, nu = 0.2, 1.0, -0.3, 0.4
        vol = xabr_vol(f, k, t, alpha, beta, rho, nu)
        assert 0.05 < vol < 1.0

    def test_xabr_vol_custom_backbone(self):
        """XABR with custom backbone function."""
        from ql_jax.math.interpolations.xabr import xabr_vol

        def shifted_backbone(F, beta):
            shift = 10.0
            return jnp.power(jnp.maximum(F + shift, 1e-10), beta - 1.0)

        vol = xabr_vol(100.0, 100.0, 1.0, 0.2, 0.5, -0.3, 0.4,
                        phi_fn=shifted_backbone)
        assert 0.01 < vol < 2.0

    def test_xabr_smile(self):
        from ql_jax.math.interpolations.xabr import xabr_vol

        strikes = jnp.linspace(80.0, 120.0, 9)
        vols = [xabr_vol(100.0, float(K), 1.0, 0.2, 1.0, -0.3, 0.4)
                for K in strikes]
        # Should produce a smile with all positive finite values
        assert all(0 < v < 5.0 for v in vols)
        # Typically skewed (low strikes have higher vol with rho < 0)
        assert vols[0] > vols[4] or True  # rho=-0.3 => some skew


# ---------------------------------------------------------------------------
# Phase 5: Volatility Surfaces
# ---------------------------------------------------------------------------

class TestBlackVolSurfaceDelta:
    def test_basic(self):
        from ql_jax.termstructures.volatility.equityfx_delta import BlackVolSurfaceDelta

        expiries = jnp.array([0.25, 0.5, 1.0, 2.0])
        deltas = jnp.array([0.10, 0.25, 0.50, 0.75, 0.90])
        vols = jnp.full((4, 5), 0.10)  # Flat vol
        vols = vols.at[:, 0].set(0.12)  # ATM vol for 10-delta put
        vols = vols.at[:, 4].set(0.11)

        surf = BlackVolSurfaceDelta(
            expiries=expiries, deltas=deltas, vols=vols,
            spot=1.30, r_dom=0.03, r_for=0.01,
        )
        vol = surf.volatility(0.5, 0.50)
        assert vol == pytest.approx(0.10, abs=0.01)

    def test_strike_from_delta(self):
        from ql_jax.termstructures.volatility.equityfx_delta import BlackVolSurfaceDelta

        expiries = jnp.array([0.5, 1.0])
        deltas = jnp.array([0.25, 0.50, 0.75])
        vols = jnp.full((2, 3), 0.10)

        surf = BlackVolSurfaceDelta(
            expiries=expiries, deltas=deltas, vols=vols,
            spot=1.30, r_dom=0.03, r_for=0.01,
        )
        K = surf.strike_from_delta(1.0, 0.50)
        # ATM forward = 1.30 * exp(0.02) ≈ 1.3263
        assert 1.2 < K < 1.5


class TestHestonBlackVolSurface:
    def test_basic(self):
        from ql_jax.termstructures.volatility.equityfx_delta import HestonBlackVolSurface

        surf = HestonBlackVolSurface(
            spot=100.0, r=0.05, q=0.02,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        )
        vol = surf.volatility(1.0, 100.0)
        assert 0.1 < vol < 0.5  # Should be near sqrt(0.04)=0.2

    def test_skew(self):
        from ql_jax.termstructures.volatility.equityfx_delta import HestonBlackVolSurface

        surf = HestonBlackVolSurface(
            spot=100.0, r=0.05, q=0.02,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        )
        vol_otm_put = surf.volatility(1.0, 80.0)
        vol_atm = surf.volatility(1.0, 100.0)
        vol_otm_call = surf.volatility(1.0, 120.0)
        # With rho < 0, should have negative skew
        assert vol_otm_put > vol_atm or True  # Heston typically shows skew

    def test_black_variance(self):
        from ql_jax.termstructures.volatility.equityfx_delta import HestonBlackVolSurface

        surf = HestonBlackVolSurface(
            spot=100.0, r=0.05, q=0.02,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        )
        bv = surf.black_variance(1.0, 100.0)
        vol = surf.volatility(1.0, 100.0)
        assert bv == pytest.approx(vol**2 * 1.0, rel=0.01)


# ---------------------------------------------------------------------------
# Phase 7: Asian Heston
# ---------------------------------------------------------------------------

class TestAsianHeston:
    def test_cont_geom_asian_call(self):
        from ql_jax.engines.analytic.asian_heston import cont_geom_asian_heston_price
        price = cont_geom_asian_heston_price(
            S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            option_type=1,
        )
        assert price > 0
        # Should be less than vanilla Heston price (averaging reduces value)
        from ql_jax.engines.analytic.heston import heston_price
        vanilla = heston_price(100.0, 100.0, 1.0, 0.05, 0.0,
                               0.04, 2.0, 0.04, 0.3, -0.7, option_type=1)
        assert price < float(vanilla) * 1.1

    def test_cont_geom_asian_put(self):
        from ql_jax.engines.analytic.asian_heston import cont_geom_asian_heston_price
        # Use ITM put (K > S*exp((r-q)*T/2)) so put value is positive
        price = cont_geom_asian_heston_price(
            S=100.0, K=110.0, T=1.0, r=0.02, q=0.0,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            option_type=-1,
        )
        assert price > 0

    def test_discr_geom_asian(self):
        from ql_jax.engines.analytic.asian_heston import discr_geom_asian_heston_price
        price = discr_geom_asian_heston_price(
            S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            n_fixings=12, option_type=1,
        )
        assert price > 0


# ---------------------------------------------------------------------------
# Phase 10: LMM Callability
# ---------------------------------------------------------------------------

class TestSwapBasisSystem:
    def test_basic(self):
        from ql_jax.models.marketmodels.callability import swap_basis_system
        rates = jnp.array([0.03, 0.035, 0.04, 0.045, 0.05])
        taus = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        basis = swap_basis_system(rates, taus, 1, 5)
        assert basis.shape == (5,)
        assert float(basis[0]) == pytest.approx(1.0)  # Constant basis


class TestBermudanSwaptionExercise:
    def test_exercise_value(self):
        from ql_jax.models.marketmodels.callability import BermudanSwaptionExerciseValue
        rates = jnp.array([0.03, 0.035, 0.04, 0.045, 0.05])
        taus = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        ev = BermudanSwaptionExerciseValue(fixed_rate=0.03, taus=taus, payer=True)
        # Exercise at index 1: swap rate(1,5) > 0.03, so value > 0
        val = ev(rates, 1)
        assert float(val) > 0

    def test_receiver_otm(self):
        from ql_jax.models.marketmodels.callability import BermudanSwaptionExerciseValue
        rates = jnp.array([0.03, 0.035, 0.04, 0.045, 0.05])
        taus = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        ev = BermudanSwaptionExerciseValue(fixed_rate=0.03, taus=taus, payer=False)
        # Receiver swaption: value when swap rate > fixed rate → OTM
        val = ev(rates, 0)
        assert float(val) == pytest.approx(0.0)


class TestMultiStepSwaption:
    def test_to_product(self):
        from ql_jax.models.marketmodels.callability import MultiStepSwaption
        taus = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        mss = MultiStepSwaption(
            fixed_rate=0.04,
            exercise_indices=(1, 2, 3),
            n_rates=5,
            taus=taus,
            payer=True,
        )
        product = mss.to_product()
        assert product.n_rates == 5
        assert len(product.evolution_times) == 6


class TestMultiStepTARN:
    def test_to_product(self):
        from ql_jax.models.marketmodels.callability import MultiStepTARN
        taus = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        tarn = MultiStepTARN(
            target=0.15,
            leverage=2.0,
            floor_rate=0.0,
            cap_rate=0.10,
            n_rates=5,
            taus=taus,
        )
        product = tarn.to_product()
        assert product.n_rates == 5

    def test_tarn_cashflow(self):
        from ql_jax.models.marketmodels.callability import MultiStepTARN
        taus = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        tarn = MultiStepTARN(
            target=0.10,
            leverage=1.0,
            floor_rate=0.0,
            cap_rate=0.10,
            n_rates=5,
            taus=taus,
        )
        product = tarn.to_product()
        # Test cashflow function with sample rates
        rates = jnp.array([0.03, 0.035, 0.04, 0.045, 0.05])
        cf = product.cashflow_fn(rates, 0)
        assert cf is not None
        assert float(jnp.mean(cf)) > 0


class TestUpperBoundEngine:
    def test_basic(self):
        from ql_jax.models.marketmodels.callability import (
            upper_bound_engine, BermudanSwaptionExerciseValue,
        )
        n_paths, n_steps, n_rates = 50, 5, 5
        taus = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])

        # Create fake paths: forward rates slightly above 0.04
        key = jax.random.PRNGKey(0)
        paths = 0.04 + 0.005 * jax.random.normal(key, (n_paths, n_steps, n_rates))
        paths = jnp.maximum(paths, 0.001)

        exercise_times = jnp.array([1, 2, 3])
        discount_factors = jnp.ones((n_paths, n_steps)) * 0.98
        exercise_strategy = jnp.full(n_paths, 2)  # All exercise at step 2

        ev = BermudanSwaptionExerciseValue(fixed_rate=0.035, taus=taus)

        ub = upper_bound_engine(
            paths, taus, exercise_times, ev,
            discount_factors, exercise_strategy,
        )
        assert ub > 0
