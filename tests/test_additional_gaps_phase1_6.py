"""Tests for additional_gaps Phases 1-6.

Phase 1: Missing indexes, day counters, ASX dates, BMA index
Phase 2: Copula framework
Phase 3: ODE solver, matrix utilities
Phase 4: SVI and no-arb SABR
Phase 5: Variance gamma process and model
Phase 6: CMS spread coupon and pricer
"""

import jax
import jax.numpy as jnp
import pytest
import math

jax.config.update("jax_enable_x64", True)


# =========================================================================
# Phase 1: Missing indexes, day counters, ASX, BMA
# =========================================================================

class TestPhase1MissingIborIndexes:
    """Test new IBOR index factory functions."""

    def test_bibor(self):
        from ql_jax.indexes.ibor import Bibor
        idx = Bibor(3)
        assert idx.family_name == "Bibor"
        assert idx.currency_code == "HUF"
        assert idx.tenor_months == 3

    def test_bkbm(self):
        from ql_jax.indexes.ibor import BKBM
        idx = BKBM(6)
        assert idx.family_name == "BKBM"
        assert idx.currency_code == "NZD"

    def test_mosprime(self):
        from ql_jax.indexes.ibor import Mosprime
        idx = Mosprime()
        assert idx.family_name == "Mosprime"
        assert idx.currency_code == "RUB"

    def test_robor(self):
        from ql_jax.indexes.ibor import Robor
        idx = Robor()
        assert idx.family_name == "Robor"
        assert idx.currency_code == "RON"

    def test_thbfix(self):
        from ql_jax.indexes.ibor import Thbfix
        idx = Thbfix()
        assert idx.family_name == "Thbfix"
        assert idx.currency_code == "THB"

    def test_trlibor(self):
        from ql_jax.indexes.ibor import Trlibor
        idx = Trlibor()
        assert idx.family_name == "Trlibor"
        assert idx.currency_code == "TRY"

    def test_zibor(self):
        from ql_jax.indexes.ibor import Zibor
        idx = Zibor()
        assert idx.family_name == "Zibor"
        assert idx.currency_code == "HRK"

    def test_all_new_indexes_have_names(self):
        from ql_jax.indexes.ibor import Bibor, BKBM, Mosprime, Robor, Thbfix, Trlibor, Zibor
        for factory in [Bibor, BKBM, Mosprime, Robor, Thbfix, Trlibor, Zibor]:
            idx = factory()
            assert len(idx.name) > 0


class TestPhase1MissingInflationIndexes:
    def test_zacpi(self):
        from ql_jax.indexes.inflation import ZACPI
        idx = ZACPI()
        assert idx.region == "ZA"
        assert idx.family_name == "CPI"

    def test_ukhicp(self):
        from ql_jax.indexes.inflation import UKHICP
        idx = UKHICP()
        assert idx.region == "UK"
        assert idx.family_name == "HICP"


class TestPhase1DayCounters:
    def test_actual_365_25(self):
        from ql_jax.time.daycounter import year_fraction
        from ql_jax.time.date import Date
        d1 = Date(1, 1, 2024)
        d2 = Date(1, 1, 2025)
        # 2024 is leap year -> 366 days
        yf = year_fraction(d1, d2, "Actual36525")
        assert yf == pytest.approx(366 / 365.25, rel=1e-10)

    def test_actual_366(self):
        from ql_jax.time.daycounter import year_fraction
        from ql_jax.time.date import Date
        d1 = Date(1, 1, 2024)
        d2 = Date(1, 1, 2025)
        yf = year_fraction(d1, d2, "Actual366")
        assert yf == pytest.approx(366 / 366.0, rel=1e-10)

    def test_actual_365_25_convention_alias(self):
        from ql_jax.time.daycounter import year_fraction
        from ql_jax.time.date import Date
        d1 = Date(15, 3, 2025)
        d2 = Date(15, 9, 2025)
        yf = year_fraction(d1, d2, "Actual365.25")
        days = d2.serial - d1.serial
        assert yf == pytest.approx(days / 365.25, rel=1e-10)


class TestPhase1ASXDates:
    def test_is_asx_date(self):
        from ql_jax.time.asx import is_asx_date
        from ql_jax.time.date import Date
        # 2nd Friday of March 2025 = March 14, 2025
        d = Date(14, 3, 2025)
        assert is_asx_date(d, main_cycle=True)

    def test_is_not_asx_date(self):
        from ql_jax.time.asx import is_asx_date
        from ql_jax.time.date import Date
        d = Date(15, 3, 2025)  # Saturday
        assert not is_asx_date(d)

    def test_next_asx_date(self):
        from ql_jax.time.asx import next_asx_date
        from ql_jax.time.date import Date
        d = Date(1, 1, 2025)
        nxt = next_asx_date(d, main_cycle=True)
        # Next main cycle ASX after Jan 1 is 2nd Friday of March 2025
        assert nxt.month == 3
        assert nxt.year == 2025
        assert 8 <= nxt.day <= 14
        assert nxt.weekday() == 6  # Friday

    def test_asx_code(self):
        from ql_jax.time.asx import asx_code, is_asx_date
        from ql_jax.time.date import Date
        d = Date(14, 3, 2025)
        assert asx_code(d) == "H5"

    def test_asx_date_from_code(self):
        from ql_jax.time.asx import asx_date_from_code, is_asx_date
        from ql_jax.time.date import Date
        d = asx_date_from_code("H5", Date(1, 1, 2025))
        assert d.month == 3
        assert d.year == 2025
        assert is_asx_date(d)


class TestPhase1BmaIndex:
    def test_bma_index_creation(self):
        from ql_jax.indexes.base import BmaIndex
        idx = BmaIndex()
        assert idx.family_name == "BMA"
        assert idx.currency_code == "USD"

    def test_bma_index_fixing(self):
        from ql_jax.indexes.base import BmaIndex
        from ql_jax.time.date import Date
        idx = BmaIndex()
        d = Date(15, 1, 2025)
        idx.add_fixing(d, 0.035)
        assert idx.fixing(d) == pytest.approx(0.035)


# =========================================================================
# Phase 2: Copula framework
# =========================================================================

class TestCopulas:
    def test_gaussian_copula(self):
        from ql_jax.math.copulas import gaussian_copula
        # Independence case (rho=0): C(u,v) approx u*v
        val = gaussian_copula(0.5, 0.5, 0.0)
        assert val == pytest.approx(0.25, abs=0.01)

    def test_gaussian_copula_perfect_correlation(self):
        from ql_jax.math.copulas import gaussian_copula
        # Perfect correlation: C(u,v) = min(u,v)
        val = gaussian_copula(0.3, 0.7, 0.999)
        assert val == pytest.approx(0.3, abs=0.02)

    def test_clayton_copula(self):
        from ql_jax.math.copulas import clayton_copula
        val = clayton_copula(0.5, 0.5, 2.0)
        # Should be > u*v = 0.25 (positive dependence)
        assert val > 0.25
        assert val < 0.5

    def test_frank_copula_independence(self):
        from ql_jax.math.copulas import frank_copula
        # theta=0 gives independence
        val = frank_copula(0.5, 0.5, 0.0)
        assert val == pytest.approx(0.25, abs=1e-6)

    def test_gumbel_copula(self):
        from ql_jax.math.copulas import gumbel_copula
        # theta=1 gives independence
        val = gumbel_copula(0.5, 0.5, 1.0)
        assert val == pytest.approx(0.25, abs=0.01)

    def test_independent_copula(self):
        from ql_jax.math.copulas import independent_copula
        assert independent_copula(0.3, 0.7) == pytest.approx(0.21)

    def test_min_copula(self):
        from ql_jax.math.copulas import min_copula
        assert min_copula(0.3, 0.7) == pytest.approx(0.3)

    def test_max_copula(self):
        from ql_jax.math.copulas import max_copula
        assert max_copula(0.3, 0.7) == pytest.approx(0.0)
        assert max_copula(0.8, 0.9) == pytest.approx(0.7)

    def test_plackett_copula_independence(self):
        from ql_jax.math.copulas import plackett_copula
        val = plackett_copula(0.5, 0.5, 1.0)
        assert val == pytest.approx(0.25, abs=1e-6)

    def test_fgm_copula(self):
        from ql_jax.math.copulas import fgm_copula
        # theta=0 gives independence
        val = fgm_copula(0.5, 0.5, 0.0)
        assert val == pytest.approx(0.25)
        # theta=1 should give positive dependence
        val2 = fgm_copula(0.5, 0.5, 1.0)
        assert val2 > 0.25

    def test_ali_mikhail_haq_copula(self):
        from ql_jax.math.copulas import ali_mikhail_haq_copula
        # theta=0 gives independence
        val = ali_mikhail_haq_copula(0.5, 0.5, 0.0)
        assert val == pytest.approx(0.25)

    def test_galambos_copula(self):
        from ql_jax.math.copulas import galambos_copula
        val = galambos_copula(0.5, 0.5, 1.0)
        assert 0.25 <= val <= 0.5

    def test_marshall_olkin_copula(self):
        from ql_jax.math.copulas import marshall_olkin_copula
        val = marshall_olkin_copula(0.5, 0.5, 0.5, 0.5)
        assert 0.0 <= val <= 0.5

    def test_copula_boundary_u0(self):
        """All copulas should have C(0,v) = 0."""
        from ql_jax.math.copulas import (
            independent_copula, clayton_copula, frank_copula,
            fgm_copula, ali_mikhail_haq_copula, plackett_copula,
        )
        eps = 1e-12
        assert independent_copula(eps, 0.5) == pytest.approx(0.0, abs=1e-8)
        assert clayton_copula(eps, 0.5, 2.0) == pytest.approx(0.0, abs=1e-4)
        assert frank_copula(eps, 0.5, 5.0) == pytest.approx(0.0, abs=1e-4)

    def test_copula_boundary_u1(self):
        """All copulas should have C(1,v) = v."""
        from ql_jax.math.copulas import independent_copula, fgm_copula
        assert independent_copula(1.0, 0.7) == pytest.approx(0.7)
        assert fgm_copula(1.0, 0.7, 0.5) == pytest.approx(0.7, abs=1e-10)


# =========================================================================
# Phase 3: ODE solver, matrix utilities
# =========================================================================

class TestAdaptiveRungeKutta:
    def test_exponential_decay(self):
        """dy/dt = -y, y(0)=1 => y(t) = e^{-t}."""
        from ql_jax.math.ode import adaptive_runge_kutta
        y = adaptive_runge_kutta(lambda t, y: -y, 1.0, 0.0, 2.0)
        assert y == pytest.approx(math.exp(-2.0), rel=1e-6)

    def test_sine_ode(self):
        """dy/dt = cos(t), y(0)=0 => y(t) = sin(t)."""
        from ql_jax.math.ode import adaptive_runge_kutta
        y = adaptive_runge_kutta(lambda t, y: jnp.cos(t), 0.0, 0.0, math.pi / 2)
        assert y == pytest.approx(1.0, rel=1e-5)

    def test_system_ode(self):
        """2D system: dy1/dt = -y2, dy2/dt = y1 (rotation)."""
        from ql_jax.math.ode import adaptive_runge_kutta
        y0 = jnp.array([1.0, 0.0])

        def f(t, y):
            return jnp.array([-y[1], y[0]])

        y = adaptive_runge_kutta(f, y0, 0.0, jnp.pi / 2)
        assert float(y[0]) == pytest.approx(0.0, abs=1e-5)
        assert float(y[1]) == pytest.approx(1.0, abs=1e-5)


class TestBiCGStab:
    def test_simple_system(self):
        from ql_jax.math.matrix_utilities import bicgstab
        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        b = jnp.array([1.0, 2.0])
        x = bicgstab(A, b)
        residual = jnp.linalg.norm(A @ x - b)
        assert float(residual) < 1e-8

    def test_callable_matvec(self):
        from ql_jax.math.matrix_utilities import bicgstab
        A = jnp.array([[2.0, -1.0], [-1.0, 2.0]])
        b = jnp.array([1.0, 1.0])
        x = bicgstab(lambda v: A @ v, b)
        assert float(jnp.linalg.norm(A @ x - b)) < 1e-8


class TestGMRES:
    def test_simple_system(self):
        from ql_jax.math.matrix_utilities import gmres
        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        b = jnp.array([1.0, 2.0])
        x = gmres(A, b)
        assert float(jnp.linalg.norm(A @ x - b)) < 1e-6

    def test_larger_system(self):
        from ql_jax.math.matrix_utilities import gmres
        n = 10
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (n, n))
        A = A @ A.T + 5.0 * jnp.eye(n)  # SPD
        b = jnp.ones(n)
        x = gmres(A, b)
        assert float(jnp.linalg.norm(A @ x - b)) < 1e-5


class TestMatrixUtilities:
    def test_qr_decomposition(self):
        from ql_jax.math.matrix_utilities import qr_decomposition
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        Q, R = qr_decomposition(A)
        assert jnp.allclose(Q @ R, A, atol=1e-10)

    def test_matrix_exponential(self):
        from ql_jax.math.matrix_utilities import matrix_exponential
        A = jnp.zeros((2, 2))
        expA = matrix_exponential(A)
        assert jnp.allclose(expA, jnp.eye(2), atol=1e-10)

    def test_pseudoinverse(self):
        from ql_jax.math.matrix_utilities import pseudoinverse
        A = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        pinv = pseudoinverse(A)
        assert jnp.allclose(pinv, jnp.array([[1.0, 0.0], [0.0, 0.5]]), atol=1e-10)

    def test_factor_reduction(self):
        from ql_jax.math.matrix_utilities import factor_reduction
        # Simple correlation matrix
        C = jnp.array([[1.0, 0.9], [0.9, 1.0]])
        factors = factor_reduction(C, 1)
        assert factors.shape == (2, 1)
        # Reconstruct approx correlation
        C_approx = factors @ factors.T
        assert jnp.allclose(C_approx, C, atol=0.15)

    def test_autocovariance(self):
        from ql_jax.math.matrix_utilities import autocovariance, autocorrelation
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        acov = autocovariance(x, 2)
        assert len(acov) == 3
        assert float(acov[0]) > 0  # variance is positive
        acorr = autocorrelation(x, 2)
        assert float(acorr[0]) == pytest.approx(1.0)


# =========================================================================
# Phase 4: SVI and No-Arb SABR
# =========================================================================

class TestSVI:
    def test_svi_total_variance(self):
        from ql_jax.termstructures.volatility.svi import svi_total_variance
        w = svi_total_variance(0.0, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
        # At k=0: w = a + b*(rho*0 + sqrt(0 + sigma^2)) = a + b*sigma
        expected = 0.04 + 0.1 * 0.2
        assert float(w) == pytest.approx(expected, rel=1e-10)

    def test_svi_implied_vol(self):
        from ql_jax.termstructures.volatility.svi import svi_implied_vol
        vol = svi_implied_vol(0.0, t=1.0, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
        # w = 0.06, vol = sqrt(0.06) ≈ 0.2449
        assert float(vol) == pytest.approx(math.sqrt(0.06), rel=1e-6)

    def test_svi_smile_section(self):
        from ql_jax.termstructures.volatility.svi import SviSmileSection
        ss = SviSmileSection(1.0, forward=100.0, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
        vol_atm = ss.volatility(100.0)
        vol_otm = ss.volatility(120.0)
        # SVI should produce a smile
        assert float(vol_atm) > 0
        assert float(vol_otm) > 0

    def test_svi_skew(self):
        """SVI with negative rho should produce left skew."""
        from ql_jax.termstructures.volatility.svi import svi_implied_vol
        vol_low = svi_implied_vol(-0.2, t=1.0, a=0.04, b=0.2, rho=-0.5, m=0.0, sigma=0.2)
        vol_high = svi_implied_vol(0.2, t=1.0, a=0.04, b=0.2, rho=-0.5, m=0.0, sigma=0.2)
        assert float(vol_low) > float(vol_high)  # negative skew


class TestNoArbSABR:
    def test_noarb_sabr_vol_basic(self):
        from ql_jax.termstructures.volatility.noarb_sabr import noarb_sabr_vol
        vol = noarb_sabr_vol(100.0, 100.0, 1.0, 0.3, 0.5, -0.3, 0.4)
        assert float(vol) > 0.0
        assert float(vol) < 2.0

    def test_noarb_sabr_smile_section(self):
        from ql_jax.termstructures.volatility.noarb_sabr import NoArbSabrSmileSection
        ss = NoArbSabrSmileSection(1.0, 100.0, 0.3, 0.5, -0.3, 0.4)
        vol = ss.volatility(100.0)
        assert float(vol) > 0.0

    def test_noarb_sabr_non_negative(self):
        """No-arb SABR should not produce negative vols at low strikes."""
        from ql_jax.termstructures.volatility.noarb_sabr import noarb_sabr_vol
        vol = noarb_sabr_vol(100.0, 1.0, 1.0, 0.3, 0.5, -0.3, 0.4)
        assert float(vol) >= 0.0


# =========================================================================
# Phase 5: Variance Gamma
# =========================================================================

class TestVarianceGamma:
    def test_process_creation(self):
        from ql_jax.models.equity.variance_gamma import VarianceGammaProcess
        vg = VarianceGammaProcess(s0=100, r=0.05, q=0.02, sigma=0.2, nu=0.5, theta=-0.1)
        assert vg.s0 == 100
        assert float(vg.omega) != 0.0

    def test_characteristic_exponent(self):
        from ql_jax.models.equity.variance_gamma import VarianceGammaProcess
        vg = VarianceGammaProcess(s0=100, r=0.05, q=0.0, sigma=0.2, nu=0.5, theta=-0.1)
        # At u=0, char exponent should be 0
        psi = vg.characteristic_exponent(0.0)
        assert abs(complex(psi)) < 1e-10

    def test_model_creation(self):
        from ql_jax.models.equity.variance_gamma import VarianceGammaModel
        model = VarianceGammaModel.from_params(100, 0.05, 0.0, 0.2, 0.5, -0.1)
        assert model.process.sigma == 0.2

    def test_analytic_vg_price_positive(self):
        from ql_jax.models.equity.variance_gamma import (
            VarianceGammaProcess, analytic_vg_price,
        )
        vg = VarianceGammaProcess(s0=100, r=0.05, q=0.0, sigma=0.2, nu=0.5, theta=-0.1)
        price = analytic_vg_price(vg, strike=100.0, expiry=1.0, is_call=True)
        assert price > 0.0

    def test_vg_put_call_parity(self):
        """Put-call parity: C - P = disc * (F - K)."""
        from ql_jax.models.equity.variance_gamma import (
            VarianceGammaProcess, analytic_vg_price,
        )
        vg = VarianceGammaProcess(s0=100, r=0.05, q=0.02, sigma=0.2, nu=0.3, theta=-0.05)
        K = 105.0
        T = 0.5
        call = analytic_vg_price(vg, K, T, True)
        put = analytic_vg_price(vg, K, T, False)
        fwd = vg.s0 * math.exp((vg.r - vg.q) * T)
        disc = math.exp(-vg.r * T)
        # C - P = disc * (F - K)
        assert call - put == pytest.approx(disc * (fwd - K), abs=0.5)


# =========================================================================
# Phase 6: CMS spread coupon and pricer
# =========================================================================

class TestSwapSpreadIndex:
    def test_creation(self):
        from ql_jax.indexes.swap_spread import SwapSpreadIndex
        from ql_jax.indexes.swap import EuriborSwapIsdaFixA
        idx1 = EuriborSwapIsdaFixA(10)
        idx2 = EuriborSwapIsdaFixA(2)
        spread_idx = SwapSpreadIndex(idx1, idx2)
        assert "EuriborSwapIsdaFixA" in spread_idx.name


class TestCmsSpreadCoupon:
    def test_coupon_amount(self):
        from ql_jax.cashflows.cms_spread import CmsSpreadCoupon
        cpn = CmsSpreadCoupon(
            notional=1e6, accrual_fraction=0.5,
            cms_rate_1=0.03, cms_rate_2=0.01,
        )
        # Spread = 0.03 - 0.01 = 0.02, amount = 1e6 * 0.5 * 0.02 = 10000
        assert float(cpn.amount) == pytest.approx(10000.0)

    def test_capped_coupon(self):
        from ql_jax.cashflows.cms_spread import CmsSpreadCoupon
        cpn = CmsSpreadCoupon(
            notional=1e6, accrual_fraction=0.5,
            cms_rate_1=0.03, cms_rate_2=0.005,
            cap=0.01,
        )
        # Spread = 0.025, capped at 0.01
        assert float(cpn.adjusted_rate) == pytest.approx(0.01)

    def test_floored_coupon(self):
        from ql_jax.cashflows.cms_spread import CmsSpreadCoupon
        cpn = CmsSpreadCoupon(
            notional=1e6, accrual_fraction=0.5,
            cms_rate_1=0.02, cms_rate_2=0.03,
            floor=0.0,
        )
        # Spread = -0.01, floored at 0.0
        assert float(cpn.adjusted_rate) == pytest.approx(0.0)

    def test_digital_coupon(self):
        from ql_jax.cashflows.cms_spread import DigitalCmsSpreadCoupon
        cpn = DigitalCmsSpreadCoupon(
            notional=1e6, accrual_fraction=0.5,
            cms_rate_1=0.03, cms_rate_2=0.01,
            strike=0.01, digital_rate=0.005,
            is_call=True,
        )
        # Spread = 0.02 > strike 0.01, pays digital_rate
        assert float(cpn.amount) == pytest.approx(1e6 * 0.5 * 0.005)


class TestLognormalCmsSpreadPricer:
    def test_positive_spread_call(self):
        from ql_jax.cashflows.cms_spread import lognormal_cms_spread_price
        price = lognormal_cms_spread_price(
            forward1=0.03, forward2=0.01,
            vol1=0.3, vol2=0.25, rho=0.7,
            expiry=1.0, strike=0.01,
            is_call=True, discount=0.95,
        )
        # CMS1 - CMS2 = 0.02, K = 0.01 -> deeply ITM
        assert price > 0.0

    def test_cms_spread_floor(self):
        from ql_jax.cashflows.cms_spread import lognormal_cms_spread_price
        price = lognormal_cms_spread_price(
            forward1=0.02, forward2=0.025,
            vol1=0.3, vol2=0.25, rho=0.7,
            expiry=1.0, strike=0.0,
            is_call=False, discount=0.95,
        )
        # Floor on spread (put), spread is negative
        assert price > 0.0

    def test_intrinsic_value(self):
        """Deep ITM call should be close to intrinsic."""
        from ql_jax.cashflows.cms_spread import lognormal_cms_spread_price
        price = lognormal_cms_spread_price(
            forward1=0.05, forward2=0.01,
            vol1=0.01, vol2=0.01, rho=0.9,
            expiry=0.01, strike=0.0,
            is_call=True, discount=1.0,
        )
        # Near-zero vol, near-zero time: should be close to 0.04
        assert price == pytest.approx(0.04, abs=0.005)
