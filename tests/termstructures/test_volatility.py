"""Tests for volatility term structures."""

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.termstructures.volatility.black_vol import (
    BlackConstantVol, BlackVarianceCurve, BlackVarianceSurface,
    LocalConstantVol, ImpliedVolTermStructure,
)
from ql_jax.termstructures.volatility.smile_section import (
    FlatSmileSection, InterpolatedSmileSection, SABRSmileSection,
)
from ql_jax.termstructures.volatility.sabr_functions import sabr_vol
from ql_jax.termstructures.volatility.swaption_vol import (
    SwaptionConstantVol, SwaptionVolMatrix,
)
from ql_jax.termstructures.volatility.capfloor_vol import (
    ConstantCapFloorTermVol, CapFloorTermVolCurve,
)


class TestBlackConstantVol:
    def test_vol(self):
        ref = Date(1, 1, 2023)
        vol_surface = BlackConstantVol(ref, 0.20)
        assert abs(float(vol_surface.black_vol(1.0)) - 0.20) < 1e-12

    def test_variance(self):
        ref = Date(1, 1, 2023)
        vol_surface = BlackConstantVol(ref, 0.20)
        var = float(vol_surface.black_variance(2.0))
        assert abs(var - 0.04 * 2.0) < 1e-10

    def test_forward_vol(self):
        ref = Date(1, 1, 2023)
        vol_surface = BlackConstantVol(ref, 0.20)
        fwd = float(vol_surface.black_forward_vol(1.0, 2.0))
        assert abs(fwd - 0.20) < 1e-6


class TestBlackVarianceCurve:
    def test_interpolation(self):
        ref = Date(1, 1, 2023)
        dates = [Date(1, 7, 2023), Date(1, 1, 2024), Date(1, 1, 2025)]
        vols = [0.18, 0.20, 0.22]
        curve = BlackVarianceCurve(ref, dates, vols)
        t = float(curve.time_from_reference(Date(1, 1, 2024)))
        vol = float(curve.black_vol(t))
        assert abs(vol - 0.20) < 0.01


class TestBlackVarianceSurface:
    def test_atm_vol(self):
        ref = Date(1, 1, 2023)
        dates = [Date(1, 7, 2023), Date(1, 1, 2024)]
        strikes = [90.0, 100.0, 110.0]
        vol_matrix = [
            [0.25, 0.24],  # strike=90
            [0.20, 0.19],  # strike=100
            [0.22, 0.21],  # strike=110
        ]
        surface = BlackVarianceSurface(ref, dates, strikes, vol_matrix)
        t = float(surface.time_from_reference(Date(1, 7, 2023)))
        vol = float(surface.black_vol(t, 100.0))
        assert abs(vol - 0.20) < 0.02


class TestLocalConstantVol:
    def test_value(self):
        ref = Date(1, 1, 2023)
        lv = LocalConstantVol(ref, 0.30)
        assert abs(float(lv.local_vol(1.0, 100.0)) - 0.30) < 1e-12


class TestFlatSmileSection:
    def test_constant_vol(self):
        smile = FlatSmileSection(1.0, 0.20)
        assert abs(float(smile.volatility(90.0)) - 0.20) < 1e-12
        assert abs(float(smile.volatility(110.0)) - 0.20) < 1e-12

    def test_variance(self):
        smile = FlatSmileSection(2.0, 0.20)
        var = float(smile.variance(100.0))
        assert abs(var - 0.04 * 2.0) < 1e-10


class TestInterpolatedSmileSection:
    def test_exact_points(self):
        strikes = [90.0, 100.0, 110.0]
        vols = [0.25, 0.20, 0.22]
        smile = InterpolatedSmileSection(1.0, strikes, vols)
        assert abs(float(smile.volatility(100.0)) - 0.20) < 1e-10


class TestSABRFunctions:
    def test_atm_vol(self):
        """SABR vol at ATM should be close to alpha."""
        vol = float(sabr_vol(
            strike=100.0, forward=100.0, t=1.0,
            alpha=0.20, beta=1.0, rho=-0.2, nu=0.4,
        ))
        assert abs(vol - 0.20) < 0.05  # Close to alpha for beta=1

    def test_smile_shape(self):
        """SABR with negative rho should show negative skew."""
        forward = 100.0
        strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = sabr_vol(strikes, forward, 1.0, 0.20, 0.5, -0.3, 0.4)
        # With negative rho, low strikes should have higher vol
        assert float(vols[0]) > float(vols[2])


class TestSwaptionVol:
    def test_constant(self):
        ref = Date(1, 1, 2023)
        sv = SwaptionConstantVol(ref, 0.15)
        assert abs(float(sv.volatility(1.0, 5.0)) - 0.15) < 1e-12

    def test_matrix(self):
        ref = Date(1, 1, 2023)
        opt_tenors = [0.25, 0.5, 1.0, 2.0]
        swap_lengths = [2.0, 5.0, 10.0]
        vol_matrix = [
            [0.20, 0.19, 0.18],
            [0.19, 0.18, 0.17],
            [0.18, 0.17, 0.16],
            [0.17, 0.16, 0.15],
        ]
        matrix = SwaptionVolMatrix(ref, opt_tenors, swap_lengths, vol_matrix)
        vol = float(matrix.volatility(1.0, 5.0))
        assert abs(vol - 0.17) < 1e-10


class TestCapFloorVol:
    def test_constant(self):
        ref = Date(1, 1, 2023)
        v = ConstantCapFloorTermVol(ref, 0.18)
        assert abs(float(v.volatility(1.0)) - 0.18) < 1e-12

    def test_curve(self):
        ref = Date(1, 1, 2023)
        dates = [Date(1, 1, 2024), Date(1, 1, 2025), Date(1, 1, 2026)]
        vols = [0.18, 0.19, 0.20]
        curve = CapFloorTermVolCurve(ref, dates, vols)
        t = float(curve.time_from_reference(Date(1, 1, 2024)))
        vol = float(curve.volatility(t))
        assert abs(vol - 0.18) < 0.01
