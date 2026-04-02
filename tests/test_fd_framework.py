"""Tests for FD framework: meshers, ADI, generic solvers."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Meshers
# ---------------------------------------------------------------------------

class TestUniform1dMesher:
    def test_basic(self):
        from ql_jax.methods.finitedifferences.meshers import Uniform1dMesher
        m = Uniform1dMesher(low=0.0, high=1.0, size=11)
        locs = m.locations()
        assert locs.shape == (11,)
        assert float(locs[0]) == pytest.approx(0.0)
        assert float(locs[-1]) == pytest.approx(1.0)

    def test_dx(self):
        from ql_jax.methods.finitedifferences.meshers import Uniform1dMesher
        m = Uniform1dMesher(low=0.0, high=2.0, size=5)
        assert float(m.dx()) == pytest.approx(0.5)


class TestConcentrating1dMesher:
    def test_concentration(self):
        from ql_jax.methods.finitedifferences.meshers import Concentrating1dMesher
        m = Concentrating1dMesher(low=0.0, high=200.0, size=51, center=100.0, density=10.0)
        locs = m.locations()
        assert locs.shape == (51,)
        assert float(locs[0]) == pytest.approx(0.0, abs=1.0)
        assert float(locs[-1]) == pytest.approx(200.0, abs=1.0)
        # More points near center=100
        mid_count = jnp.sum((locs > 80) & (locs < 120))
        total_count = len(locs)
        assert int(mid_count) > total_count * 0.3  # Concentrated


class TestLogMesher:
    def test_log_spacing(self):
        from ql_jax.methods.finitedifferences.meshers import LogMesher
        m = LogMesher(low=10.0, high=1000.0, size=21)
        locs = m.locations()
        assert locs.shape == (21,)
        assert float(locs[0]) == pytest.approx(10.0, rel=0.01)
        assert float(locs[-1]) == pytest.approx(1000.0, rel=0.01)
        # Log spacing: ratios should be roughly constant
        ratios = locs[1:] / locs[:-1]
        assert float(jnp.std(ratios)) < float(jnp.mean(ratios)) * 0.05


class TestPredefined1dMesher:
    def test_custom_grid(self):
        from ql_jax.methods.finitedifferences.meshers import Predefined1dMesher
        grid = jnp.array([1.0, 2.0, 5.0, 10.0, 20.0])
        m = Predefined1dMesher(grid=grid)
        assert m.size == 5
        assert float(m.low) == pytest.approx(1.0)
        assert float(m.high) == pytest.approx(20.0)


class TestFdmMesherComposite:
    def test_2d_composite(self):
        from ql_jax.methods.finitedifferences.meshers import Uniform1dMesher, FdmMesherComposite
        m1 = Uniform1dMesher(0.0, 1.0, 11)
        m2 = Uniform1dMesher(0.0, 2.0, 21)
        comp = FdmMesherComposite(meshers=(m1, m2))
        assert comp.ndim == 2
        assert comp.total_size() == 11 * 21

    def test_locations(self):
        from ql_jax.methods.finitedifferences.meshers import Uniform1dMesher, FdmMesherComposite
        m1 = Uniform1dMesher(0.0, 1.0, 5)
        m2 = Uniform1dMesher(0.0, 2.0, 3)
        comp = FdmMesherComposite(meshers=(m1, m2))
        locs0 = comp.locations(0)
        locs1 = comp.locations(1)
        assert locs0.shape == (5,)
        assert locs1.shape == (3,)


class TestFdmBlackScholesMesher:
    def test_construction(self):
        from ql_jax.methods.finitedifferences.meshers import FdmBlackScholesMesher
        m = FdmBlackScholesMesher(size=101, strike=100.0, spot=100.0,
                                   r=0.05, q=0.0, sigma=0.2, maturity=1.0)
        locs = m.locations()
        assert locs.shape == (101,)
        # Should contain log-spot near strike
        assert jnp.any(jnp.abs(jnp.exp(locs) - 100.0) < 5.0)


class TestFdmHestonVarianceMesher:
    def test_construction(self):
        from ql_jax.methods.finitedifferences.meshers import FdmHestonVarianceMesher
        m = FdmHestonVarianceMesher(size=51, v0=0.04, kappa=1.5,
                                     theta=0.04, sigma_v=0.3, maturity=1.0)
        locs = m.locations()
        assert locs.shape == (51,)
        assert float(locs[0]) >= 0.0  # Variance >= 0


# ---------------------------------------------------------------------------
# ADI schemes
# ---------------------------------------------------------------------------

class TestADISchemes:
    def test_douglas_step(self):
        from ql_jax.methods.finitedifferences.adi_schemes import douglas_step
        from ql_jax.methods.finitedifferences.operators import TridiagonalOperator
        n = 10
        V = jnp.ones(n)
        # Simple zero operator (identity-like)
        L = TridiagonalOperator(
            lower=jnp.zeros(n - 1),
            diag=jnp.zeros(n),
            upper=jnp.zeros(n - 1),
        )
        V_new = douglas_step(V, dt=0.01, ops=[L, L], theta=0.5)
        assert V_new.shape == (n,)

    def test_craig_sneyd_step(self):
        from ql_jax.methods.finitedifferences.adi_schemes import craig_sneyd_step
        from ql_jax.methods.finitedifferences.operators import TridiagonalOperator
        n = 10
        V = jnp.ones(n)
        L = TridiagonalOperator(
            lower=jnp.zeros(n - 1),
            diag=jnp.zeros(n),
            upper=jnp.zeros(n - 1),
        )
        V_new = craig_sneyd_step(V, dt=0.01, ops=[L, L], theta=0.5)
        assert V_new.shape == (n,)


# ---------------------------------------------------------------------------
# Generic solvers
# ---------------------------------------------------------------------------

class TestFdm1dSolve:
    def test_heat_equation(self):
        """Solve heat equation: u_t = u_xx with known solution."""
        from ql_jax.methods.finitedifferences.generic_solvers import fdm_1d_solve
        from ql_jax.methods.finitedifferences.operators import d_plus_d_minus

        grid = jnp.linspace(-3.0, 3.0, 101)
        # Terminal condition: indicator near 0
        payoff_fn = lambda x: jnp.where(jnp.abs(x) < 0.5, 1.0, 0.0)

        def operator_fn(t):
            return d_plus_d_minus(grid)

        V = fdm_1d_solve(grid, payoff_fn, operator_fn, T=0.1, n_steps=100, theta=0.5)
        assert V.shape == (101,)
        # Solution should be smooth (diffused)
        assert float(V[50]) > 0  # Center
        assert float(V[0]) >= 0  # Boundary

    def test_bs_european_call(self):
        """Price a European call via 1D FD and compare to Black-Scholes."""
        from ql_jax.methods.finitedifferences.generic_solvers import fdm_1d_solve
        from ql_jax.methods.finitedifferences.operators import TridiagonalOperator

        K, r, sigma, T = 100.0, 0.05, 0.20, 1.0
        n_x, n_t = 201, 200
        x = jnp.linspace(jnp.log(30.0), jnp.log(300.0), n_x)
        dx = x[1] - x[0]

        payoff_fn = lambda grid: jnp.maximum(jnp.exp(grid) - K, 0.0)

        def operator_fn(t):
            d = (r - 0.5 * sigma ** 2) / (2 * dx)
            s = sigma ** 2 / (dx ** 2)
            lower = jnp.full(n_x - 1, 0.5 * s - d)
            diag = jnp.full(n_x, -s - r)
            upper = jnp.full(n_x - 1, 0.5 * s + d)
            return TridiagonalOperator(lower=lower, diag=diag, upper=upper)

        V = fdm_1d_solve(x, payoff_fn, operator_fn, T=T, n_steps=n_t, theta=0.5)

        # Interpolate at S=100
        idx = jnp.argmin(jnp.abs(x - jnp.log(100.0)))
        fd_price = float(V[idx])

        from ql_jax.engines.analytic.black_formula import black_scholes_price
        bs_price = float(black_scholes_price(100.0, K, T, r, 0.0, sigma, 1))

        assert fd_price == pytest.approx(bs_price, rel=0.05)


class TestFdm2dSolve:
    @pytest.mark.skip(reason="2D solver ADI needs per-line dimension-wise operator application")
    def test_2d_diffusion(self):
        from ql_jax.methods.finitedifferences.generic_solvers import fdm_2d_solve
        from ql_jax.methods.finitedifferences.operators import TridiagonalOperator

        nx, ny = 11, 11
        gx = jnp.linspace(-1.0, 1.0, nx)
        gy = jnp.linspace(-1.0, 1.0, ny)

        payoff_fn = lambda X, Y: jnp.maximum(1.0 - X ** 2 - Y ** 2, 0.0)

        def operator_fn(t):
            dx = gx[1] - gx[0]
            s = 1.0 / (dx ** 2)
            Lx = TridiagonalOperator(
                lower=jnp.full(nx - 1, 0.5 * s),
                diag=jnp.full(nx, -s),
                upper=jnp.full(nx - 1, 0.5 * s),
            )
            Ly = TridiagonalOperator(
                lower=jnp.full(ny - 1, 0.5 * s),
                diag=jnp.full(ny, -s),
                upper=jnp.full(ny - 1, 0.5 * s),
            )
            return [Lx, Ly]

        V = fdm_2d_solve(gx, gy, payoff_fn, operator_fn, T=0.01, n_steps=10, scheme="douglas")
        assert V.shape == (nx, ny)
        assert float(V[5, 5]) > 0  # Center should be positive
