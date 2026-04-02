"""Tests for Phase 15 process extensions."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestGSRProcess:
    def test_drift(self):
        from ql_jax.processes.extended import GSRProcess
        proc = GSRProcess(sigma_fn=0.01, kappa=0.1)
        # drift = -kappa * x
        assert float(proc.drift(0.0, 0.5)) == pytest.approx(-0.05)
        assert float(proc.drift(0.0, 0.0)) == pytest.approx(0.0)

    def test_diffusion(self):
        from ql_jax.processes.extended import GSRProcess
        proc = GSRProcess(sigma_fn=0.015, kappa=0.2)
        assert float(proc.diffusion(0.0, 0.0)) == pytest.approx(0.015)

    def test_evolve(self):
        from ql_jax.processes.extended import GSRProcess
        proc = GSRProcess(sigma_fn=0.01, kappa=0.1)
        x_new = proc.evolve(0.0, 0.0, 0.01, 1.0)  # dw=1.0
        assert isinstance(float(x_new), float)

    def test_variance(self):
        from ql_jax.processes.extended import GSRProcess
        proc = GSRProcess(sigma_fn=0.01, kappa=0.1)
        v = proc.variance(0.0, 1.0)
        assert float(v) > 0


class TestHybridHestonHullWhiteProcess:
    def test_construction(self):
        from ql_jax.processes.extended import HybridHestonHullWhiteProcess
        proc = HybridHestonHullWhiteProcess(
            S0=100.0, v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho_sv=-0.7,
            a=0.1, sigma_r=0.01,
        )
        assert proc.S0 == 100.0

    def test_correlation_matrix(self):
        from ql_jax.processes.extended import HybridHestonHullWhiteProcess
        proc = HybridHestonHullWhiteProcess(
            S0=100.0, v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho_sv=-0.7,
            a=0.1, sigma_r=0.01, rho_sr=0.1, rho_vr=-0.2,
        )
        corr = proc.correlation_matrix()
        assert corr.shape == (3, 3)
        # Diagonal is 1
        assert float(corr[0, 0]) == pytest.approx(1.0)
        assert float(corr[1, 1]) == pytest.approx(1.0)
        assert float(corr[2, 2]) == pytest.approx(1.0)
        # Off diagonal
        assert float(corr[0, 1]) == pytest.approx(-0.7)

    def test_evolve(self):
        from ql_jax.processes.extended import HybridHestonHullWhiteProcess
        proc = HybridHestonHullWhiteProcess(
            S0=100.0, v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho_sv=-0.7,
            a=0.1, sigma_r=0.01,
        )
        state = (100.0, 0.04, 0.05)
        dw = (0.1, -0.05, 0.02)
        new_state = proc.evolve(0.0, state, 0.01, dw)
        assert len(new_state) == 3
        assert float(new_state[0]) > 0  # Spot positive


class TestForwardMeasureProcess:
    def test_drift_adjustment(self):
        from ql_jax.processes.extended import ForwardMeasureProcess

        class MockProcess:
            def drift(self, t, x):
                return 0.0
            def diffusion(self, t, x):
                return 0.01

        base = MockProcess()
        fmp = ForwardMeasureProcess(base_process=base, bond_vol_fn=lambda t: 0.005)
        d = fmp.drift(0.5, 0.0)
        # Girsanov drift = sigma * bond_vol
        assert float(d) == pytest.approx(0.01 * 0.005, abs=1e-8)


class TestMfStateProcess:
    def test_martingale_drift(self):
        from ql_jax.processes.extended import MfStateProcess
        proc = MfStateProcess(sigma_fn=0.01)
        assert float(proc.drift(0.0, 0.0)) == pytest.approx(0.0)

    def test_diffusion(self):
        from ql_jax.processes.extended import MfStateProcess
        proc = MfStateProcess(sigma_fn=0.02)
        assert float(proc.diffusion(0.5, 1.0)) == pytest.approx(0.02)
