"""Tests for MC framework."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestPathPricers:
    def test_european_call(self):
        from ql_jax.methods.montecarlo.framework import EuropeanCallPathPricer
        pricer = EuropeanCallPathPricer(strike=100.0, discount=0.95)
        # Path ending at 110 => payoff = 0.95*(110-100) = 9.5
        path = jnp.array([100.0, 105.0, 110.0])
        assert float(pricer(path)) == pytest.approx(9.5)

    def test_european_call_otm(self):
        from ql_jax.methods.montecarlo.framework import EuropeanCallPathPricer
        pricer = EuropeanCallPathPricer(strike=100.0, discount=0.95)
        path = jnp.array([100.0, 95.0, 90.0])
        assert float(pricer(path)) == pytest.approx(0.0)

    def test_european_put(self):
        from ql_jax.methods.montecarlo.framework import EuropeanPutPathPricer
        pricer = EuropeanPutPathPricer(strike=100.0, discount=0.95)
        path = jnp.array([100.0, 95.0, 90.0])
        assert float(pricer(path)) == pytest.approx(0.95 * 10.0)

    def test_arithmetic_avg(self):
        from ql_jax.methods.montecarlo.framework import ArithmeticAvgPathPricer
        pricer = ArithmeticAvgPathPricer(strike=100.0, discount=1.0)
        path = jnp.array([100.0, 110.0, 120.0])
        avg = (100 + 110 + 120) / 3.0
        expected = max(avg - 100.0, 0.0)
        assert float(pricer(path)) == pytest.approx(expected, abs=0.1)


class TestMonteCarloModel:
    def test_gbm_call(self):
        from ql_jax.methods.montecarlo.framework import monte_carlo_model, EuropeanCallPathPricer

        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

        def path_generator(key, n_paths):
            n_steps = 50
            dt = T / n_steps
            keys = jax.random.split(key, n_steps)
            paths = jnp.zeros((n_paths, n_steps + 1))
            paths = paths.at[:, 0].set(S0)
            S = jnp.full(n_paths, S0)
            for i in range(n_steps):
                Z = jax.random.normal(keys[i], (n_paths,))
                S = S * jnp.exp((r - 0.5 * sigma ** 2) * dt + sigma * jnp.sqrt(dt) * Z)
                paths = paths.at[:, i + 1].set(S)
            return paths

        pricer = EuropeanCallPathPricer(strike=K, discount=jnp.exp(-r * T))
        result = monte_carlo_model(path_generator, pricer, n_paths=50000, key=jax.random.PRNGKey(42))

        # Compare to Black-Scholes
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        bs = black_scholes_price(S0, K, T, r, 0.0, sigma, 1)
        assert float(result.mean) == pytest.approx(float(bs), rel=0.10)
        assert result.std_error > 0
        assert result.n_paths == 50000


class TestLSMBasisSystem:
    def test_monomial(self):
        from ql_jax.methods.montecarlo.framework import LSMBasisSystem
        basis = LSMBasisSystem(basis_type='monomial', order=3)
        x = jnp.array([1.0, 2.0, 3.0])
        vals = basis(x)
        assert vals.shape == (3, 4)  # order+1

    def test_laguerre(self):
        from ql_jax.methods.montecarlo.framework import LSMBasisSystem
        basis = LSMBasisSystem(basis_type='laguerre', order=2)
        x = jnp.array([1.0, 2.0])
        vals = basis(x)
        assert vals.shape == (2, 3)

    def test_hermite(self):
        from ql_jax.methods.montecarlo.framework import LSMBasisSystem
        basis = LSMBasisSystem(basis_type='hermite', order=2)
        x = jnp.array([0.0, 1.0])
        vals = basis(x)
        assert vals.shape == (2, 3)


class TestGenericLSRegression:
    def test_american_put(self):
        from ql_jax.methods.montecarlo.framework import generic_ls_regression, LSMBasisSystem

        # Simple 2-step American put
        n_paths = 5000
        key = jax.random.PRNGKey(42)
        K, S0, r, sigma = 100.0, 100.0, 0.05, 0.20
        dt = 0.5

        # Generate GBM paths: 3 time points (t=0, t=0.5, t=1.0)
        keys = jax.random.split(key, 2)
        Z1 = jax.random.normal(keys[0], (n_paths,))
        Z2 = jax.random.normal(keys[1], (n_paths,))
        S1 = S0 * jnp.exp((r - 0.5 * sigma ** 2) * dt + sigma * jnp.sqrt(dt) * Z1)
        S2 = S1 * jnp.exp((r - 0.5 * sigma ** 2) * dt + sigma * jnp.sqrt(dt) * Z2)

        paths = jnp.column_stack([jnp.full(n_paths, S0), S1, S2])
        cashflows = jnp.maximum(K - paths[:, 1:], 0.0)
        discount_factors = jnp.array([jnp.exp(-r * dt), jnp.exp(-r * 2 * dt)])

        basis = LSMBasisSystem(basis_type='monomial', order=3)
        values = generic_ls_regression(paths, cashflows, discount_factors, basis=basis)
        am_price = float(jnp.mean(values))
        assert am_price > 0
        assert am_price < 20  # Reasonable bound
