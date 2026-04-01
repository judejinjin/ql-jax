"""Smoke tests: verify JAX works and ql_jax can be imported."""

import jax
import jax.numpy as jnp
import ql_jax


def test_import():
    assert hasattr(ql_jax, "__version__")
    assert ql_jax.__version__ == "0.1.0"


def test_jit():
    f = jax.jit(lambda x: x + 1.0)
    result = f(1.0)
    assert float(result) == 2.0


def test_grad():
    f = lambda x: x ** 2
    df = jax.grad(f)
    assert float(df(3.0)) == 6.0


def test_vmap():
    f = lambda x: x ** 2
    xs = jnp.arange(5, dtype=jnp.float64)
    result = jax.vmap(f)(xs)
    expected = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
    assert jnp.allclose(result, expected)


def test_float64_enabled():
    x = jnp.array(1.0)
    assert x.dtype == jnp.float64
