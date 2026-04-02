"""Tests for random number generation."""

import jax.numpy as jnp
from ql_jax.math.random import pseudo, quasi


class TestPseudoRandom:
    def test_uniform_range(self):
        key = pseudo.make_key(42)
        samples = pseudo.uniform(key, shape=(10000,))
        assert float(jnp.min(samples)) >= 0.0
        assert float(jnp.max(samples)) < 1.0

    def test_normal_mean_std(self):
        key = pseudo.make_key(42)
        samples = pseudo.normal(key, shape=(100000,))
        assert abs(float(jnp.mean(samples))) < 0.02
        assert abs(float(jnp.std(samples)) - 1.0) < 0.02

    def test_reproducible(self):
        key1 = pseudo.make_key(42)
        key2 = pseudo.make_key(42)
        s1 = pseudo.uniform(key1, shape=(10,))
        s2 = pseudo.uniform(key2, shape=(10,))
        assert jnp.allclose(s1, s2)

    def test_split_independent(self):
        key = pseudo.make_key(42)
        keys = pseudo.split(key, 3)
        s1 = pseudo.normal(keys[0], shape=(100,))
        s2 = pseudo.normal(keys[1], shape=(100,))
        # Should be different
        assert not jnp.allclose(s1, s2)


class TestQuasiRandom:
    def test_halton_range(self):
        seq = quasi.halton_sequence(2, 100)
        assert seq.shape == (100, 2)
        assert float(jnp.min(seq)) >= 0.0
        assert float(jnp.max(seq)) <= 1.0

    def test_halton_dimension(self):
        seq = quasi.halton_sequence(5, 50)
        assert seq.shape == (50, 5)
