"""Pytest configuration and shared fixtures for QL-JAX tests."""

import pytest
import jax


@pytest.fixture(autouse=True, scope="session")
def _enable_float64():
    """Enable float64 precision for all tests."""
    jax.config.update("jax_enable_x64", True)
