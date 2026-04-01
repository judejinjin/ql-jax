"""JAX utility helpers for QL-JAX: pytree registration, precision config, etc."""

from functools import partial
from typing import Any, Type

import jax
import jax.numpy as jnp


def enable_float64():
    """Enable 64-bit floating point in JAX (must be called before any JAX computation)."""
    jax.config.update("jax_enable_x64", True)


def ensure_array(x: Any, dtype=jnp.float64) -> jnp.ndarray:
    """Convert input to a JAX array with the specified dtype."""
    return jnp.asarray(x, dtype=dtype)


def register_pytree_dataclass(cls: Type) -> Type:
    """Register a frozen dataclass as a JAX pytree node.

    The dataclass fields become the children (leaves) of the pytree.
    """
    field_names = [f.name for f in cls.__dataclass_fields__.values()]

    def tree_flatten(obj):
        children = [getattr(obj, name) for name in field_names]
        return children, None

    def tree_unflatten(aux_data, children):
        return cls(**dict(zip(field_names, children)))

    jax.tree_util.register_pytree_node(cls, tree_flatten, tree_unflatten)
    return cls


def is_gpu_available() -> bool:
    """Check if a GPU backend is available."""
    return len(jax.devices("gpu")) > 0


def default_backend() -> str:
    """Return the default JAX backend name ('cpu', 'gpu', or 'tpu')."""
    return jax.default_backend()
