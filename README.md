# QL-JAX

A JAX re-implementation of [QuantLib](https://www.quantlib.org/).

## Features

- **Automatic Greeks** via `jax.grad` — exact algorithmic differentiation replaces finite-difference bumping
- **GPU/TPU acceleration** via `jax.jit` — XLA compilation to CPU, GPU, or TPU with zero code changes
- **Vectorized portfolio pricing** via `jax.vmap` — price thousands of instruments in a single batched call
- **End-to-end differentiable** — gradient-based calibration, hedging optimization, and ML integration

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# Install with dev tools
pip install -e ".[dev]"

# Install with GPU support
pip install -e ".[gpu]"
```

## Quick Start

```python
import ql_jax
print(ql_jax.__version__)
```

## Development

```bash
make install-dev   # Install with dev dependencies
make test          # Run tests
make lint          # Run linters
make format        # Auto-format code
make bench         # Run benchmarks
```

## License

BSD-3-Clause
