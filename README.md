# QL-JAX

A JAX re-implementation of [QuantLib](https://www.quantlib.org/) — the most widely-used library for quantitative finance.

[![Test (CPU)](https://github.com/judejinjin/ql-jax/actions/workflows/test-cpu.yml/badge.svg)](https://github.com/judejinjin/ql-jax/actions/workflows/test-cpu.yml)

## Features

- **Automatic Greeks** via `jax.grad` — exact algorithmic differentiation replaces finite-difference bumping
- **GPU/TPU acceleration** via `jax.jit` — XLA compilation to CPU, GPU, or TPU with zero code changes
- **Vectorized portfolio pricing** via `jax.vmap` — price thousands of instruments in a single batched call
- **End-to-end differentiable** — gradient-based calibration, hedging optimization, and ML integration
- **Pure functional design** — frozen dataclasses, no mutation, JAX-traceable through all pricing paths

## What's Implemented

| Module | Contents |
|--------|----------|
| **Time** | Dates, calendars (30+), day counters (7+), schedules, business day conventions |
| **Math** | Interpolation (linear, log-linear, cubic spline), root solvers (Brent, Newton), quadrature, FFT, distributions, random sequences (Sobol, Halton) |
| **Term Structures** | Yield curves (flat, piecewise bootstrap, fitted bond, Nelson-Siegel, Svensson), credit curves, volatility surfaces (Black, SABR, SVI) |
| **Instruments** | Bonds (fixed, zero, floating, amortizing), swaps, FRAs, caps/floors, swaptions, CDS, FX forwards, options (European, American, barrier, Asian) |
| **Pricing Engines** | Black-Scholes analytic, Heston (CF + COS), binomial trees (5 types), Crank-Nicolson FD, Monte Carlo (European, American LSM, barrier, Asian), bond discounting, callable bonds |
| **Models** | Vasicek, CIR, Hull-White, G2++, GARCH(1,1), Heston calibration, LIBOR Market Model (BGM) |
| **Credit** | CDS midpoint pricing, fair spread, hazard rate bootstrapping |
| **Inflation** | YoY inflation caps/floors (Black), zero-coupon inflation swaps |

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

### Price a European Option

```python
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from ql_jax.engines.analytic.black_formula import black_scholes_price

# Price a European call
price = black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20, option_type=1)
print(f"Call price: {price:.4f}")

# Compute delta via AD
delta = jax.grad(lambda s: black_scholes_price(s, 100.0, 1.0, 0.05, 0.0, 0.20, 1))(100.0)
print(f"Delta: {delta:.4f}")

# Batch-price 1000 strikes in parallel
strikes = jnp.linspace(80, 120, 1000)
prices = jax.vmap(lambda k: black_scholes_price(100.0, k, 1.0, 0.05, 0.0, 0.20, 1))(strikes)
```

### Price a Bond

```python
from ql_jax.time.date import Date
from ql_jax.time.schedule import MakeSchedule
from ql_jax.time.calendar import NullCalendar
from ql_jax._util.types import Frequency, BusinessDayConvention
from ql_jax.termstructures.yield_.flat_forward import FlatForward
from ql_jax.instruments.bond import make_fixed_rate_bond
from ql_jax.engines.bond.discounting import discounting_bond_clean_price

curve = FlatForward(Date(15, 1, 2025), 0.045)
schedule = (MakeSchedule().from_date(Date(15, 1, 2025)).to_date(Date(15, 1, 2035))
            .with_frequency(Frequency.Semiannual).with_calendar(NullCalendar())
            .with_convention(BusinessDayConvention.Unadjusted).build())
bond = make_fixed_rate_bond(settlement_days=0, face_amount=100.0,
                             schedule=schedule, coupons=0.05)
clean_price = discounting_bond_clean_price(bond, curve)
```

### Price a CDS

```python
from ql_jax.instruments.cds import make_cds
from ql_jax.engines.credit.midpoint import midpoint_cds_npv, survival_probability

cds = make_cds(notional=10e6, spread=0.01, maturity=5.0)
discount_fn = lambda t: jnp.exp(-0.03 * t)
survival_fn = lambda t: survival_probability(0.02, t)
npv = midpoint_cds_npv(cds, discount_fn, survival_fn)
```

## Project Structure

```
ql_jax/
├── time/               # Date arithmetic, calendars, day counters
├── math/               # Interpolation, solvers, integration, distributions
├── termstructures/     # Yield curves, volatility surfaces, credit curves
├── instruments/        # Bonds, swaps, options, CDS, forwards
├── engines/            # Pricing engines (analytic, tree, FD, MC, credit)
├── models/             # Short-rate, equity, market models, GARCH
├── processes/          # Stochastic processes (GBM, Heston)
├── cashflows/          # Cash flow generation and analytics
├── methods/            # Monte Carlo framework
└── patterns/           # Observable, lazy evaluation
tests/                  # 366+ unit tests
examples/               # Runnable examples
benchmarks/             # Performance benchmarks
```

## Development

```bash
make install-dev   # Install with dev dependencies
make test          # Run tests
make lint          # Run linters
make format        # Auto-format code
make bench         # Run benchmarks
```

## Running Tests

```bash
# All tests (CPU)
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_equity.py -v

# Integration tests
python -m pytest tests/integration/ -v
```

### Running Tests on GPU

```bash
JAX_PLATFORMS=cuda python -m pytest tests/ -v
```

| Flag | Purpose |
|------|---------|
| `JAX_PLATFORMS=cuda` | Tells JAX to use the CUDA (NVIDIA GPU) backend instead of the default CPU. All array operations and JIT-compiled functions run on the GPU. No code changes are needed — JAX dispatches transparently. |

**Requirements**: `pip install -e ".[gpu]"` (installs `jax[cuda12]` with NVIDIA CUDA 12 runtime libraries).

**Note**: The first run is slower because XLA must JIT-compile every kernel for the GPU target. Subsequent runs with cached compilations are faster.

## Examples

```bash
python examples/equity_option.py   # Equity option pricing + AD Greeks
python examples/bonds.py           # Bond pricing + DV01
python examples/cds.py             # CDS pricing + AD sensitivities
```

## License

BSD-3-Clause
