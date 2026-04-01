# QL-JAX: A JAX Re-implementation of QuantLib

## 1. Executive Summary

This project re-implements the open-source [QuantLib](https://www.quantlib.org/) C++ library (located at `/mnt/c/finance/QuantLib`) using the [JAX](https://github.com/jax-ml/jax) framework (located at `/home/jude/jax`). JAX provides built-in automatic differentiation (forward and reverse mode), automatic vectorization (`vmap`), just-in-time compilation (`jit`), and hardware acceleration across CPU, GPU (NVIDIA CUDA / AMD ROCm), and TPU backends.

Re-implementing QuantLib in JAX unlocks:

- **Hardware acceleration** — offload heavy pricing, calibration, and risk computations to GPU/TPU via XLA compilation.
- **Automatic Greeks** — replace finite-difference bumping with exact algorithmic differentiation (`jax.grad`, `jax.jacrev`, `jax.hessian`) for sensitivities at any order.
- **Vectorized portfolio pricing** — price thousands of instruments in a single batched call via `jax.vmap`, eliminating Python loops.
- **End-to-end differentiable finance** — enable gradient-based calibration, hedging optimization, and machine-learning integration with no code changes.

All classes, methods, examples, and tests in QuantLib will be re-implemented. JAX versions of tests and examples must match the numerical output of their QuantLib C++ counterparts within acceptable floating-point tolerance.

---

## 2. Background & Motivation

### 2.1 QuantLib Overview

QuantLib is a production-grade, open-source C++ framework for quantitative finance comprising approximately **1,000+ source files** organized into **25+ major modules**. It provides:

| Area | Scope |
|------|-------|
| **Instruments** | 87+ files — vanilla/exotic options, bonds, swaps, CDS, swaptions, caps/floors, FRAs, convertibles |
| **Pricing Engines** | 100+ engines — analytic (Black-Scholes, Heston, SABR), binomial/trinomial trees, finite differences, Monte Carlo |
| **Term Structures** | Yield curves (piecewise bootstrap, fitted bond), volatility surfaces/cubes, credit curves, inflation curves |
| **Stochastic Processes** | 21 types — GBM, Heston, CIR, Hull-White, G2++, Bates, GJRGARCH |
| **Numerical Methods** | Solvers (Newton, Brent), integration (Gauss quadrature), interpolation (24 methods), optimization |
| **Random Numbers** | 32+ generators — Mersenne Twister, Sobol, Halton, Faure, Box-Muller, Ziggurat |
| **Market Data** | 48+ calendars, 50+ IBOR/RFR index definitions, day counters, schedule generation |
| **Examples** | 18 complete programs (equity options, bonds, swaptions, CDS, callable bonds, etc.) |
| **Tests** | 150+ test files with comprehensive coverage |

Key design patterns: **Observer/Observable** (market data propagation), **LazyObject** (deferred computation and caching), **Handle** (relinkable smart pointers), **Strategy** (pricing engine selection), **Visitor** (instrument traversal).

### 2.2 JAX Overview

JAX (version 0.10.0) is a composable program transformation system for numerical computing. Its core transforms are:

| Transform | Purpose |
|-----------|---------|
| `jax.grad` / `jax.jacrev` / `jax.jacfwd` / `jax.hessian` | Automatic differentiation (forward & reverse mode) |
| `jax.jit` | Just-in-time XLA compilation to CPU/GPU/TPU native code |
| `jax.vmap` | Automatic vectorization (batch dimension lifting) |
| `jax.pmap` / `jax.shard_map` | Multi-device parallelism |
| `jax.custom_jvp` / `jax.custom_vjp` | User-defined derivative rules |
| `jax.checkpoint` / `jax.remat` | Gradient checkpointing (memory/compute trade-off) |

JAX provides NumPy-compatible (`jax.numpy`), SciPy-compatible (`jax.scipy`), and low-level (`jax.lax`) APIs, all fully differentiable and JIT-compilable. Its compilation pipeline traces Python → Jaxpr IR → StableHLO (MLIR) → XLA → device code (LLVM/CUDA/cuDNN).

### 2.3 Why Re-implement QuantLib in JAX

| Challenge in C++ QuantLib | JAX Solution |
|--------------------------|--------------|
| Greeks via finite-difference bumping (slow, inaccurate, O(n) per Greek) | `jax.grad` gives exact AD Greeks in a single backward pass |
| Sequential instrument pricing in loops | `jax.vmap` vectorizes across portfolios in one kernel launch |
| CPU-only execution | `jax.jit` compiles to GPU/TPU via XLA with zero code changes |
| Manual parallelism (OpenMP/MPI) | `jax.pmap` / `jax.shard_map` for automatic multi-device distribution |
| Calibration via derivative-free optimizers | Gradient-based optimization with `jax.grad` for faster convergence |
| No native ML integration | Seamless interop with Flax/Optax for hybrid ML-quant models |

---

## 3. Project Scope

### 3.1 In Scope

All QuantLib functionality will be ported, organized into the following module groups:

#### 3.1.1 Foundation Layer

| Module | QuantLib Source | QL-JAX Target | Key Items |
|--------|----------------|---------------|-----------|
| **Time** | `ql/time/` | `ql_jax/time/` | Date, Period, Calendar (48+ markets), DayCounter (10+ conventions), Schedule |
| **Quotes** | `ql/quotes/` | `ql_jax/quotes/` | SimpleQuote, CompositeQuote, DerivedQuote, ForwardValueQuote |
| **Math — Core** | `ql/math/` | `ql_jax/math/` | Matrix ops, Array, comparison, rounding, functional |
| **Math — Interpolation** | `ql/math/interpolations/` | `ql_jax/math/interpolations/` | Linear, cubic, log-linear, SABR, Chebyshev, B-spline (24 methods) |
| **Math — Solvers** | `ql/math/solvers1d/` | `ql_jax/math/solvers/` | Brent, Newton, Bisection, Secant, Ridder |
| **Math — Integration** | `ql/math/integrals/` | `ql_jax/math/integrals/` | Gauss-Legendre, Gauss-Laguerre, adaptive Simpson, Kronrod |
| **Math — Distributions** | `ql/math/distributions/` | `ql_jax/math/distributions/` | Normal, Poisson, Gamma, Chi-squared, inverse CDF |
| **Math — Optimization** | `ql/math/optimization/` | `ql_jax/math/optimization/` | Levenberg-Marquardt, Simplex, BFGS, differential evolution |
| **Math — RNG/LDS** | `ql/math/randomnumbers/` | `ql_jax/math/random/` | MT19937, Sobol, Halton, Box-Muller, inverse-normal (mapped to `jax.random`) |
| **Math — Statistics** | `ql/math/statistics/` | `ql_jax/math/statistics/` | GeneralStatistics, IncrementalStatistics, RiskStatistics |
| **Patterns** | `ql/patterns/` | `ql_jax/patterns/` | Observable, LazyObject (adapted to JAX's functional paradigm) |

#### 3.1.2 Market Data Layer

| Module | QuantLib Source | QL-JAX Target | Key Items |
|--------|----------------|---------------|-----------|
| **Yield Term Structures** | `ql/termstructures/yield/` | `ql_jax/termstructures/yield/` | Piecewise bootstrap, fitted bond curve, discount/forward/zero curves |
| **Volatility Surfaces** | `ql/termstructures/volatility/` | `ql_jax/termstructures/volatility/` | Black vol, local vol, SmileSection, SwaptionVolCube, SABR smile |
| **Credit Curves** | `ql/termstructures/credit/` | `ql_jax/termstructures/credit/` | Default probability, hazard rate, survival probability curves |
| **Inflation Curves** | `ql/termstructures/inflation/` | `ql_jax/termstructures/inflation/` | Zero-coupon and year-on-year inflation curves |
| **Indexes** | `ql/indexes/` | `ql_jax/indexes/` | 50+ IBOR variants, modern RFRs (SOFR, SONIA, ESTR, TONA, SARON), swap indexes |
| **Currencies & FX** | `ql/currencies/` | `ql_jax/currencies/` | Currency definitions, exchange rates |

#### 3.1.3 Instruments & Cash Flows

| Module | QuantLib Source | QL-JAX Target | Key Items |
|--------|----------------|---------------|-----------|
| **Cash Flows** | `ql/cashflows/` | `ql_jax/cashflows/` | FixedRateCoupon, IborCoupon, CMSCoupon, CapFlooredCoupon, digital coupons (50+ files) |
| **Equity Instruments** | `ql/instruments/` | `ql_jax/instruments/` | VanillaOption, BarrierOption, AsianOption, LookbackOption, CliquetOption, BasketOption |
| **Fixed Income** | `ql/instruments/` + `ql/instruments/bonds/` | `ql_jax/instruments/` | Bond (fixed, floating, zero, callable, convertible), Swap (vanilla, basis, OIS), FRA |
| **Interest Rate Derivatives** | `ql/instruments/` | `ql_jax/instruments/` | Swaption, CapFloor, CreditDefaultSwap |
| **Composite** | `ql/instruments/` | `ql_jax/instruments/` | CompositeInstrument, payoff definitions |

#### 3.1.4 Stochastic Processes

| Module | QuantLib Source | QL-JAX Target | Key Items |
|--------|----------------|---------------|-----------|
| **Equity Processes** | `ql/processes/` | `ql_jax/processes/` | BlackScholesProcess, HestonProcess, BatesProcess, Merton76, GJRGARCH |
| **Rate Processes** | `ql/processes/` | `ql_jax/processes/` | HullWhiteProcess, G2Process, CIRProcess |
| **General** | `ql/processes/` | `ql_jax/processes/` | OrnsteinUhlenbeck, GeometricBrownianMotion, StochasticProcessArray |

#### 3.1.5 Pricing Engines

| Module | QuantLib Source | QL-JAX Target | Key Items |
|--------|----------------|---------------|-----------|
| **Analytic Engines** | `ql/pricingengines/vanilla/` | `ql_jax/engines/analytic/` | Black-Scholes-Merton, Heston (semi-analytic), Bates, SABR, Kirk's spread |
| **Tree/Lattice Engines** | `ql/pricingengines/vanilla/` + `ql/methods/lattices/` | `ql_jax/engines/lattice/` | CRR, JR, Trigeorgis, Tian binomial; trinomial trees |
| **Finite Difference Engines** | `ql/pricingengines/vanilla/` + `ql/methods/finitedifferences/` | `ql_jax/engines/fd/` | Explicit/implicit/Crank-Nicolson PDE schemes, operators, boundary conditions |
| **Monte Carlo Engines** | `ql/pricingengines/vanilla/` + `ql/methods/montecarlo/` | `ql_jax/engines/mc/` | Path generation, MC European/American/Asian/Barrier, Longstaff-Schwartz regression |
| **Bond Engines** | `ql/pricingengines/bond/` | `ql_jax/engines/bond/` | DiscountingBondEngine, TreeCallableFixedRateBondEngine |
| **Swap Engines** | `ql/pricingengines/swap/` | `ql_jax/engines/swap/` | DiscountingSwapEngine |
| **Swaption Engines** | `ql/pricingengines/swaption/` | `ql_jax/engines/swaption/` | Black, Bachelier, G2/HW tree, Jamshidian |
| **Credit Engines** | `ql/pricingengines/credit/` | `ql_jax/engines/credit/` | MidPointCdsEngine, IntegralCdsEngine |

#### 3.1.6 Models

| Module | QuantLib Source | QL-JAX Target | Key Items |
|--------|----------------|---------------|-----------|
| **Short-Rate Models** | `ql/models/shortrate/` | `ql_jax/models/shortrate/` | Vasicek, CIR, Hull-White (1F/2F), Black-Karasinski, G2++ |
| **Equity Models** | `ql/models/equity/` | `ql_jax/models/equity/` | HestonModel, BatesModel |
| **Market Models** | `ql/models/marketmodels/` | `ql_jax/models/marketmodels/` | LIBOR Market Model, Swap Market Model |
| **Calibration** | `ql/models/` | `ql_jax/models/` | CalibratedModel base, calibration helpers (gradient-based via `jax.grad`) |

#### 3.1.7 Examples (18 Programs)

All QuantLib examples will be re-implemented as JAX equivalents:

| # | Example | Description |
|---|---------|-------------|
| 1 | EquityOption | Pricing European/American/Bermudan equity options with multiple engines |
| 2 | Bonds | Fixed-rate, floating-rate, zero-coupon bond pricing |
| 3 | BermudanSwaption | Bermudan swaption pricing with short-rate models |
| 4 | FittedBondCurve | Yield curve fitting (Nelson-Siegel, Svensson, etc.) |
| 5 | FRA | Forward rate agreement pricing |
| 6 | Repo | Repo instrument pricing |
| 7 | CDS | Credit default swap pricing |
| 8 | CallableBonds | Callable fixed-rate bond pricing with Hull-White |
| 9 | ConvertibleBonds | Convertible bond pricing |
| 10 | DiscreteHedging | Discrete hedging simulation for Black-Scholes |
| 11 | BasketLosses | Portfolio credit loss modeling |
| 12 | LatentModel | Latent variable credit model |
| 13 | MarketModels | LIBOR market model calibration and pricing |
| 14 | Gaussian1dModels | Gaussian short-rate model examples |
| 15 | MulticurveBootstrapping | Modern multi-curve framework |
| 16 | GlobalOptimizer | Global optimization for model calibration |
| 17 | MultidimIntegral | Multi-dimensional numerical integration |
| 18 | Replication | Static replication strategies |
| 19 | CVAIRS | CVA for interest-rate swaps |
| 20 | AsianOption | Asian option pricing (arithmetic/geometric) |

Each example will also include a **JAX-enhanced variant** demonstrating AD Greeks, `vmap` portfolio pricing, and GPU acceleration.

#### 3.1.8 Test Suite (150+ Files)

All QuantLib test files will be ported. Each test will:
- Reproduce the QuantLib C++ test logic in Python/JAX.
- Assert numerical agreement with C++ outputs within tolerance (typically `1e-10` for analytic, `1e-4` for MC).
- Add JAX-specific tests for AD correctness, `vmap` batch consistency, and `jit` compilation.

### 3.2 Out of Scope (Initial Release)

- GUI/Excel integration layers
- Real-time market data feeds
- Database persistence
- The `ql/experimental/` directory (30+ subdirectories) — deferred to a future phase

---

## 4. Architecture & Design

### 4.1 Design Philosophy

QuantLib's C++ architecture relies on OOP patterns (inheritance, virtual dispatch, mutable observer state). JAX requires **pure functions** — no side effects, no in-place mutation, deterministic outputs. The re-implementation must bridge this fundamental gap.

**Core translation principles:**

| C++ QuantLib Pattern | JAX/Python Equivalent |
|----------------------|----------------------|
| Class with mutable state | Immutable dataclass / named tuple + pure functions |
| Observer/Observable | Explicit dependency passing; recompute on input change (or use `jax.checkpoint`) |
| Virtual dispatch (pricing engine) | Function dispatch via Python dict/registry; `functools.singledispatch` |
| LazyObject caching | `jax.jit` compilation cache; `functools.lru_cache` for Python-level memoization |
| Handle/relinkable pointer | Explicit parameter passing; closure capture |
| Template metaprogramming | Python generics / protocols; `jax.lax` polymorphic primitives |
| `boost::shared_ptr` | Standard Python references (GC-managed) |
| Inheritance hierarchy | Python `dataclass` + `Protocol` for structural typing; composition over inheritance |

### 4.2 Package Structure

```
ql_jax/
├── __init__.py
├── time/
│   ├── date.py              # Date, Period, DateGeneration
│   ├── calendar.py          # Calendar base + 48 market calendars
│   ├── daycounter.py        # Actual360, Actual365, 30/360, ActualActual, etc.
│   ├── schedule.py          # Schedule generation
│   └── businessdayconvention.py
├── quotes/
│   ├── simplequote.py
│   └── compositequote.py
├── math/
│   ├── interpolations/      # 24 interpolation methods
│   │   ├── linear.py
│   │   ├── cubic.py
│   │   ├── sabr.py
│   │   └── ...
│   ├── solvers/             # 1D root finders
│   ├── integrals/           # Numerical quadrature
│   ├── distributions/       # Statistical distributions
│   ├── optimization/        # Optimizers
│   ├── random/              # RNG wrappers around jax.random
│   └── statistics/          # Descriptive / risk statistics
├── termstructures/
│   ├── yield_/              # Yield curves (piecewise, fitted, flat)
│   ├── volatility/          # Vol surfaces & cubes
│   ├── credit/              # Credit term structures
│   └── inflation/           # Inflation curves
├── indexes/
│   ├── ibor.py              # IBOR indexes
│   ├── rfr.py               # RFR indexes (SOFR, SONIA, etc.)
│   └── swap.py              # Swap indexes
├── currencies/
│   └── currency.py
├── cashflows/
│   ├── fixedratecoupon.py
│   ├── iborcoupon.py
│   ├── cmscoupon.py
│   └── ...
├── instruments/
│   ├── option.py            # VanillaOption, payoffs, exercise types
│   ├── barrier.py           # BarrierOption
│   ├── asian.py             # AsianOption
│   ├── bond.py              # Bond types
│   ├── swap.py              # Swap types
│   ├── swaption.py          # Swaption
│   ├── capfloor.py          # Cap/Floor
│   ├── cds.py               # CreditDefaultSwap
│   └── ...
├── processes/
│   ├── blackscholesprocess.py
│   ├── hestonprocess.py
│   ├── hullwhiteprocess.py
│   └── ...
├── models/
│   ├── shortrate/           # Vasicek, CIR, HW, G2++
│   ├── equity/              # Heston, Bates
│   ├── marketmodels/        # LMM, SMM
│   └── calibration.py       # Gradient-based calibration via jax.grad
├── engines/
│   ├── analytic/            # Closed-form pricing
│   ├── lattice/             # Tree methods
│   ├── fd/                  # Finite difference PDE
│   ├── mc/                  # Monte Carlo
│   ├── bond/                # Bond engines
│   ├── swap/                # Swap engines
│   ├── swaption/            # Swaption engines
│   └── credit/              # CDS engines
├── methods/
│   ├── finitedifferences/   # FD schemes, operators, meshers
│   ├── lattices/            # Tree construction
│   └── montecarlo/          # Path generation, regression
└── _util/
    ├── types.py             # Type aliases, enums
    └── jax_helpers.py       # JAX utility wrappers
```

### 4.3 Data Representation

All numerical data will use **JAX arrays** (`jnp.ndarray`) to ensure compatibility with `jit`, `grad`, and `vmap`:

```python
import jax.numpy as jnp
from dataclasses import dataclass

# Instrument state as immutable JAX-compatible structure
@dataclass(frozen=True)
class VanillaOptionData:
    strike: jnp.ndarray       # scalar or batched
    maturity: float            # time to maturity in years
    option_type: int           # 1 = call, -1 = put (static)

# Market data as immutable structure
@dataclass(frozen=True)
class BlackScholesMarket:
    spot: jnp.ndarray
    rate: jnp.ndarray
    dividend: jnp.ndarray
    volatility: jnp.ndarray
```

Structures will be registered as **JAX pytrees** to enable seamless use with all transforms.

### 4.4 AD Strategy for Greeks

QuantLib computes Greeks via finite-difference bumping. QL-JAX will provide **exact algorithmic Greeks** via JAX's AD:

```python
# Pricing function: pure function of market data → price
def black_scholes_price(market: BlackScholesMarket, option: VanillaOptionData) -> jnp.ndarray:
    ...

# First-order Greeks via grad
delta = jax.grad(black_scholes_price, argnums=0)  # ∂V/∂S
vega  = jax.grad(black_scholes_price, argnums=0)   # ∂V/∂σ (w.r.t. vol field)

# Second-order Greeks
gamma = jax.grad(jax.grad(black_scholes_price))     # ∂²V/∂S²

# Full Jacobian across all market inputs
jacobian = jax.jacrev(black_scholes_price, argnums=0)

# Hessian for second-order risk
hessian = jax.hessian(black_scholes_price, argnums=0)
```

For algorithms with non-differentiable components (Monte Carlo control flow, early exercise), `jax.custom_vjp` will define numerically stable custom derivative rules.

### 4.5 Vectorization Strategy

Portfolio-level pricing via `vmap` eliminates Python loops:

```python
# Price a single bond
def price_bond(market, bond_data):
    ...

# Price 10,000 bonds in one batched kernel
batch_price = jax.vmap(price_bond, in_axes=(None, 0))
prices = batch_price(shared_market, portfolio_of_bonds)  # shape: (10000,)

# Greeks for entire portfolio
batch_delta = jax.vmap(jax.grad(price_bond), in_axes=(None, 0))
```

### 4.6 Custom Derivatives for Numerical Stability

Several QuantLib algorithms require custom derivative rules:

| Algorithm | Why Custom Derivative |
|-----------|----------------------|
| **Root finding** (Newton, Brent) | Implicit function theorem: differentiate through the fixed point, not the iteration |
| **Curve bootstrapping** | Differentiate through the bootstrap solution via implicit differentiation |
| **Monte Carlo pricing** | Pathwise derivatives or likelihood ratio method for discontinuous payoffs |
| **PDE solvers** | Adjoint PDE method for efficient sensitivity computation |
| **Optimization / calibration** | Implicit differentiation through optimality conditions |

These will use `jax.custom_jvp` / `jax.custom_vjp` with mathematically correct derivative rules.

---

## 5. Implementation Plan

The project is organized into 8 phases. Each phase produces a shippable increment with its own tests and examples.

### Phase 1: Foundation (Time, Math Utilities, Core Patterns)

**Modules:** `time/`, `math/` (core, interpolations, solvers, distributions, integration, statistics, random), `_util/`, `patterns/`

**Key deliverables:**
- Date arithmetic, calendar engine (48 markets), day counters (10+ conventions), schedule generation
- 24 interpolation methods (all differentiable via JAX)
- 1D root finders (Brent, Newton, etc.) with implicit-function-theorem AD
- Gauss quadrature, adaptive integration
- Normal, Poisson, Gamma, Chi-squared distributions via `jax.scipy.stats`
- Mersenne Twister / Sobol / Halton mapped to `jax.random` PRNG
- Observable/LazyObject patterns adapted to functional style
- Statistics (moments, VaR, expected shortfall)

**Tests:** Port all QuantLib date, calendar, day-counter, interpolation, solver, and statistics tests.

### Phase 2: Market Data (Term Structures, Quotes, Indexes)

**Modules:** `quotes/`, `termstructures/yield_/`, `termstructures/volatility/`, `indexes/`, `currencies/`

**Key deliverables:**
- Quote types (Simple, Composite, Derived)
- Flat and interpolated yield curves
- Piecewise yield curve bootstrap (with AD through the bootstrap via `custom_vjp`)
- Fitted bond curve (Nelson-Siegel, Svensson)
- Black volatility term structures and smile sections
- SABR smile interpolation
- IBOR indexes (50+ variants) and RFR indexes (SOFR, SONIA, ESTR, etc.)
- Currency definitions

**Tests:** Port yield curve, vol surface, and index tests. Verify bootstrap outputs match C++.

**Example:** `MulticurveBootstrapping`

### Phase 3: Cash Flows & Fixed Income Instruments

**Modules:** `cashflows/`, `instruments/` (bonds, swaps, FRA)

**Key deliverables:**
- FixedRateCoupon, IborCoupon, CMSCoupon, CapFlooredCoupon, digital coupons
- Bond types: FixedRateBond, FloatingRateBond, ZeroCouponBond
- Swaps: VanillaSwap, OvernightIndexedSwap, BasisSwap
- FRAs
- DiscountingBondEngine, DiscountingSwapEngine

**Tests:** Port all bond, swap, FRA, and coupon tests.

**Examples:** `Bonds`, `FRA`, `Repo`

### Phase 4: Equity Instruments & Core Pricing Engines

**Modules:** `instruments/` (options), `processes/` (Black-Scholes, Heston), `engines/analytic/`, `engines/lattice/`, `engines/fd/`

**Key deliverables:**
- VanillaOption, payoffs, exercise types (European, American, Bermudan)
- Barrier, Asian, Lookback, Cliquet options
- BlackScholesProcess, HestonProcess
- Analytic engines: BSM, Heston semi-analytic, Bates, SABR
- Binomial tree engines (CRR, JR, Tian, Trigeorgis)
- Finite difference engines (explicit, implicit, Crank-Nicolson)
- AD Greeks out-of-the-box for all engines

**Tests:** Port all equity option, barrier, Asian, etc. tests.

**Example:** `EquityOption` (with JAX-enhanced variant showing AD Greeks + `vmap` batch pricing)

### Phase 5: Monte Carlo Framework

**Modules:** `methods/montecarlo/`, `engines/mc/`, `math/random/` (LDS)

**Key deliverables:**
- Path generation (GBM, Heston, multi-factor)
- MC European, American (Longstaff-Schwartz), Asian, Barrier engines
- Sobol / Halton low-discrepancy sequences
- Antithetic and control variate variance reduction
- Pathwise AD for Monte Carlo Greeks via `jax.grad`

**Tests:** Port MC test suite. Verify convergence rates.

**Examples:** `DiscreteHedging`, `AsianOption`

### Phase 6: Interest Rate Models & Derivatives

**Modules:** `models/shortrate/`, `engines/swaption/`, `instruments/` (swaption, cap/floor), `models/equity/`

**Key deliverables:**
- Short-rate models: Vasicek, CIR, Hull-White (1F/2F), G2++, Black-Karasinski
- Swaption engines: Black, Bachelier, tree-based (HW/G2), Jamshidian
- Cap/Floor pricing
- Callable bond pricing (TreeCallableFixedRateBondEngine)
- Heston/Bates model classes
- **Gradient-based model calibration** using `jax.grad` + optimizers

**Tests:** Port swaption, cap/floor, callable bond, and model calibration tests.

**Examples:** `BermudanSwaption`, `CallableBonds`, `ConvertibleBonds`, `Gaussian1dModels`

### Phase 7: Credit, Inflation & Advanced Models

**Modules:** `termstructures/credit/`, `termstructures/inflation/`, `instruments/` (CDS), `engines/credit/`, `models/marketmodels/`

**Key deliverables:**
- Credit term structures (hazard rate, default probability, survival probability)
- CDS pricing (mid-point, integral, ISDA)
- Inflation curves (zero-coupon, year-on-year)
- LIBOR Market Model (LMM) and Swap Market Model (SMM)

**Tests:** Port CDS, credit, inflation, and LMM tests.

**Examples:** `CDS`, `BasketLosses`, `LatentModel`, `MarketModels`, `CVAIRS`

### Phase 8: Integration, Optimization & Polish

**Key deliverables:**
- Remaining examples: `GlobalOptimizer`, `MultidimIntegral`, `Replication`
- End-to-end integration tests (full pricing workflows matching C++ outputs)
- Performance benchmarks (CPU vs GPU, QL-JAX vs C++ QuantLib)
- API documentation (Sphinx + NumPy-style docstrings)
- CI/CD pipeline (pytest, GPU test runner)
- Package distribution (`pyproject.toml`, PyPI-ready)

---

## 6. Testing Strategy

### 6.1 Test Categories

| Category | Purpose | Tolerance |
|----------|---------|-----------|
| **Correctness** | Match QuantLib C++ outputs for all tests/examples | `1e-10` analytic, `1e-4` MC |
| **AD Correctness** | Verify AD Greeks against finite-difference bumping | `1e-6` relative |
| **vmap Consistency** | Batch results equal loop-over-single results | exact (bitwise) |
| **jit Invariance** | JIT-compiled output equals eager output | exact (bitwise) |
| **Numerical Stability** | No NaN/Inf in gradients across reasonable input ranges | `jax.config.debug_nans` |
| **Regression** | Golden-value snapshots for key pricing scenarios | `1e-12` |

### 6.2 Test Infrastructure

- **Framework:** `pytest` with `jax.config.update("jax_enable_x64", True)` for double precision.
- **Reference values:** Generated by running QuantLib C++ tests and capturing expected outputs.
- **Parametric tests:** `pytest.mark.parametrize` over instrument types, engines, market scenarios.
- **CI matrix:** Test on CPU and GPU (CUDA) backends.

---

## 7. Performance Targets

| Benchmark | QuantLib C++ (baseline) | QL-JAX CPU | QL-JAX GPU |
|-----------|------------------------|------------|------------|
| Single European option (BSM analytic) | ~1 µs | ~1 µs (jitted) | ~5 µs (kernel launch overhead) |
| 100,000 European options (batched) | ~100 ms (loop) | ~5 ms (vmap+jit) | ~0.5 ms (vmap+jit+GPU) |
| Heston calibration (10 strikes × 5 maturities) | ~500 ms (derivative-free) | ~50 ms (gradient-based, jit) | ~10 ms (GPU) |
| MC American option (100k paths, 50 steps) | ~200 ms | ~50 ms (jit) | ~5 ms (GPU) |
| Portfolio Greeks (1000 instruments, 6 Greeks each) | ~6s (6 bumps × 1000 prices) | ~200 ms (vmap + jacrev) | ~20 ms (GPU) |
| Yield curve bootstrap (20 instruments) | ~1 ms | ~2 ms (jit, first call) | ~1 ms (cached) |

**Note:** First-call JIT compilation overhead is excluded. Subsequent calls benefit from XLA's compilation cache.

### 7.1 CPU vs GPU JIT Backend Benchmark Requirement

A dedicated benchmark suite must be developed to systematically compare JAX's CPU and GPU JIT backends across all major computation categories. This is required to quantify the real-world speedup from GPU offloading and identify workloads where CPU JIT remains competitive (e.g., small-scale or latency-sensitive calculations where GPU kernel launch overhead dominates).

**Benchmark dimensions:**

| Dimension | Variations |
|-----------|------------|
| **Backend** | `jax.devices("cpu")` vs `jax.devices("gpu")` — each benchmark run on both |
| **Problem size** | Small (1–100), medium (1k–10k), large (100k–1M) instruments/paths/grid points |
| **Precision** | `float32` vs `float64` (`jax_enable_x64`) — GPU float64 throughput is typically lower |
| **Computation type** | Analytic, tree/lattice, finite difference, Monte Carlo, curve bootstrap, calibration |
| **Transform overhead** | Raw `jit` vs `jit + vmap` vs `jit + vmap + grad` |

**Required benchmarks (CPU JIT vs GPU JIT):**

| Benchmark | What It Measures |
|-----------|-----------------|
| **Analytic pricing** (BSM, Heston) at varying batch sizes | Crossover point where GPU surpasses CPU |
| **Binomial tree** option pricing (10–10,000 steps) | Tree traversal scaling on CPU vs GPU |
| **Finite difference PDE** (varying grid sizes: 100–10,000 spatial points) | Dense linear algebra / tri-diagonal solve performance |
| **Monte Carlo simulation** (1k–1M paths, 10–500 time steps) | Massively parallel path generation and averaging |
| **AD Greeks** — `jax.grad` / `jax.jacrev` on each engine type | Differentiation overhead on CPU vs GPU |
| **Batched portfolio pricing** — `vmap` over 100–1M instruments | Vectorization throughput scaling |
| **Yield curve bootstrap** (5–50 instruments) | Iterative solvers with small array sizes |
| **Model calibration** (Heston, Hull-White) — gradient-based optimizer | End-to-end optimization loop, many small JIT calls |
| **JIT compilation latency** — first-call wall time per engine | Compilation cost on CPU vs GPU backend |
| **Memory transfer overhead** — host↔device data movement | Quantify transfer cost relative to computation time |

**Benchmark methodology:**

- Use `pytest-benchmark` or a custom harness that records **wall time**, **peak memory**, and **FLOP utilization**.
- Each benchmark must run **warm-up iterations** (to trigger JIT compilation) followed by **timed iterations** (minimum 10 repetitions).
- Report **median**, **p5**, and **p95** latencies.
- Record hardware specs (CPU model, GPU model, CUDA version, JAX/jaxlib version) in every report.
- Produce a markdown summary table and plots (latency vs problem size, CPU vs GPU speedup ratio) for each benchmark category.

**Crossover analysis:**

For each computation type, identify the **problem-size crossover** where GPU JIT becomes faster than CPU JIT. This informs users when to select `jax.devices("gpu")` vs `jax.devices("cpu")` for optimal performance.

**Deliverable:** A `benchmarks/` directory containing:
- `benchmarks/runner.py` — configurable benchmark runner (backend, size, precision as CLI args)
- `benchmarks/bench_analytic.py`, `bench_mc.py`, `bench_fd.py`, `bench_tree.py`, `bench_calibration.py`, `bench_bootstrap.py`, `bench_portfolio.py`
- `benchmarks/results/` — raw JSON results + auto-generated markdown comparison tables
- `benchmarks/README.md` — instructions to reproduce benchmarks on user hardware

---

## 8. Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **Core framework** | JAX 0.10+ (`jax`, `jaxlib`) |
| **Array library** | `jax.numpy` (NumPy-compatible) |
| **Scientific functions** | `jax.scipy` (SciPy-compatible) |
| **AD** | `jax.grad`, `jax.jacrev`, `jax.jacfwd`, `jax.hessian`, `jax.custom_vjp`, `jax.custom_jvp` |
| **Vectorization** | `jax.vmap` |
| **Compilation** | `jax.jit` → XLA (CPU/GPU/TPU) |
| **GPU backend** | NVIDIA CUDA (via `jax-cuda12-plugin`) or AMD ROCm |
| **Testing** | `pytest`, `pytest-benchmark` |
| **Documentation** | Sphinx, NumPy-style docstrings |
| **Build** | `pyproject.toml`, `setuptools` |
| **CI/CD** | GitHub Actions (CPU) + self-hosted GPU runner |
| **Precision** | 64-bit floats (`jax_enable_x64`) for financial accuracy |
| **Virtual environment** | `venv` (or `conda`) — all development and testing must run inside an isolated virtual environment |

### 8.1 Virtual Environment Requirement

All development, testing, benchmarking, and CI/CD execution **must** use a Python virtual environment to ensure reproducible, isolated builds. No system-wide Python packages should be relied upon.

**Setup:**

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install the project in editable mode with all dev/test dependencies
pip install -e ".[dev,test]"
```

**`pyproject.toml` extras:**

```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-benchmark",
    "sphinx",
    "black",
    "ruff",
    "mypy",
    "pre-commit",
]
test = [
    "pytest",
    "pytest-xdist",
    "pytest-benchmark",
]
gpu = [
    "jax[cuda12]",
]
```

**Rules:**

- The `.venv/` directory must be listed in `.gitignore`.
- CI/CD pipelines must create a fresh virtual environment per job.
- A `Makefile` (or `justfile`) must provide convenience targets: `make venv`, `make test`, `make bench`, `make docs`.
- Lock files (`requirements-lock.txt` or `uv.lock`) must be committed to pin exact dependency versions for reproducibility.
- GPU-specific dependencies (`jax[cuda12]`) are installed via the `gpu` extra, keeping the base environment lightweight.

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **JAX's functional constraint vs. QuantLib's mutable state** | High — many QuantLib algorithms rely on mutable observer state and caching | Design immutable data structures upfront; use pytree registration; explicit dependency passing |
| **Non-differentiable operations** (integer logic, date arithmetic, branching) | Medium — calendar/schedule logic is inherently discrete | Separate differentiable (pricing, curves) from non-differentiable (dates, calendars) layers; use `jax.lax.cond`/`switch` where possible |
| **XLA compilation overhead** for complex functions | Medium — first-call latency can be seconds | Cache compiled functions aggressively; `static_argnums` for non-array args; profile and decompose large computations |
| **Numerical precision differences** (float64 behavior across CPU/GPU) | Medium — GPU float64 may differ in last ULP | Test with tolerances; validate on CPU first; use `jax_enable_x64` everywhere |
| **Monte Carlo variance** makes test matching difficult | Low — inherent randomness | Fix PRNG seeds; use large sample sizes for convergence; match moments/distribution rather than point values |
| **Scope creep** from ~1,000 source files | High — massive library | Strict phase gating; prioritize most-used instruments first; defer `experimental/` |
| **JAX breaking changes** across versions | Low — JAX API stabilizing | Pin JAX version; use only public API; add compatibility shims |

---

## 10. Success Criteria

1. **Numerical fidelity** — All 150+ test files pass, matching QuantLib C++ outputs within specified tolerances.
2. **All 18 examples** run and produce equivalent results to their C++ counterparts.
3. **AD Greeks** — Every differentiable pricing function supports `jax.grad` and produces Greeks that agree with finite-difference values to `1e-6` relative tolerance.
4. **Vectorization** — `jax.vmap` over any pricing function produces results identical to a sequential loop.
5. **GPU acceleration** — Key benchmarks (batched option pricing, MC simulation, model calibration) demonstrate >10× speedup on GPU vs. single-threaded CPU.
6. **CPU vs GPU JIT benchmarks** — All computation categories benchmarked on both CPU and GPU JIT backends; crossover analysis identifies the problem-size threshold where GPU outperforms CPU for each engine type.
7. **JIT compilation** — All pricing engines execute correctly under `jax.jit`.
8. **Package quality** — Installable via `pip install ql-jax`; documented API; CI green on CPU and GPU.

---

## 11. Deliverables Summary

| Deliverable | Description |
|-------------|-------------|
| `ql_jax` Python package | Full re-implementation of QuantLib in JAX |
| Test suite | 150+ test modules matching C++ outputs + AD/vmap/jit tests |
| Examples | 18+ example scripts with JAX-enhanced variants |
| Benchmarks | Performance comparison suite: CPU JIT vs GPU JIT crossover analysis, plus comparison against C++ QuantLib |
| Documentation | API reference (Sphinx), user guide, migration guide from C++ QuantLib |
| CI/CD pipeline | Automated testing on CPU and GPU |

