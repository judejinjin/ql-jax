# QL-JAX Detailed Implementation Plan

This document provides a file-by-file implementation plan for re-implementing QuantLib in JAX, derived from the project specification in `project.md`. The plan is organized into 8 phases, each broken into numbered tasks with specific source→target file mappings, dependencies, acceptance criteria, and testing requirements.

---

## Phase 0: Project Scaffolding

### Task 0.1: Repository & Environment Setup

**Goal:** Establish the project skeleton, virtual environment, CI/CD, and build configuration.

**Deliverables:**

```
ql-jax/
├── .gitignore
├── .github/
│   └── workflows/
│       ├── test-cpu.yml
│       └── test-gpu.yml
├── Makefile
├── pyproject.toml
├── README.md
├── ql_jax/
│   ├── __init__.py
│   └── _util/
│       ├── __init__.py
│       ├── types.py          # Enums, type aliases, constants
│       └── jax_helpers.py    # JAX utility wrappers (pytree registration, etc.)
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # pytest fixtures, jax_enable_x64, reference values
│   └── test_smoke.py         # Basic import and jit/grad/vmap smoke test
├── benchmarks/
│   ├── __init__.py
│   ├── runner.py
│   └── README.md
└── examples/
    └── README.md
```

**Steps:**
1. Create virtual environment: `python3 -m venv .venv`
2. Write `pyproject.toml` with `[project]`, `[project.optional-dependencies]` (dev, test, gpu)
3. Write `Makefile` with targets: `venv`, `install`, `test`, `bench`, `docs`, `lint`
4. Write `.gitignore` (`.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `benchmarks/results/`)
5. Write `ql_jax/__init__.py` with version string
6. Write `ql_jax/_util/types.py` — define `OptionType`, `ExerciseType`, `BarrierType` enums; `Array = jnp.ndarray` alias
7. Write `ql_jax/_util/jax_helpers.py` — pytree registration utilities, `ensure_array()`, precision config
8. Write `tests/conftest.py` — enable float64, provide `@pytest.fixture` for common market data
9. Write CI workflows for CPU and GPU test matrix

**Acceptance criteria:**
- `make install && make test` passes
- `import ql_jax` works
- `jax.jit(lambda x: x + 1)(1.0)` runs in test suite
- Virtual environment is isolated (no system packages)

---

## Phase 1: Foundation — Time, Math, Patterns

### Task 1.1: Date & Period Arithmetic

**QuantLib source:** `ql/time/date.hpp`, `ql/time/date.cpp`, `ql/time/period.hpp`, `ql/time/period.cpp`, `ql/time/weekday.hpp`, `ql/time/weekday.cpp`, `ql/time/timeunit.hpp`, `ql/time/timeunit.cpp`, `ql/time/frequency.hpp`, `ql/time/frequency.cpp`

**Target:** `ql_jax/time/date.py`, `ql_jax/time/period.py`

**Implementation notes:**
- Date as integer serial number (days since epoch), matching QuantLib's convention
- Period as `(length, timeunit)` named tuple
- Weekday enum, TimeUnit enum, Frequency enum
- Date arithmetic: `+`, `-`, comparison operators
- Conversion: `date_to_ymd()`, `ymd_to_date()`, `weekday()`, `is_leap()`
- These are non-differentiable (integer operations) — no JAX transforms needed
- Use plain Python/NumPy for date logic; JAX arrays only for year-fractions in pricing

**Tests to port:** `test-suite/dates.cpp`, `test-suite/period.cpp`

**Acceptance criteria:** All serial date round-trips, weekday calculations, period addition/subtraction match C++ outputs exactly.

---

### Task 1.2: Calendars (48 Markets)

**QuantLib source:** `ql/time/calendar.hpp`, `ql/time/calendar.cpp`, `ql/time/calendars/` (47 hpp + 44 cpp = 91 files)

**Target:** `ql_jax/time/calendar.py` (base class + all market implementations)

**Calendar implementations (each a function or class):**
- Argentina, Australia, Austria, Botswana, Brazil, Canada, Chile, China
- CzechRepublic, Denmark, Finland, France, Germany, HongKong, Hungary, Iceland
- India, Indonesia, Israel, Italy, Japan, Mexico, NewZealand, Norway
- Poland, Romania, Russia, SaudiArabia, Singapore, Slovakia, SouthAfrica
- SouthKorea, Sweden, Switzerland, Taiwan, TARGET, Thailand, Turkey, Ukraine
- UnitedKingdom, UnitedStates, WeekendsOnly, NullCalendar, BespokeCalendar
- JointCalendar (combine multiple calendars)

**Implementation notes:**
- Base: `is_business_day(date, calendar) -> bool`
- Derived: `advance(date, period, calendar, convention)`, `business_days_between()`
- Holiday lists as frozen sets per calendar per year
- `BusinessDayConvention` enum: Following, ModifiedFollowing, Preceding, etc.
- Non-differentiable — pure Python

**Tests to port:** `test-suite/calendars.cpp`, `test-suite/businessdayconventions.cpp`

**Acceptance criteria:** Every holiday date for every calendar matches C++ QuantLib for years 1900–2100.

---

### Task 1.3: Day Counters

**QuantLib source:** `ql/time/daycounters/` (13 hpp + 6 cpp = 19 files)

**Target:** `ql_jax/time/daycounter.py`

**Day counter implementations:**
- Actual360, Actual364, Actual365Fixed, Actual36525, Actual366
- ActualActual (ISDA, ISMA/Bond, AFB variants)
- Thirty360 (BondBasis, EurobondBasis, USA, European, Italian variants)
- Thirty365
- Business252 (uses calendar)
- SimpleDayCounter, One
- YearFractionToDate

**Key function:** `year_fraction(d1, d2, convention) -> float` — this is the bridge to JAX (returns float for pricing)

**Tests to port:** `test-suite/daycounters.cpp`

**Acceptance criteria:** Year fractions match C++ for all convention/date combinations in the test suite.

---

### Task 1.4: Schedule Generation

**QuantLib source:** `ql/time/schedule.hpp`, `ql/time/schedule.cpp`, `ql/time/dategenerationrule.hpp`, `ql/time/imm.hpp`, `ql/time/imm.cpp`, `ql/time/ecb.hpp`, `ql/time/ecb.cpp`

**Target:** `ql_jax/time/schedule.py`, `ql_jax/time/imm.py`, `ql_jax/time/ecb.py`

**Implementation notes:**
- DateGeneration rules: Forward, Backward, Zero, ThirdWednesday, Twentieth, etc.
- IMM date calculations
- ECB meeting dates
- Schedule: generate payment/reset date sequences given start, end, tenor, calendar, convention, rule
- Non-differentiable

**Tests to port:** `test-suite/schedule.cpp`

---

### Task 1.5: Core Math — Array, Matrix, Special Functions

**QuantLib source:** `ql/math/` root-level (26 hpp + 15 cpp = 41 files)

**Target:** `ql_jax/math/core.py`

**Key items:**
- `Array` → `jnp.ndarray` (direct mapping)
- `Matrix` → `jnp.ndarray` (2D)
- Beta function → `jax.scipy.special.beta`
- Incomplete gamma → `jax.scipy.special.gammainc`
- Error function → `jax.scipy.special.erf`
- Modified Bessel functions → `jax.scipy.special.i0`, `i1`, `k0`, `k1`
- Factorial → precomputed table or `jax.scipy.special.factorial`
- B-spline basis functions
- Bernstein polynomials
- Richardson extrapolation
- FFT → `jnp.fft.fft`
- Rounding → `jnp.round`, `jnp.floor`, `jnp.ceil`

**Implementation notes:**
- Most map directly to `jax.numpy` or `jax.scipy.special` — thin wrappers
- All differentiable via JAX's built-in rules
- Register as pytrees where needed

**Tests to port:** `test-suite/array.cpp`, `test-suite/matrices.cpp`, `test-suite/functions.cpp`, `test-suite/fastfouriertransform.cpp`

---

### Task 1.6: Interpolation Methods (24 Methods)

**QuantLib source:** `ql/math/interpolations/` (23 hpp + 1 cpp = 24 files)

**Target:** `ql_jax/math/interpolations/`

**Interpolation methods to implement:**

| # | Method | Source File | Target File |
|---|--------|-------------|-------------|
| 1 | Linear | `linearinterpolation.hpp` | `linear.py` |
| 2 | Log-linear | `loginterpolation.hpp` | `log.py` |
| 3 | Backward flat | `backwardflatinterpolation.hpp` | `backward_flat.py` |
| 4 | Forward flat | `forwardflatinterpolation.hpp` | `forward_flat.py` |
| 5 | Cubic spline | `cubicinterpolation.hpp` | `cubic.py` |
| 6 | Bicubic spline | `bicubicsplineinterpolation.hpp` | `bicubic.py` |
| 7 | Bilinear | `bilinearinterpolation.hpp` | `bilinear.py` |
| 8 | Chebyshev | `chebyshevinterpolation.hpp/.cpp` | `chebyshev.py` |
| 9 | Convex monotone | `convexmonotoneinterpolation.hpp` | `convex_monotone.py` |
| 10 | Lagrange | `lagrangeinterpolation.hpp` | `lagrange.py` |
| 11 | Kernel | `kernelinterpolation.hpp` | `kernel.py` |
| 12 | Kernel 2D | `kernelinterpolation2d.hpp` | `kernel2d.py` |
| 13 | SABR | `sabrinterpolation.hpp` | `sabr.py` |
| 14 | XABR | `xabrinterpolation.hpp` | `xabr.py` |
| 15 | ZABR | `zabrinterpolation.hpp` | `zabr.py` |
| 16 | Mixed | `mixedinterpolation.hpp` | `mixed.py` |
| 17 | Multi-cubic spline | `multicubicspline.hpp` | `multicubic.py` |
| 18 | ABCD | `abcdinterpolation.hpp` | `abcd.py` |
| 19 | Backward flat linear | `backwardflatlinearinterpolation.hpp` | `backward_flat_linear.py` |
| 20 | Flat extrapolation 2D | `flatextrapolation2d.hpp` | `flat_extrapolation_2d.py` |
| 21 | Interpolation 2D base | `interpolation2d.hpp` | `base2d.py` |
| 22 | Extrapolation base | `extrapolation.hpp` | `base.py` |

**Implementation notes:**
- Each interpolation: `build(xs, ys) -> InterpolationState`, `evaluate(state, x) -> y`
- All must be differentiable via `jax.grad` for yield curve sensitivity
- Use `jax.lax.cond` or `jnp.where` for piecewise selection (JIT-compatible)
- `custom_jvp` for SABR/ZABR where analytic derivatives are known

**Tests to port:** `test-suite/interpolations.cpp`

---

### Task 1.7: 1D Root Solvers

**QuantLib source:** `ql/math/solvers1d/` (10 hpp = 10 files), `ql/math/solver1d.hpp`

**Target:** `ql_jax/math/solvers/`

**Solvers:**
| Solver | Source | Target |
|--------|--------|--------|
| Brent | `brent.hpp` | `brent.py` |
| Newton | `newton.hpp` | `newton.py` |
| Bisection | `bisection.hpp` | `bisection.py` |
| Secant | `secant.hpp` | `secant.py` |
| Ridder | `ridder.hpp` | `ridder.py` |
| False position | `falseposition.hpp` | `false_position.py` |
| Newton safe | `newtonsafe.hpp` | `newton_safe.py` |
| Halley | `halley.hpp` | `halley.py` |
| FD Newton safe | `finitedifferencenewtonsafe.hpp` | `fd_newton_safe.py` |

**Implementation notes:**
- Implement as `jax.lax.while_loop` for JIT compatibility
- AD through solver: use `jax.custom_vjp` with implicit function theorem: `dx*/dp = -(∂f/∂x)⁻¹ · (∂f/∂p)`
- This is critical — curve bootstrapping and implied vol depend on differentiable solvers

**Tests to port:** `test-suite/solvers.cpp`

**Acceptance criteria:** Root solutions match C++ to `1e-12`; `jax.grad` through solver produces correct sensitivities verified against finite differences.

---

### Task 1.8: Numerical Integration

**QuantLib source:** `ql/math/integrals/` (17 hpp + 9 cpp = 26 files)

**Target:** `ql_jax/math/integrals/`

**Integration methods:**
| Method | Source | Target |
|--------|--------|--------|
| Gauss-Legendre | `gaussianquadratures.hpp` | `gauss.py` |
| Gauss-Laguerre | `gaussianquadratures.hpp` | `gauss.py` |
| Gauss-Hermite | `gaussianquadratures.hpp` | `gauss.py` |
| Gauss-Jacobi | `gaussianquadratures.hpp` | `gauss.py` |
| Gauss-Lobatto | `gausslobattointegral.hpp` | `gauss_lobatto.py` |
| Gauss-Kronrod | `kronrodintegral.hpp` | `kronrod.py` |
| Simpson | `simpsonintegral.hpp` | `simpson.py` |
| Trapezoid | `trapezoidintegral.hpp` | `trapezoid.py` |
| Segment | `segmentintegral.hpp` | `segment.py` |
| Discrete (sum, trapezoid, Simpson) | `discreteintegrals.hpp` | `discrete.py` |
| Filon | `filonintegral.hpp` | `filon.py` |
| TanhSinh | `tanhsinhintegral.hpp` | `tanh_sinh.py` |
| ExpSinh | `expsinhintegral.hpp` | `exp_sinh.py` |
| Two-dimensional | `twodimensionalintegral.hpp` | `integral2d.py` |
| Exponential integrals | `exponentialintegrals.hpp` | `exponential.py` |
| Gaussian orthogonal polynomials | `gaussianorthogonalpolynomial.hpp` | `orthogonal_poly.py` |

**Implementation notes:**
- Gauss quadrature: precompute nodes/weights, then `jnp.dot(weights, f(nodes))`
- All differentiable through `jax.grad` (Leibniz rule)
- `custom_jvp` for quadrature to avoid differentiating through node computation

**Tests to port:** `test-suite/integrals.cpp`, `test-suite/gaussianquadratures.cpp`

---

### Task 1.9: Distributions

**QuantLib source:** `ql/math/distributions/` (9 hpp + 6 cpp = 15 files)

**Target:** `ql_jax/math/distributions/`

**Distributions:**
| Distribution | Source | Target |
|--------------|--------|--------|
| Normal (CDF, PDF, inverse) | `normaldistribution.hpp` | `normal.py` |
| Bivariate normal | `bivariatenormaldistribution.hpp` | `bivariate_normal.py` |
| Bivariate Student-t | `bivariatestudenttdistribution.hpp` | `bivariate_student_t.py` |
| Chi-squared | `chisquaredistribution.hpp` | `chi_squared.py` |
| Gamma | `gammadistribution.hpp` | `gamma.py` |
| Poisson | `poissondistribution.hpp` | `poisson.py` |
| Binomial | `binomialdistribution.hpp` | `binomial.py` |
| Student-t | `studenttdistribution.hpp` | `student_t.py` |

**Implementation notes:**
- Map to `jax.scipy.stats` where possible (normal, chi2, gamma, t)
- Bivariate normal: implement Drezner-Wesolowsky or use Gauss quadrature
- Inverse CDF: `jax.scipy.stats.norm.ppf` or Beasley-Springer-Moro
- All differentiable

**Tests to port:** `test-suite/distributions.cpp`

---

### Task 1.10: Optimization

**QuantLib source:** `ql/math/optimization/` (23 hpp + 17 cpp = 40 files)

**Target:** `ql_jax/math/optimization/`

**Optimizers:**
| Optimizer | Source | Target |
|-----------|--------|--------|
| Levenberg-Marquardt | `levenbergmarquardt.hpp/.cpp` | `levenberg_marquardt.py` |
| Simplex (Nelder-Mead) | `simplex.hpp/.cpp` | `simplex.py` |
| BFGS | `bfgs.hpp/.cpp` | `bfgs.py` |
| Conjugate gradient | `conjugategradient.hpp/.cpp` | `conjugate_gradient.py` |
| Steepest descent | `steepestdescent.hpp/.cpp` | `steepest_descent.py` |
| Differential evolution | `differentialevolution.hpp/.cpp` | `differential_evolution.py` |
| Simulated annealing | `simulatedannealing.hpp` | `simulated_annealing.py` |
| EndCriteria | `endcriteria.hpp/.cpp` | `end_criteria.py` |
| Constraint | `constraint.hpp/.cpp` | `constraint.py` |
| CostFunction | `costfunction.hpp` | `cost_function.py` |
| Problem | `problem.hpp` | `problem.py` |
| Line search (Armijo, Goldstein) | `armijo.hpp/.cpp`, `goldstein.hpp/.cpp` | `line_search.py` |
| Projected constraint/cost | `projected*.hpp/.cpp` | `projected.py` |

**Implementation notes:**
- Gradient-based optimizers (BFGS, CG, steepest descent) can use `jax.grad` for automatic gradients
- Levenberg-Marquardt: Jacobian via `jax.jacfwd` or `jax.jacrev`
- Implement via `jax.lax.while_loop` for JIT compatibility
- `custom_vjp` to differentiate through optimization (implicit function theorem at optimum)

**Tests to port:** `test-suite/optimizers.cpp`

---

### Task 1.11: Random Number Generators

**QuantLib source:** `ql/math/randomnumbers/` (20 hpp + 13 cpp = 33 files)

**Target:** `ql_jax/math/random/`

**RNG implementations:**
| Generator | Source | Target | JAX Mapping |
|-----------|--------|--------|-------------|
| MT19937 | `mt19937uniformrng.hpp/.cpp` | `mt19937.py` | `jax.random.PRNGKey` (Threefry default) |
| Sobol | `sobolrsg.hpp/.cpp` | `sobol.py` | Custom or `jax.random` with Sobol key |
| Halton | `haltonrsg.cpp` | `halton.py` | Custom implementation |
| Faure | `faurersg.cpp` | `faure.py` | Custom implementation |
| Knuth | `knuthuniformrng.hpp/.cpp` | `knuth.py` | Map to `jax.random` |
| L'Ecuyer | `lecuyeruniformrng.hpp/.cpp` | `lecuyer.py` | Map to `jax.random` |
| Xoshiro256** | `xoshiro256starstaruniformrng.hpp/.cpp` | `xoshiro.py` | Map to `jax.random` |
| Box-Muller | `boxmullergaussianrng.hpp` | `box_muller.py` | `jax.random.normal` |
| Ziggurat | `zigguratgaussianrng.hpp` | `ziggurat.py` | `jax.random.normal` |
| Inverse cumulative | `inversecumulativerng.hpp` | `inverse_cumulative.py` | `jax.scipy.stats.norm.ppf` |
| Sobol-Brownian bridge | `sobolbrownianbridgersg.hpp/.cpp` | `sobol_brownian_bridge.py` | Custom |
| Stochastic collocation | `stochasticcollocationinvcdf.hpp/.cpp` | `stochastic_collocation.py` | Custom |

**Implementation notes:**
- JAX's `jax.random` replaces most PRNGs — use functional key-splitting API
- Sobol/Halton quasi-random sequences need custom implementation (state-free)
- All generators return `jnp.ndarray` for downstream differentiability (pathwise derivatives)

**Tests to port:** `test-suite/mersennetwister.cpp`, `test-suite/lowdiscrepancysequences.cpp`, `test-suite/rngtraits.cpp`, `test-suite/zigguratgaussian.cpp`, `test-suite/xoshiro256starstar.cpp`

---

### Task 1.12: Statistics

**QuantLib source:** `ql/math/statistics/` (10 hpp + 4 cpp = 14 files)

**Target:** `ql_jax/math/statistics/`

**Statistics classes:**
| Class | Source | Target |
|-------|--------|--------|
| GeneralStatistics | `generalstatistics.hpp/.cpp` | `general.py` |
| IncrementalStatistics | `incrementalstatistics.hpp/.cpp` | `incremental.py` |
| RiskStatistics | `riskstatistics.hpp` | `risk.py` |
| GaussianStatistics | `gaussianstatistics.hpp` | `gaussian.py` |
| SequenceStatistics | `sequencestatistics.hpp` | `sequence.py` |
| ConvergenceStatistics | `convergencestatistics.hpp` | `convergence.py` |
| DiscrepancyStatistics | `discrepancystatistics.hpp/.cpp` | `discrepancy.py` |
| Histogram | `histogram.hpp/.cpp` | `histogram.py` |

**Implementation notes:**
- Statistics as pure functions operating on arrays: `mean(samples)`, `variance(samples)`, `var(samples, level)`, `expected_shortfall(samples, level)`
- Use `jnp.mean`, `jnp.var`, `jnp.sort`, `jnp.percentile`

**Tests to port:** `test-suite/stats.cpp`, `test-suite/riskstats.cpp`

---

### Task 1.13: Patterns (Observable, LazyObject)

**QuantLib source:** `ql/patterns/` (6 hpp + 1 cpp = 7 files)

**Target:** `ql_jax/patterns/`

**Patterns:**
| Pattern | Source | Target | Translation |
|---------|--------|--------|-------------|
| Observable | `observable.hpp/.cpp` | `observable.py` | Explicit dependency passing (no mutation) |
| LazyObject | `lazyobject.hpp` | `lazy.py` | `jax.jit` caching + `functools.lru_cache` |
| Singleton | `singleton.hpp` | `singleton.py` | Module-level globals or `Settings` dataclass |
| Visitor | `visitor.hpp` | `visitor.py` | `functools.singledispatch` |
| CRTP | `curiouslyrecurring.hpp` | Not needed | Python duck typing / Protocol |

**Implementation notes:**
- The Observer pattern is fundamentally about mutable state notification — in JAX, we replace this with explicit parameter passing. When market data changes, the user creates new immutable state and re-calls pricing functions.
- LazyObject → `@jax.jit` provides compilation caching; `@functools.lru_cache` for Python-level memoization

**Tests to port:** `test-suite/observable.cpp`, `test-suite/lazyobject.cpp`

---

## Phase 2: Market Data — Term Structures, Quotes, Indexes

**Dependencies:** Phase 1 (time, interpolations, solvers)

### Task 2.1: Quotes

**QuantLib source:** `ql/quotes/` (11 hpp + 7 cpp = 18 files)

**Target:** `ql_jax/quotes/`

**Quote types:**
| Quote | Source | Target |
|-------|--------|--------|
| SimpleQuote | `simplequote.hpp` | `simple.py` |
| CompositeQuote | `compositequote.hpp` | `composite.py` |
| DerivedQuote | `derivedquote.hpp` | `derived.py` |
| DeltaVolQuote | `deltavolquote.hpp/.cpp` | `delta_vol.py` |
| ForwardValueQuote | `forwardvaluequote.hpp/.cpp` | `forward_value.py` |
| ForwardSwapQuote | `forwardswapquote.hpp/.cpp` | `forward_swap.py` |
| EurodollarFuturesQuote | `eurodollarfuturesquote.hpp/.cpp` | `eurodollar_futures.py` |
| ImpliedStdDevQuote | `impliedstddevquote.hpp/.cpp` | `implied_stddev.py` |
| FuturesConvAdjustmentQuote | `futuresconvadjustmentquote.hpp/.cpp` | `futures_conv.py` |
| LastFixingQuote | `lastfixingquote.hpp/.cpp` | `last_fixing.py` |

**Implementation notes:**
- In QuantLib, Quote is Observable + returns a value. In QL-JAX, quote is simply `jnp.ndarray` (scalar).
- CompositeQuote: `f(q1, q2)` → just a function composition
- Thin wrappers — most quotes become direct array values

**Tests to port:** `test-suite/quotes.cpp`

---

### Task 2.2: Currency Definitions

**QuantLib source:** `ql/currencies/` (8 hpp + 7 cpp = 15 files)

**Target:** `ql_jax/currencies/currency.py`

**Currency groups:** Africa, America, Asia, Europe, Oceania, Crypto

**Implementation notes:**
- Currency as frozen dataclass: `name`, `code`, `numeric_code`, `symbol`, `fractions_per_unit`, `rounding`
- ExchangeRateManager: dict-based lookup
- Non-differentiable metadata

**Tests to port:** `test-suite/currency.cpp`, `test-suite/exchangerate.cpp`, `test-suite/money.cpp`

---

### Task 2.3: Interest Rate Indexes — IBOR

**QuantLib source:** `ql/indexes/` root (9 hpp + 8 cpp) + `ql/indexes/ibor/` (43 hpp + 15 cpp) = 75 files

**Target:** `ql_jax/indexes/`

**Index base classes:**
- `InterestRateIndex` → `ql_jax/indexes/base.py`
- `IborIndex` → `ql_jax/indexes/ibor.py`
- `IndexManager` → `ql_jax/indexes/manager.py`

**IBOR implementations (all in `ql_jax/indexes/ibor_defs.py`):**
- USD: USDLibor, FedFunds, SOFR
- EUR: Euribor, EURLibor, Eonia, ESTR
- GBP: GBPLibor, SONIA
- JPY: JPYLibor, Tibor, TONA, TONAR
- CHF: CHFLibor, SARON
- Others: AONIA, BBSW, BKBM, CDI, CDOR, CORRA, Pribor, Robor, Wibor, Zibor, etc.
- SEK: SEKLibor, SWESTR
- DKK: DKKLibor
- Custom: user-defined index

**Implementation notes:**
- Index as frozen dataclass with `fixing_calendar`, `tenor`, `day_counter`, `currency`
- Fixing lookup: dict of `{date: value}`
- Rate forecast: requires term structure link → passed as function argument

**Tests to port:** `test-suite/indexes.cpp`, `test-suite/sofrfutures.cpp`

---

### Task 2.4: Swap Indexes

**QuantLib source:** `ql/indexes/swap/` (7 hpp + 6 cpp = 13 files)

**Target:** `ql_jax/indexes/swap.py`

**Swap indexes:** CHFLiborSwap, EuriborSwap, EURLiborSwap, GBPLiborSwap, JPYLiborSwap, USDLiborSwap

---

### Task 2.5: Yield Term Structures — Flat & Interpolated Curves

**QuantLib source:** `ql/termstructures/yieldtermstructure.hpp/.cpp`, `ql/termstructures/yield/` (28 hpp + 10 cpp = 38 files)

**Target:** `ql_jax/termstructures/yield_/`

**Implementation order:**

**2.5a: Base and simple curves**
| Curve | Source | Target |
|-------|--------|--------|
| YieldTermStructure base | `yieldtermstructure.hpp` | `base.py` |
| FlatForward | `flatforward.hpp/.cpp` | `flat_forward.py` |
| DiscountCurve (interpolated) | `discountcurve.hpp` | `discount_curve.py` |
| ZeroCurve (interpolated) | `zerocurve.hpp` | `zero_curve.py` |
| ForwardCurve (interpolated) | `forwardcurve.hpp` | `forward_curve.py` |
| ImpliedTermStructure | `impliedtermstructure.hpp` | `implied.py` |
| ForwardSpreadedTermStructure | `forwardspreadedtermstructure.hpp` | `forward_spreaded.py` |
| ZeroSpreadedTermStructure | `zerospreadedtermstructure.hpp` | `zero_spreaded.py` |

**Core functions:** `discount(t)`, `zero_rate(t)`, `forward_rate(t1, t2)` — all differentiable via JAX

**2.5b: Piecewise bootstrap**
| Component | Source | Target |
|-----------|--------|--------|
| PiecewiseYieldCurve | `piecewiseyieldcurve.hpp` | `piecewise.py` |
| BootstrapTraits | `bootstraptraits.hpp` | `bootstrap_traits.py` |
| IterativeBootstrap | `iterativebootstrap.hpp` | `iterative_bootstrap.py` |
| RateHelpers (Deposit, FRA, Swap, Future, Bond) | `ratehelpers.hpp/.cpp` | `rate_helpers.py` |
| BondHelpers | `bondhelpers.hpp/.cpp` | `bond_helpers.py` |
| OISRateHelper | `oisratehelper.hpp/.cpp` | `ois_rate_helper.py` |
| OvernightIndexFutureRateHelper | `overnightindexfutureratehelper.hpp/.cpp` | `ois_future_helper.py` |

**Implementation notes:**
- Bootstrap: iteratively solve for discount factors using 1D root solver (Task 1.7)
- AD through bootstrap: `jax.custom_vjp` using implicit differentiation — d(curve)/d(quotes) via implicit function theorem
- This is the most critical differentiable component in the project

**2.5c: Fitted bond curves**
| Component | Source | Target |
|-----------|--------|--------|
| FittedBondDiscountCurve | `fittedbonddiscountcurve.hpp/.cpp` | `fitted_bond_curve.py` |
| Fitting methods (Nelson-Siegel, Svensson, etc.) | `nonlinearfittingmethods.hpp/.cpp` | `fitting_methods.py` |

**2.5d: Remaining yield structures**
- CompositeZeroYieldStructure, QuantoTermStructure
- PiecewiseSpreadYieldCurve, PiecewiseZeroSpreadedTermStructure
- UltimateForwardTermStructure, MultipleResetsSwapHelper

**Tests to port:** `test-suite/piecewiseyieldcurve.cpp`, `test-suite/fittedbonddiscountcurve.cpp`, `test-suite/termstructures.cpp`, `test-suite/piecewisezerospreadedtermstructure.cpp`, `test-suite/ultimateforwardtermstructure.cpp`

**Example to port:** `MulticurveBootstrapping`

---

### Task 2.6: Volatility Term Structures

**QuantLib source:** `ql/termstructures/volatility/` (all subdirs, 141 files total)

**Target:** `ql_jax/termstructures/volatility/`

**2.6a: Smile sections (root-level)**
| Component | Source | Target |
|-----------|--------|--------|
| SmileSection base | `smilesection.hpp/.cpp` | `smile_section.py` |
| FlatSmileSection | `flatsmilesection.hpp/.cpp` | `flat_smile.py` |
| InterpolatedSmileSection | `interpolatedsmilesection.hpp` | `interpolated_smile.py` |
| SABRSmileSection | `sabrsmilesection.hpp/.cpp` | `sabr_smile.py` |
| KahaleSmileSection | `kahalesmilesection.hpp/.cpp` | `kahale_smile.py` |
| SpreadedSmileSection | `spreadedsmilesection.hpp/.cpp` | `spreaded_smile.py` |
| SABR functions | `sabr.hpp/.cpp` | `sabr_functions.py` |
| ZABR functions | `zabr.hpp/.cpp`, `zabrinterpolatedsmilesection.hpp` | `zabr_functions.py` |
| ABCD functions | `abcd.hpp/.cpp`, `abcdcalibration.hpp/.cpp` | `abcd_functions.py` |

**2.6b: Equity/FX vol structures** (`ql/termstructures/volatility/equityfx/`, 34 files)
| Component | Source | Target |
|-----------|--------|--------|
| BlackVolTermStructure base | `blackvoltermstructure.hpp/.cpp` | `black_vol_base.py` |
| BlackConstantVol | `blackconstantvol.hpp` | `black_constant_vol.py` |
| BlackVarianceCurve | `blackvariancecurve.hpp/.cpp` | `black_variance_curve.py` |
| BlackVarianceSurface | `blackvariancesurface.hpp/.cpp` | `black_variance_surface.py` |
| BlackVolSurfaceDelta | `blackvolsurfacedelta.hpp/.cpp` | `black_vol_delta.py` |
| LocalVolTermStructure | `localvoltermstructure.hpp/.cpp` | `local_vol_base.py` |
| LocalConstantVol | `localconstantvol.hpp` | `local_constant_vol.py` |
| LocalVolSurface | `localvolsurface.hpp/.cpp` | `local_vol_surface.py` |
| LocalVolCurve | `localvolcurve.hpp` | `local_vol_curve.py` |
| HestonBlackVolSurface | `hestonblackvolsurface.hpp/.cpp` | `heston_black_vol.py` |
| ImpliedVolTermStructure | `impliedvoltermstructure.hpp` | `implied_vol.py` |
| Andreasen-Huge adapter | `andreasen*.hpp/.cpp` | `andreasen_huge.py` |
| PiecewiseBlackVarianceSurface | `piecewiseblackvariancesurface.hpp/.cpp` | `piecewise_black_var.py` |

**2.6c: Swaption vol structures** (`ql/termstructures/volatility/swaption/`, 23 files)
| Component | Source | Target |
|-----------|--------|--------|
| SwaptionVolStructure base | `swaptionvolstructure.hpp/.cpp` | `swaption_vol_base.py` |
| SwaptionVolMatrix | `swaptionvolmatrix.hpp/.cpp` | `swaption_vol_matrix.py` |
| SwaptionConstantVol | `swaptionconstantvol.hpp/.cpp` | `swaption_constant_vol.py` |
| SwaptionVolCube | `swaptionvolcube.hpp/.cpp` | `swaption_vol_cube.py` |
| SABRSwaptionVolCube | `sabrswaptionvolatilitycube.hpp` | `sabr_swaption_cube.py` |
| CMSMarket | `cmsmarket.hpp/.cpp`, `cmsmarketcalibration.hpp/.cpp` | `cms_market.py` |
| SpreadedSwaptionVol | `spreadedswaptionvol.hpp/.cpp` | `spreaded_swaption_vol.py` |

**2.6d: Cap/Floor vol structures** (`ql/termstructures/volatility/capfloor/`, 9 files)
- CapFloorTermVolatilityStructure, CapFloorTermVolCurve, CapFloorTermVolSurface, ConstantCapFloorTermVol

**2.6e: Optionlet vol structures** (`ql/termstructures/volatility/optionlet/`, 19 files)
- OptionletVolatilityStructure, ConstantOptionletVol, StrippedOptionlet, OptionletStripper1/2, StrippedOptionletAdapter, SpreadedOptionletVol

**Tests to port:** `test-suite/swaptionvolatilitycube.cpp`, `test-suite/swaptionvolatilitymatrix.cpp`, `test-suite/interpolatedsmilesection.cpp`, `test-suite/blackvolsurfacedelta.cpp`, `test-suite/piecewiseblackvariancesurface.cpp`, `test-suite/optionletstripper.cpp`, `test-suite/svivolatility.cpp`, `test-suite/noarbsabr.cpp`, `test-suite/zabr.cpp`

---

### Task 2.7: Credit Term Structures

**QuantLib source:** `ql/termstructures/credit/` (11 hpp + 5 cpp = 16 files)

**Target:** `ql_jax/termstructures/credit/`

| Component | Source | Target |
|-----------|--------|--------|
| DefaultDensityStructure | `defaultdensitystructure.hpp/.cpp` | `default_density.py` |
| HazardRateStructure | `hazardratestructure.hpp/.cpp` | `hazard_rate.py` |
| SurvivalProbabilityStructure | `survivalprobabilitystructure.hpp/.cpp` | `survival_prob.py` |
| FlatHazardRate | `flathazardrate.hpp/.cpp` | `flat_hazard.py` |
| InterpolatedHazardRateCurve | `interpolatedhazardratecurve.hpp` | `interpolated_hazard.py` |
| InterpolatedDefaultDensityCurve | `interpolateddefaultdensitycurve.hpp` | `interpolated_default.py` |
| InterpolatedSurvivalProbabilityCurve | `interpolatedsurvivalprobabilitycurve.hpp` | `interpolated_survival.py` |
| PiecewiseDefaultCurve | `piecewisedefaultcurve.hpp` | `piecewise_default.py` |
| DefaultProbabilityHelpers | `defaultprobabilityhelpers.hpp/.cpp` | `default_helpers.py` |
| ProbabilityTraits | `probabilitytraits.hpp` | `probability_traits.py` |

**Tests to port:** `test-suite/defaultprobabilitycurves.cpp`

---

### Task 2.8: Inflation Term Structures

**QuantLib source:** `ql/termstructures/inflation/` (8 hpp + 2 cpp = 10 files), `ql/termstructures/inflationtermstructure.hpp/.cpp`

**Target:** `ql_jax/termstructures/inflation/`

| Component | Target |
|-----------|--------|
| InflationTermStructure base | `base.py` |
| InterpolatedZeroInflationCurve | `zero_inflation_curve.py` |
| InterpolatedYoYInflationCurve | `yoy_inflation_curve.py` |
| PiecewiseZeroInflationCurve | `piecewise_zero.py` |
| PiecewiseYoYInflationCurve | `piecewise_yoy.py` |
| InflationHelpers | `helpers.py` |
| Seasonality | `seasonality.py` |

**Tests to port:** `test-suite/inflation.cpp`, `test-suite/inflationvolatility.cpp`

---

## Phase 3: Cash Flows & Fixed Income Instruments

**Dependencies:** Phase 1 (time, day counters), Phase 2 (yield curves, indexes)

### Task 3.1: Simple & Fixed Rate Cash Flows

**QuantLib source (subset of `ql/cashflows/`, 71 files total):**
- `simplecashflow.hpp/.cpp`, `coupon.hpp/.cpp`, `fixedratecoupon.hpp/.cpp`
- `cashflows.hpp/.cpp` (static analytics: NPV, BPS, yield, duration, convexity)
- `cashflowvectors.hpp/.cpp`
- `duration.hpp/.cpp`

**Target:** `ql_jax/cashflows/simple.py`, `ql_jax/cashflows/fixed_rate.py`, `ql_jax/cashflows/analytics.py`

**Implementation notes:**
- Cashflow as frozen dataclass: `{date, amount}` or `{date, notional, rate, accrual_start, accrual_end, day_counter}`
- NPV = `jnp.sum(amounts * discount_factors)` — trivially differentiable
- Duration, convexity: AD via `jax.grad` of NPV w.r.t. yield

---

### Task 3.2: Floating Rate Cash Flows

**QuantLib source:**
- `floatingratecoupon.hpp/.cpp`, `iborcoupon.hpp/.cpp`
- `overnightindexedcoupon.hpp/.cpp`, `overnightindexedcouponpricer.hpp/.cpp`, `blackovernightindexedcouponpricer.hpp/.cpp`
- `cmscoupon.hpp/.cpp`, `conundrumpricer.hpp/.cpp`, `lineartsrpricer.hpp/.cpp`
- `multipleresetscoupon.hpp/.cpp`
- `couponpricer.hpp/.cpp`
- `rateaveraging.hpp`

**Target:** `ql_jax/cashflows/floating_rate.py`, `ql_jax/cashflows/ibor.py`, `ql_jax/cashflows/overnight.py`, `ql_jax/cashflows/cms.py`, `ql_jax/cashflows/pricer.py`

**Implementation notes:**
- IborCoupon: rate = forward from yield curve → differentiable
- OvernightIndexedCoupon: compounded/averaged overnight rates
- CMSCoupon: convexity adjustment via replication or TSR pricer

---

### Task 3.3: Capped/Floored & Digital Coupons

**QuantLib source:**
- `capflooredcoupon.hpp/.cpp`, `digitalcoupon.hpp/.cpp`
- `digitaliborcoupon.hpp/.cpp`, `digitalcmscoupon.hpp/.cpp`
- `rangeaccrual.hpp/.cpp`, `replication.hpp/.cpp`
- `stickyratchet.hpp`

**Target:** `ql_jax/cashflows/capped_floored.py`, `ql_jax/cashflows/digital.py`, `ql_jax/cashflows/range_accrual.py`

---

### Task 3.4: Inflation Cash Flows

**QuantLib source:**
- `inflationcoupon.hpp/.cpp`, `inflationcouponpricer.hpp/.cpp`
- `cpicoupon.hpp/.cpp`, `cpicouponpricer.hpp/.cpp`
- `yoyinflationcoupon.hpp/.cpp`
- `zeroinflationcashflow.hpp/.cpp`
- `capflooredinflationcoupon.hpp/.cpp`
- `indexedcashflow.hpp/.cpp`

**Target:** `ql_jax/cashflows/inflation.py`, `ql_jax/cashflows/cpi.py`, `ql_jax/cashflows/yoy.py`

---

### Task 3.5: Other Cash Flows

**QuantLib source:**
- `averagebmacoupon.hpp/.cpp`
- `dividend.hpp/.cpp`
- `equitycashflow.hpp/.cpp`
- `timebasket.hpp/.cpp`

**Target:** `ql_jax/cashflows/bma.py`, `ql_jax/cashflows/dividend.py`, `ql_jax/cashflows/equity.py`

**Tests to port:** `test-suite/cashflows.cpp`, `test-suite/capflooredcoupon.cpp`, `test-suite/digitalcoupon.cpp`, `test-suite/overnightindexedcoupon.cpp`, `test-suite/multipleresetscoupons.cpp`, `test-suite/rangeaccrual.cpp`, `test-suite/cms.cpp`, `test-suite/cms_normal.cpp`, `test-suite/cmsspread.cpp`, `test-suite/equitycashflow.cpp`

---

### Task 3.6: Bond Instruments

**QuantLib source:** `ql/instruments/bond.hpp/.cpp`, `ql/instruments/bonds/` (11 hpp + 10 cpp = 21 files)

**Target:** `ql_jax/instruments/bond.py`, `ql_jax/instruments/bonds/`

**Bond types:**
| Bond | Source | Target |
|------|--------|--------|
| Bond base | `bond.hpp/.cpp` | `bond.py` |
| FixedRateBond | `fixedratebond.hpp/.cpp` | `bonds/fixed_rate.py` |
| FloatingRateBond | `floatingratebond.hpp/.cpp` | `bonds/floating_rate.py` |
| ZeroCouponBond | `zerocouponbond.hpp/.cpp` | `bonds/zero_coupon.py` |
| AmortizingFixedRateBond | `amortizingfixedratebond.hpp/.cpp` | `bonds/amortizing_fixed.py` |
| AmortizingFloatingRateBond | `amortizingfloatingratebond.hpp/.cpp` | `bonds/amortizing_floating.py` |
| CMSRateBond | `cmsratebond.hpp/.cpp` | `bonds/cms_rate.py` |
| CPIBond | `cpibond.hpp/.cpp` | `bonds/cpi.py` |
| ConvertibleBonds | `convertiblebonds.hpp/.cpp` | `bonds/convertible.py` |
| BTP | `btp.hpp/.cpp` | `bonds/btp.py` |

**Tests to port:** `test-suite/bonds.cpp`, `test-suite/amortizingbond.cpp`, `test-suite/inflationcpibond.cpp`

**Example to port:** `Bonds`

---

### Task 3.7: Swap Instruments

**QuantLib source:**
- `swap.hpp/.cpp`, `vanillaswap.hpp/.cpp`, `fixedvsfloatingswap.hpp/.cpp`
- `overnightindexedswap.hpp/.cpp`, `bmaswap.hpp/.cpp`
- `multipleresetsswap.hpp/.cpp`, `nonstandardswap.hpp/.cpp`
- `zerocouponswap.hpp/.cpp`, `floatfloatswap.hpp/.cpp`
- `cpiswap.hpp/.cpp`, `yearonyearinflationswap.hpp/.cpp`, `zerocouponinflationswap.hpp/.cpp`
- `equitytotalreturnswap.hpp/.cpp`
- Makers: `makevanillaswap.hpp/.cpp`, `makeois.hpp/.cpp`, `makemultipleresetsswap.hpp/.cpp`

**Target:** `ql_jax/instruments/swap.py`, `ql_jax/instruments/ois.py`, `ql_jax/instruments/inflation_swap.py`

**Tests to port:** `test-suite/swap.cpp`, `test-suite/overnightindexedswap.cpp`, `test-suite/multipleresetsswap.cpp`, `test-suite/zerocouponswap.cpp`, `test-suite/inflationcpiswap.cpp`

---

### Task 3.8: Forward Rate Agreement

**QuantLib source:** `forwardrateagreement.hpp/.cpp`

**Target:** `ql_jax/instruments/fra.py`

**Tests to port:** `test-suite/forwardrateagreement.cpp`

**Example to port:** `FRA`, `Repo`

---

### Task 3.9: Bond & Swap Pricing Engines

**QuantLib source:**
- `ql/pricingengines/bond/` (6 hpp + 4 cpp = 10 files)
- `ql/pricingengines/swap/` (5 hpp + 4 cpp = 9 files)

**Target:** `ql_jax/engines/bond/`, `ql_jax/engines/swap/`

**Bond engines:**
| Engine | Source | Target |
|--------|--------|--------|
| DiscountingBondEngine | `discountingbondengine.hpp/.cpp` | `discounting.py` |
| BondFunctions | `bondfunctions.hpp/.cpp` | `bond_functions.py` |
| RiskyBondEngine | `riskybondengine.hpp/.cpp` | `risky.py` |

**Swap engines:**
| Engine | Source | Target |
|--------|--------|--------|
| DiscountingSwapEngine | `discountingswapengine.hpp/.cpp` | `discounting.py` |
| CVASwapEngine | `cvaswapengine.hpp/.cpp` | `cva.py` |
| TreeSwapEngine | `treeswapengine.hpp/.cpp` | `tree.py` |
| DiscretizedSwap | `discretizedswap.hpp/.cpp` | `discretized.py` |

**Implementation notes:**
- DiscountingBondEngine: NPV = Σ cashflow_amount × discount(cashflow_date)
- Trivially differentiable: `jax.grad(npv, argnums=curve_params)` gives DV01, key rate durations

---

## Phase 4: Equity Instruments & Core Pricing Engines

**Dependencies:** Phase 1 (math, interpolations, solvers), Phase 2 (vol surfaces)

### Task 4.1: Payoffs & Exercise Types

**QuantLib source:** `ql/instruments/payoffs.hpp/.cpp`, `ql/instruments/oneassetoption.hpp/.cpp`, `ql/instruments/europeanoption.hpp/.cpp`

**Target:** `ql_jax/instruments/payoff.py`, `ql_jax/instruments/exercise.py`

**Payoff types:** PlainVanillaPayoff, CashOrNothingPayoff, AssetOrNothingPayoff, GapPayoff, SuperSharePayoff, SuperFundPayoff
**Exercise types:** EuropeanExercise, AmericanExercise, BermudanExercise

**Implementation notes:**
- Payoff as pure function: `payoff(spot, strike, option_type) -> jnp.ndarray`
- Use `jnp.where` for call/put logic (JIT-compatible, differentiable)

---

### Task 4.2: Vanilla Option & Core Black-Scholes Engine

**QuantLib source:**
- `ql/instruments/vanillaoption.hpp/.cpp`
- `ql/processes/blackscholesprocess.hpp/.cpp`
- `ql/pricingengines/vanilla/analyticeuropeanengine.hpp/.cpp`
- `ql/pricingengines/blackcalculator.hpp/.cpp`
- `ql/pricingengines/blackformula.hpp/.cpp`
- `ql/pricingengines/blackscholescalculator.hpp/.cpp`

**Target:**
- `ql_jax/instruments/vanilla_option.py`
- `ql_jax/processes/black_scholes.py`
- `ql_jax/engines/analytic/european.py`
- `ql_jax/engines/analytic/black_calculator.py`
- `ql_jax/engines/analytic/black_formula.py`

**Implementation notes:**
- BlackScholesProcess as frozen dataclass: `{spot, rate_curve, dividend_curve, vol_surface}`
- `black_scholes_price(S, K, T, r, q, sigma, option_type) -> price`
- Greeks via `jax.grad`: delta = ∂V/∂S, gamma = ∂²V/∂S², vega = ∂V/∂σ, theta = ∂V/∂T, rho = ∂V/∂r
- This is the first end-to-end differentiable pricing function — template for all others

**Tests to port:** `test-suite/europeanoption.cpp`

**Example to port:** `EquityOption` (European portion)

---

### Task 4.3: Additional Analytic Engines

**QuantLib source:** `ql/pricingengines/vanilla/` (subset)

**Target:** `ql_jax/engines/analytic/`

| Engine | Source | Target |
|--------|--------|--------|
| Heston (semi-analytic) | `analytichestonengine.hpp/.cpp` | `heston.py` |
| Piecewise time-dep Heston | `analyticptdhestonengine.hpp/.cpp` | `heston_ptd.py` |
| Heston expansion | `hestonexpansionengine.hpp/.cpp` | `heston_expansion.py` |
| Heston PDF | `analyticpdfhestonengine.hpp/.cpp` | `heston_pdf.py` |
| COS Heston | `coshestonengine.hpp/.cpp` | `heston_cos.py` |
| Exponential fitting Heston | `exponentialfittinghestonengine.hpp/.cpp` | `heston_exp_fit.py` |
| Bates | `batesengine.hpp/.cpp` | `bates.py` |
| GJRGARCH | `analyticgjrgarchengine.hpp/.cpp` | `gjrgarch.py` |
| CEV | `analyticcevengine.hpp/.cpp` | `cev.py` |
| Jump diffusion | `jumpdiffusionengine.hpp/.cpp` | `jump_diffusion.py` |
| BSM-HW hybrid | `analyticbsmhullwhiteengine.hpp/.cpp` | `bsm_hull_white.py` |
| Heston-HW hybrid | `analytichestonhullwhiteengine.hpp/.cpp`, `analytich1hwengine.hpp/.cpp` | `heston_hull_white.py` |
| BSM with Vasicek | `analyticeuropeanvasicekengine.hpp/.cpp` | `european_vasicek.py` |
| Dividend European | `analyticdividendeuropeanengine.hpp/.cpp`, `cashdividendeuropeanengine.hpp/.cpp` | `dividend_european.py` |
| Digital American | `analyticdigitalamericanengine.hpp/.cpp` | `digital_american.py` |
| Barone-Adesi-Whaley | `baroneadesiwhaleyengine.hpp/.cpp` | `barone_adesi_whaley.py` |
| Bjerksund-Stensland | `bjerksundstenslandengine.hpp/.cpp` | `bjerksund_stensland.py` |
| Ju quadratic | `juquadraticengine.hpp/.cpp` | `ju_quadratic.py` |
| QD+ American | `qdplusamericanengine.hpp/.cpp`, `qdfpamericanengine.hpp/.cpp` | `qdplus_american.py` |
| Integral | `integralengine.hpp/.cpp` | `integral.py` |

**Tests to port:** `test-suite/americanoption.cpp`, `test-suite/hestonmodel.cpp`, `test-suite/batesmodel.cpp`, `test-suite/gjrgarchmodel.cpp`, `test-suite/jumpdiffusion.cpp`, `test-suite/dividendoption.cpp`, `test-suite/digitaloption.cpp`

---

### Task 4.4: Stochastic Processes

**QuantLib source:** `ql/processes/` (22 hpp + 21 cpp = 43 files)

**Target:** `ql_jax/processes/`

| Process | Source | Target |
|---------|--------|--------|
| StochasticProcess base | `stochasticprocess.hpp` (in ql/ root) | `base.py` |
| BlackScholesProcess | `blackscholesprocess.hpp/.cpp` | `black_scholes.py` |
| HestonProcess | `hestonprocess.hpp/.cpp` | `heston.py` |
| BatesProcess | `batesprocess.hpp/.cpp` | `bates.py` |
| Merton76Process | `merton76process.hpp/.cpp` | `merton76.py` |
| GJRGARCHProcess | `gjrgarchprocess.hpp/.cpp` | `gjrgarch.py` |
| HullWhiteProcess | `hullwhiteprocess.hpp/.cpp` | `hull_white.py` |
| G2Process | `g2process.hpp/.cpp` | `g2.py` |
| CoxIngersollRossProcess | `coxingersollrossprocess.hpp/.cpp` | `cir.py` |
| GSRProcess | `gsrprocess.hpp/.cpp`, `gsrprocesscore.hpp/.cpp` | `gsr.py` |
| OrnsteinUhlenbeckProcess | `ornsteinuhlenbeckprocess.hpp/.cpp` | `ornstein_uhlenbeck.py` |
| GeometricBrownianProcess | `geometricbrownianprocess.hpp/.cpp` | `gbm.py` |
| SquareRootProcess | `squarerootprocess.hpp/.cpp` | `square_root.py` |
| HestonSLVProcess | `hestonslvprocess.hpp/.cpp` | `heston_slv.py` |
| HybridHestonHWProcess | `hybridhestonhullwhiteprocess.hpp/.cpp` | `hybrid_heston_hw.py` |
| StochasticProcessArray | `stochasticprocessarray.hpp/.cpp` | `process_array.py` |
| JointStochasticProcess | `jointstochasticprocess.hpp/.cpp` | `joint.py` |
| EulerDiscretization | `eulerdiscretization.hpp/.cpp` | `euler.py` |
| EndEulerDiscretization | `endeulerdiscretization.hpp/.cpp` | `end_euler.py` |
| MfStateProcess | `mfstateprocess.hpp/.cpp` | `mf_state.py` |
| ForwardMeasureProcess | `forwardmeasureprocess.hpp/.cpp` | `forward_measure.py` |

**Implementation notes:**
- Process: `drift(t, x)`, `diffusion(t, x)`, `evolve(t0, x0, dt, dw)` — all pure functions
- All differentiable via `jax.grad` (pathwise sensitivities)
- Use `jax.vmap` over paths for Monte Carlo

**Tests to port:** `test-suite/hybridhestonhullwhiteprocess.cpp`

---

### Task 4.5: Binomial/Lattice Engines

**QuantLib source:**
- `ql/methods/lattices/` (9 hpp + 2 cpp = 11 files)
- `ql/pricingengines/vanilla/binomialengine.hpp`

**Target:** `ql_jax/methods/lattices/`, `ql_jax/engines/lattice/`

**Tree types:**
| Tree | Source | Target |
|------|--------|--------|
| BinomialTree base | `binomialtree.hpp/.cpp` | `binomial_tree.py` |
| CRR (Cox-Ross-Rubinstein) | `binomialtree.hpp` | `binomial_tree.py` |
| JR (Jarrow-Rudd) | `binomialtree.hpp` | `binomial_tree.py` |
| Tian | `binomialtree.hpp` | `binomial_tree.py` |
| Trigeorgis | `binomialtree.hpp` | `binomial_tree.py` |
| Leisen-Reimer | `binomialtree.hpp` | `binomial_tree.py` |
| Joshi4 | `binomialtree.hpp` | `binomial_tree.py` |
| TrinomialTree | `trinomialtree.hpp/.cpp` | `trinomial_tree.py` |
| BSMLattice | `bsmlattice.hpp` | `bsm_lattice.py` |
| Lattice1D, Lattice2D | `lattice1d.hpp`, `lattice2d.hpp` | `lattice.py` |

**Implementation notes:**
- Tree: backward induction via `jax.lax.fori_loop` (JIT-compatible)
- American exercise: `jnp.maximum(continuation_value, exercise_value)` at each node
- Differentiable via AD (backward pass through the tree)
- `vmap` over strike/spot for batched tree pricing

**Tests to port:** `test-suite/extendedtrees.cpp`

---

### Task 4.6: Finite Difference Engines

**QuantLib source:**
- `ql/methods/finitedifferences/` (209 files across operators, meshers, schemes, solvers, stepconditions, utilities)
- `ql/pricingengines/vanilla/fd*.hpp/.cpp` (12 files)

**Target:** `ql_jax/methods/finitedifferences/`, `ql_jax/engines/fd/`

**Implementation groups:**

**4.6a: Core FDM infrastructure**
| Component | Source Dir | Target |
|-----------|-----------|--------|
| LinearOp base | `operators/fdmlinearop.hpp` | `operators/base.py` |
| TripleBandLinearOp | `operators/triplebandlinearop.hpp/.cpp` | `operators/triple_band.py` |
| FirstDerivativeOp | `operators/firstderivativeop.hpp/.cpp` | `operators/first_derivative.py` |
| SecondDerivativeOp | `operators/secondderivativeop.hpp/.cpp` | `operators/second_derivative.py` |
| MixedDerivativeOp | `operators/secondordermixedderivativeop.hpp/.cpp` | `operators/mixed_derivative.py` |
| NinePointLinearOp | `operators/ninepointlinearop.hpp/.cpp` | `operators/nine_point.py` |
| NumericalDifferentiation | `operators/numericaldifferentiation.hpp/.cpp` | `operators/numerical_diff.py` |
| FdmLinearOpLayout | `operators/fdmlinearoplayout.hpp/.cpp` | `operators/layout.py` |
| TridiagonalOperator | `tridiagonaloperator.hpp/.cpp` | `operators/tridiagonal.py` |

**4.6b: Process-specific operators**
- FdmBlackScholesOp, FdmHestonOp, FdmBatesOp, FdmSABROp, FdmCEVOp, FdmCIROp, FdmHullWhiteOp, FdmG2Op, etc.

**4.6c: Meshers**
- Concentrating1dMesher, FdmBlackScholesMesher, FdmHestonVarianceMesher, Uniform1dMesher, etc.

**4.6d: Time-stepping schemes**
- ExplicitEuler, ImplicitEuler, CrankNicolson, Douglas, Hundsdorfer, Craig-Sneyd, ModifiedCraig-Sneyd, TRBDF2, MethodOfLines

**4.6e: FD solvers**
- Fdm1DimSolver, Fdm2DimSolver, Fdm3DimSolver, FdmNDimSolver, FdmBackwardSolver
- Process-specific: FdmBlackScholesSolver, FdmHestonSolver, FdmBatesSolver, etc.

**4.6f: Step conditions (American, Bermudan, arithmetic average, shout)**

**4.6g: Utilities (boundary conditions, Dirichlet, dividend handling, inner value calculators)**

**4.6h: Vanilla FD engines**
| Engine | Source | Target |
|--------|--------|--------|
| FdBlackScholesVanillaEngine | `fdblackscholesvanillaengine.hpp/.cpp` | `fd/black_scholes.py` |
| FdHestonVanillaEngine | `fdhestonvanillaengine.hpp/.cpp` | `fd/heston.py` |
| FdBatesVanillaEngine | `fdbatesvanillaengine.hpp/.cpp` | `fd/bates.py` |
| FdSabrVanillaEngine | `fdsabrvanillaengine.hpp/.cpp` | `fd/sabr.py` |
| FdCevVanillaEngine | `fdcevvanillaengine.hpp/.cpp` | `fd/cev.py` |
| FdCirVanillaEngine | `fdcirvanillaengine.hpp/.cpp` | `fd/cir.py` |
| FdHestonHullWhiteVanillaEngine | `fdhestonhullwhitevanillaengine.hpp/.cpp` | `fd/heston_hull_white.py` |
| FdBlackScholesShoutEngine | `fdblackscholesshoutengine.hpp/.cpp` | `fd/shout.py` |
| FdSimpleBSSwingEngine | `fdsimplebsswingengine.hpp/.cpp` | `fd/swing.py` |

**Implementation notes:**
- PDE discretization on JAX arrays: operators as sparse-matrix-vector products
- Tridiagonal solve: `jax.scipy.linalg.solve_banded` or custom Thomas algorithm
- Time stepping: `jax.lax.fori_loop` over time steps
- AD: adjoint method via `jax.custom_vjp` for efficient PDE sensitivities
- 2D/3D PDE: ADI (alternating direction implicit) splitting

**Tests to port:** `test-suite/fdmlinearop.cpp`, `test-suite/fdheston.cpp`, `test-suite/fdcev.cpp`, `test-suite/fdcir.cpp`, `test-suite/fdsabr.cpp`, `test-suite/nthorderderivativeop.cpp`, `test-suite/numericaldifferentiation.cpp`, `test-suite/operators.cpp`

---

### Task 4.7: Exotic Option Instruments

**QuantLib source (from `ql/instruments/`):**
- `barrieroption.hpp/.cpp`, `doublebarrieroption.hpp/.cpp`
- `asianoption.hpp/.cpp`
- `lookbackoption.hpp/.cpp`
- `cliquetoption.hpp/.cpp`
- `basketoption.hpp/.cpp`, `multiassetoption.hpp/.cpp`
- `quantovanillaoption.hpp/.cpp`, `quantobarrieroption.hpp/.cpp`, `quantoforwardvanillaoption.hpp/.cpp`
- `compoundoption.hpp/.cpp`, `simplechooseroption.hpp/.cpp`, `complexchooseroption.hpp/.cpp`
- `holderextensibleoption.hpp/.cpp`, `writerextensibleoption.hpp/.cpp`
- `margrabeoption.hpp/.cpp`, `twoassetbarrieroption.hpp/.cpp`, `twoassetcorrelationoption.hpp/.cpp`
- `partialtimebarrieroption.hpp/.cpp`, `softbarrieroption.hpp/.cpp`
- `forwardvanillaoption.hpp/.cpp`, `forward.hpp/.cpp`
- `varianceswap.hpp/.cpp`

**Target:** `ql_jax/instruments/barrier.py`, `ql_jax/instruments/asian.py`, `ql_jax/instruments/lookback.py`, `ql_jax/instruments/cliquet.py`, `ql_jax/instruments/basket.py`, `ql_jax/instruments/quanto.py`, `ql_jax/instruments/compound.py`, `ql_jax/instruments/chooser.py`, `ql_jax/instruments/extensible.py`, `ql_jax/instruments/margrabe.py`, `ql_jax/instruments/two_asset.py`, `ql_jax/instruments/forward_option.py`, `ql_jax/instruments/variance_swap.py`

---

### Task 4.8: Exotic Pricing Engines

**QuantLib source:** `ql/pricingengines/barrier/` (30 files), `ql/pricingengines/asian/` (26 files), `ql/pricingengines/basket/` (27 files), `ql/pricingengines/lookback/` (11 files), `ql/pricingengines/cliquet/` (7 files), `ql/pricingengines/exotic/` (17 files), `ql/pricingengines/forward/` (12 files), `ql/pricingengines/quanto/` (2 files)

**Target:** `ql_jax/engines/barrier/`, `ql_jax/engines/asian/`, `ql_jax/engines/basket/`, `ql_jax/engines/lookback/`, `ql_jax/engines/cliquet/`, `ql_jax/engines/exotic/`, `ql_jax/engines/forward/`, `ql_jax/engines/quanto/`

**Key engines (analytic):**
- Barrier: AnalyticBarrierEngine, AnalyticBinaryBarrierEngine, AnalyticDoubleBarrierEngine, AnalyticSoftBarrier
- Asian: AnalyticContGeomAvPrice, AnalyticDiscrGeomAvPrice, ChoiAsianEngine, TurnbullWakemanEngine, ContinuousArithmeticAsianLevyEngine
- Basket: KirkEngine, StulzEngine, BjerksundStenslandSpreadEngine, SingleFactorBSMBasket, ChoiBasketEngine, DengLiZhouBasketEngine
- Lookback: AnalyticContinuousFixed/FloatingLookback, AnalyticContinuousPartialFixed/FloatingLookback
- Cliquet: AnalyticCliquetEngine, AnalyticPerformanceEngine
- Exotic: AnalyticMargrabe, AnalyticCompound, AnalyticSimple/ComplexChooser, AnalyticHolderExtensible, AnalyticWriterExtensible, AnalyticTwoAssetCorrelation, AnalyticTwoAssetBarrier

**Key engines (FD):**
- FdBlackScholesBarrierEngine, FdHestonBarrierEngine, FdHestonDoubleBarrierEngine
- FdBlackScholesAsianEngine
- Fd2dBlackScholesVanillaEngine, FdNdimBlackScholesVanillaEngine

**Key engines (MC):**
- MCBarrierEngine, MCLookbackEngine, MCPerformanceEngine
- MCAmericanBasketEngine, MCEuropeanBasketEngine

**Tests to port:** `test-suite/barrieroption.cpp`, `test-suite/doublebarrieroption.cpp`, `test-suite/asianoptions.cpp`, `test-suite/lookbackoptions.cpp`, `test-suite/cliquetoption.cpp`, `test-suite/basketoption.cpp`, `test-suite/quantooption.cpp`, `test-suite/compoundoption.cpp`, `test-suite/chooseroption.cpp`, `test-suite/extensibleoptions.cpp`, `test-suite/margrabeoption.cpp`, `test-suite/twoassetbarrieroption.cpp`, `test-suite/twoassetcorrelationoption.cpp`, `test-suite/partialtimebarrieroption.cpp`, `test-suite/softbarrieroption.cpp`, `test-suite/forwardoption.cpp`, `test-suite/varianceswaps.cpp`, `test-suite/binaryoption.cpp`, `test-suite/doublebinaryoption.cpp`

**Example to port:** `EquityOption` (full version with all engine types), `AsianOption`

---

## Phase 5: Monte Carlo Framework

**Dependencies:** Phase 1 (RNG, statistics), Phase 4 (processes, payoffs)

### Task 5.1: Path Generation

**QuantLib source:** `ql/methods/montecarlo/` (17 hpp + 4 cpp = 21 files)

**Target:** `ql_jax/methods/montecarlo/`

| Component | Source | Target |
|-----------|--------|--------|
| Path | `path.hpp` | `path.py` |
| MultiPath | `multipath.hpp` | `multi_path.py` |
| PathGenerator | `pathgenerator.hpp` | `path_generator.py` |
| MultiPathGenerator | `multipathgenerator.hpp` | `multi_path_generator.py` |
| BrownianBridge | `brownianbridge.hpp/.cpp` | `brownian_bridge.py` |
| Sample | `sample.hpp` | `sample.py` |
| MCTraits | `mctraits.hpp` | `mc_traits.py` |
| MonteCarloModel | `montecarlomodel.hpp` | `mc_model.py` |

**Implementation notes:**
- Path generation: `evolve(process, t_grid, rng_key) -> paths[n_paths, n_steps]`
- Use `jax.random.normal(key, shape=(n_paths, n_steps))` for Gaussian increments
- Use `jax.vmap` over paths for vectorized simulation
- BrownianBridge: construct path by filling in midpoints (used with Sobol sequences)
- All paths are JAX arrays — differentiable via pathwise method

---

### Task 5.2: MC Pricing Engines — Vanilla

**QuantLib source:** `ql/pricingengines/vanilla/mc*.hpp/.cpp`

**Target:** `ql_jax/engines/mc/`

| Engine | Source | Target |
|--------|--------|--------|
| MCEuropeanEngine | `mceuropeanengine.hpp` | `european.py` |
| MCAmericanEngine | `mcamericanengine.hpp/.cpp` | `american.py` |
| MCDigitalEngine | `mcdigitalengine.hpp/.cpp` | `digital.py` |
| MCVanillaEngine base | `mcvanillaengine.hpp` | `vanilla_base.py` |
| MCEuropeanHestonEngine | `mceuropeanhestonengine.hpp` | `european_heston.py` |
| MCEuropeanGJRGARCHEngine | `mceuropeangjrgarchengine.hpp` | `european_gjrgarch.py` |
| MCHestonHullWhiteEngine | `mchestonhullwhiteengine.hpp/.cpp` | `heston_hull_white.py` |

---

### Task 5.3: MC Pricing Engines — Exotic

**Target:** `ql_jax/engines/mc/`

| Engine | Source | Target |
|--------|--------|--------|
| MCBarrierEngine | `barrier/mcbarrierengine.hpp/.cpp` | `barrier.py` |
| MCLookbackEngine | `lookback/mclookbackengine.hpp/.cpp` | `lookback.py` |
| MCPerformanceEngine | `cliquet/mcperformanceengine.hpp/.cpp` | `performance.py` |
| MCAmericanBasketEngine | `basket/mcamericanbasketengine.hpp/.cpp` | `american_basket.py` |
| MCEuropeanBasketEngine | `basket/mceuropeanbasketengine.hpp/.cpp` | `european_basket.py` |
| MCForwardEuropeanBSEngine | `forward/mcforwardeuropeanbsengine.hpp/.cpp` | `forward_european.py` |
| MCForwardEuropeanHestonEngine | `forward/mcforwardeuropeanhestonengine.hpp/.cpp` | `forward_heston.py` |
| MCVarianceSwapEngine | `forward/mcvarianceswapengine.hpp` | `variance_swap.py` |
| MC Asian engines | `asian/mc_discr_arith_av_*.hpp/.cpp`, etc. | `asian.py` |

---

### Task 5.4: Longstaff-Schwartz (American MC)

**QuantLib source:**
- `ql/methods/montecarlo/longstaffschwartzpathpricer.hpp`
- `ql/methods/montecarlo/genericlsregression.hpp/.cpp`
- `ql/methods/montecarlo/lsmbasissystem.hpp/.cpp`
- `ql/methods/montecarlo/earlyexercisepathpricer.hpp`
- `ql/pricingengines/mclongstaffschwartzengine.hpp`

**Target:** `ql_jax/methods/montecarlo/longstaff_schwartz.py`, `ql_jax/methods/montecarlo/lsm_basis.py`

**Implementation notes:**
- Regression: `jnp.linalg.lstsq` for polynomial regression at each exercise date
- Backward induction through exercise dates via `jax.lax.fori_loop`
- Differentiable via pathwise AD + `custom_vjp` for the exercise boundary
- Basis functions: monomials, Laguerre, Hermite polynomials

**Tests to port:** `test-suite/mclongstaffschwartzengine.cpp`

---

### Task 5.5: Variance Reduction

**Implementation notes (no separate QuantLib files — embedded in engine logic):**
- Antithetic variates: generate `(W, -W)` pairs → average
- Control variates: use analytic geometric average as control for arithmetic average Asian
- Importance sampling: shift drift for rare event pricing
- All implemented as JAX array operations — transparent to AD

**Tests to port:** Part of MC engine tests (convergence rate checks)

**Examples to port:** `DiscreteHedging`, `AsianOption`

---

## Phase 6: Interest Rate Models & Derivatives

**Dependencies:** Phase 2 (yield curves, vol), Phase 3 (swaps, bonds), Phase 4 (trees, FD)

### Task 6.1: Short-Rate Models — One Factor

**QuantLib source:** `ql/models/shortrate/onefactormodels/` (9 hpp + 8 cpp = 17 files), `ql/models/shortrate/onefactormodel.hpp/.cpp`

**Target:** `ql_jax/models/shortrate/`

| Model | Source | Target |
|-------|--------|--------|
| OneFactor base | `onefactormodel.hpp/.cpp` | `one_factor.py` |
| Vasicek | `vasicek.hpp/.cpp` | `vasicek.py` |
| CoxIngersollRoss | `coxingersollross.hpp/.cpp` | `cir.py` |
| ExtendedCIR | `extendedcoxingersollross.hpp/.cpp` | `extended_cir.py` |
| HullWhite | `hullwhite.hpp/.cpp` | `hull_white.py` |
| BlackKarasinski | `blackkarasinski.hpp/.cpp` | `black_karasinski.py` |
| GSR (Gaussian Short Rate) | `gsr.hpp/.cpp` | `gsr.py` |
| Gaussian1dModel | `gaussian1dmodel.hpp/.cpp` | `gaussian1d.py` |
| MarkovFunctional | `markovfunctional.hpp/.cpp` | `markov_functional.py` |

**Implementation notes:**
- Model parameters as JAX arrays (mean reversion, vol, etc.)
- Bond price P(t,T) as closed-form where available (Vasicek, CIR, HW)
- Calibration: `jax.grad` of calibration error w.r.t. model params → gradient-based optimizer

---

### Task 6.2: Short-Rate Models — Two Factor

**QuantLib source:** `ql/models/shortrate/twofactormodels/` (2 hpp + 1 cpp), `ql/models/shortrate/twofactormodel.hpp/.cpp`

**Target:** `ql_jax/models/shortrate/g2.py`, `ql_jax/models/shortrate/two_factor.py`

| Model | Source | Target |
|-------|--------|--------|
| TwoFactorModel base | `twofactormodel.hpp/.cpp` | `two_factor.py` |
| G2++ | `g2.hpp/.cpp` | `g2.py` |

---

### Task 6.3: Calibration Helpers

**QuantLib source:** `ql/models/shortrate/calibrationhelpers/` (3 hpp + 2 cpp), `ql/models/calibrationhelper.hpp/.cpp`, `ql/models/model.hpp/.cpp`, `ql/models/parameter.hpp`

**Target:** `ql_jax/models/calibration.py`, `ql_jax/models/helpers.py`

| Component | Source | Target |
|-----------|--------|--------|
| CalibratedModel | `model.hpp/.cpp` | `calibration.py` |
| CalibrationHelper | `calibrationhelper.hpp/.cpp` | `helpers.py` |
| Parameter | `parameter.hpp` | `parameter.py` |
| SwaptionHelper | `swaptionhelper.hpp/.cpp` | `swaption_helper.py` |
| CapHelper | `caphelper.hpp/.cpp` | `cap_helper.py` |

**Implementation notes:**
- `calibrate(model_params, helpers, market_data) -> optimized_params`
- Objective: minimize squared pricing error across calibration instruments
- Use `jax.grad` for Jacobian → Levenberg-Marquardt or BFGS
- `jax.jit` the entire calibration loop for speed

**Tests to port:** `test-suite/shortratemodels.cpp`, `test-suite/gsr.cpp`, `test-suite/markovfunctional.cpp`

---

### Task 6.4: Equity Models (Heston, Bates, GJRGARCH)

**QuantLib source:** `ql/models/equity/` (8 hpp + 7 cpp = 15 files)

**Target:** `ql_jax/models/equity/`

| Model | Source | Target |
|-------|--------|--------|
| HestonModel | `hestonmodel.hpp/.cpp` | `heston.py` |
| HestonModelHelper | `hestonmodelhelper.hpp/.cpp` | `heston_helper.py` |
| BatesModel | `batesmodel.hpp/.cpp` | `bates.py` |
| GJRGARCHModel | `gjrgarchmodel.hpp/.cpp` | `gjrgarch.py` |
| PiecewiseTimeDependentHeston | `piecewisetimedependenthestonmodel.hpp/.cpp` | `piecewise_heston.py` |
| HestonSLVFDMModel | `hestonslvfdmmodel.hpp/.cpp` | `heston_slv_fdm.py` |
| HestonSLVMCModel | `hestonslvmcmodel.hpp/.cpp` | `heston_slv_mc.py` |

**Tests to port:** `test-suite/hestonmodel.cpp`, `test-suite/hestonslvmodel.cpp`, `test-suite/batesmodel.cpp`, `test-suite/gjrgarchmodel.cpp`

---

### Task 6.5: Volatility Models

**QuantLib source:** `ql/models/volatility/` (5 hpp + 2 cpp = 7 files)

**Target:** `ql_jax/models/volatility/`

- ConstantEstimator, GARCH, GarmanKlass, SimpleLocalEstimator

**Tests to port:** `test-suite/volatilitymodels.cpp`, `test-suite/garch.cpp`

---

### Task 6.6: Cap/Floor Instruments & Engines

**QuantLib source:**
- `ql/instruments/capfloor.hpp/.cpp`, `ql/instruments/makecapfloor.hpp/.cpp`
- `ql/pricingengines/capfloor/` (8 hpp + 7 cpp = 15 files)

**Target:** `ql_jax/instruments/capfloor.py`, `ql_jax/engines/capfloor/`

**Engines:**
| Engine | Source | Target |
|--------|--------|--------|
| BlackCapFloorEngine | `blackcapfloorengine.hpp/.cpp` | `black.py` |
| BachelierCapFloorEngine | `bacheliercapfloorengine.hpp/.cpp` | `bachelier.py` |
| AnalyticCapFloorEngine | `analyticcapfloorengine.hpp/.cpp` | `analytic.py` |
| TreeCapFloorEngine | `treecapfloorengine.hpp/.cpp` | `tree.py` |
| Gaussian1dCapFloorEngine | `gaussian1dcapfloorengine.hpp/.cpp` | `gaussian1d.py` |
| MCHullWhiteCapFloorEngine | `mchullwhiteengine.hpp/.cpp` | `mc_hull_white.py` |

**Tests to port:** `test-suite/capfloor.cpp`

---

### Task 6.7: Swaption Instruments & Engines

**QuantLib source:**
- `ql/instruments/swaption.hpp/.cpp`, `ql/instruments/makeswaption.hpp/.cpp`
- `ql/instruments/nonstandardswaption.hpp/.cpp`, `ql/instruments/floatfloatswaption.hpp/.cpp`
- `ql/pricingengines/swaption/` (13 hpp + 11 cpp = 24 files)

**Target:** `ql_jax/instruments/swaption.py`, `ql_jax/engines/swaption/`

**Engines:**
| Engine | Source | Target |
|--------|--------|--------|
| BlackSwaptionEngine | `blackswaptionengine.hpp/.cpp` | `black.py` |
| JamshidianSwaptionEngine | `jamshidianswaptionengine.hpp/.cpp` | `jamshidian.py` |
| TreeSwaptionEngine | `treeswaptionengine.hpp/.cpp` | `tree.py` |
| G2SwaptionEngine | `g2swaptionengine.hpp/.cpp` | `g2.py` |
| FdG2SwaptionEngine | `fdg2swaptionengine.hpp/.cpp` | `fd_g2.py` |
| FdHullWhiteSwaptionEngine | `fdhullwhiteswaptionengine.hpp/.cpp` | `fd_hull_white.py` |
| Gaussian1dSwaptionEngine | `gaussian1dswaptionengine.hpp/.cpp` | `gaussian1d.py` |
| Gaussian1dJamshidianSwaptionEngine | `gaussian1djamshidianswaptionengine.hpp/.cpp` | `gaussian1d_jamshidian.py` |
| Gaussian1dNonstandardSwaptionEngine | `gaussian1dnonstandardswaptionengine.hpp/.cpp` | `gaussian1d_nonstandard.py` |
| Gaussian1dFloatFloatSwaptionEngine | `gaussian1dfloatfloatswaptionengine.hpp/.cpp` | `gaussian1d_float_float.py` |
| BasketGeneratingEngine | `basketgeneratingengine.hpp/.cpp` | `basket_generating.py` |

**Tests to port:** `test-suite/swaption.cpp`, `test-suite/bermudanswaption.cpp`

**Examples to port:** `BermudanSwaption`, `Gaussian1dModels`

---

### Task 6.8: Callable Bond Engines

**QuantLib source:**
- `ql/instruments/callabilityschedule.hpp`
- `ql/pricingengines/bond/binomialconvertibleengine.hpp`
- `ql/pricingengines/bond/discretizedconvertible.hpp/.cpp`

**Target:** `ql_jax/instruments/callable.py`, `ql_jax/engines/bond/callable.py`, `ql_jax/engines/bond/convertible.py`

**Tests to port:** `test-suite/callablebonds.cpp`, `test-suite/convertiblebonds.cpp`

**Examples to port:** `CallableBonds`, `ConvertibleBonds`

---

## Phase 7: Credit, Inflation & Market Models

**Dependencies:** Phase 2 (credit/inflation curves), Phase 3 (CDS instruments), Phase 5 (MC framework)

### Task 7.1: Credit Default Swap Instruments

**QuantLib source:**
- `ql/instruments/creditdefaultswap.hpp/.cpp`
- `ql/instruments/makecds.hpp/.cpp`
- `ql/instruments/claim.hpp/.cpp`

**Target:** `ql_jax/instruments/cds.py`

---

### Task 7.2: CDS Pricing Engines

**QuantLib source:** `ql/pricingengines/credit/` (4 hpp + 3 cpp = 7 files)

**Target:** `ql_jax/engines/credit/`

| Engine | Source | Target |
|--------|--------|--------|
| MidPointCdsEngine | `midpointcdsengine.hpp/.cpp` | `midpoint.py` |
| IntegralCdsEngine | `integralcdsengine.hpp/.cpp` | `integral.py` |
| IsdaCdsEngine | `isdacdsengine.hpp/.cpp` | `isda.py` |

**Tests to port:** `test-suite/creditdefaultswap.cpp`, `test-suite/cdsoption.cpp`

**Example to port:** `CDS`

---

### Task 7.3: Inflation Instruments & Engines

**QuantLib source:**
- `ql/instruments/inflationcapfloor.hpp/.cpp`, `ql/instruments/makeyoyinflationcapfloor.hpp/.cpp`
- `ql/instruments/cpicapfloor.hpp/.cpp`
- `ql/pricingengines/inflation/inflationcapfloorengines.hpp/.cpp`

**Target:** `ql_jax/instruments/inflation_capfloor.py`, `ql_jax/engines/inflation/`

**Tests to port:** `test-suite/inflationcapfloor.cpp`, `test-suite/inflationcapflooredcoupon.cpp`, `test-suite/inflationcpicapfloor.cpp`

---

### Task 7.4: LIBOR Market Model (LMM)

**QuantLib source:** `ql/models/marketmodels/` (19 hpp + 13 cpp = 32 files + additional subdirectories)

**Target:** `ql_jax/models/marketmodels/`

**Key components:**
| Component | Source | Target |
|-----------|--------|--------|
| MarketModel base | `marketmodel.hpp/.cpp` | `base.py` |
| CurveState | `curvestate.hpp/.cpp` | `curve_state.py` |
| EvolutionDescription | `evolutiondescription.hpp/.cpp` | `evolution.py` |
| Evolver | `evolver.hpp` | `evolver.py` |
| BrownianGenerator | `browniangenerator.hpp` | `brownian_generator.py` |
| MultiProduct | `multiproduct.hpp` | `multi_product.py` |
| Discounter | `discounter.hpp/.cpp` | `discounter.py` |
| ForwardForwardMappings | `forwardforwardmappings.hpp/.cpp` | `forward_mappings.py` |
| SwapForwardMappings | `swapforwardmappings.hpp/.cpp` | `swap_mappings.py` |
| PiecewiseConstantCorrelation | `piecewiseconstantcorrelation.hpp` | `correlation.py` |
| PathwiseAccountingEngine | `pathwiseaccountingengine.hpp/.cpp` | `pathwise_accounting.py` |
| ProxyGreekEngine | `proxygreekengine.hpp/.cpp` | `proxy_greeks.py` |
| Utilities | `utilities.hpp/.cpp` | `utilities.py` |

**Implementation notes:**
- LMM simulates forward LIBOR rates under a chosen numeraire
- Use `jax.vmap` over paths for vectorized simulation
- Correlation matrix parametrization differentiable for calibration
- AD replaces the need for ProxyGreekEngine (adjoint method via `jax.grad`)

**Tests to port:** `test-suite/marketmodel.cpp`, `test-suite/marketmodel_cms.cpp`, `test-suite/marketmodel_smm.cpp`, `test-suite/marketmodel_smmcapletalphacalibration.cpp`, `test-suite/marketmodel_smmcapletcalibration.cpp`, `test-suite/marketmodel_smmcaplethomocalibration.cpp`, `test-suite/libormarketmodel.cpp`, `test-suite/libormarketmodelprocess.cpp`, `test-suite/swapforwardmappings.cpp`, `test-suite/curvestates.cpp`

**Examples to port:** `MarketModels`, `BasketLosses`, `LatentModel`, `CVAIRS`

---

### Task 7.5: Additional Credit Models

**Tests to port:** `test-suite/cdo.cpp`, `test-suite/nthtodefault.cpp`

---

### Task 7.6: Remaining Instruments

**QuantLib source (miscellaneous):**
- `ql/instruments/stock.hpp/.cpp`
- `ql/instruments/bondforward.hpp/.cpp`
- `ql/instruments/fxforward.hpp/.cpp`
- `ql/instruments/futures.hpp/.cpp`, `ql/instruments/perpetualfutures.hpp/.cpp`
- `ql/instruments/overnightindexfuture.hpp/.cpp`
- `ql/instruments/assetswap.hpp/.cpp`
- `ql/instruments/vanillaswingoption.hpp/.cpp`, `ql/instruments/vanillastorageoption.hpp/.cpp`
- `ql/instruments/impliedvolatility.hpp/.cpp`
- `ql/instruments/compositeinstrument.hpp/.cpp`
- `ql/instruments/dividendschedule.hpp`

**Target:** Various files under `ql_jax/instruments/`

**Tests to port:** `test-suite/bondforward.cpp`, `test-suite/fxforward.cpp`, `test-suite/perpetualfutures.cpp`, `test-suite/equitytotalreturnswap.cpp`, `test-suite/assetswap.cpp`, `test-suite/swingoption.cpp`, `test-suite/instruments.cpp`

---

### Task 7.7: Remaining Pricing Engines

**From `ql/pricingengines/`:**
- `forward/replicatingvarianceswapengine.hpp` → `ql_jax/engines/forward/replicating_variance.py`
- `forward/discountingfxforwardengine.hpp/.cpp` → `ql_jax/engines/forward/fx_forward.py`
- `inflation/inflationcapfloorengines.hpp/.cpp` → `ql_jax/engines/inflation/capfloor.py`
- `futures/discountingperpetualfuturesengine.hpp/.cpp` → `ql_jax/engines/futures/perpetual.py`
- `greeks.hpp/.cpp` → `ql_jax/engines/greeks.py`
- `latticeshortratemodelengine.hpp` → `ql_jax/engines/lattice/short_rate_model.py`

---

## Phase 8: Integration, Benchmarks & Polish

**Dependencies:** All previous phases

### Task 8.1: Remaining Examples

**All 20 QuantLib examples to port:**

| # | Example | Phase Dependency | JAX-Enhanced Features |
|---|---------|-----------------|----------------------|
| 1 | EquityOption | Phase 4 | AD Greeks, vmap batch pricing, GPU comparison |
| 2 | Bonds | Phase 3 | AD key-rate durations, vmap portfolio |
| 3 | BermudanSwaption | Phase 6 | AD calibration sensitivities |
| 4 | FittedBondCurve | Phase 2 | AD curve sensitivities |
| 5 | FRA | Phase 3 | AD |
| 6 | Repo | Phase 3 | AD |
| 7 | CDS | Phase 7 | AD hazard rate sensitivity |
| 8 | CallableBonds | Phase 6 | AD |
| 9 | ConvertibleBonds | Phase 6 | AD |
| 10 | DiscreteHedging | Phase 5 | GPU MC acceleration |
| 11 | BasketLosses | Phase 7 | vmap over scenarios |
| 12 | LatentModel | Phase 7 | GPU-accelerated simulation |
| 13 | MarketModels | Phase 7 | AD calibration |
| 14 | Gaussian1dModels | Phase 6 | AD |
| 15 | MulticurveBootstrapping | Phase 2 | AD through bootstrap |
| 16 | GlobalOptimizer | Phase 1 | jax.grad-based optimization comparison |
| 17 | MultidimIntegral | Phase 1 | GPU-accelerated quadrature |
| 18 | Replication | Phase 4 | AD replication sensitivities |
| 19 | CVAIRS | Phase 7 | GPU MC CVA computation |
| 20 | AsianOption | Phase 5 | AD Asian Greeks, GPU MC |

Each example will have two versions:
1. **`examples/XX_classic.py`** — faithful port matching C++ output
2. **`examples/XX_jax_enhanced.py`** — demonstrates AD, vmap, and GPU capabilities

---

### Task 8.2: CPU vs GPU JIT Benchmark Suite

**Target:** `benchmarks/`

**Benchmark files:**

| File | What It Benchmarks |
|------|--------------------|
| `bench_analytic.py` | BSM, Heston analytic at batch sizes 1–1M (CPU vs GPU) |
| `bench_tree.py` | Binomial tree pricing at 10–10,000 steps (CPU vs GPU) |
| `bench_fd.py` | FD PDE solving at grid sizes 100–10,000 (CPU vs GPU) |
| `bench_mc.py` | Monte Carlo at 1k–1M paths (CPU vs GPU) |
| `bench_greeks.py` | AD Greeks via grad/jacrev on each engine (CPU vs GPU) |
| `bench_portfolio.py` | vmap portfolio pricing at 100–1M instruments (CPU vs GPU) |
| `bench_bootstrap.py` | Yield curve bootstrap at 5–50 instruments (CPU vs GPU) |
| `bench_calibration.py` | Heston/HW calibration loops (CPU vs GPU) |
| `bench_jit_latency.py` | First-call JIT compilation time per engine (CPU vs GPU) |
| `bench_transfer.py` | Host↔device data transfer overhead measurement |
| `runner.py` | CLI runner: `python runner.py --backend cpu --size large --precision float64` |

**Methodology:**
- Warm-up: 3 iterations (trigger JIT compilation)
- Timed: minimum 10 iterations
- Metrics: median, p5, p95 wall time; peak memory; FLOP estimate
- Hardware recorded: CPU model, GPU model, CUDA version, JAX version
- Output: `benchmarks/results/` with JSON + auto-generated markdown tables

**Crossover analysis:**
- For each engine type, identify problem size N* where GPU becomes faster than CPU
- Generate plots: latency vs problem size, speedup ratio vs problem size

---

### Task 8.3: End-to-End Integration Tests

**Target:** `tests/integration/`

| Test | Description |
|------|-------------|
| `test_equity_workflow.py` | Build market → price options → compute Greeks → verify vs C++ |
| `test_bond_workflow.py` | Bootstrap curve → price bonds → compute DV01/KRD → verify |
| `test_swaption_workflow.py` | Build curves/vols → calibrate model → price swaptions → verify |
| `test_cds_workflow.py` | Build credit curve → price CDS → compute spread sensitivity |
| `test_mc_convergence.py` | MC pricing converges to analytic at large N for all engines |
| `test_portfolio_vmap.py` | vmap batch pricing matches sequential loop (exact) |
| `test_jit_correctness.py` | JIT output matches eager output for all engines (bitwise) |
| `test_ad_greeks.py` | AD Greeks match finite-difference bumps for all engines |
| `test_float64_precision.py` | All results use float64 and match C++ to 1e-10 |

---

### Task 8.4: API Documentation

**Target:** `docs/`

**Documentation structure:**
```
docs/
├── conf.py                  # Sphinx configuration
├── index.rst                # Landing page
├── getting_started.rst      # Installation, first example
├── api/
│   ├── time.rst             # Time module API
│   ├── math.rst             # Math module API
│   ├── termstructures.rst   # Term structures API
│   ├── instruments.rst      # Instruments API
│   ├── engines.rst          # Pricing engines API
│   ├── models.rst           # Models API
│   └── processes.rst        # Stochastic processes API
├── tutorials/
│   ├── ad_greeks.rst        # AD Greeks tutorial
│   ├── vmap_portfolio.rst   # vmap portfolio tutorial
│   ├── gpu_acceleration.rst # GPU setup and usage
│   └── calibration.rst      # Gradient-based calibration
├── migration_guide.rst      # C++ QuantLib → QL-JAX migration
└── benchmarks.rst           # Performance results
```

---

### Task 8.5: CI/CD Pipeline

**Target:** `.github/workflows/`

| Workflow | Trigger | What It Does |
|----------|---------|--------------|
| `test-cpu.yml` | Push/PR | Install in venv, run full test suite on CPU |
| `test-gpu.yml` | Push/PR | Install with GPU extra, run tests on CUDA runner |
| `lint.yml` | Push/PR | ruff, mypy, black --check |
| `docs.yml` | Push to main | Build Sphinx docs, deploy to GitHub Pages |
| `benchmark.yml` | Weekly / manual | Run benchmark suite, commit results |
| `release.yml` | Tag v* | Build wheel, publish to PyPI |

---

### Task 8.6: Package Distribution

**`pyproject.toml` final structure:**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ql-jax"
version = "0.1.0"
description = "QuantLib re-implemented in JAX"
requires-python = ">=3.10"
dependencies = [
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-benchmark", "sphinx", "black", "ruff", "mypy", "pre-commit"]
test = ["pytest", "pytest-xdist", "pytest-benchmark"]
gpu = ["jax[cuda12]"]
```

---

## Appendix A: Complete Test File Mapping

All 138 QuantLib C++ test files mapped to QL-JAX Python test modules:

| C++ Test File | QL-JAX Test File | Phase |
|---------------|-----------------|-------|
| `dates.cpp` | `tests/time/test_dates.py` | 1 |
| `period.cpp` | `tests/time/test_period.py` | 1 |
| `calendars.cpp` | `tests/time/test_calendars.py` | 1 |
| `businessdayconventions.cpp` | `tests/time/test_bdc.py` | 1 |
| `daycounters.cpp` | `tests/time/test_daycounters.py` | 1 |
| `schedule.cpp` | `tests/time/test_schedule.py` | 1 |
| `timegrid.cpp` | `tests/time/test_timegrid.py` | 1 |
| `array.cpp` | `tests/math/test_array.py` | 1 |
| `matrices.cpp` | `tests/math/test_matrices.py` | 1 |
| `functions.cpp` | `tests/math/test_functions.py` | 1 |
| `fastfouriertransform.cpp` | `tests/math/test_fft.py` | 1 |
| `interpolations.cpp` | `tests/math/test_interpolations.py` | 1 |
| `solvers.cpp` | `tests/math/test_solvers.py` | 1 |
| `integrals.cpp` | `tests/math/test_integrals.py` | 1 |
| `gaussianquadratures.cpp` | `tests/math/test_gauss_quad.py` | 1 |
| `distributions.cpp` | `tests/math/test_distributions.py` | 1 |
| `optimizers.cpp` | `tests/math/test_optimizers.py` | 1 |
| `mersennetwister.cpp` | `tests/math/test_rng.py` | 1 |
| `lowdiscrepancysequences.cpp` | `tests/math/test_lds.py` | 1 |
| `rngtraits.cpp` | `tests/math/test_rng_traits.py` | 1 |
| `xoshiro256starstar.cpp` | `tests/math/test_xoshiro.py` | 1 |
| `zigguratgaussian.cpp` | `tests/math/test_ziggurat.py` | 1 |
| `stats.cpp` | `tests/math/test_stats.py` | 1 |
| `riskstats.cpp` | `tests/math/test_risk_stats.py` | 1 |
| `observable.cpp` | `tests/patterns/test_observable.py` | 1 |
| `lazyobject.cpp` | `tests/patterns/test_lazy.py` | 1 |
| `quotes.cpp` | `tests/test_quotes.py` | 2 |
| `currency.cpp` | `tests/test_currency.py` | 2 |
| `exchangerate.cpp` | `tests/test_exchange_rate.py` | 2 |
| `money.cpp` | `tests/test_money.py` | 2 |
| `indexes.cpp` | `tests/test_indexes.py` | 2 |
| `sofrfutures.cpp` | `tests/test_sofr_futures.py` | 2 |
| `interestrates.cpp` | `tests/test_interest_rates.py` | 2 |
| `piecewiseyieldcurve.cpp` | `tests/termstructures/test_piecewise_yield.py` | 2 |
| `fittedbonddiscountcurve.cpp` | `tests/termstructures/test_fitted_bond.py` | 2 |
| `termstructures.cpp` | `tests/termstructures/test_termstructures.py` | 2 |
| `piecewisezerospreadedtermstructure.cpp` | `tests/termstructures/test_zero_spreaded.py` | 2 |
| `ultimateforwardtermstructure.cpp` | `tests/termstructures/test_ultimate_forward.py` | 2 |
| `defaultprobabilitycurves.cpp` | `tests/termstructures/test_default_curves.py` | 2 |
| `inflation.cpp` | `tests/termstructures/test_inflation.py` | 2 |
| `inflationvolatility.cpp` | `tests/termstructures/test_inflation_vol.py` | 2 |
| `swaptionvolatilitycube.cpp` | `tests/termstructures/test_swaption_vol_cube.py` | 2 |
| `swaptionvolatilitymatrix.cpp` | `tests/termstructures/test_swaption_vol_matrix.py` | 2 |
| `interpolatedsmilesection.cpp` | `tests/termstructures/test_smile_section.py` | 2 |
| `blackvolsurfacedelta.cpp` | `tests/termstructures/test_black_vol_delta.py` | 2 |
| `piecewiseblackvariancesurface.cpp` | `tests/termstructures/test_piecewise_black_var.py` | 2 |
| `optionletstripper.cpp` | `tests/termstructures/test_optionlet_stripper.py` | 2 |
| `noarbsabr.cpp` | `tests/termstructures/test_noarb_sabr.py` | 2 |
| `zabr.cpp` | `tests/termstructures/test_zabr.py` | 2 |
| `svivolatility.cpp` | `tests/termstructures/test_svi.py` | 2 |
| `cashflows.cpp` | `tests/cashflows/test_cashflows.py` | 3 |
| `capflooredcoupon.cpp` | `tests/cashflows/test_capped_floored.py` | 3 |
| `digitalcoupon.cpp` | `tests/cashflows/test_digital.py` | 3 |
| `overnightindexedcoupon.cpp` | `tests/cashflows/test_overnight.py` | 3 |
| `multipleresetscoupons.cpp` | `tests/cashflows/test_multiple_resets.py` | 3 |
| `rangeaccrual.cpp` | `tests/cashflows/test_range_accrual.py` | 3 |
| `cms.cpp` | `tests/cashflows/test_cms.py` | 3 |
| `cms_normal.cpp` | `tests/cashflows/test_cms_normal.py` | 3 |
| `cmsspread.cpp` | `tests/cashflows/test_cms_spread.py` | 3 |
| `equitycashflow.cpp` | `tests/cashflows/test_equity_cashflow.py` | 3 |
| `bonds.cpp` | `tests/instruments/test_bonds.py` | 3 |
| `amortizingbond.cpp` | `tests/instruments/test_amortizing_bond.py` | 3 |
| `inflationcpibond.cpp` | `tests/instruments/test_cpi_bond.py` | 3 |
| `swap.cpp` | `tests/instruments/test_swap.py` | 3 |
| `overnightindexedswap.cpp` | `tests/instruments/test_ois.py` | 3 |
| `multipleresetsswap.cpp` | `tests/instruments/test_multiple_resets_swap.py` | 3 |
| `zerocouponswap.cpp` | `tests/instruments/test_zero_coupon_swap.py` | 3 |
| `inflationcpiswap.cpp` | `tests/instruments/test_cpi_swap.py` | 3 |
| `forwardrateagreement.cpp` | `tests/instruments/test_fra.py` | 3 |
| `europeanoption.cpp` | `tests/engines/test_european.py` | 4 |
| `americanoption.cpp` | `tests/engines/test_american.py` | 4 |
| `hestonmodel.cpp` | `tests/models/test_heston.py` | 4 |
| `batesmodel.cpp` | `tests/models/test_bates.py` | 4 |
| `gjrgarchmodel.cpp` | `tests/models/test_gjrgarch.py` | 4 |
| `jumpdiffusion.cpp` | `tests/engines/test_jump_diffusion.py` | 4 |
| `dividendoption.cpp` | `tests/engines/test_dividend.py` | 4 |
| `digitaloption.cpp` | `tests/engines/test_digital_option.py` | 4 |
| `extendedtrees.cpp` | `tests/engines/test_trees.py` | 4 |
| `barrieroption.cpp` | `tests/engines/test_barrier.py` | 4 |
| `doublebarrieroption.cpp` | `tests/engines/test_double_barrier.py` | 4 |
| `asianoptions.cpp` | `tests/engines/test_asian.py` | 4 |
| `lookbackoptions.cpp` | `tests/engines/test_lookback.py` | 4 |
| `cliquetoption.cpp` | `tests/engines/test_cliquet.py` | 4 |
| `basketoption.cpp` | `tests/engines/test_basket.py` | 4 |
| `quantooption.cpp` | `tests/engines/test_quanto.py` | 4 |
| `compoundoption.cpp` | `tests/engines/test_compound.py` | 4 |
| `chooseroption.cpp` | `tests/engines/test_chooser.py` | 4 |
| `extensibleoptions.cpp` | `tests/engines/test_extensible.py` | 4 |
| `margrabeoption.cpp` | `tests/engines/test_margrabe.py` | 4 |
| `twoassetbarrieroption.cpp` | `tests/engines/test_two_asset_barrier.py` | 4 |
| `twoassetcorrelationoption.cpp` | `tests/engines/test_two_asset_corr.py` | 4 |
| `partialtimebarrieroption.cpp` | `tests/engines/test_partial_time_barrier.py` | 4 |
| `softbarrieroption.cpp` | `tests/engines/test_soft_barrier.py` | 4 |
| `forwardoption.cpp` | `tests/engines/test_forward_option.py` | 4 |
| `varianceswaps.cpp` | `tests/engines/test_variance_swap.py` | 4 |
| `binaryoption.cpp` | `tests/engines/test_binary.py` | 4 |
| `doublebinaryoption.cpp` | `tests/engines/test_double_binary.py` | 4 |
| `fdheston.cpp` | `tests/engines/test_fd_heston.py` | 4 |
| `fdcev.cpp` | `tests/engines/test_fd_cev.py` | 4 |
| `fdcir.cpp` | `tests/engines/test_fd_cir.py` | 4 |
| `fdsabr.cpp` | `tests/engines/test_fd_sabr.py` | 4 |
| `fdmlinearop.cpp` | `tests/methods/test_fdm_linear_op.py` | 4 |
| `nthorderderivativeop.cpp` | `tests/methods/test_nth_order.py` | 4 |
| `numericaldifferentiation.cpp` | `tests/methods/test_numerical_diff.py` | 4 |
| `operators.cpp` | `tests/methods/test_operators.py` | 4 |
| `hybridhestonhullwhiteprocess.cpp` | `tests/processes/test_hybrid.py` | 4 |
| `mclongstaffschwartzengine.cpp` | `tests/engines/test_ls_mc.py` | 5 |
| `brownianbridge.cpp` | `tests/methods/test_brownian_bridge.py` | 5 |
| `pathgenerator.cpp` | `tests/methods/test_path_generator.py` | 5 |
| `shortratemodels.cpp` | `tests/models/test_short_rate.py` | 6 |
| `gsr.cpp` | `tests/models/test_gsr.py` | 6 |
| `markovfunctional.cpp` | `tests/models/test_markov_functional.py` | 6 |
| `hestonslvmodel.cpp` | `tests/models/test_heston_slv.py` | 6 |
| `volatilitymodels.cpp` | `tests/models/test_vol_models.py` | 6 |
| `garch.cpp` | `tests/models/test_garch.py` | 6 |
| `capfloor.cpp` | `tests/instruments/test_capfloor.py` | 6 |
| `swaption.cpp` | `tests/instruments/test_swaption.py` | 6 |
| `bermudanswaption.cpp` | `tests/instruments/test_bermudan_swaption.py` | 6 |
| `callablebonds.cpp` | `tests/instruments/test_callable_bonds.py` | 6 |
| `convertiblebonds.cpp` | `tests/instruments/test_convertible_bonds.py` | 6 |
| `creditdefaultswap.cpp` | `tests/instruments/test_cds.py` | 7 |
| `cdsoption.cpp` | `tests/instruments/test_cds_option.py` | 7 |
| `inflationcapfloor.cpp` | `tests/instruments/test_inflation_capfloor.py` | 7 |
| `inflationcapflooredcoupon.cpp` | `tests/instruments/test_inflation_capped.py` | 7 |
| `inflationcpicapfloor.cpp` | `tests/instruments/test_cpi_capfloor.py` | 7 |
| `marketmodel.cpp` | `tests/models/test_market_model.py` | 7 |
| `marketmodel_cms.cpp` | `tests/models/test_mm_cms.py` | 7 |
| `marketmodel_smm.cpp` | `tests/models/test_mm_smm.py` | 7 |
| `libormarketmodel.cpp` | `tests/models/test_lmm.py` | 7 |
| `libormarketmodelprocess.cpp` | `tests/models/test_lmm_process.py` | 7 |
| `swapforwardmappings.cpp` | `tests/models/test_swap_fwd_map.py` | 7 |
| `curvestates.cpp` | `tests/models/test_curve_states.py` | 7 |
| `cdo.cpp` | `tests/instruments/test_cdo.py` | 7 |
| `nthtodefault.cpp` | `tests/instruments/test_nth_to_default.py` | 7 |
| `bondforward.cpp` | `tests/instruments/test_bond_forward.py` | 7 |
| `fxforward.cpp` | `tests/instruments/test_fx_forward.py` | 7 |
| `perpetualfutures.cpp` | `tests/instruments/test_perpetual_futures.py` | 7 |
| `equitytotalreturnswap.cpp` | `tests/instruments/test_equity_trs.py` | 7 |
| `assetswap.cpp` | `tests/instruments/test_asset_swap.py` | 7 |
| `swingoption.cpp` | `tests/instruments/test_swing.py` | 7 |
| `instruments.cpp` | `tests/instruments/test_instruments.py` | 7 |

**Remaining test files (utilities, settings, misc):**
- `autocovariances.cpp` → `tests/math/test_autocovariance.py`
- `covariance.cpp` → `tests/math/test_covariance.py`
- `linearleastsquaresregression.cpp` → `tests/math/test_linear_regression.py`
- `riskneutraldensitycalculator.cpp` → `tests/methods/test_rnd_calculator.py`
- `rounding.cpp` → `tests/math/test_rounding.py`
- `prices.cpp` → `tests/test_prices.py`
- `settings.cpp` → `tests/test_settings.py`
- `timeseries.cpp` → `tests/time/test_timeseries.py`
- `tracing.cpp` → `tests/test_tracing.py`

---

## Appendix B: File Count Summary

| Phase | New Python Files (est.) | C++ Files Covered | Tests Ported |
|-------|------------------------|-------------------|-------------|
| 0 — Scaffolding | 10 | 0 | 1 smoke test |
| 1 — Foundation | 60 | ~280 | 26 |
| 2 — Market Data | 55 | ~340 | 24 |
| 3 — Cash Flows & FI | 35 | ~270 | 18 |
| 4 — Equity & Engines | 80 | ~530 | 30 |
| 5 — Monte Carlo | 20 | ~55 | 3 |
| 6 — IR Models & Derivs | 40 | ~150 | 12 |
| 7 — Credit/Inflation/LMM | 25 | ~120 | 20 |
| 8 — Integration & Polish | 30 | — | 9 integration |
| **Total** | **~355** | **~1,745** | **~143 + 9** |

---

## Appendix C: Dependency Graph

```
Phase 0: Scaffolding
    │
    ▼
Phase 1: Foundation (time, math, patterns)
    │
    ├──────────────────┐
    ▼                  ▼
Phase 2: Market Data   Phase 1 (cont.)
    │
    ├──────────────────┐
    ▼                  ▼
Phase 3: Cash Flows    Phase 4: Equity & Engines
    │                  │
    ├──────────────────┤
    ▼                  ▼
Phase 5: Monte Carlo   Phase 6: IR Models
    │                  │
    ├──────────────────┤
    ▼                  ▼
Phase 7: Credit/Inflation/LMM
    │
    ▼
Phase 8: Integration & Polish
```

**Critical path:** Phase 0 → 1 → 2 → 3 → 4 → 5 → 8 (equity/MC track)
**Parallel track:** Phase 0 → 1 → 2 → 6 → 7 → 8 (rates/credit track)

Phases 4 and 6 can proceed in parallel after Phase 2 completes.
