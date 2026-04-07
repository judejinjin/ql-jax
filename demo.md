# QL-JAX Demo Notebooks: Proposal & Implementation Plan

## Vision

Build a comprehensive set of Jupyter notebooks that teach users ql-jax by working through real financial instruments — from basic bonds to exotic credit portfolios. Each notebook directly compares ql-jax against QuantLib (Python SWIG 1.42) across three dimensions: **API usage**, **numerical accuracy**, and **performance**. Users can switch between CPU and GPU backends with a single toggle.

---

## Design Principles

1. **Side-by-side comparison**: Every pricing example runs in both QuantLib-SWIG and ql-jax, printing results in aligned tables so numerical agreement is immediately visible.
2. **One instrument per notebook**: Each notebook covers one instrument or a tightly related set (e.g., cap + floor + collar). No notebook exceeds ~30 minutes of runtime.
3. **CPU/GPU backend toggle**: A shared utility cell at the top of each notebook lets users select `cpu` or `gpu` via a dropdown widget or environment variable. All JIT-compiled functions automatically target the chosen backend.
4. **Progressive learning path**: Notebooks are numbered and ordered from foundational (dates, curves) to advanced (portfolio credit, CVA), forming a complete curriculum.
5. **Self-contained**: Each notebook can be run independently. Shared helpers live in a `notebooks/_common.py` module.

---

## Shared Infrastructure

### `notebooks/_common.py` — Utility Module

```python
# Backend selector (called at top of every notebook)
def setup_backend(backend="cpu"):
    """Configure JAX backend and enable float64."""
    import os
    os.environ["JAX_PLATFORMS"] = backend
    import jax
    jax.config.update("jax_enable_x64", True)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices:     {jax.devices()}")

# QuantLib SWIG loader
def load_quantlib():
    """Import QuantLib with correct LD_LIBRARY_PATH."""
    import sys, os
    os.environ["LD_LIBRARY_PATH"] = "/home/jude/QuantLib/build/ql:" + os.environ.get("LD_LIBRARY_PATH", "")
    sys.path.insert(0, "/home/jude/QuantLib-SWIG/Python/build/lib.linux-x86_64-cpython-312")
    import QuantLib
    return QuantLib

# Comparison table formatter
def compare(label, ql_val, jax_val, unit=""):
    """Print side-by-side comparison with absolute and relative error."""
    ...

# Timing decorator
def timed(fn, *args, warmup=2, runs=5):
    """Benchmark with warmup, return (avg_seconds, result)."""
    ...

# Speedup bar chart
def plot_speedup(labels, ql_times, jax_times):
    """Matplotlib bar chart of QuantLib vs ql-jax timings."""
    ...
```

### Standard Notebook Header Cell

Every notebook starts with:
```python
# Select backend: "cpu" or "gpu"
BACKEND = "cpu"

import sys; sys.path.insert(0, "..")
from notebooks._common import setup_backend, load_quantlib, compare, timed
setup_backend(BACKEND)
QL = load_quantlib()

import jax
import jax.numpy as jnp
```

---

## Notebook Curriculum

### Part 1: Foundations (Notebooks 01–04)

| # | Notebook | Topics | QL-JAX Modules | QuantLib Ref |
|---|----------|--------|----------------|--------------|
| 01 | **Dates, Calendars & Day Counters** | Date arithmetic, 48 calendars, 10+ day counters, schedule generation | `time.date`, `time.calendar`, `time.schedule`, `time.daycounter` | `ql.Date`, `ql.TARGET()`, `ql.Schedule` |
| 02 | **Yield Curves & Discounting** | Flat forward, piecewise bootstrap (deposit+swap helpers), fitted bond curves (Nelson-Siegel, Svensson), zero/forward/discount relationships | `termstructures.yield_`, `math.interpolations`, `math.solvers` | `ql.FlatForward`, `ql.PiecewiseYieldCurve`, `ql.FittedBondDiscountCurve` |
| 03 | **Fixed-Income Basics: Bonds & FRA** | Zero-coupon bond, fixed-rate bond (par/discount/premium), clean/dirty price, duration/convexity via `jax.grad`, FRA pricing & settlement | `instruments.bond`, `instruments.fra`, `engines.bond.discounting` | `ql.FixedRateBond`, `ql.ForwardRateAgreement` |
| 04 | **Interest Rate Swaps** | Vanilla IRS NPV, fair rate, DV01 via AD, payer vs receiver, cashflow extraction, multicurve (OIS + EURIBOR) bootstrap | `instruments.swap`, `engines.swap.discounting`, `cashflows` | `ql.VanillaSwap`, `ql.DiscountingSwapEngine` |

**Key demos in Part 1:**
- `jax.grad` of bond price w.r.t. yield → analytic duration verification
- `jax.jacrev` through piecewise curve bootstrap → zero-rate Jacobian
- vmap across 1000 swap scenarios in a single call

---

### Part 2: Vanilla Derivatives (Notebooks 05–09)

| # | Notebook | Topics | QL-JAX Modules | QuantLib Ref |
|---|----------|--------|----------------|--------------|
| 05 | **European Options: Multi-Engine Pricing** | BSM analytic, Heston semi-analytic, integral engine, FD (Crank-Nicolson), 7 binomial tree types (CRR, JR, Tian, LR, Trigeorgis, Joshi4, Leisen-Reimer) | `engines.analytic.black_formula`, `engines.analytic.heston`, `engines.fd.black_scholes`, `engines.lattice.binomial` | `ql.AnalyticEuropeanEngine`, `ql.BinomialVanillaEngine` |
| 06 | **European Greeks & Hessians** | Delta, gamma, vega, theta, rho via `jax.grad`; 4×4 Hessian (gamma, vanna, volga, charm) via `jax.hessian`; comparison to analytic BSM Greeks | `engines.analytic.black_formula` + `jax.grad`, `jax.hessian` | `ql.AnalyticEuropeanEngine` (FD bumping) |
| 07 | **American Options** | Barone-Adesi-Whaley analytic, FD (implicit, Crank-Nicolson), 5 binomial trees; early exercise boundary; American Greeks via AD | `engines.analytic.american`, `engines.fd.black_scholes`, `engines.lattice.binomial` | `ql.BaroneAdesiWhaleyApproximationEngine`, `ql.FdBlackScholesVanillaEngine` |
| 08 | **Caps, Floors & Collars** | Black cap/floor pricing, cap-floor parity, Hull-White cap pricing, implied vol stripping; `jax.grad` of cap price w.r.t. vol and rate | `instruments.capfloor`, `engines.capfloor.black`, `engines.capfloor.hull_white` | `ql.BlackCapFloorEngine` |
| 09 | **Swaptions** | European swaption (Black, Bachelier), Bermudan swaption (HW tree, G2++), Gaussian 1D model; calibration to market vols; dV/dVol via AD | `instruments.swaption`, `engines.swaption.black`, `engines.swaption.hull_white`, `engines.swaption.g2` | `ql.BlackSwaptionEngine`, `ql.TreeSwaptionEngine` |

**Key demos in Part 2:**
- Side-by-side table of 10 engines pricing the same European call
- `jax.hessian` producing 4×4 second-order Greek matrix in one call
- Bermudan swaption calibration: `jax.grad`-based vs QuantLib's LM optimizer

---

### Part 3: Exotic Derivatives (Notebooks 10–14)

| # | Notebook | Topics | QL-JAX Modules | QuantLib Ref |
|---|----------|--------|----------------|--------------|
| 10 | **Asian Options** | Geometric (analytic), arithmetic (Turnbull-Wakeman), discrete monitoring; MC convergence study | `instruments.asian`, `engines.analytic.asian_choi`, `engines.mc.asian` | `ql.AnalyticContinuousGeometricAsianEngine` |
| 11 | **Barrier & Double-Barrier Options** | Analytic barrier, vanna-volga barrier, perturbative barrier; static replication via vanilla portfolio | `instruments.barrier`, `engines.analytic.barrier`, `engines.analytic.double_barrier` | `ql.AnalyticBarrierEngine` |
| 12 | **Lookback & Cliquet Options** | Floating-strike lookback, fixed-strike lookback, partial lookback; cliquet (ratchet) option pricing | `instruments.lookback`, `instruments.cliquet`, `engines.analytic.lookback`, `engines.analytic.cliquet` | `ql.AnalyticContinuousFloatingLookbackEngine` |
| 13 | **Basket Options** | 2-asset Stulz analytic, moment-matching, Longstaff-Schwartz MC for American basket; correlation sensitivity via AD | `instruments.basket`, `engines.analytic.basket`, `engines.mc.american` | `ql.MCAmericanBasketEngine` |
| 14 | **Swing Options** | FD pricing on energy grid, exercise-count sensitivity, Greeks via AD; JIT speedup demonstration | `instruments.swing`, `engines.fd.swing` | `ql.FdSimpleExtOUJumpSwingEngine` |

**Key demos in Part 3:**
- MC convergence plot: paths vs price ± 2σ band
- Barrier replication error analysis
- Swing option: 14,000× JIT speedup visualization

---

### Part 4: Fixed-Income Advanced (Notebooks 15–18)

| # | Notebook | Topics | QL-JAX Modules | QuantLib Ref |
|---|----------|--------|----------------|--------------|
| 15 | **Callable Bonds** | Hull-White tree pricing, call schedule impact, dPrice/dVol and dPrice/dRate via AD | `instruments.callable_bond`, `engines.bond.callable` | `ql.CallableFixedRateBond`, `ql.TreeCallableFixedRateBondEngine` |
| 16 | **Convertible Bonds** | Binomial model, conversion ratio sensitivity, equity-credit interaction | `instruments.convertible_bond` | `ql.BinomialConvertibleEngine` |
| 17 | **Bond Forwards & Repos** | Dirty/clean forward price, spot income, repo rate sensitivity, AD Greeks, vmap batch | `instruments.forward`, `engines.bond.discounting` (bond_forward_*) | `ql.BondForward` |
| 18 | **Fitted Bond Curves** | Nelson-Siegel (4-param), Svensson (6-param), cubic B-spline; residual analysis; AD sensitivity of curve parameters | `termstructures.yield_` | `ql.FittedBondDiscountCurve` |

**Key demos in Part 4:**
- Callable bond price surface: rate × vol heatmap
- Repo forward price vs repo rate: analytical and AD verification

---

### Part 5: Credit Derivatives (Notebooks 19–22)

| # | Notebook | Topics | QL-JAX Modules | QuantLib Ref |
|---|----------|--------|----------------|--------------|
| 19 | **CDS Pricing & Hazard Curves** | Fair spread, NPV, protection/coupon legs; hazard rate bootstrap; ISDA standard model | `instruments.cds`, `engines.credit.isda`, `engines.credit.midpoint`, `termstructures.credit` | `ql.CreditDefaultSwap`, `ql.IsdaCdsEngine` |
| 20 | **CDS Greeks & Jacobians** | 6-input AD Greeks, CDS-spread Jacobian (4×4), hazard-rate Jacobian; 100-scenario batch via vmap | `engines.credit` + `jax.grad`, `jax.jacrev`, `jax.vmap` | `ql.IsdaCdsEngine` (FD bumps) |
| 21 | **Risky Bonds & CVA** | Risky bond NPV (discount + survival), risky bond Hessian (3×3); CVA-adjusted IRS (Brigo-Masetti swaption approach); AD d(CVA)/d(vol) | `engines.bond.risky`, `engines.swap.cva` | `ql.RiskyBondEngine`, `ql.CounterpartyAdjSwapEngine` |
| 22 | **Portfolio Credit: Copulas & Latent Models** | Gaussian copula, Clayton copula, one-factor Gaussian default model; expected loss, VaR, tail dependence; correlation sensitivity | `math.copulas`, latent model functions | `ql.GaussianDefProbLM` |

**Key demos in Part 5:**
- CDS spread Jacobian heatmap
- CVA term structure across 3 credit risk levels (Brigo-Masetti Table 2)
- Portfolio loss distribution histogram: independent vs correlated

---

### Part 6: Stochastic Models & Calibration (Notebooks 23–26)

| # | Notebook | Topics | QL-JAX Modules | QuantLib Ref |
|---|----------|--------|----------------|--------------|
| 23 | **Hull-White Short-Rate Model** | Bond pricing, caplet/swaption pricing; mean reversion & vol calibration via `jax.grad`-based optimizer; Monte Carlo simulation | `models.shortrate.hull_white`, `engines.fd.hull_white` | `ql.HullWhite`, `ql.Gaussian1dModel` |
| 24 | **Heston Stochastic Volatility** | Semi-analytic pricing, FD 2D PDE, Monte Carlo; calibration to market smile; implied vol surface | `models.equity.heston`, `engines.analytic.heston`, `engines.fd.heston`, `engines.mc.heston` | `ql.HestonModel`, `ql.AnalyticHestonEngine` |
| 25 | **Heston Stochastic-Local Vol (SLV)** | Leverage function, MC pricing, put-call parity, convergence to pure Heston | `models.equity.heston_slv` | `ql.HestonSLVProcess` |
| 26 | **LIBOR Market Model (LMM)** | Forward rate simulation, caplet pricing, drift correction; forward rate correlation structure | `models.marketmodels.lmm` | `ql.LiborForwardModel` |

**Key demos in Part 6:**
- Heston calibration: JAX gradient-based (seconds) vs QuantLib LM (seconds) — convergence comparison
- Implied vol smile before/after calibration overlay plot
- SLV leverage surface 3D plot

---

### Part 7: Performance & Batch Computing (Notebooks 27–29)

| # | Notebook | Topics | QL-JAX Modules | QuantLib Ref |
|---|----------|--------|----------------|--------------|
| 27 | **Vectorized Portfolio Pricing** | vmap across 10,000 options, 1,000 swap scenarios, 100 CDS scenarios; single-call batch pricing vs Python loops | `jax.vmap` + all engines | Python loops over ql.* |
| 28 | **JIT Compilation Deep Dive** | First-call overhead, trace caching, static vs dynamic shapes; when JIT helps vs hurts; recompilation triggers | `jax.jit` + all engines | N/A |
| 29 | **CPU vs GPU Scaling** | Batch size scaling (10 → 1M), GPU crossover point, memory considerations; FD grid scaling; MC path scaling | `jax.devices`, backend selection | N/A |

**Key demos in Part 7:**
- Log-log plot: batch size vs wall-clock time (CPU vs GPU)
- GPU crossover point identification
- JIT trace time vs execution time breakdown

---

### Part 8: Advanced AD Techniques (Notebooks 30–32)

| # | Notebook | Topics | QL-JAX Modules | QuantLib Ref |
|---|----------|--------|----------------|--------------|
| 30 | **First-Order Greeks: grad & jacrev** | Delta, vega, rho for options/swaps/CDS; full Jacobian matrices (zero-rate, par-rate, CDS-spread, hazard-rate); vmap(grad) for scenario risk | `jax.grad`, `jax.jacrev`, `jax.vmap` | FD bumping |
| 31 | **Second-Order Greeks: Hessian** | Gamma, vanna, volga for options; 17×17 IRS Hessian; 6×6 CDS Hessian; 18×18 cap Hessian; Hessian symmetry verification | `jax.hessian` | FD-over-FD bumping |
| 32 | **Differentiating Through Calibration** | AD through piecewise curve bootstrap; AD through model calibration (Heston); sensitivity of calibrated params to market inputs | `jax.grad` through bootstrap, calibration loops | Not possible in QL C++ |

**Key demos in Part 8:**
- 17×17 IRS Hessian heatmap
- Speedup table: AD vs FD for n-input sensitivity computation
- "Differentiating through the solver" — unique JAX capability demo

---

### Part 9: Numerical Methods (Notebook 33)

| # | Notebook | Topics | QL-JAX Modules | QuantLib Ref |
|---|----------|--------|----------------|--------------|
| 33 | **Math Toolkit: Integration, Optimization & Interpolation** | Gauss-Legendre, Simpson, adaptive quadrature; DE, SA, BFGS on test functions; linear, cubic, monotone interpolation; Sobol/Halton sequences | `math.integrals`, `math.optimization`, `math.interpolations`, `math.random` | `ql.GaussLegendreIntegration`, `ql.DifferentialEvolution` |

---

## Summary: 33 Notebooks Across 9 Parts

| Part | Notebooks | Focus |
|------|-----------|-------|
| 1. Foundations | 01–04 | Dates, curves, bonds, swaps |
| 2. Vanilla Derivatives | 05–09 | Options, caps, swaptions |
| 3. Exotic Derivatives | 10–14 | Asian, barrier, basket, swing |
| 4. Fixed-Income Advanced | 15–18 | Callable, convertible, repo, fitted curves |
| 5. Credit Derivatives | 19–22 | CDS, risky bonds, CVA, portfolio credit |
| 6. Stochastic Models | 23–26 | HW, Heston, SLV, LMM |
| 7. Performance | 27–29 | vmap, JIT, CPU/GPU scaling |
| 8. Advanced AD | 30–32 | Jacobians, Hessians, AD through calibration |
| 9. Numerical Methods | 33 | Integration, optimization, interpolation |

---

## Implementation Plan

### Phase 1: Infrastructure & Foundations (Notebooks 01–04)

**Tasks:**
1. Create `notebooks/` directory and `_common.py` shared utilities
2. Implement backend selector widget (ipywidgets dropdown for cpu/gpu)
3. Build comparison table formatter with color-coded pass/fail
4. Build timing harness with warmup and statistical summary
5. Write notebooks 01–04

**Per-notebook structure:**
```
1. Setup cell (backend selector, imports)
2. Theory cell (brief mathematical background in LaTeX)
3. QuantLib implementation (3–5 code cells)
4. ql-jax implementation (3–5 code cells)
5. Accuracy comparison table
6. Performance benchmark (timing + bar chart)
7. JAX-unique features demo (AD Greeks, vmap batch, JIT speedup)
8. Exercises (2–3 suggested extensions)
```

**Acceptance criteria:**
- All ql-jax prices match QuantLib within 1e-6 (absolute) or 0.01% (relative)
- Timing comparison shows at least 1 JIT-compiled operation
- Backend toggle works: changing `BACKEND = "gpu"` and re-running produces correct results

### Phase 2: Vanilla & Exotic Derivatives (Notebooks 05–14)

**Tasks:**
1. Write notebooks 05–09 (vanilla derivatives)
2. Write notebooks 10–14 (exotic derivatives)
3. Each notebook includes at least one `jax.grad` demo and one `jax.vmap` demo

**Key implementation notes:**
- Notebook 05: build master comparison table of all 10+ engines
- Notebook 06: visualize Hessian as heatmap
- Notebook 13: LSM convergence study (paths vs price)
- Notebook 14: demonstrate 14,000× JIT speedup on swing option

### Phase 3: Fixed-Income & Credit (Notebooks 15–22)

**Tasks:**
1. Write notebooks 15–18 (fixed-income advanced)
2. Write notebooks 19–22 (credit derivatives)
3. Include cross-instrument demos (e.g., risky bond uses both yield and credit curves)

**Key implementation notes:**
- Notebook 17: repo forward pricing uses the new `bond_forward_*` functions
- Notebook 21: CVA uses `cva_swap_brigo_masetti`, reproduce Brigo-Masetti Table 2
- Notebook 22: portfolio loss distribution comparison (Gaussian vs Clayton)

### Phase 4: Models, Performance & AD (Notebooks 23–32)

**Tasks:**
1. Write notebooks 23–26 (stochastic models & calibration)
2. Write notebooks 27–29 (performance deep dives)
3. Write notebooks 30–32 (advanced AD techniques)

**Key implementation notes:**
- Notebook 24: Heston calibration race — `jax.grad` optimizer vs QuantLib LM
- Notebook 27: scaling study up to 1M instruments
- Notebook 29: GPU crossover point analysis with log-log plot
- Notebook 32: differentiating through bootstrap — not possible in QuantLib C++

### Phase 5: Numerical Methods & Polish (Notebook 33 + review)

**Tasks:**
1. Write notebook 33 (math toolkit)
2. Review all notebooks for consistency
3. Add table of contents notebook (00_index.ipynb)
4. Test all notebooks on both CPU and GPU backends
5. Write `notebooks/README.md` with installation and running instructions

---

## File Structure

```
notebooks/
├── _common.py                        # Shared utilities
├── README.md                         # Installation & usage guide
├── 00_index.ipynb                    # Table of contents
├── 01_dates_calendars.ipynb
├── 02_yield_curves.ipynb
├── 03_bonds_fra.ipynb
├── 04_interest_rate_swaps.ipynb
├── 05_european_options.ipynb
├── 06_european_greeks_hessians.ipynb
├── 07_american_options.ipynb
├── 08_caps_floors.ipynb
├── 09_swaptions.ipynb
├── 10_asian_options.ipynb
├── 11_barrier_options.ipynb
├── 12_lookback_cliquet.ipynb
├── 13_basket_options.ipynb
├── 14_swing_options.ipynb
├── 15_callable_bonds.ipynb
├── 16_convertible_bonds.ipynb
├── 17_bond_forwards_repos.ipynb
├── 18_fitted_bond_curves.ipynb
├── 19_cds_pricing.ipynb
├── 20_cds_greeks_jacobians.ipynb
├── 21_risky_bonds_cva.ipynb
├── 22_portfolio_credit.ipynb
├── 23_hull_white_model.ipynb
├── 24_heston_model.ipynb
├── 25_heston_slv.ipynb
├── 26_libor_market_model.ipynb
├── 27_vectorized_portfolio.ipynb
├── 28_jit_deep_dive.ipynb
├── 29_cpu_gpu_scaling.ipynb
├── 30_first_order_greeks.ipynb
├── 31_second_order_greeks.ipynb
├── 32_ad_through_calibration.ipynb
└── 33_math_toolkit.ipynb
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `jupyter` / `jupyterlab` | Notebook runtime |
| `ipywidgets` | Backend selector widget |
| `matplotlib` | Charts (convergence, scaling, heatmaps) |
| `seaborn` | Heatmaps for Jacobian/Hessian matrices |
| `pandas` | Comparison tables |
| `ql-jax` | JAX quantitative finance library |
| `QuantLib-Python` (SWIG) | Reference implementation for comparison |
| `jaxlib[cuda]` (optional) | GPU backend |
