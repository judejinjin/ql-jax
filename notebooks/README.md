# ql-jax Demo Notebooks

33 Jupyter notebooks demonstrating **ql-jax** — a JAX reimplementation of QuantLib C++.

Every notebook compares ql-jax against QuantLib SWIG, showing identical numerical results with the added benefits of:
- **Automatic Differentiation** — exact Greeks via `jax.grad`, `jax.hessian`
- **Vectorization** — batch pricing via `jax.vmap`
- **JIT Compilation** — XLA-compiled execution via `jax.jit`
- **GPU Acceleration** — seamless CPU/GPU backend switching

## Quick Start

```bash
cd ql-jax
source .venv/bin/activate
jupyter lab notebooks/
```

## Notebook Index

### Part 1 — Foundations
| # | Notebook | Topics |
|---|----------|--------|
| 01 | [Dates & Calendars](notebooks/01_dates_calendars.ipynb) | Date arithmetic, calendars, day counters, schedules |
| 02 | [Yield Curves](notebooks/02_yield_curves.ipynb) | Flat forward, piecewise bootstrap, AD through curves |

### Part 2 — Fixed Income
| # | Notebook | Topics |
|---|----------|--------|
| 03 | [Bonds & FRA](notebooks/03_bonds_fra.ipynb) | Zero/fixed bonds, AD duration/convexity, FRA |
| 04 | [Interest Rate Swaps](notebooks/04_interest_rate_swaps.ipynb) | Vanilla IRS, cashflows, DV01 via AD |

### Part 3 — Vanilla Options
| # | Notebook | Topics |
|---|----------|--------|
| 05 | [European Options](notebooks/05_european_options.ipynb) | BSM, Heston, FD, binomial, MC — 10 engines |
| 06 | [Greeks & Hessians](notebooks/06_european_greeks_hessians.ipynb) | Full Hessian, Greek surfaces, AD vs FD |
| 07 | [American Options](notebooks/07_american_options.ipynb) | BAW, Bjerksund-Stensland, FD, LSM |
| 08 | [Caps, Floors & Collars](notebooks/08_caps_floors.ipynb) | Black model, put-call parity, vol sensitivity |
| 09 | [Swaptions](notebooks/09_swaptions.ipynb) | Black, Bachelier, Hull-White, vol surface |

### Part 4 — Exotic Options
| # | Notebook | Topics |
|---|----------|--------|
| 10 | [Asian Options](notebooks/10_asian_options.ipynb) | Geometric, Turnbull-Wakeman, Choi, Lévy, MC |
| 11 | [Barrier Options](notebooks/11_barrier_options.ipynb) | Single/double barrier, FD, MC, in-out parity |
| 12 | [Lookback & Cliquet](notebooks/12_lookback_cliquet.ipynb) | Floating/fixed lookback, forward-start, cliquet |
| 13 | [Basket Options](notebooks/13_basket_options.ipynb) | Stulz, moment matching, MC, Kirk's spread |
| 14 | [Swing Options](notebooks/14_swing_options.ipynb) | FD swing, multiple exercises, JIT speedup |

### Part 5 — Advanced Fixed Income
| # | Notebook | Topics |
|---|----------|--------|
| 15 | [Callable Bonds](notebooks/15_callable_bonds.ipynb) | Black model, Hull-White tree, yield vol |
| 16 | [Convertible Bonds](notebooks/16_convertible_bonds.ipynb) | Binomial tree, parity, stock sensitivity |
| 17 | [Bond Forwards](notebooks/17_bond_forwards.ipynb) | Forward dirty/clean pricing, rate sensitivities |
| 18 | [Fitted Curves](notebooks/18_fitted_curves.ipynb) | Nelson-Siegel, Svensson, exponential splines |

### Part 6 — Credit
| # | Notebook | Topics |
|---|----------|--------|
| 19 | [CDS](notebooks/19_cds.ipynb) | Midpoint, analytics, ISDA engines, bootstrapping |
| 20 | [CDS Greeks](notebooks/20_cds_greeks.ipynb) | CS01, IR01, recovery, Hessian cross-gammas |
| 21 | [Risky Bonds & CVA](notebooks/21_risky_bonds_cva.ipynb) | Risky bond NPV, CVA swap, counterparty risk |
| 22 | [Portfolio Credit](notebooks/22_portfolio_credit.ipynb) | CDO tranches, LHP, NthToDefault, correlation |

### Part 7 — Stochastic Models
| # | Notebook | Topics |
|---|----------|--------|
| 23 | [Hull-White](notebooks/23_hull_white.ipynb) | ZCB, caplet, bond option, BSM-HW hybrid |
| 24 | [Heston](notebooks/24_heston.ipynb) | Fourier, COS, expansion, MC QE, FD ADI, Bates |
| 25 | [Stochastic Local Vol](notebooks/25_slv.ipynb) | Leverage calibration, SLV MC, Heston-HW hybrid |
| 26 | [LIBOR Market Model](notebooks/26_lmm.ipynb) | Path simulation, Euler vs PC, correlation, AD |

### Part 8 — Performance
| # | Notebook | Topics |
|---|----------|--------|
| 27 | [Vectorized Portfolio](notebooks/27_vectorized_portfolio.ipynb) | vmap batch pricing, portfolio Greeks, scaling |
| 28 | [JIT Deep Dive](notebooks/28_jit_deep_dive.ipynb) | Tracing, static args, recompilation, HLO |
| 29 | [CPU/GPU Scaling](notebooks/29_cpu_gpu_scaling.ipynb) | Cross-over point, MC scaling, transfer overhead |

### Part 9 — Advanced AD
| # | Notebook | Topics |
|---|----------|--------|
| 30 | [First-Order Greeks](notebooks/30_first_order_greeks.ipynb) | jacfwd, Greek surfaces, AD vs FD accuracy |
| 31 | [Second-Order Greeks](notebooks/31_second_order_greeks.ipynb) | Full Hessian, Gamma surface, third-order |
| 32 | [AD Through Calibration](notebooks/32_ad_through_calibration.ipynb) | End-to-end differentiation, implicit function theorem |
| 33 | [Math Toolkit](notebooks/33_math_toolkit.ipynb) | Interpolation, solvers, optimization, copulas, distributions |

## Requirements

- Python 3.12+
- JAX (with optional GPU support)
- QuantLib-SWIG 1.42+ (for comparison cells)
- matplotlib, numpy
