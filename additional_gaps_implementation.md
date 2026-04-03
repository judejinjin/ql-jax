# Additional Gaps Implementation Plan

**Baseline**: 860 tests, commit 32e151e  
**QuantLib C++ reference**: `/mnt/c/finance/QuantLib`  
**ql-jax implementation**: `/home/jude/ql-jax`

---

## Summary of Remaining Gaps

### Coverage Status by Category

| Category | ql-jax | QuantLib C++ | Coverage | Notes |
|----------|--------|-------------|----------|-------|
| Calendars | 46 | 46 | 100% | Complete |
| Day counters | ~10 | 12 | ~83% | Missing Actual/365.25, Actual/366 |
| IBOR/Overnight indexes | 33 factories | 43 files | ~77% | Missing bibor, bkbm, mosprime, robor, thbfix, trlibor, zibor |
| Inflation indexes | 6 | 7 | 86% | Missing ZACPI, UKHICP |
| BMA index | 0 | 1 | 0% | Completely missing |
| ASX dates | 0 | 1 | 0% | Completely missing |
| Processes | 15 | 22 | 68% | Missing hybrid Heston-HW, joint, MF state, GSR core, extended processes |
| Copulas | 0 | 13 | 0% | Completely missing |
| ODE solvers | 0 | 1 | 0% | Missing adaptive Runge-Kutta |
| Matrix utilities | 0 | 17 | 0% | Only SVD/Cholesky via JAX; missing BiCGStab, GMRES, QR, expm, etc. |
| FD operators | 3 (flat files) | 29 | ~10% | Have BS, Heston, local vol operators in flat files |
| FD meshers | 1 (flat file) | 14 | ~7% | Have basic mesher in flat file |
| FD solvers | 1 (flat file) | 16 | ~6% | Have basic solver in flat file |
| FD step conditions | 1 (flat file) | 8 | ~13% | Have American step condition |
| FD utilities | 0 | 23 | 0% | Missing all FD utility classes |
| Vanilla engines | 17 | 40+ | ~43% | Missing CEV, QD+, integral, FD Bates/CEV/CIR/SABR/Shout, hybrids |
| Asian engines | 4 | 14 | ~29% | Missing Choi, Lévy, FD, Heston MC/discrete geom |
| Barrier engines | 5 | 15 | ~33% | Missing binomial, FD Heston, perturbative, vanna-volga |
| Basket engines | 4 | 12 | ~33% | Missing Choi, DLZ, FD 2D, MC American, operator splitting |
| Swaption engines | 5 | 12 | ~42% | Missing FD G2/HW, Gaussian1d variants, basket generating |
| Cap/floor engines | 3 | 7 | ~43% | Missing Gaussian1d, MC HW, tree |
| Forward engines | 0 | 5 | 0% | Missing MC forward European (BS/Heston), forward/performance |
| Instruments | 30 | 80+ | ~38% | Missing many swap variants, callable schedule, BTP, etc. |
| Credit experimental | 0 | 45 | 0% | Missing CDO, CDS options, basket credit |
| Variance gamma | 0 | 7 | 0% | Completely missing |
| SVI volatility | 0 | 3 | 0% | Completely missing |
| No-arb SABR | 0 | 5 | 0% | Completely missing |
| CMS spread coupon | 0 | 4 | 0% | Completely missing |
| Cat bonds | 0 | 5 | 0% | Completely missing |
| CLV models | 0 | 2 | 0% | Completely missing |
| Commodities | 0 | 23 | 0% | Completely missing |
| Experimental FD | 0 | 25 | 0% | Completely missing |

---

## Phase 1: Missing IBOR/Overnight Indexes and Day Counters (Small)

**Files to create/modify**: `ql_jax/indexes/ibor.py`, `ql_jax/time/daycounter.py`
**Est. lines**: ~150

### 1a. Missing IBOR indexes (factory functions)
Add factory functions for the 10 missing indexes:
- `Bibor` (DKK IBOR — Bahrain uses BHD but QuantLib has it for Budapest)
- `BKBM` (NZD bank bill rate)
- `Mosprime` (RUB Moscow Prime)
- `Robor` (RON Romanian)
- `Thbfix` (THB Thai baht)
- `Trlibor` (TRY Turkish lira)
- `Zibor` (HRK Croatian — now EUR zone)

### 1b. Missing inflation indexes
- `ZACPI` (South Africa CPI)
- `UKHICP` (UK HICP — Harmonised Index)

### 1c. Missing day counters
- `Actual365_25` (Actual/365.25 — leap year average)
- `Actual366` (Actual/366 — always 366 denominator)

### 1d. ASX dates utility
- `ql_jax/time/asx.py` — ASX futures date calculations (like IMM dates but for Australian Securities Exchange)

### 1e. BMA index
- Add `BmaIndex` class to `ql_jax/indexes/base.py` or new file

**Tests**: ~15 tests  
**Reference**: `ql/indexes/ibor/bibor.hpp`, `ql/indexes/ibor/bkbm.hpp`, etc., `ql/time/asx.hpp`, `ql/indexes/bmaindex.hpp`

---

## Phase 2: Copula Framework (Math)

**Files to create**: `ql_jax/math/copulas/__init__.py`, individual copula files  
**Est. lines**: ~400

Implement the complete copula framework:
- `GaussianCopula` — bivariate Gaussian copula
- `ClaytonCopula` — lower tail dependence
- `FrankCopula` — symmetric dependence
- `GumbelCopula` — upper tail dependence
- `IndependentCopula` — product copula
- `MinCopula` / `MaxCopula` — Fréchet-Hoeffding bounds
- `PlackettCopula` — Plackett family
- `FarlieGumbelMorgensternCopula` — FGM copula
- `GalambosCoula` — extreme value copula
- `HuslerReissCopula` — extreme value copula
- `AliMikhailHaqCopula` — AMH copula
- `MarshallOlkinCopula` — bivariate Marshall-Olkin

Each copula implements: `operator()(u, v)` density/CDF, parameter bounds.

**Tests**: ~25 tests  
**Reference**: `ql/math/copulas/*.hpp` (574 lines total)

---

## Phase 3: ODE Solver and Matrix Utilities (Math)

**Files to create**: `ql_jax/math/ode.py`, `ql_jax/math/matrix_utilities.py`  
**Est. lines**: ~350

### 3a. Adaptive Runge-Kutta ODE solver
- 4th/5th order Dormand-Prince (RK45) with adaptive step size
- Configurable tolerance, max steps
- Used by: Heston characteristic function integration, various calibrations

### 3b. Matrix utilities
- `BiCGStab` — biconjugate gradient stabilized iterative solver
- `GMRES` — generalized minimal residual method
- `QRDecomposition` — QR factorization (Householder reflections)
- `matrix_exponential` — matrix exp via Padé approximation (or use `jax.scipy.linalg.expm`)
- `moore_penrose_inverse` — pseudoinverse (or use `jnp.linalg.pinv`)
- `factor_reduction` — PCA/factor reduction for correlation matrices

### 3c. Additional statistics
- `autocovariance` / `autocorrelation` functions
- `convolved_student_t` — convolution of Student-t distributions

**Tests**: ~20 tests  
**Reference**: `ql/math/ode/adaptiverungekutta.hpp`, `ql/math/matrixutilities/*.hpp`

---

## Phase 4: SVI and No-Arbitrage SABR Volatility (Volatility)

**Files to create**: `ql_jax/termstructures/volatility/svi.py`, `ql_jax/termstructures/volatility/noarb_sabr.py`  
**Est. lines**: ~600

### 4a. SVI (Stochastic Volatility Inspired) smile
- `SviInterpolation` — raw SVI parameterization: $w(k) = a + b(\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2})$
- `SviSmileSection` — SmileSection using SVI
- `SviInterpolatedSmileSection` — calibrated SVI smile to market quotes

### 4b. No-arbitrage SABR
- `NoArbSabrModel` — absorption-at-zero SABR with no-arbitrage correction
- `NoArbSabrInterpolation` — interpolation using no-arb SABR
- `NoArbSabrSmileSection` — calibrated smile section
- `NoArbSabrSwaptionVolatilityCube` — SABR cube with no-arb correction

**Tests**: ~20 tests  
**Reference**: `ql/experimental/volatility/svi*.hpp` (513 lines), `ql/experimental/volatility/noarb*.hpp` (799 lines)

---

## Phase 5: Variance Gamma Process and Engines (Experimental)

**Files to create**: `ql_jax/processes/variance_gamma.py`, `ql_jax/models/equity/variance_gamma.py`, `ql_jax/engines/analytic/variance_gamma.py`  
**Est. lines**: ~350

### 5a. Variance gamma process
- `VarianceGammaProcess` — subordinated Brownian motion: $X(t) = \theta G(t) + \sigma W(G(t))$ where G is gamma process
- Parameters: sigma (volatility), nu (variance rate of gamma), theta (drift of BM)

### 5b. Variance gamma model
- `VarianceGammaModel` — calibrateable model wrapping the process

### 5c. Engines
- `AnalyticVarianceGammaEngine` — closed-form European option pricing via characteristic function
- `FFTVarianceGammaEngine` — FFT-based pricing for variance gamma

**Tests**: ~15 tests  
**Reference**: `ql/experimental/variancegamma/*.hpp` (324 lines total)

---

## Phase 6: CMS Spread Coupon and Pricer (Cashflows)

**Files to create**: `ql_jax/cashflows/cms_spread.py`, `ql_jax/indexes/swap_spread.py`  
**Est. lines**: ~300

### 6a. Swap spread index
- `SwapSpreadIndex` — index defined as difference/spread of two swap rates

### 6b. CMS spread coupon
- `CmsSpreadCoupon` — coupon paying CMS spread (S1 - S2)
- `CappedFlooredCmsSpreadCoupon` — with cap/floor
- `DigitalCmsSpreadCoupon` — digital payoff on CMS spread

### 6c. Log-normal CMS spread pricer
- `LognormalCmsSpreadPricer` — copula-based pricer using bivariate normal for CMS spread coupons

**Tests**: ~15 tests  
**Reference**: `ql/experimental/coupons/cmsspreadcoupon.hpp`, `lognormalcmsspreadpricer.hpp` (291 lines total)

---

## Phase 7: Missing Vanilla Option Engines — Analytic (Engines)

**Files to create/modify**: New engine files under `ql_jax/engines/analytic/`  
**Est. lines**: ~600

### 7a. CEV analytic engine
- `AnalyticCEVEngine` — constant elasticity of variance model: $dS = \mu S dt + \sigma S^\beta dW$
- Uses non-central chi-squared distribution for pricing
- File: `ql_jax/engines/analytic/cev.py`

### 7b. Integral engine
- `IntegralEngine` — European option via direct integration of Black-Scholes PDE
- File: `ql_jax/engines/analytic/integral.py`

### 7c. QD+ American engine
- `QdPlusAmericanEngine` — Andersen-Lake-Offengenden QD+ method for American options
- High-accuracy analytic approximation with iterative boundary refinement
- File: `ql_jax/engines/analytic/qdplus_american.py`

### 7d. QDFP American engine  
- `QdFpAmericanEngine` — fixed-point iteration variant of QD method
- File: `ql_jax/engines/analytic/qdfp_american.py`

### 7e. Cash dividend European engine
- `AnalyticDividendEuropeanEngine` — European options with known cash dividends (escrowed model)
- File: `ql_jax/engines/analytic/cash_dividend_european.py`

### 7f. BSM-Hull-White hybrid engine
- `AnalyticBSMHullWhiteEngine` — equity-rates hybrid: BSM equity + HW short rate
- File: `ql_jax/engines/analytic/bsm_hull_white.py`

### 7g. Heston-Hull-White hybrid engine
- `AnalyticHestonHullWhiteEngine` — Heston stochastic vol + HW stochastic rates
- `MCHestonHullWhiteEngine` — Monte Carlo variant
- File: `ql_jax/engines/analytic/heston_hull_white.py`

### 7h. Exponential fitting Heston engine
- `ExponentialFittingHestonEngine` — Heston pricing via exponential fitting of the characteristic function
- File: `ql_jax/engines/analytic/exp_fitting_heston.py`

### 7i. Heston expansion engine
- Already exists: `ql_jax/engines/analytic/heston_variants.py`
- Verify `HestonExpansionEngine` is complete

### 7j. Analytic European Vasicek engine
- `AnalyticEuropeanVasicekEngine` — European option under Vasicek short rate
- File: `ql_jax/engines/analytic/vasicek_european.py`

**Tests**: ~30 tests  
**Reference**: `ql/pricingengines/vanilla/analyticcevengine.hpp`, `qdplusamericanengine.hpp`, etc.

---

## Phase 8: Missing Asian Engines (Engines)

**Files to modify**: `ql_jax/engines/analytic/asian.py`, `ql_jax/engines/mc/asian.py`; new FD engine  
**Est. lines**: ~500

### 8a. Choi Asian engine
- `ChoiAsianEngine` — semi-analytic arithmetic Asian via Choi (2018) — Laplace transform method
- File: `ql_jax/engines/analytic/asian_choi.py`

### 8b. Lévy arithmetic Asian engine
- `ContinuousArithmeticAsianLevyEngine` — continuous arithmetic avg price using Lévy approximation
- Add to `ql_jax/engines/analytic/asian.py`

### 8c. Turnbull-Wakeman Asian engine
- `TurnbullWakemanAsianEngine` — moment matching for arithmetic average
- File: `ql_jax/engines/analytic/asian_turnbull_wakeman.py`

### 8d. Discrete geometric average price (Heston)
- `AnalyticDiscreteGeometricAveragePriceHestonEngine` — discrete geometric avg under Heston
- `MCDiscreteGeometricAveragePriceHestonEngine` — MC variant
- Add to `ql_jax/engines/analytic/asian.py` or `ql_jax/engines/mc/asian.py`

### 8e. MC discrete arithmetic average (Heston)
- `MCDiscreteArithmeticAveragePriceHestonEngine`
- Add to `ql_jax/engines/mc/asian.py`

### 8f. FD Black-Scholes Asian engine
- `FdBlackScholesAsianEngine` — finite difference for Asian options
- File: `ql_jax/engines/fd/asian.py`

**Tests**: ~25 tests  
**Reference**: `ql/pricingengines/asian/*.hpp`

---

## Phase 9: Missing Barrier and Double Barrier Engines (Engines)

**Files to create/modify**: New files under `ql_jax/engines/`  
**Est. lines**: ~500

### 9a. Binomial barrier engine
- `BinomialBarrierEngine` — barrier options on binomial tree with barrier adjustment
- File: `ql_jax/engines/lattice/barrier_binomial.py`

### 9b. FD Heston barrier engine
- `FdHestonBarrierEngine` — FD Heston for single barrier
- `FdHestonDoubleBarrierEngine` — FD Heston for double barrier
- `FdHestonRebateEngine` — rebate pricing under Heston
- File: `ql_jax/engines/fd/heston_barrier.py`

### 9c. FD Black-Scholes rebate engine
- `FdBlackScholesRebateEngine` — rebate-only pricing
- Add to `ql_jax/engines/fd/barrier.py`

### 9d. Perturbative barrier engine
- `PerturbativeBarrierOptionEngine` — asymptotic expansion for barrier options
- File: `ql_jax/engines/analytic/barrier_perturbative.py`

### 9e. Vanna-Volga barrier engines
- `VannaVolgaBarrierEngine` — vanna-volga method for single barrier
- `VannaVolgaDoubleBarrierEngine` — double barrier variant  
- File: `ql_jax/engines/analytic/barrier_vanna_volga.py`

### 9f. MC double barrier engine
- `MCDoubleBarrierEngine` — Monte Carlo for double barriers
- File: `ql_jax/engines/mc/double_barrier.py`

### 9g. Analytical soft barrier engine
- Already have: `ql_jax/engines/analytic/barrier.py` — verify soft barrier coverage

### 9h. Suo-Wang double barrier engine
- `SuoWangDoubleBarrierEngine` — analytic double barrier via Suo-Wang (2004)
- Add to `ql_jax/engines/analytic/double_barrier.py`

### 9i. Binomial double barrier engine
- `BinomialDoubleBarrierEngine` — tree-based double barrier
- File: `ql_jax/engines/lattice/double_barrier_binomial.py`

**Tests**: ~25 tests  
**Reference**: `ql/pricingengines/barrier/*.hpp`, `ql/experimental/barrieroption/*.hpp`

---

## Phase 10: Missing Basket and Spread Engines (Engines)

**Files to create/modify**: New engine files  
**Est. lines**: ~500

### 10a. Choi basket engine
- `ChoiBasketEngine` — semi-analytic basket via Choi method
- File: `ql_jax/engines/analytic/basket_choi.py`

### 10b. Deng-Li-Zhou (DLZ) basket engine
- `DengLiZhouBasketEngine` — analytic approximation for basket options
- File: `ql_jax/engines/analytic/basket_dlz.py`

### 10c. FD 2D Black-Scholes basket engine
- `Fd2dBlackScholesVanillaEngine` — 2D finite difference for two-asset basket
- File: `ql_jax/engines/fd/basket_2d.py`

### 10d. FD N-dimensional Black-Scholes engine
- `FdNdimBlackScholesVanillaEngine` — N-dimensional FD basket engine
- File: `ql_jax/engines/fd/basket_nd.py`

### 10e. MC American basket engine
- `MCAmericanBasketEngine` — least-squares MC for American baskets
- File: `ql_jax/engines/mc/basket_american.py`

### 10f. Operator splitting spread engine
- `OperatorSplittingSpreadEngine` — operator splitting method for spread options
- File: `ql_jax/engines/analytic/spread_operator_splitting.py`

### 10g. Spread Black-Scholes vanilla engine
- `SpreadBlackScholesVanillaEngine` — BS-based spread engine
- File: `ql_jax/engines/analytic/spread_bs.py`

### 10h. Single-factor BSM basket engine
- `SingleFactorBsmBasketEngine` — moment-matching single factor
- File: `ql_jax/engines/analytic/basket_single_factor.py`

**Tests**: ~25 tests  
**Reference**: `ql/pricingengines/basket/*.hpp`

---

## Phase 11: Missing Swaption and Cap/Floor Engines (Engines)

**Files to create/modify**: New engine files  
**Est. lines**: ~500

### 11a. FD G2 swaption engine
- `FdG2SwaptionEngine` — finite difference G2++ model swaption pricing
- File: `ql_jax/engines/swaption/fd_g2.py`

### 11b. FD Hull-White swaption engine
- `FdHullWhiteSwaptionEngine` — FD Hull-White swaption
- File: `ql_jax/engines/swaption/fd_hull_white.py`

### 11c. Gaussian1d swaption engine
- `Gaussian1dSwaptionEngine` — generic Gaussian 1-factor swaption engine
- File: `ql_jax/engines/swaption/gaussian1d.py`

### 11d. Gaussian1d Jamshidian swaption engine
- `Gaussian1dJamshidianSwaptionEngine` — Jamshidian decomposition for Gaussian 1-factor
- Add to `ql_jax/engines/swaption/gaussian1d.py`

### 11e. Gaussian1d non-standard swaption engine
- `Gaussian1dNonstandardSwaptionEngine` — for non-standard swaptions (amortizing, etc.)
- File: `ql_jax/engines/swaption/gaussian1d_nonstandard.py`

### 11f. Gaussian1d float-float swaption engine
- `Gaussian1dFloatFloatSwaptionEngine` — float-float swaption pricing
- File: `ql_jax/engines/swaption/gaussian1d_float_float.py`

### 11g. Basket generating engine (Bermudan swaption)
- `BasketGeneratingEngine` — generates basket of European swaptions approximating a Bermudan
- File: `ql_jax/engines/swaption/basket_generating.py`

### 11h. Gaussian1d cap/floor engine
- `Gaussian1dCapFloorEngine` — cap/floor pricing under Gaussian 1-factor
- File: `ql_jax/engines/capfloor/gaussian1d.py`

### 11i. MC Hull-White cap/floor engine
- `MCHullWhiteCapFloorEngine` — Monte Carlo Hull-White for caps/floors
- File: `ql_jax/engines/capfloor/mc_hull_white.py`

### 11j. Tree cap/floor engine
- `TreeCapFloorEngine` — short-rate tree pricing for caps/floors
- File: `ql_jax/engines/capfloor/tree.py`

### 11k. Tree swap engine
- `TreeSwapEngine` — short-rate tree pricing for swaps
- File: `ql_jax/engines/swap/tree.py`

**Tests**: ~30 tests  
**Reference**: `ql/pricingengines/swaption/*.hpp`, `ql/pricingengines/capfloor/*.hpp`

---

## Phase 12: Forward Engines (Engines)

**Files to create**: New engine files  
**Est. lines**: ~300

### 12a. Forward engine
- `ForwardEngine` — generic forward-starting option pricer
- `ForwardPerformanceEngine` — performance (return) option pricer
- File: `ql_jax/engines/forward/__init__.py`, `ql_jax/engines/forward/forward.py`

### 12b. MC forward European BS engine
- `MCForwardEuropeanBSEngine` — MC forward-start European under Black-Scholes
- File: `ql_jax/engines/mc/forward_european.py`

### 12c. MC forward European Heston engine
- `MCForwardEuropeanHestonEngine` — MC forward-start European under Heston
- Add to `ql_jax/engines/mc/forward_european.py`

### 12d. Analytic Heston forward European engine
- `AnalyticHestonForwardEuropeanEngine` — semi-analytic Heston forward start
- File: `ql_jax/engines/analytic/heston_forward.py`

**Tests**: ~15 tests  
**Reference**: `ql/pricingengines/forward/*.hpp`, `ql/experimental/forward/*.hpp`

---

## Phase 13: FD Framework — Operators (Methods)

**Files to create**: Restructure `ql_jax/methods/finitedifferences/operators/` as proper subpackage  
**Est. lines**: ~800

Currently operators live in flat files (`bs_operator.py`, `heston_operator.py`, `operators.py`). Need to add:

### 13a. Core operator infrastructure
- `FdmLinearOp` — abstract base for all FD operators
- `FdmLinearOpLayout` — grid layout descriptor
- `FdmLinearOpIterator` — grid iterator
- `FdmLinearOpComposite` — composition of operators
- `TripleBandLinearOp` — tridiagonal banded operator
- `NinePointLinearOp` — 9-point stencil for 2D
- `FirstDerivativeOp` / `SecondDerivativeOp` — standard derivative operators
- `SecondOrderMixedDerivativeOp` — cross-derivative
- `NthOrderDerivativeOp` — higher-order derivatives
- File: `ql_jax/methods/finitedifferences/operators/base.py`

### 13b. Model-specific FD operators
- `FdmBlackScholesOp` — 1D BS operator (refactor from `bs_operator.py`)
- `Fdm2dBlackScholesOp` — 2D BS operator for basket FD
- `FdmBatesOp` — Bates model operator (Heston + jumps)
- `FdmBlackScholesFwdOp` — forward BS operator (Dupire)
- `FdmCevOp` — CEV model operator
- `FdmCirOp` — CIR model operator
- `FdmG2Op` — G2++ model operator
- `FdmHestonFwdOp` — forward Heston operator
- `FdmHestonHullWhiteOp` — Heston-HW hybrid operator
- `FdmHestonOp` — Heston operator (refactor from `heston_operator.py`)
- `FdmHullWhiteOp` — Hull-White operator
- `FdmLocalVolFwdOp` — forward local vol operator
- `FdmOrnsteinUhlenbeckOp` — OU process operator
- `FdmSabrOp` — SABR model operator
- `FdmSquareRootFwdOp` — square root forward operator
- `FdmWienerOp` — Wiener/diffusion operator
- Files: `ql_jax/methods/finitedifferences/operators/bs.py`, `heston.py`, `bates.py`, etc.

**Tests**: ~25 tests  
**Reference**: `ql/methods/finitedifferences/operators/*.hpp`

---

## Phase 14: FD Framework — Meshers (Methods)

**Files to create**: `ql_jax/methods/finitedifferences/meshers/` subpackage  
**Est. lines**: ~400

### 14a. Base mesher classes
- `Fdm1dMesher` — abstract 1D mesh
- `FdmMesher` — abstract N-D mesh
- `FdmMesherComposite` — composite of 1D meshers for N-D

### 14b. Standard 1D meshers
- `Concentrating1dMesher` — mesh concentrated around specified points
- `Uniform1dMesher` — uniform grid
- `Predefined1dMesher` — user-specified grid points
- `ExponentialJump1dMesher` — for jump processes

### 14c. Model-specific meshers
- `FdmBlackScholesMesher` — log-spot mesher for BS (refactor existing)
- `FdmBlackScholesMultiStrikeMesher` — mesher for multiple strikes
- `FdmCev1dMesher` — CEV-adapted mesh
- `FdmHestonVarianceMesher` — variance dimension mesh for Heston
- `FdmSimpleProcess1dMesher` — generic 1D process mesher
- `UniformGridMesher` — uniform N-D grid

**Tests**: ~15 tests  
**Reference**: `ql/methods/finitedifferences/meshers/*.hpp`

---

## Phase 15: FD Framework — Solvers and Utilities (Methods)

**Files to create**: `ql_jax/methods/finitedifferences/solvers/`, `ql_jax/methods/finitedifferences/utilities/`  
**Est. lines**: ~800

### 15a. Base solver classes
- `FdmBackwardSolver` — backward PDE solver
- `Fdm1DimSolver` — 1D solver
- `Fdm2DimSolver` — 2D solver
- `Fdm3DimSolver` — 3D solver
- `FdmNDimSolver` — N-D solver
- `FdmSolverDesc` — solver description/configuration

### 15b. Model-specific solvers
- `FdmBlackScholesSolver` — complete BS FD solver
- `FdmHestonSolver` — Heston FD solver
- `FdmBatesSolver` — Bates FD solver
- `FdmCirSolver` — CIR FD solver
- `FdmG2Solver` — G2++ FD solver
- `FdmHullWhiteSolver` — HW FD solver
- `FdmHestonHullWhiteSolver` — hybrid Heston-HW solver
- `FdmSimple2dBSSolver` — simple 2D BS solver

### 15c. Step conditions
- `FdmAmericanStepCondition` — early exercise (refactor existing)
- `FdmBermudanStepCondition` — Bermudan exercise
- `FdmArithmeticAverageCondition` — for Asian FD
- `FdmSimpleStorageCondition` — storage option condition
- `FdmSimpleSwingCondition` — swing option condition
- `FdmSnapshotCondition` — snapshot for intermediate values
- `FdmStepConditionComposite` — composite step conditions

### 15d. FD utilities
- `FdmDirichletBoundary` — Dirichlet BC
- `FdmDiscountDirichletBoundary` — discounted Dirichlet BC
- `FdmTimeDependentDirichletBoundary` — time-dependent BC
- `FdmDividendHandler` — discrete dividend adjustment
- `FdmQuantoHelper` — quanto adjustment
- `FdmInnerValueCalculator` — base inner value
- `FdmLogInnerValue` / `FdmEscrowedLogInnerValue` — log inner values
- `FdmShoutLogInnerValue` — shout option inner value
- `FdmBoundaryConditionSet` — BC collection
- `FdmIndicesOnBoundary` — boundary index identification
- `FdmMesherIntegral` — numerical integration on mesh
- `FdmAffineModelSwapInnerValue` — for rate model FD
- `FdmAffineModelTermStructure` — TS for FD rate models

### 15e. Risk-neutral density (RND) calculators
- `BsmRndCalculator` — BS model RND
- `GBsmRndCalculator` — generalized BS RND
- `CevRndCalculator` — CEV model RND
- `HestonRndCalculator` — Heston model RND
- `LocalVolRndCalculator` — local vol RND
- `SquareRootProcessRndCalculator` — CIR/square root RND
- `FdmHestonGreensFct` — Heston Green's function

**Tests**: ~30 tests  
**Reference**: `ql/methods/finitedifferences/solvers/*.hpp`, `utilities/*.hpp`, `stepconditions/*.hpp`

---

## Phase 16: FD Vanilla Engines (Engines)

**Files to create/modify**: FD engine files  
**Est. lines**: ~500

Using the FD framework from Phases 13-15, implement:

### 16a. FD Bates vanilla engine
- `FdBatesVanillaEngine` — Bates model FD pricing for vanilla options
- File: `ql_jax/engines/fd/bates.py`

### 16b. FD CEV vanilla engine
- `FdCevVanillaEngine` — CEV model FD pricing
- File: `ql_jax/engines/fd/cev.py`

### 16c. FD CIR vanilla engine
- `FdCirVanillaEngine` — CIR equity model FD pricing
- File: `ql_jax/engines/fd/cir.py`

### 16d. FD SABR vanilla engine
- `FdSabrVanillaEngine` — SABR model FD pricing
- File: `ql_jax/engines/fd/sabr.py`

### 16e. FD Black-Scholes shout engine
- `FdBlackScholesShoutEngine` — shout option (lock-in) via FD
- File: `ql_jax/engines/fd/shout.py`

### 16f. FD Heston-Hull-White vanilla engine
- `FdHestonHullWhiteVanillaEngine` — 3-factor FD hybrid
- File: `ql_jax/engines/fd/heston_hull_white.py`

### 16g. FD simple BS swing engine
- `FdSimpleBsSwingEngine` — swing option FD under BS
- File: `ql_jax/engines/fd/swing.py`

**Tests**: ~25 tests  
**Reference**: `ql/pricingengines/vanilla/fd*.hpp`

---

## Phase 17: Missing Instruments (Instruments)

**Files to create/modify**: Instrument files  
**Est. lines**: ~600

### 17a. Float-float swap
- `FloatFloatSwap` — swap with two floating legs (different indexes/currencies)
- File: `ql_jax/instruments/float_float_swap.py`

### 17b. Float-float swaption
- `FloatFloatSwaption` — option on FloatFloatSwap
- Add to `ql_jax/instruments/swaption.py`

### 17c. Non-standard swap/swaption
- `NonstandardSwap` — swap with non-standard features (amortizing, step-up, etc.)
- `NonstandardSwaption` — option on NonstandardSwap
- File: `ql_jax/instruments/nonstandard_swap.py`

### 17d. Multiple resets swap
- `MultipleResetsSwap` — swap based on overnight index with multiple resets per period
- File: `ql_jax/instruments/multiple_resets_swap.py` (or extend existing `ql_jax/cashflows/multiple_resets.py`)

### 17e. BMA swap
- `BmaSwap` — swap vs BMA/SIFMA municipal bond index
- File: `ql_jax/instruments/bma_swap.py`

### 17f. Callability schedule
- `CallabilitySchedule` — schedule of call/put dates with prices for callable bonds
- `Callability` — single call/put provision
- File: `ql_jax/instruments/callability.py`

### 17g. Bond forward
- `BondForward` — forward contract on a bond
- Add to `ql_jax/instruments/bond.py` or `ql_jax/instruments/forward.py`

### 17h. Equity total return swap
- `EquityTotalReturnSwap` — TRS on equity
- File: `ql_jax/instruments/equity_trs.py`

### 17i. FX forward (if missing)
- `FxForward` — spot + forward points FX forward
- Check and add to `ql_jax/instruments/forward.py`

### 17j. Zero coupon swap
- `ZeroCouponSwap` — swap with zero coupon fixed leg
- File: `ql_jax/instruments/zero_coupon_swap.py`

### 17k. Complex chooser option
- `ComplexChooserOption` — chooser with different call/put strikes and maturities
- Add to `ql_jax/instruments/chooser.py`

### 17l. Perpetual futures
- `PerpetualFutures` — perpetual futures contract (crypto-style)
- Add to `ql_jax/instruments/futures.py`

### 17m. Vanilla storage/swing options
- `VanillaStorageOption` — energy storage option
- `VanillaSwingOption` — swing option with multiple exercise rights
- File: `ql_jax/instruments/swing.py`

### 17n. Forward vanilla option
- `ForwardVanillaOption` — forward-starting vanilla option instrument
- Check existing `ql_jax/instruments/forward_start.py` for completeness

**Tests**: ~25 tests  
**Reference**: `ql/instruments/*.hpp`

---

## Phase 18: Missing Processes (Processes)

**Files to create/modify**: Process files  
**Est. lines**: ~300

### 18a. Hybrid Heston-Hull-White process
- `HybridHestonHullWhiteProcess` — 3-factor hybrid process
- File: `ql_jax/processes/hybrid_heston_hw.py`

### 18b. Joint stochastic process
- `JointStochasticProcess` — composition of multiple stochastic processes
- File: `ql_jax/processes/joint.py`

### 18c. GSR process (core)
- `GsrProcess` — Gaussian short rate process internals
- `GsrProcessCore` — core implementation
- Verify/extend `ql_jax/models/shortrate/gsr.py`

### 18d. Markov functional state process
- `MfStateProcess` — state variable process for Markov functional model
- Add to `ql_jax/models/shortrate/markov_functional.py`

### 18e. Forward measure process
- `ForwardMeasureProcess` — process under forward measure
- File: `ql_jax/processes/forward_measure.py`

### 18f. Extended OU process
- `ExtendedOrnsteinUhlenbeckProcess` — OU with time-dependent parameters
- `ExtOUWithJumpsProcess` — OU with jumps (for energy/commodities)
- File: `ql_jax/processes/extended_ou.py`

### 18g. Vega-stressed BS process
- `VegaStressedBlackScholesProcess` — BS with stressed volatility for risk
- File: `ql_jax/processes/vega_stressed_bs.py`

### 18h. End-of-period Euler discretization
- `EndEulerDiscretization` — Euler discretization at end of period
- Add to `ql_jax/processes/discretization.py`

**Tests**: ~15 tests  
**Reference**: `ql/processes/*.hpp`, `ql/experimental/processes/*.hpp`

---

## Phase 19: Missing Term Structures — Yield (Term Structures)

**Files to create/modify**: Term structure files  
**Est. lines**: ~400

### 19a. OIS rate helper enhancements
- `OvernightIndexFutureRateHelper` — rate helper based on overnight index futures
- `MultipleResetsSwapHelper` — rate helper for multiple-reset OIS swaps
- Add to `ql_jax/termstructures/yield_/rate_helpers.py`

### 19b. Piecewise forward-spreaded term structure
- `PiecewiseForwardSpreadedTermStructure` — piecewise forward spreads
- Add to `ql_jax/termstructures/yield_/spread_curves.py`

### 19c. Piecewise spread yield curve
- `PiecewiseSpreadYieldCurve` — bootstrapped spread curve
- File: `ql_jax/termstructures/yield_/piecewise_spread.py`

### 19d. Spread discount curve
- `SpreadDiscountCurve` — discount curve built from additive spreads
- Add to `ql_jax/termstructures/yield_/spread_curves.py`

### 19e. Interpolated simple zero curve
- `InterpolatedSimpleZeroCurve` — zero curve with simple (non-continuous) compounding
- Add to `ql_jax/termstructures/yield_/zero_curve.py`

### 19f. Spread bootstrap traits
- `SpreadBootstrapTraits` — traits for spread bootstrapping
- Add to `ql_jax/termstructures/yield_/piecewise.py`

**Tests**: ~15 tests  
**Reference**: `ql/termstructures/yield/*.hpp`

---

## Phase 20: Missing Term Structures — Volatility (Term Structures)

**Files to create/modify**: Volatility term structure files  
**Est. lines**: ~500

### 20a. Gaussian1d swaption volatility
- `Gaussian1dSwaptionVolatility` — swaption vol surface from Gaussian 1-factor model
- `Gaussian1dSmileSection` — smile section from Gaussian 1-factor
- File: `ql_jax/termstructures/volatility/gaussian1d_vol.py`

### 20b. Caplet variance curve
- `CapletVarianceCurve` — term structure of caplet variances
- Add to `ql_jax/termstructures/volatility/capfloor_vol.py`

### 20c. Grid model local vol surface
- `GridModelLocalVolSurface` — local vol from pre-computed grid
- Add to `ql_jax/termstructures/volatility/local_vol_surface.py`

### 20d. Heston black vol surface
- `HestonBlackVolSurface` — implied Black vol from Heston model
- Add to or verify in `ql_jax/termstructures/volatility/heston_vol.py`

### 20e. CMS market / CMS market calibration
- `CmsMarket` — CMS market data container
- `CmsMarketCalibration` — calibrate CMS model to market
- File: `ql_jax/termstructures/volatility/cms_market.py`

### 20f. Kahale smile section
- `KahaleSmileSection` — arbitrage-free smile extrapolation (Kahale method)
- File: `ql_jax/termstructures/volatility/kahale_smile.py`

### 20g. ATM smile section
- `AtmSmileSection` — at-the-money section
- `AtmAdjustedSmileSection` — ATM-adjusted section
- Add to `ql_jax/termstructures/volatility/smile_section.py`

### 20h. Interpolated swaption volatility cube
- `InterpolatedSwaptionVolatilityCube` — interpolating swaption vol cube (non-SABR)
- Add to `ql_jax/termstructures/volatility/swaption_vol.py`

### 20i. Piecewise black variance surface
- `PiecewiseBlackVarianceSurface` — piecewise parametric variance surface
- File: `ql_jax/termstructures/volatility/piecewise_black_var.py`

### 20j. Black vol surface delta
- `BlackVolSurfaceDelta` — vol surface parameterized by delta
- Add to `ql_jax/termstructures/volatility/equityfx_extended.py`

### 20k. Black vol time extrapolation
- `BlackVolTimeExtrapolation` — flat/linear time extrapolation wrapper
- Add to `ql_jax/termstructures/volatility/black_vol.py`

**Tests**: ~25 tests  
**Reference**: `ql/termstructures/volatility/*.hpp`

---

## Phase 21: Credit Experimental (Experimental)

**Files to create**: `ql_jax/experimental/credit/`  
**Est. lines**: ~800

### 21a. CDO / Synthetic CDO
- `Cdo` — collateralized debt obligation instrument
- `SyntheticCdo` — synthetic CDO
- File: `ql_jax/experimental/credit/cdo.py`

### 21b. CDS option
- `CdsOption` — option on credit default swap
- `BlackCdsOptionEngine` — Black-formula CDS option engine
- File: `ql_jax/experimental/credit/cds_option.py`

### 21c. Credit basket and loss models
- `Basket` — credit default basket
- `DefaultLossModel` — abstract loss model
- `BinomialLossModel` — binomial loss model
- `GaussianLHPLossModel` — large homogeneous pool
- `BaseCorrelationLossModel` — base correlation mapping
- File: `ql_jax/experimental/credit/basket.py`, `loss_models.py`

### 21d. Nth-to-default swap
- Via basket + tranche definition
- Add to `ql_jax/experimental/credit/basket.py`

### 21e. Correlation structures
- `BaseCorrelationStructure` — base correlation term structure
- `CorrelationStructure` — generic correlation TS
- File: `ql_jax/experimental/credit/correlation.py`

### 21f. Default probability latent models
- `DefaultProbabilityLatentModel` — factor model for correlated defaults
- `ConstantLossLatentModel` — constant loss given default
- File: `ql_jax/experimental/credit/latent_model.py`

### 21g. Factor-spreaded hazard rate curve
- `FactorSpreadedHazardRateCurve` — hazard rate curve with factor spread
- Add to `ql_jax/termstructures/credit/`

**Tests**: ~25 tests  
**Reference**: `ql/experimental/credit/*.hpp` (45 files)

---

## Phase 22: Callable Bonds (Experimental)

**Files to create/modify**: Bond engine files  
**Est. lines**: ~350

### 22a. Callable fixed rate bond
- `CallableFixedRateBond` — bond with embedded call/put options
- File: extend `ql_jax/instruments/bond.py` or `ql_jax/instruments/callable_bond.py`

### 22b. Black callable bond engine
- `BlackCallableFixedRateBondEngine` — Black model for callable bonds
- Add to `ql_jax/engines/bond/callable.py`

### 22c. Tree callable bond engine
- `TreeCallableFixedRateBondEngine` — short-rate tree pricing for callable bonds
- File: `ql_jax/engines/bond/callable_tree.py`

### 22d. Callable bond volatility structure
- `CallableBondConstantVolatility` — constant callable bond vol
- `CallableBondVolatilityStructure` — abstract callable bond vol TS
- File: `ql_jax/termstructures/volatility/callable_bond_vol.py`

**Tests**: ~15 tests  
**Reference**: `ql/experimental/callablebonds/*.hpp` (7 files)

---

## Phase 23: Inflation Experimental (Experimental)

**Files to create/modify**: Inflation files  
**Est. lines**: ~400

### 23a. CPI cap/floor term price surface
- `CPICapFloorTermPriceSurface` — market prices for CPI caps/floors
- File: `ql_jax/experimental/inflation/cpi_term_price_surface.py`

### 23b. YoY cap/floor term price surface
- `YoYCapFloorTermPriceSurface` — market prices for YoY inflation caps/floors
- File: `ql_jax/experimental/inflation/yoy_term_price_surface.py`

### 23c. YoY optionlet stripper
- `InterpolatedYoYOptionletStripper` — strip YoY optionlet vols from cap prices
- `YoYOptionletStripper` — base YoY optionlet stripper
- File: `ql_jax/experimental/inflation/yoy_optionlet_stripper.py`

### 23d. YoY optionlet helpers
- `YoYOptionletHelper` — calibration helper for YoY optionlets
- File: `ql_jax/experimental/inflation/yoy_optionlet_helpers.py`

### 23e. Capped/floored inflation coupon
- `CappedFlooredYoYInflationCoupon` — inflation coupon with cap/floor
- `CappedFlooredZeroInflationCashFlow` — zero inflation with cap/floor
- Add to `ql_jax/cashflows/inflation.py`

### 23f. Additional inflation indexes
- Generic inflation indexes for less common regions
- Add to `ql_jax/indexes/inflation.py`

### 23g. Polynomial 2D spline interpolation
- `Polynomial2DSpline` — bivariate spline for inflation surfaces
- Add to `ql_jax/math/interpolations/`

**Tests**: ~20 tests  
**Reference**: `ql/experimental/inflation/*.hpp` (12 files)

---

## Phase 24: Exotic Options Experimental (Experimental)

**Files to create**: Exotic option files  
**Est. lines**: ~400

### 24a. Himalaya option
- `HimalayaOption` — multi-asset path-dependent option
- `MCHimalayaEngine` — Monte Carlo pricing
- File: `ql_jax/experimental/exotics/himalaya.py`

### 24b. Everest option
- `EverestOption` — worst-of basket option
- `MCEverestEngine` — Monte Carlo pricing
- File: `ql_jax/experimental/exotics/everest.py`

### 24c. Pagoda option
- `PagodaOption` — accumulated increment option
- `MCPagodaEngine` — Monte Carlo pricing
- File: `ql_jax/experimental/exotics/pagoda.py`

### 24d. Path-dependent basket / MC path basket
- `PathMultiAssetOption` — generic path-dependent multi-asset
- `MCPathBasketEngine` — MC pricing
- `MCAmericanPathEngine` — Longstaff-Schwartz for path-dependent
- File: `ql_jax/experimental/exotics/path_basket.py`

### 24e. Spread option (market-standard)
- `SpreadOption` — generic spread option
- Add to `ql_jax/instruments/basket.py` or new file

### 24f. Continuous arithmetic Asian Vecer engine
- `ContinuousArithmeticAsianVecerEngine` — Vecer PDE approach
- File: `ql_jax/engines/analytic/asian_vecer.py`

**Tests**: ~20 tests  
**Reference**: `ql/experimental/exoticoptions/*.hpp`, `ql/experimental/mcbasket/*.hpp`

---

## Phase 25: CLV Models (Experimental)

**Files to create**: CLV model files  
**Est. lines**: ~250

### 25a. Normal CLV model
- `NormalCLVModel` — collocation local vol model using normal distribution
- File: `ql_jax/models/equity/normal_clv.py`

### 25b. Square-root CLV model
- `SquareRootCLVModel` — CLV model using square root kernel
- File: `ql_jax/models/equity/sqrt_clv.py`

### 25c. Stochastic collocation inverse CDF
- Helper functions for CLV model construction
- Add to `ql_jax/math/` or within CLV model files

**Tests**: ~10 tests  
**Reference**: `ql/experimental/models/*.hpp` (214 lines total)

---

## Phase 26: Energy/Commodities Framework (Experimental)

**Files to create**: `ql_jax/experimental/commodities/`  
**Est. lines**: ~400

### 26a. Commodity types and settings
- `Commodity` — base commodity instrument
- `CommodityType` — commodity type definition
- `CommoditySettings` — singleton settings
- `CommodityIndex` — commodity price index
- `CommodityCurve` — commodity forward curve
- File: `ql_jax/experimental/commodities/base.py`

### 26b. Energy instruments
- `EnergyCommodity` — energy commodity base
- `EnergyFuture` — energy future
- `EnergySwap` — energy swap
- `EnergyVanillaSwap` — fixed-float energy swap
- `EnergyBasisSwap` — basis swap between two energy indexes
- File: `ql_jax/experimental/commodities/energy.py`

### 26c. Commodity cash flows
- `CommodityCashFlow` — commodity-linked cash flow
- `CommodityUnitCost` — unit cost for commodity accounting
- File: `ql_jax/experimental/commodities/cashflows.py`

### 26d. Related processes
- `GemanRoncoroniProcess` — mean-reverting jump-diffusion for energy
- `KlugeExtOUProcess` — Kluge model (OU + spikes)
- `ExtOUWithJumpsProcess` — extended OU with Poisson jumps
- Add to or extend `ql_jax/processes/`

**Tests**: ~15 tests  
**Reference**: `ql/experimental/commodities/*.hpp` (23 files)

---

## Phase 27: Cat Bonds (Experimental)

**Files to create**: `ql_jax/experimental/catbonds/`  
**Est. lines**: ~300

### 27a. Catastrophe risk modeling
- `CatRisk` — catastrophe event simulation
- `BetaRisk` — beta-distributed loss model
- `EventSetRisk` — event-set based risk
- File: `ql_jax/experimental/catbonds/cat_risk.py`

### 27b. Cat bond instrument
- `CatBond` — catastrophe bond instrument
- `RiskyNotional` — notional at risk
- File: `ql_jax/experimental/catbonds/cat_bond.py`

### 27c. Monte Carlo cat bond engine
- `MonteCarloCatBondEngine` — MC pricing for cat bonds
- File: `ql_jax/experimental/catbonds/mc_engine.py`

**Tests**: ~10 tests  
**Reference**: `ql/experimental/catbonds/*.hpp` (5 files)

---

## Phase 28: Average OIS and Irregular Swaptions (Experimental)

**Files to create/modify**: Various  
**Est. lines**: ~300

### 28a. Arithmetic average OIS
- `ArithmeticAverageOIS` — OIS swap with arithmetic (vs geometric) averaging
- `MakeArithmeticAverageOIS` — builder
- `ArithmeticOISRateHelper` — bootstrap helper
- File: `ql_jax/experimental/averageois/`

### 28b. Irregular swaption
- `IrregularSwap` — swap with irregular (non-standard) features
- `IrregularSwaption` — swaption on irregular swap
- `HaganIrregularSwaptionEngine` — Hagan's method for irregular swaptions
- File: `ql_jax/experimental/swaptions/`

### 28c. Basis models
- `SwaptionCashFlows` — cash flow decomposition for swaption
- `TenorOptionletVTS` — optionlet vol mapped to different tenor
- `TenorSwaptionVTS` — swaption vol mapped to different tenor
- File: `ql_jax/experimental/basismodels/`

**Tests**: ~15 tests  
**Reference**: `ql/experimental/averageois/*.hpp`, `swaptions/*.hpp`, `basismodels/*.hpp`

---

## Phase 29: Experimental FD (Energy/Commodity FD)

**Files to create**: `ql_jax/experimental/finitedifferences/`  
**Est. lines**: ~500

### 29a. Extended OU FD operators
- `FdmExtendedOrnsteinUhlenbeckOp` — FD operator for extended OU
- `FdmExtOUJumpOp` — OU with jumps operator
- `FdmKlugeExtOUOp` — Kluge model operator
- `FdmDupire1dOp` — Dupire local vol 1D operator
- `FdmZabrOp` — ZABR model operator

### 29b. Extended OU FD solvers
- `FdmExtOUJumpSolver` — solver for OU + jumps
- `FdmKlugeExtOUSolver` — Kluge solver
- `FdmSimple2dExtOUSolver` — 2D OU solver
- `FdmSimple3dExtOUJumpSolver` — 3D OU + jump solver

### 29c. Energy/commodity FD engines
- `FdExtOUJumpVanillaEngine` — vanilla on OU + jumps
- `FdSimpleExtOUStorageEngine` — storage option
- `FdSimpleExtOUJumpSwingEngine` — swing option
- `FdSimpleKlugeExtOUVPPEngine` — virtual power plant
- `FdOrnsteinUhlenbeckVanillaEngine` — vanilla on OU
- `FdKlugeExtOUSpreadEngine` — spread on Kluge + OU

### 29d. VPP (virtual power plant)
- `VanillaVPPOption` — VPP option instrument
- `DynProgVPPIntrinsicValueEngine` — dynamic programming
- `FdmVPPStepCondition` — VPP step conditions

### 29e. Glued 1D mesher
- `Glued1dMesher` — mesher combining two 1D meshers at a junction point

**Tests**: ~15 tests  
**Reference**: `ql/experimental/finitedifferences/*.hpp` (25 files)

---

## Phase 30: Experimental Math and Optimization (Experimental)

**Files to create/modify**: Math files  
**Est. lines**: ~400

### 30a. Multi-dimensional quadrature
- `MultidimIntegrator` — base N-D integrator
- `MultidimQuadrature` — Gauss quadrature in N dimensions
- File: `ql_jax/math/integrals/multidim.py`

### 30b. Copula-based RNG
- `ClaytonCopulaRng` — RNG using Clayton copula
- `FrankCopulaRng` — RNG using Frank copula
- `FarlieGumbelMorgensternCopulaRng` — FGM copula RNG
- `GaussianCopulaPolicy` / `TCopulaPolicy` — copula policies for credit models
- File: `ql_jax/math/random/copula_rng.py`

### 30c. Advanced optimization
- `FireflyAlgorithm` — nature-inspired metaheuristic
- `ParticleSwarmOptimization` — PSO optimizer
- `HybridSimulatedAnnealing` — hybrid SA with local search
- `IsotropicRandomWalk` — isotropic random walk sampler
- File: `ql_jax/math/optimization/metaheuristic.py`

### 30d. Special distributions
- `PolarStudentTRng` — polar method for Student-t
- `LevyFlightDistribution` — Lévy flight distribution
- `GaussianNonCentralChiSquaredPolynomial` — non-central chi-sq moments
- Add to `ql_jax/math/distributions/`

### 30e. Laplace interpolation
- `LaplaceInterpolation` — Laplace equation-based interpolation for missing data
- Add to `ql_jax/math/interpolations/`

### 30f. Piecewise function/integral
- `PiecewiseFunction` — piecewise-defined function
- `PiecewiseIntegral` — integral of piecewise function
- Add to `ql_jax/math/`

### 30g. Vanna-Volga interpolation
- `VannaVolgaInterpolation` — for FX barrier pricing
- Add to `ql_jax/math/interpolations/`

**Tests**: ~20 tests  
**Reference**: `ql/experimental/math/*.hpp` (23 files)

---

## Phase 31: Short Rate Model Extensions (Models)

**Files to create/modify**: Model files  
**Est. lines**: ~200

### 31a. Generalized Hull-White
- `GeneralizedHullWhite` — multi-factor generalized HW model
- File: `ql_jax/models/shortrate/generalized_hull_white.py`

### 31b. Generalized OU process
- `GeneralizedOrnsteinUhlenbeckProcess` — OU with general mean reversion
- Add to `ql_jax/processes/` or `ql_jax/experimental/shortrate/`

### 31c. Square root Andersen
- `SquareRootAndersen` — Andersen's quadratic exponential scheme for CIR/Heston
- Add to `ql_jax/processes/discretization.py`

**Tests**: ~10 tests  
**Reference**: `ql/experimental/shortrate/*.hpp`

---

## Phase 32: Miscellaneous Missing Cashflows and Coupons (Cashflows)

**Files to create/modify**: Cashflow files  
**Est. lines**: ~250

### 32a. Cap/floored inflation coupon
- `CappedFlooredInflationCoupon` — inflation coupon with embedded cap/floor
- Verify coverage in `ql_jax/cashflows/inflation.py`, add if missing

### 32b. Stripped cap/floored coupon
- `StrippedCapFlooredCoupon` — strips cap/floor from a coupon to get pure coupon vs optionlet
- File: `ql_jax/cashflows/stripped.py`

### 32c. Quanto coupon pricer
- `QuantoCouponPricer` — FX-adjusted coupon pricer
- File: `ql_jax/cashflows/quanto_pricer.py`

### 32d. Linear TSR pricer
- `LinearTsrPricer` — terminal swap rate (TSR) model for CMS coupon pricing
- Add to `ql_jax/cashflows/cms.py`

### 32e. Black overnight indexed coupon pricer
- `BlackOvernightIndexedCouponPricer` — Black model for overnight coupons
- Add to `ql_jax/cashflows/overnight_pricer.py`

### 32f. Rate averaging
- `RateAveraging` — compound vs simple averaging types
- Add to `ql_jax/cashflows/` if not present

### 32g. Dividend schedule
- `DividendSchedule` — container for discrete dividend schedule
- Add to `ql_jax/cashflows/dividend.py`

**Tests**: ~15 tests  
**Reference**: `ql/cashflows/*.hpp`, `ql/experimental/coupons/*.hpp`

---

## Phase 33: Optionlet Stripping Extensions (Term Structures)

**Files to create/modify**: Optionlet files  
**Est. lines**: ~300

### 33a. OptionletStripper2
- `OptionletStripper2` — flat vol optionlet stripper (alternative to OptionletStripper1)
- Add to `ql_jax/termstructures/volatility/optionlet_stripper.py`

### 33b. Stripped optionlet adapter
- `StrippedOptionletAdapter` — adapts stripped optionlets to OptionletVolatilityStructure
- `StrippedOptionlet` — container for stripped optionlet data
- Add to `ql_jax/termstructures/volatility/optionlet_extended.py`

### 33c. Spreaded optionlet vol
- `SpreadedOptionletVolatility` — optionlet vol with additive spread
- Add to `ql_jax/termstructures/volatility/spreaded.py`

### 33d. ZABR swaption volatility cube
- `ZabrSwaptionVolatilityCube` — swaption vol cube using ZABR interpolation
- Add to `ql_jax/termstructures/volatility/sabr_cube.py`

### 33e. YoY inflation optionlet vol structures
- `YoYInflationOptionletVolatilityStructure` — vol surface for YoY inflation
- `KInterpolatedYoYOptionletVolatilitySurface` — interpolated YoY vol
- `PiecewiseYoYOptionletVolatility` — piecewise YoY vol
- File: `ql_jax/termstructures/volatility/yoy_optionlet_vol.py`

**Tests**: ~15 tests  
**Reference**: `ql/termstructures/volatility/optionlet*.hpp`, `ql/experimental/inflation/*.hpp`

---

## Phase 34: Variance Option and Remaining Analytic Engines (Experimental)

**Files to create**: Various  
**Est. lines**: ~250

### 34a. Variance option
- `VarianceOption` — option on realized variance (NOT variance swap)
- `IntegralHestonVarianceOptionEngine` — integral method under Heston
- File: `ql_jax/experimental/varianceoption/`

### 34b. MC digital engine
- `MCDigitalEngine` — Monte Carlo for digital/binary options
- Add to `ql_jax/engines/mc/`

### 34c. MC European GJRGARCH engine
- `MCEuropeanGJRGARCHEngine` — MC for GJRGARCH model
- Add to `ql_jax/engines/mc/`

### 34d. MC performance engine
- `MCPerformanceEngine` — MC for cliquet/performance options
- Add to `ql_jax/engines/mc/`

### 34e. Discounting FX forward engine
- `DiscountingFxForwardEngine` — discounting engine for FX forwards
- File: `ql_jax/engines/forward/discounting_fx.py`

**Tests**: ~15 tests  
**Reference**: `ql/experimental/varianceoption/*.hpp`, `ql/pricingengines/vanilla/mc*.hpp`

---

## Phase 35: FX Experimental (Experimental)

**Files to create**: `ql_jax/experimental/fx/`  
**Est. lines**: ~150

### 35a. Black delta calculator
- `BlackDeltaCalculator` — delta and strike conversions for FX options
- File: `ql_jax/experimental/fx/black_delta.py`

### 35b. Garman-Kohlhagen process
- Verify `ql_jax/processes/black_scholes.py` covers GK (BS with foreign rate)
- May already be covered via `GeneralizedBlackScholesProcess`

**Tests**: ~10 tests  
**Reference**: `ql/experimental/fx/*.hpp`

---

## Phase Summary

| Phase | Description | Est. Lines | Tests |
|-------|-------------|-----------|-------|
| 1 | Missing indexes, day counters, ASX, BMA | 150 | 15 |
| 2 | Copula framework | 400 | 25 |
| 3 | ODE solver, matrix utilities | 350 | 20 |
| 4 | SVI and no-arb SABR | 600 | 20 |
| 5 | Variance gamma | 350 | 15 |
| 6 | CMS spread coupon | 300 | 15 |
| 7 | Missing vanilla engines (analytic) | 600 | 30 |
| 8 | Missing Asian engines | 500 | 25 |
| 9 | Missing barrier engines | 500 | 25 |
| 10 | Missing basket engines | 500 | 25 |
| 11 | Missing swaption/capfloor engines | 500 | 30 |
| 12 | Forward engines | 300 | 15 |
| 13 | FD operators | 800 | 25 |
| 14 | FD meshers | 400 | 15 |
| 15 | FD solvers and utilities | 800 | 30 |
| 16 | FD vanilla engines | 500 | 25 |
| 17 | Missing instruments | 600 | 25 |
| 18 | Missing processes | 300 | 15 |
| 19 | Missing yield term structures | 400 | 15 |
| 20 | Missing volatility term structures | 500 | 25 |
| 21 | Credit experimental | 800 | 25 |
| 22 | Callable bonds | 350 | 15 |
| 23 | Inflation experimental | 400 | 20 |
| 24 | Exotic options experimental | 400 | 20 |
| 25 | CLV models | 250 | 10 |
| 26 | Commodities framework | 400 | 15 |
| 27 | Cat bonds | 300 | 10 |
| 28 | Average OIS, irregular swaptions | 300 | 15 |
| 29 | Experimental FD (energy) | 500 | 15 |
| 30 | Experimental math/optimization | 400 | 20 |
| 31 | Short rate model extensions | 200 | 10 |
| 32 | Missing cashflows/coupons | 250 | 15 |
| 33 | Optionlet stripping extensions | 300 | 15 |
| 34 | Variance option, remaining engines | 250 | 15 |
| 35 | FX experimental | 150 | 10 |
| **Total** | | **~14,300** | **~680** |

---

## Implementation Priority Order

**Tier 1 — Core math and framework** (Phases 1-3, 13-15): Build foundation first — indexes, copulas, ODE, FD framework infrastructure.

**Tier 2 — Volatility modeling** (Phases 4, 20, 33): SVI, no-arb SABR, additional vol surfaces/cubes, optionlet stripping.

**Tier 3 — Core engines** (Phases 7-12, 16): Fill significant engine gaps — vanilla, Asian, barrier, basket, swaption, forward, FD engines.

**Tier 4 — Instruments and cashflows** (Phases 5-6, 17, 32): New instruments, variance gamma, CMS spread, missing cashflows.

**Tier 5 — Term structures and processes** (Phases 18-19): Fill yield/process gaps.

**Tier 6 — Experimental** (Phases 21-31, 34-35): Credit, callable bonds, inflation, exotics, CLV, commodities, cat bonds, energy FD.
