# QL-JAX vs QuantLib C++ Gap Analysis & Implementation Plan

## Executive Summary

The ql-jax project currently implements **~141 Python source files** covering the core quantitative finance functionality. The original QuantLib C++ library contains **1,000+ source files** across 25+ modules. This document identifies every significant gap between the two implementations, organized by module, with priority levels and estimated complexity.

**Coverage summary:**

| Module | QuantLib C++ | ql-jax Implemented | Coverage |
|--------|-------------|-------------------|----------|
| Time/Calendars | 48 calendars, 13 day counters | 10 calendars, ~7 day counters | ~20% |
| Instruments | 75+ instrument types | 15 instrument types | ~20% |
| Pricing Engines | 130+ engines | 20 engines | ~15% |
| Stochastic Processes | 21 process types | 2 processes | ~10% |
| Models (short-rate) | 8 models | 4 models | 50% |
| Models (equity) | 7 model variants | 1 model (Heston) | ~14% |
| Models (market models) | 125 files (LMM/SMM) | 1 file (basic LMM) | <1% |
| Term Structures | 100+ types | 25 types | ~25% |
| Math/Interpolation | 22 methods | 5 methods | ~23% |
| Math/Solvers | 9 solvers | 4 solvers | ~44% |
| Math/Optimization | 12 methods | 3 methods | ~25% |
| Math/Distributions | 8 distribution types | 3 distributions | ~38% |
| Math/Random | 22+ generators | 2 (pseudo + quasi) | ~9% |
| Indexes | 60+ index definitions | 16 indexes | ~27% |
| Cashflows | 35 coupon/cashflow types | 6 types | ~17% |
| FD Methods | 80+ files (meshers, operators, schemes) | 1 file (basic BS) | ~1% |
| Lattice Methods | 10+ tree types | 1 (binomial) | ~10% |
| Quotes | 4+ quote types | 1 (SimpleQuote) | ~25% |
| Examples | 20 programs | 3 programs | 15% |
| Tests | 150+ files | 36 files | ~24% |

---

## Phase 9: Missing Instruments

### Priority 1 — High-demand instruments

| # | Instrument | QuantLib Header(s) | Target File | Complexity |
|---|-----------|-------------------|-------------|------------|
| 9.1 | LookbackOption (fixed/floating strike) | `lookbackoption.hpp` | `ql_jax/instruments/lookback.py` | Medium |
| 9.2 | BasketOption | `basketoption.hpp`, `multiassetoption.hpp` | `ql_jax/instruments/basket.py` | Medium |
| 9.3 | CliquetOption | `cliquetoption.hpp` | `ql_jax/instruments/cliquet.py` | Medium |
| 9.4 | VarianceSwap | `varianceswap.hpp` | `ql_jax/instruments/variance_swap.py` | Medium |
| 9.5 | DoubleBarrierOption | `doublebarrieroption.hpp` | `ql_jax/instruments/double_barrier.py` | Medium |
| 9.6 | ConvertibleBond | `convertiblebond.hpp` (experimental) | `ql_jax/instruments/convertible_bond.py` | High |
| 9.7 | CallableBond (enhanced) | `callablebonds/` | `ql_jax/instruments/callable_bond.py` | Medium |
| 9.8 | OvernightIndexedSwap (enhanced) | `overnightindexedswap.hpp` | update `ql_jax/instruments/swap.py` | Low |
| 9.9 | CPISwap / ZeroCouponInflationSwap / YoYInflationSwap | `cpiswap.hpp`, `zerocouponinflationswap.hpp`, `yearonyearinflationswap.hpp` | `ql_jax/instruments/inflation_swap.py` | Medium |
| 9.10 | CPICapFloor | `cpicapfloor.hpp` | `ql_jax/instruments/cpi_capfloor.py` | Medium |

### Priority 2 — Exotic & structured products

| # | Instrument | QuantLib Header(s) | Target File | Complexity |
|---|-----------|-------------------|-------------|------------|
| 9.11 | CompoundOption | `compoundoption.hpp` | `ql_jax/instruments/compound.py` | Medium |
| 9.12 | ChooserOption (simple + complex) | `simplechooseroption.hpp`, `complexchooseroption.hpp` | `ql_jax/instruments/chooser.py` | Medium |
| 9.13 | QuantoVanillaOption / QuantoBarrierOption | `quantovanillaoption.hpp`, `quantobarrieroption.hpp` | `ql_jax/instruments/quanto.py` | Medium |
| 9.14 | MargrabeOption (exchange) | `margrabeoption.hpp` | `ql_jax/instruments/margrabe.py` | Low |
| 9.15 | TwoAssetBarrierOption / TwoAssetCorrelationOption | `twoassetbarrieroption.hpp`, `twoassetcorrelationoption.hpp` | `ql_jax/instruments/two_asset.py` | Medium |
| 9.16 | SoftBarrierOption / PartialTimeBarrierOption | `softbarrieroption.hpp`, `partialtimebarrieroption.hpp` | `ql_jax/instruments/partial_barrier.py` | Medium |
| 9.17 | WriterExtensibleOption / HolderExtensibleOption | `writerextensibleoption.hpp`, `holderextensibleoption.hpp` | `ql_jax/instruments/extensible.py` | Medium |
| 9.18 | AssetSwap | `assetswap.hpp` | `ql_jax/instruments/asset_swap.py` | Medium |
| 9.19 | BMASwap / FloatFloatSwap | `bmaswap.hpp`, `floatfloatswap.hpp` | `ql_jax/instruments/basis_swap.py` | Medium |
| 9.20 | NonStandardSwap / NonStandardSwaption | `nonstandardswap.hpp`, `nonstandardswaption.hpp` | `ql_jax/instruments/nonstandard_swap.py` | High |
| 9.21 | EquityTotalReturnSwap | `equitytotalreturnswap.hpp` | `ql_jax/instruments/equity_trs.py` | Medium |
| 9.22 | VanillaSwingOption / VanillaStorageOption | `vanillaswingoption.hpp`, `vanillastorageoption.hpp` | `ql_jax/instruments/swing.py` | High |
| 9.23 | ForwardVanillaOption | `forwardvanillaoption.hpp` | `ql_jax/instruments/forward_start.py` | Low |
| 9.24 | StickyRatchet | `stickyratchet.hpp` | `ql_jax/instruments/sticky_ratchet.py` | High |

**Tests:** `tests/test_lookback.py`, `tests/test_basket.py`, `tests/test_exotics.py`, `tests/test_inflation_instruments.py`

---

## Phase 10: Missing Stochastic Processes

| # | Process | QuantLib Header | Target File | Complexity |
|---|---------|----------------|-------------|------------|
| 10.1 | BatesProcess | `batesprocess.hpp` | `ql_jax/processes/bates.py` | Medium |
| 10.2 | Merton76Process (jump-diffusion) | `merton76process.hpp` | `ql_jax/processes/merton76.py` | Medium |
| 10.3 | GJRGARCHProcess | `gjrgarchprocess.hpp` | `ql_jax/processes/gjrgarch.py` | Medium |
| 10.4 | HullWhiteProcess | `hullwhiteprocess.hpp` | `ql_jax/processes/hull_white.py` | Medium |
| 10.5 | G2Process | `g2process.hpp` | `ql_jax/processes/g2.py` | Medium |
| 10.6 | CoxIngersollRossProcess | `coxingersollrossprocess.hpp` | `ql_jax/processes/cir.py` | Low |
| 10.7 | OrnsteinUhlenbeckProcess | `ornsteinuhlenbeckprocess.hpp` | `ql_jax/processes/ornstein_uhlenbeck.py` | Low |
| 10.8 | GeometricBrownianMotionProcess | `geometricbrownianprocess.hpp` | `ql_jax/processes/gbm.py` | Low |
| 10.9 | SquareRootProcess | `squarerootprocess.hpp` | `ql_jax/processes/square_root.py` | Low |
| 10.10 | HestonSLVProcess | `hestonslvprocess.hpp` | `ql_jax/processes/heston_slv.py` | High |
| 10.11 | HybridHestonHullWhiteProcess | `hybridhestonhullwhiteprocess.hpp` | `ql_jax/processes/hybrid_heston_hw.py` | High |
| 10.12 | GSRProcess | `gsrprocess.hpp` | `ql_jax/processes/gsr.py` | Medium |
| 10.13 | StochasticProcessArray | `stochasticprocessarray.hpp` | `ql_jax/processes/process_array.py` | Medium |
| 10.14 | Euler/EndEuler Discretization | `eulerdiscretization.hpp`, `endeulerdiscretization.hpp` | `ql_jax/processes/discretization.py` | Low |
| 10.15 | ForwardMeasureProcess | `forwardmeasureprocess.hpp` | `ql_jax/processes/forward_measure.py` | Medium |
| 10.16 | MfStateProcess | `mfstateprocess.hpp` | `ql_jax/processes/mf_state.py` | Medium |

**Tests:** `tests/test_processes.py`

---

## Phase 11: Missing Pricing Engines

### 11A — Analytic Engines

| # | Engine | QuantLib Header | Target File | Complexity |
|---|--------|----------------|-------------|------------|
| 11.1 | BatesEngine | `batesengine.hpp` | `ql_jax/engines/analytic/bates.py` | High |
| 11.2 | AnalyticBarrierEngine (enhanced) | `analyticbarrierengine.hpp` | update `ql_jax/engines/analytic/` | Low |
| 11.3 | AnalyticDoubleBarrierEngine | `analyticdoublebarrierengine.hpp` | `ql_jax/engines/analytic/double_barrier.py` | Medium |
| 11.4 | AnalyticCliquetEngine | `analyticcliquetengine.hpp` | `ql_jax/engines/analytic/cliquet.py` | Medium |
| 11.5 | AnalyticCompoundOptionEngine | `analyticcompoundoptionengine.hpp` | `ql_jax/engines/analytic/compound.py` | Medium |
| 11.6 | AnalyticSimpleChooserEngine / ComplexChooser | `analyticsimplechooserengine.hpp`, `analyticcomplexchooserengine.hpp` | `ql_jax/engines/analytic/chooser.py` | Medium |
| 11.7 | Analytic lookback engines (fixed/floating/partial) | `analyticcontinuousfixedlookback.hpp` etc. | `ql_jax/engines/analytic/lookback.py` | Medium |
| 11.8 | AnalyticEuropeanMargrabeEngine / AmericanMargrabe | `analyticeuropeanmargrabeengine.hpp` | `ql_jax/engines/analytic/margrabe.py` | Low |
| 11.9 | AnalyticTwoAssetBarrierEngine / CorrelationEngine | `analytictwoassetbarrierengine.hpp`, `analytictwoassetcorrelationengine.hpp` | `ql_jax/engines/analytic/two_asset.py` | Medium |
| 11.10 | AnalyticPerformanceEngine / ForwardPerformance | `analyticperformanceengine.hpp` | `ql_jax/engines/analytic/performance.py` | Low |
| 11.11 | KirkEngine (spread option) | `kirkengine.hpp` | `ql_jax/engines/analytic/kirk.py` | Low |
| 11.12 | AnalyticHolderExtensibleEngine / WriterExtensible | `analyticholderextensibleoptionengine.hpp` | `ql_jax/engines/analytic/extensible.py` | Medium |
| 11.13 | AnalyticSoftBarrierEngine | `analyticsoftbarrierengine.hpp` | `ql_jax/engines/analytic/soft_barrier.py` | Medium |
| 11.14 | AnalyticPartialTimeBarrierEngine | `analyticpartialtimebarrieroptionengine.hpp` | `ql_jax/engines/analytic/partial_barrier.py` | Medium |
| 11.15 | AnalyticDigitalAmericanEngine | `analyticdigitalamericanengine.hpp` | `ql_jax/engines/analytic/digital_american.py` | Medium |
| 11.16 | AnalyticGJRGARCHEngine | `analyticgjrgarchengine.hpp` | `ql_jax/engines/analytic/gjrgarch.py` | Medium |
| 11.17 | AnalyticCEVEngine | `analyticcevengine.hpp` | `ql_jax/engines/analytic/cev.py` | Medium |
| 11.18 | JumpDiffusionEngine (Merton76) | `jumpdiffusionengine.hpp` | `ql_jax/engines/analytic/jump_diffusion.py` | Medium |
| 11.19 | AnalyticEuropeanVasicekEngine | `analyticeuropeanvasicekengine.hpp` | `ql_jax/engines/analytic/vasicek_option.py` | Medium |
| 11.20 | AnalyticBSMHullWhiteEngine / H1HWEngine | `analyticbsmhullwhiteengine.hpp`, `analytich1hwengine.hpp` | `ql_jax/engines/analytic/bsm_hw.py` | High |
| 11.21 | HestonExpansionEngine | `hestonexpansionengine.hpp` | `ql_jax/engines/analytic/heston_expansion.py` | Medium |
| 11.22 | CosHestonEngine | `coshestonengine.hpp` | update existing heston engine | Low |
| 11.23 | AnalyticPTDHestonEngine | `analyticptdhestonengine.hpp` | `ql_jax/engines/analytic/ptd_heston.py` | High |
| 11.24 | ChoiAsianEngine / TurnbullWakemanAsianEngine | `choiasianengine.hpp`, `turnbullwakemanasianengine.hpp` | `ql_jax/engines/analytic/asian_advanced.py` | Medium |
| 11.25 | QdPlusAmericanEngine / QdFpAmericanEngine | `qdplusamericanengine.hpp` | `ql_jax/engines/analytic/american_analytic.py` | High |
| 11.26 | BaroneAdesiWhaleyEngine / BjerksundStenslandEngine | `baroneadesiwhaleyengine.hpp`, `bjerksundstenslandengine.hpp` | `ql_jax/engines/analytic/american_approx.py` | Medium |
| 11.27 | JuQuadraticEngine | `juquadraticengine.hpp` | `ql_jax/engines/analytic/ju_quadratic.py` | Medium |
| 11.28 | IntegralEngine | `integralengine.hpp` | `ql_jax/engines/analytic/integral.py` | Low |
| 11.29 | Basket engines (Stulz, DengLiZhou, SingleFactorBSM) | `stulzengine.hpp`, `denglizhoubasketengine.hpp` | `ql_jax/engines/analytic/basket.py` | High |
| 11.30 | ReplicatingVarianceSwapEngine / MCVarianceSwapEngine | `replicatingvarianceswapengine.hpp`, `mcvarianceswapengine.hpp` | `ql_jax/engines/analytic/variance_swap.py` | Medium |
| 11.31 | BachelierCapFloorEngine | `bacheliercapfloorengine.hpp` | update `ql_jax/engines/capfloor/` | Low |
| 11.32 | BlackDeltaCalculator / BachelierCalculator | `blackdeltacalculator.hpp`, `bacheliercalculator.hpp` | `ql_jax/engines/analytic/delta_calculator.py` | Low |

### 11B — Monte Carlo Engines

| # | Engine | QuantLib Header | Target File | Complexity |
|---|--------|----------------|-------------|------------|
| 11.33 | MCEuropeanHestonEngine | `mceuropeanhestonengine.hpp` | update `ql_jax/engines/mc/european.py` | Low |
| 11.34 | MCEuropeanGJRGARCHEngine | `mceuropeangjrgarchengine.hpp` | `ql_jax/engines/mc/gjrgarch.py` | Medium |
| 11.35 | MCLookbackEngine | `mclookbackengine.hpp` | `ql_jax/engines/mc/lookback.py` | Medium |
| 11.36 | MCDigitalEngine | `mcdigitalengine.hpp` | `ql_jax/engines/mc/digital.py` | Low |
| 11.37 | MCPerformanceEngine | `mcperformanceengine.hpp` | `ql_jax/engines/mc/performance.py` | Low |
| 11.38 | MCEuropeanBasketEngine / MCAmericanBasketEngine | `mceuropeanbasketengine.hpp`, `mcamericanbasketengine.hpp` | `ql_jax/engines/mc/basket.py` | High |
| 11.39 | MCForwardEuropeanBSEngine / HestonForward | `mcforwardeuropeanbsengine.hpp`, `mcforwardeuropeanhestonengine.hpp` | `ql_jax/engines/mc/forward_start.py` | Medium |
| 11.40 | MCHestonHullWhiteEngine | `mchestonhullwhiteengine.hpp` | `ql_jax/engines/mc/heston_hw.py` | High |
| 11.41 | MCHullWhiteEngine (callable bonds) | `mchullwhiteengine.hpp` | `ql_jax/engines/mc/hull_white.py` | Medium |
| 11.42 | MC Discrete Asian (Arithmetic/Geometric) Heston | `mc_discr_arith_av_price_heston.hpp` | update `ql_jax/engines/mc/asian.py` | Medium |

### 11C — Finite Difference Engines

| # | Engine | QuantLib Header | Target File | Complexity |
|---|--------|----------------|-------------|------------|
| 11.43 | FdBlackScholesBarrierEngine | `fdblackscholesbarrierengine.hpp` | `ql_jax/engines/fd/barrier.py` | Medium |
| 11.44 | FdBlackScholesAsianEngine | `fdblackscholesasianengine.hpp` | `ql_jax/engines/fd/asian.py` | Medium |
| 11.45 | FdBlackScholesRebateEngine | `fdblackscholesrebateengine.hpp` | `ql_jax/engines/fd/rebate.py` | Medium |
| 11.46 | FdBlackScholesShoutEngine | `fdblackscholesshoutengine.hpp` | `ql_jax/engines/fd/shout.py` | Medium |
| 11.47 | FdHestonVanillaEngine | `fdhestonvanillaengine.hpp` | `ql_jax/engines/fd/heston.py` | High |
| 11.48 | FdHestonBarrierEngine / DoubleBarrier | `fdhestonbarrierengine.hpp`, `fdhestondoublebarrierengine.hpp` | `ql_jax/engines/fd/heston_barrier.py` | High |
| 11.49 | FdBatesVanillaEngine | `fdbatesvanillaengine.hpp` | `ql_jax/engines/fd/bates.py` | High |
| 11.50 | FdCEVVanillaEngine | `fdcevvanillaengine.hpp` | `ql_jax/engines/fd/cev.py` | Medium |
| 11.51 | FdSABRVanillaEngine | `fdsabrvanillaengine.hpp` | `ql_jax/engines/fd/sabr.py` | High |
| 11.52 | FdCIRVanillaEngine | `fdcirvanillaengine.hpp` | `ql_jax/engines/fd/cir.py` | Medium |
| 11.53 | FdHullWhiteSwaptionEngine | `fdhullwhiteswaptionengine.hpp` | `ql_jax/engines/fd/hw_swaption.py` | High |
| 11.54 | FdG2SwaptionEngine | `fdg2swaptionengine.hpp` | `ql_jax/engines/fd/g2_swaption.py` | High |
| 11.55 | FdHestonHullWhiteVanillaEngine | `fdhestonhullwhitevanillaengine.hpp` | `ql_jax/engines/fd/heston_hw.py` | Very High |
| 11.56 | Fd2dBlackScholesVanillaEngine | `fd2dblackscholesvanillaengine.hpp` | `ql_jax/engines/fd/bs2d.py` | High |
| 11.57 | FdNdimBlackScholesVanillaEngine | `fdndimblackscholesvanillaengine.hpp` | `ql_jax/engines/fd/bs_ndim.py` | Very High |
| 11.58 | FdSimpleBSSwingEngine | `fdsimplebsswingengine.hpp` | `ql_jax/engines/fd/swing.py` | High |

### 11D — Lattice / Tree Engines

| # | Engine | QuantLib Header | Target File | Complexity |
|---|--------|----------------|-------------|------------|
| 11.59 | TrinomialTree | `trinomialtree.hpp` | `ql_jax/engines/lattice/trinomial.py` | Medium |
| 11.60 | BinomialBarrierEngine | `binomialbarrierengine.hpp` | `ql_jax/engines/lattice/barrier.py` | Medium |
| 11.61 | BinomialConvertibleEngine | `binomialconvertibleengine.hpp` | `ql_jax/engines/lattice/convertible.py` | High |
| 11.62 | TreeSwapEngine | `treeswapengine.hpp` | `ql_jax/engines/lattice/swap.py` | Medium |
| 11.63 | TreeSwaptionEngine | `treeswaptionengine.hpp` | `ql_jax/engines/lattice/swaption.py` | Medium |
| 11.64 | TreeCapFloorEngine | `treecapfloorengine.hpp` | `ql_jax/engines/lattice/capfloor.py` | Medium |
| 11.65 | LatticeShortRateModelEngine | `latticeshortratemodelengine.hpp` | `ql_jax/engines/lattice/short_rate.py` | Medium |

### 11E — Swaption / IR Engines

| # | Engine | QuantLib Header | Target File | Complexity |
|---|--------|----------------|-------------|------------|
| 11.66 | G2SwaptionEngine (analytic) | `g2swaptionengine.hpp` | `ql_jax/engines/swaption/g2.py` | Medium |
| 11.67 | JamshidianSwaptionEngine | `jamshidianswaptionengine.hpp` | `ql_jax/engines/swaption/jamshidian.py` | Medium |
| 11.68 | Gaussian1dSwaptionEngine / Jamshidian / NonstandardSwaption | `gaussian1dswaptionengine.hpp`, `gaussian1djamshidianswaptionengine.hpp`, `gaussian1dnonstandardswaptionengine.hpp` | `ql_jax/engines/swaption/gaussian1d.py` | High |
| 11.69 | Gaussian1dCapFloorEngine | `gaussian1dcapfloorengine.hpp` | `ql_jax/engines/capfloor/gaussian1d.py` | Medium |
| 11.70 | Gaussian1dFloatFloatSwaptionEngine | `gaussian1dfloatfloatswaptionengine.hpp` | `ql_jax/engines/swaption/float_float.py` | High |

### 11F — Credit / Forward / Other Engines

| # | Engine | QuantLib Header | Target File | Complexity |
|---|--------|----------------|-------------|------------|
| 11.71 | IntegralCdsEngine | `integralcdsengine.hpp` | `ql_jax/engines/credit/integral.py` | Medium |
| 11.72 | IsdaCdsEngine | `isdacdsengine.hpp` | `ql_jax/engines/credit/isda.py` | High |
| 11.73 | DiscountingFXForwardEngine | `discountingfxforwardengine.hpp` | `ql_jax/engines/forward/fx_forward.py` | Low |
| 11.74 | ForwardEngine / ForwardPerformanceEngine | `forwardengine.hpp`, `forwardperformanceengine.hpp` | `ql_jax/engines/forward/forward.py` | Low |
| 11.75 | CVASwapEngine | `cvaswapengine.hpp` | `ql_jax/engines/swap/cva.py` | High |
| 11.76 | QuantoEngine | `quantoengine.hpp` | `ql_jax/engines/analytic/quanto.py` | Medium |
| 11.77 | OperatorSplittingSpreadEngine | `operatorsplittingspreadengine.hpp` | `ql_jax/engines/analytic/spread_splitting.py` | High |
| 11.78 | InflationCapFloorEngines (enhanced) | `inflationcapfloorengines.hpp` | update `ql_jax/engines/inflation/` | Medium |
| 11.79 | RiskyBondEngine | `riskybondengine.hpp` | `ql_jax/engines/bond/risky.py` | Medium |

**Tests:** `tests/test_engines_analytic.py`, `tests/test_engines_mc.py`, `tests/test_engines_fd.py`, `tests/test_engines_lattice.py`

---

## Phase 12: Missing Models

### 12A — Short-Rate Models

| # | Model | QuantLib Header | Target File | Complexity |
|---|-------|----------------|-------------|------------|
| 12.1 | BlackKarasinski | `blackkarasinski.hpp` | `ql_jax/models/shortrate/black_karasinski.py` | Medium |
| 12.2 | ExtendedCoxIngersollRoss | `extendedcoxingersollross.hpp` | `ql_jax/models/shortrate/extended_cir.py` | Medium |
| 12.3 | Gaussian1dModel (base) | `gaussian1dmodel.hpp` | `ql_jax/models/shortrate/gaussian1d.py` | Medium |
| 12.4 | GSR (Gaussian short-rate) | `gsr.hpp` | `ql_jax/models/shortrate/gsr.py` | High |
| 12.5 | MarkovFunctional | `markovfunctional.hpp` | `ql_jax/models/shortrate/markov_functional.py` | Very High |

### 12B — Equity / Volatility Models

| # | Model | QuantLib Header | Target File | Complexity |
|---|-------|----------------|-------------|------------|
| 12.6 | BatesModel (with doubly-stochastic model) | `batesmodel.hpp` | `ql_jax/models/equity/bates.py` | High |
| 12.7 | GJRGARCHModel | `gjrgarchmodel.hpp` | `ql_jax/models/equity/gjrgarch.py` | Medium |
| 12.8 | HestonSLVFdmModel | `hestonslvfdmmodel.hpp` | `ql_jax/models/equity/heston_slv_fdm.py` | Very High |
| 12.9 | HestonSLVMCModel | `hestonslvmcmodel.hpp` | `ql_jax/models/equity/heston_slv_mc.py` | High |
| 12.10 | PiecewiseTimeDependentHestonModel | `piecewisetimedependenthestonmodel.hpp` | `ql_jax/models/equity/ptd_heston.py` | High |

### 12C — Market Models (LMM/SMM expansion)

The current ql-jax implementation has a single `lmm.py` with basic forward simulation. QuantLib's market models library has **125 files**, including:

| # | Component Group | Key QuantLib Files | Target File(s) | Complexity |
|---|---------------|-------------------|----------------|------------|
| 12.11 | CurveState (LMM & Coterminal) | `curvestates/` (6 files) | `ql_jax/models/marketmodels/curvestate.py` | Medium |
| 12.12 | Drift computation | `driftcomputation/`, `driftcalculator.hpp` | `ql_jax/models/marketmodels/drift.py` | Medium |
| 12.13 | Evolvers (Euler/PC for LMM and SMM) | `evolvers/` (8+ files) | `ql_jax/models/marketmodels/evolvers.py` | High |
| 12.14 | Brownian generators (Sobol, general) | `browniangenerators/` (4 files) | `ql_jax/models/marketmodels/brownian.py` | Medium |
| 12.15 | Multi-product composites | `products/`, `multiproduct*` (10+ files) | `ql_jax/models/marketmodels/products.py` | High |
| 12.16 | Correlation models | `correlations/` (5+ files) | `ql_jax/models/marketmodels/correlations.py` | Medium |
| 12.17 | Volatility models (abcd, calibrated) | `models/` (8+ files) | `ql_jax/models/marketmodels/vol_models.py` | Medium |
| 12.18 | Pathwise accounting engine + proxy greeks | `pathwiseaccountingengine.hpp`, `proxygreekengine.hpp` | `ql_jax/models/marketmodels/pathwise.py` | Very High |
| 12.19 | SMM (Swap Market Model) | `swapforwardmappings.hpp`, evolvers | `ql_jax/models/marketmodels/smm.py` | High |

### 12D — Calibration Helpers

| # | Helper | QuantLib Header | Target File | Complexity |
|---|--------|----------------|-------------|------------|
| 12.20 | HestonModelHelper (enhanced) | `hestonmodelhelper.hpp` | update `ql_jax/models/equity/heston.py` | Low |
| 12.21 | SwaptionHelper | `calibrationhelper.hpp` | `ql_jax/models/calibration_helpers.py` | Medium |
| 12.22 | CapHelper | `calibrationhelper.hpp` | update calibration_helpers | Medium |

**Tests:** `tests/test_models_advanced.py`, `tests/test_market_models.py`, `tests/test_gaussian1d.py`

---

## Phase 13: Finite Difference Framework

The current ql-jax has a single `fd/black_scholes.py` with basic Crank-Nicolson. QuantLib's FD framework (`ql/methods/finitedifferences/`) has **80+ files**:

| # | Component | QuantLib Files | Target File(s) | Complexity |
|---|-----------|---------------|----------------|------------|
| 13.1 | 1D Mesher framework | `fdm1dmesher.hpp`, `uniform1dmesher.hpp`, `concentrating1dmesher.hpp`, `fdmblackscholesmesher.hpp`, `fdmcev1dmesher.hpp` | `ql_jax/methods/fd/meshers.py` | Medium |
| 13.2 | Composite mesher (multi-dim) | `fdmmeshercomposite.hpp`, `uniformgridmesher.hpp` | `ql_jax/methods/fd/mesher_composite.py` | Medium |
| 13.3 | Linear operators (tridiagonal, 9-point) | `tridiagonaloperator.hpp`, `triplebandlinearop.hpp`, `ninepointlinearop.hpp`, `fdmlinearop*.hpp` | `ql_jax/methods/fd/operators.py` | High |
| 13.4 | BS operator (1D/2D) | `fdmblackscholesop.hpp`, `fdm2dblackscholesop.hpp`, `fdmblackscholesfwdop.hpp` | `ql_jax/methods/fd/bs_operator.py` | Medium |
| 13.5 | Heston operator | `fdmhestonop.hpp`, `fdmhestonfwdop.hpp`, `fdmhestonvariancemesher.hpp` | `ql_jax/methods/fd/heston_operator.py` | High |
| 13.6 | Other model operators (HW, G2, CIR, CEV, SABR, OU) | `fdmhullwhiteop.hpp`, `fdmg2op.hpp`, `fdmcirop.hpp`, `fdmcevop.hpp`, `fdmsabrop.hpp`, `fdmornsteinuhlenbeckop.hpp` | `ql_jax/methods/fd/model_operators.py` | High |
| 13.7 | Time-stepping schemes | `cranknicolsonscheme.hpp`, `douglasscheme.hpp`, `hundsdorferscheme.hpp`, `craigsneydscheme.hpp`, `impliciteulerscheme.hpp`, `expliciteulerscheme.hpp`, `trbdf2scheme.hpp`, `methodoflinesscheme.hpp` | `ql_jax/methods/fd/schemes.py` | High |
| 13.8 | Boundary conditions | `fdmdirichletboundary.hpp`, `fdmdiscountdirichletboundary.hpp`, `fdmtimedepdirichletboundary.hpp`, `boundarycondition.hpp` | `ql_jax/methods/fd/boundary.py` | Medium |
| 13.9 | Step conditions (American, Bermudan, dividend) | `fdmamericanstepcondition.hpp`, `fdmbermudanstepcondition.hpp`, `fdmdividendhandler.hpp`, `fdmsnapshotcondition.hpp` | `ql_jax/methods/fd/step_conditions.py` | Medium |
| 13.10 | Generic solver (1D, 2D, 3D, N-dim) | `fdm1dimsolver.hpp`, `fdm2dimsolver.hpp`, `fdm3dimsolver.hpp`, `fdmndimsolver.hpp`, `fdmbackwardsolver.hpp` | `ql_jax/methods/fd/solvers.py` | High |
| 13.11 | Inner value calculators | `fdminnervaluecalculator.hpp`, `fdmescrowedloginnervaluecalculator.hpp`, `fdmshoutloginnervaluecalculator.hpp` | `ql_jax/methods/fd/inner_value.py` | Low |
| 13.12 | Storage/swing conditions | `fdmsimplestoragecondition.hpp`, `fdmsimpleswingcondition.hpp` | `ql_jax/methods/fd/storage.py` | Medium |
| 13.13 | Quanto helper | `fdmquantohelper.hpp` | `ql_jax/methods/fd/quanto.py` | Low |
| 13.14 | RND calculators | `bsmlattice.hpp`, `hestonrndcalculator.hpp`, `gbsmrndcalculator.hpp`, `cevrndcalculator.hpp`, `localvolrndcalculator.hpp` | `ql_jax/methods/fd/rnd.py` | Medium |

**Tests:** `tests/test_fd_framework.py`, `tests/test_fd_engines.py`

---

## Phase 14: Lattice / Tree Framework Enhancement

| # | Component | QuantLib Files | Target File(s) | Complexity |
|---|-----------|---------------|----------------|------------|
| 14.1 | Trinomial tree | `trinomialtree.hpp` | `ql_jax/methods/lattice/trinomial.py` | Medium |
| 14.2 | Tree base + lattice (1D/2D) | `tree.hpp`, `lattice.hpp`, `lattice1d.hpp`, `lattice2d.hpp` | `ql_jax/methods/lattice/base.py` | Medium |
| 14.3 | BSM lattice | `bsmlattice.hpp` | `ql_jax/methods/lattice/bsm.py` | Medium |
| 14.4 | TF lattice | `tflattice.hpp` | `ql_jax/methods/lattice/tf_lattice.py` | Medium |

**Tests:** `tests/test_lattice.py`

---

## Phase 15: Missing Term Structures

### 15A — Yield Term Structures

| # | Component | QuantLib Header | Target File | Complexity |
|---|-----------|----------------|-------------|------------|
| 15.1 | CompositeZeroYieldStructure | `compositezeroyieldstructure.hpp` | `ql_jax/termstructures/yield_/composite.py` | Low |
| 15.2 | PiecewiseZeroSpreadedTermStructure | `piecewisezerospreadedtermstructure.hpp` | `ql_jax/termstructures/yield_/zero_spreaded.py` | Medium |
| 15.3 | PiecewiseSpreadYieldCurve | `piecewisespreadyieldcurve.hpp` | `ql_jax/termstructures/yield_/spread_yield_curve.py` | Medium |
| 15.4 | UltimateForwardTermStructure | `ultimateforwardtermstructure.hpp` | `ql_jax/termstructures/yield_/ultimate_forward.py` | Medium |
| 15.5 | NonLinearFittingMethods (additional) | `nonlinearfittingmethods.hpp` | update `ql_jax/termstructures/yield_/fitted_bond_curve.py` | Medium |
| 15.6 | BondHelper / FixedRateBondHelper | `bondhelpers.hpp` | `ql_jax/termstructures/yield_/bond_helpers.py` | Medium |
| 15.7 | OISRateHelper (enhanced) / OvernightIndexFutureRateHelper | `oisratehelper.hpp`, `overnightindexfutureratehelper.hpp` | update `ql_jax/termstructures/yield_/rate_helpers.py` | Medium |
| 15.8 | MultipleResetsSwapHelper | `multipleresetsswaphelper.hpp` | `ql_jax/termstructures/yield_/multiple_resets_helper.py` | Medium |
| 15.9 | GlobalBootstrap | `globalbootstrap.hpp` | `ql_jax/termstructures/yield_/global_bootstrap.py` | High |

### 15B — Volatility Surfaces

| # | Component | QuantLib Header | Target File | Complexity |
|---|-----------|----------------|-------------|------------|
| 15.10 | LocalVolSurface / LocalVolCurve | `localvolsurface.hpp`, `localvolcurve.hpp` | `ql_jax/termstructures/volatility/local_vol_surface.py` | Medium |
| 15.11 | FixedLocalVolSurface | `fixedlocalvolsurface.hpp` | `ql_jax/termstructures/volatility/fixed_local_vol.py` | Medium |
| 15.12 | HestonBlackVolSurface | `hestonblackvolsurface.hpp` | `ql_jax/termstructures/volatility/heston_vol.py` | Medium |
| 15.13 | AndreasenHugeVolatilityAdapter / LocalVolAdapter | `andreasenhugevolatilityadapter.hpp`, `andreasenhugelocalvoladapter.hpp` | `ql_jax/termstructures/volatility/andreasen_huge.py` | High |
| 15.14 | BlackVolSurfaceDelta | `blackvolsurfacedelta.hpp` | `ql_jax/termstructures/volatility/delta_surface.py` | Medium |
| 15.15 | PiecewiseBlackVarianceSurface | `piecewiseblackvariancesurface.hpp` | `ql_jax/termstructures/volatility/piecewise_variance.py` | Medium |
| 15.16 | QuantoTermStructure | `quantotermstructure.hpp` | `ql_jax/termstructures/volatility/quanto.py` | Medium |
| 15.17 | KahaleSmileSection / ZABR | `kahalesmilesection.hpp`, `zabr.hpp`, `zabrsmilesection.hpp` | `ql_jax/termstructures/volatility/kahale_zabr.py` | High |
| 15.18 | Gaussian1dSmileSection / SwaptionVol | `gaussian1dsmilesection.hpp`, `gaussian1dswaptionvolatility.hpp` | `ql_jax/termstructures/volatility/gaussian1d_vol.py` | Medium |
| 15.19 | SABR Swaption Vol Cube (interpolated) | `sabrswaptionvolatilitycube.hpp`, `interpolatedswaptionvolatilitycube.hpp` | `ql_jax/termstructures/volatility/sabr_cube.py` | High |
| 15.20 | OptionletStripper1 / OptionletStripper2 | `optionletstripper1.hpp`, `optionletstripper2.hpp` | `ql_jax/termstructures/volatility/optionlet_stripper.py` | High |
| 15.21 | CapletVarianceCurve | `capletvariancecurve.hpp` | `ql_jax/termstructures/volatility/caplet_variance.py` | Medium |
| 15.22 | SpreadedSmileSection / SpreadedOptionletVol / SpreadedSwaptionVol | various `spreaded*.hpp` | `ql_jax/termstructures/volatility/spreaded.py` | Medium |
| 15.23 | CPI volatility structure | `constantcpivolatility.hpp`, `cpivolatilitystructure.hpp` | `ql_jax/termstructures/volatility/cpi_vol.py` | Medium |
| 15.24 | YoY Inflation Optionlet Vol | `yoyinflationoptionletvolatilitystructure.hpp` | `ql_jax/termstructures/volatility/yoy_optionlet.py` | Medium |
| 15.25 | ATM-adjusted / ATM SmileSection | `atmadjustedsmilesection.hpp`, `atmsmilesection.hpp` | update smile_section.py | Low |

### 15C — Credit Term Structures

| # | Component | QuantLib Header | Target File | Complexity |
|---|-----------|----------------|-------------|------------|
| 15.26 | DefaultDensityStructure / InterpolatedDefaultDensityCurve | `defaultdensitystructure.hpp`, `interpolateddefaultdensitycurve.hpp` | `ql_jax/termstructures/credit/density_curve.py` | Medium |
| 15.27 | DefaultProbabilityHelpers | `defaultprobabilityhelpers.hpp` | `ql_jax/termstructures/credit/helpers.py` | Medium |

### 15D — Inflation Term Structures

| # | Component | QuantLib Header | Target File | Complexity |
|---|-----------|----------------|-------------|------------|
| 15.28 | PiecewiseZeroInflationCurve / PiecewiseYoYInflationCurve | `piecewisezeroinflationcurve.hpp`, `piecewiseyoyinflationcurve.hpp` | `ql_jax/termstructures/inflation/piecewise.py` | Medium |
| 15.29 | InflationHelpers | `inflationhelpers.hpp` | `ql_jax/termstructures/inflation/helpers.py` | Medium |
| 15.30 | Seasonality (enhanced) | `seasonality.hpp` | update `ql_jax/termstructures/inflation/curves.py` | Low |

**Tests:** `tests/test_term_structures_advanced.py`, `tests/test_vol_surfaces.py`

---

## Phase 16: Missing Math Components

### 16A — Interpolation

| # | Method | QuantLib Header | Target File | Complexity |
|---|--------|----------------|-------------|------------|
| 16.1 | SABR interpolation | `sabrinterpolation.hpp` | `ql_jax/math/interpolations/sabr.py` | High |
| 16.2 | ZABR interpolation | `zabrinterpolation.hpp` | `ql_jax/math/interpolations/zabr.py` | High |
| 16.3 | ABCD interpolation | `abcdinterpolation.hpp` | `ql_jax/math/interpolations/abcd.py` | Medium |
| 16.4 | Chebyshev interpolation | `chebyshevinterpolation.hpp` | `ql_jax/math/interpolations/chebyshev.py` | Medium |
| 16.5 | Lagrange interpolation | `lagrangeinterpolation.hpp` | `ql_jax/math/interpolations/lagrange.py` | Low |
| 16.6 | ConvexMonotone interpolation | `convexmonotoneinterpolation.hpp` | `ql_jax/math/interpolations/convex_monotone.py` | Medium |
| 16.7 | Kernel interpolation (1D + 2D) | `kernelinterpolation.hpp`, `kernelinterpolation2d.hpp` | `ql_jax/math/interpolations/kernel.py` | Medium |
| 16.8 | Mixed interpolation | `mixedinterpolation.hpp` | `ql_jax/math/interpolations/mixed.py` | Low |
| 16.9 | Bicubic spline / Bilinear (2D) | `bicubicsplineinterpolation.hpp`, `bilinearinterpolation.hpp` | `ql_jax/math/interpolations/interp2d.py` | Medium |
| 16.10 | MultiCubicSpline | `multicubicspline.hpp` | `ql_jax/math/interpolations/multicubic.py` | High |
| 16.11 | Flat extrapolation 2D | `flatextrapolation2d.hpp` | `ql_jax/math/interpolations/flat_extrap_2d.py` | Low |

### 16B — Solvers

| # | Solver | QuantLib Header | Target File | Complexity |
|---|--------|----------------|-------------|------------|
| 16.12 | Ridder | `ridder.hpp` | `ql_jax/math/solvers/ridder.py` | Low |
| 16.13 | FalsePosition | `falseposition.hpp` | `ql_jax/math/solvers/false_position.py` | Low |
| 16.14 | Halley | `halley.hpp` | `ql_jax/math/solvers/halley.py` | Low |
| 16.15 | NewtonSafe / FiniteDifferenceNewtonSafe | `newtonsafe.hpp`, `finitedifferencenewtonsafe.hpp` | `ql_jax/math/solvers/newton_safe.py` | Low |

### 16C — Distributions

| # | Distribution | QuantLib Header | Target File | Complexity |
|---|-------------|----------------|-------------|------------|
| 16.16 | Gamma distribution | `gammadistribution.hpp` | `ql_jax/math/distributions/gamma.py` | Low |
| 16.17 | Binomial distribution | `binomialdistribution.hpp` | `ql_jax/math/distributions/binomial.py` | Low |
| 16.18 | BivariateNormal / BivariateStudentT | `bivariatenormaldistribution.hpp`, `bivariatestudenttdistribution.hpp` | `ql_jax/math/distributions/bivariate.py` | Medium |
| 16.19 | StudentT distribution | `studenttdistribution.hpp` | `ql_jax/math/distributions/student_t.py` | Low |

### 16D — Optimization

| # | Method | QuantLib Header | Target File | Complexity |
|---|--------|----------------|-------------|------------|
| 16.20 | DifferentialEvolution | `differentialevolution.hpp` | `ql_jax/math/optimization/differential_evolution.py` | Medium |
| 16.21 | SimulatedAnnealing | `simulatedannealing.hpp` | `ql_jax/math/optimization/simulated_annealing.py` | Medium |
| 16.22 | ConjugateGradient | `conjugategradient.hpp` | `ql_jax/math/optimization/conjugate_gradient.py` | Medium |
| 16.23 | SteepestDescent | `steepestdescent.hpp` | `ql_jax/math/optimization/steepest_descent.py` | Low |
| 16.24 | Armijo / Goldstein line search | `armijo.hpp`, `goldstein.hpp` | `ql_jax/math/optimization/line_search.py` | Low |
| 16.25 | Constraint / ProjectedConstraint / Projection | `constraint.hpp`, `projectedconstraint.hpp`, `projection.hpp` | `ql_jax/math/optimization/constraint.py` | Medium |
| 16.26 | SphereCylinder | `spherecylinder.hpp` | `ql_jax/math/optimization/sphere_cylinder.py` | Low |

### 16E — Random Numbers

| # | Generator | QuantLib Header | Target File | Complexity |
|---|----------|----------------|-------------|------------|
| 16.27 | Faure low-discrepancy | `faurersg.hpp` | `ql_jax/math/random/faure.py` | Medium |
| 16.28 | Lattice RSG | `latticersg.hpp` | `ql_jax/math/random/lattice.py` | Medium |
| 16.29 | Randomized LDS | `randomizedlds.hpp` | `ql_jax/math/random/randomized_lds.py` | Medium |
| 16.30 | Sobol-Brownian bridge RSG | `sobolbrownianbridgersg.hpp` | `ql_jax/math/random/sobol_brownian.py` | Medium |
| 16.31 | Stochastic collocation inverse CDF | `stochasticcollocationinvcdf.hpp` | `ql_jax/math/random/stochastic_collocation.py` | Medium |
| 16.32 | Box-Muller / Ziggurat wrappers | `boxmullergaussianrng.hpp`, `zigguratgaussianrng.hpp` | note: JAX handles internally | Low |
| 16.33 | InverseCumulative RNG/RSG | `inversecumulativerng.hpp`, `inversecumulativersg.hpp` | `ql_jax/math/random/inverse_cumulative.py` | Low |
| 16.34 | Burley2020 Sobol | `burley2020sobolrsg.hpp` | `ql_jax/math/random/burley_sobol.py` | Medium |

**Tests:** `tests/math/test_interpolation_advanced.py`, `tests/math/test_solvers.py`, `tests/math/test_optimization.py`

---

## Phase 17: Missing Cashflows & Coupons

| # | Cashflow Type | QuantLib Header | Target File | Complexity |
|---|-------------|----------------|-------------|------------|
| 17.1 | CMSCoupon + CMS pricer (conundrum, linear TSR) | `cmscoupon.hpp`, `conundrumpricer.hpp`, `lineartsrpricer.hpp` | `ql_jax/cashflows/cms.py` | High |
| 17.2 | DigitalCoupon / DigitalIborCoupon / DigitalCMSCoupon | `digitalcoupon.hpp`, `digitaliborcoupon.hpp`, `digitalcmscoupon.hpp` | `ql_jax/cashflows/digital.py` | Medium |
| 17.3 | RangeAccrual | `rangeaccrual.hpp` | `ql_jax/cashflows/range_accrual.py` | High |
| 17.4 | AverageBMACoupon | `averagebmacoupon.hpp` | `ql_jax/cashflows/bma.py` | Medium |
| 17.5 | OvernightIndexedCoupon pricer (Black) | `blackovernightindexedcouponpricer.hpp`, `overnightindexedcouponpricer.hpp` | `ql_jax/cashflows/overnight_pricer.py` | Medium |
| 17.6 | CPICoupon pricer | `cpicouponpricer.hpp` | `ql_jax/cashflows/cpi_pricer.py` | Medium |
| 17.7 | InflationCouponPricer | `inflationcouponpricer.hpp` | `ql_jax/cashflows/inflation_pricer.py` | Medium |
| 17.8 | MultipleResetsCoupon | `multipleresetscoupon.hpp` | `ql_jax/cashflows/multiple_resets.py` | Medium |
| 17.9 | EquityCashFlow | `equitycashflow.hpp` | `ql_jax/cashflows/equity.py` | Low |
| 17.10 | CouponPricer framework | `couponpricer.hpp` | `ql_jax/cashflows/pricer.py` | Medium |
| 17.11 | Replication | `replication.hpp` | `ql_jax/cashflows/replication.py` | Medium |
| 17.12 | RateAveraging | `rateaveraging.hpp` | update `ql_jax/cashflows/floating_rate.py` | Low |
| 17.13 | IndexedCashFlow | `indexedcashflow.hpp` | `ql_jax/cashflows/indexed.py` | Low |
| 17.14 | TimeBasket | `timebasket.hpp` | `ql_jax/cashflows/time_basket.py` | Low |
| 17.15 | Duration enhancements | `duration.hpp` | update `ql_jax/cashflows/analytics.py` | Low |

**Tests:** `tests/test_cashflows_advanced.py`

---

## Phase 18: Missing Calendars, Day Counters & Indexes

### 18A — Calendars (38 missing)

Currently implemented: NullCalendar, WeekendsOnly, TARGET, UnitedStates, UnitedKingdom, Japan, Germany, JointCalendar, BespokeCalendar (9 total)

Missing calendars to add to `ql_jax/time/calendar.py` or `ql_jax/time/calendars/`:

| # | Calendars (batch) | Complexity |
|---|-------------------|------------|
| 18.1 | Major G10: Australia, Canada, Switzerland, France, Sweden, Norway, Denmark | Low each |
| 18.2 | Europe: Austria, Finland, Hungary, Iceland, Italy, Poland, Romania, Slovakia, Czech Republic | Low each |
| 18.3 | Americas: Argentina, Brazil, Chile, Mexico | Low each |
| 18.4 | Asia-Pacific: China, Hong Kong, India, Indonesia, Singapore, South Korea, Taiwan, Thailand | Low each |
| 18.5 | Other: Israel, Russia, Saudi Arabia, South Africa, Turkey, Ukraine, Botswana, New Zealand | Low each |

### 18B — Day Counters (missing variants)

| # | Day Counter | QuantLib Header | Complexity |
|---|-----------|----------------|------------|
| 18.6 | Actual364 | `actual364.hpp` | Low |
| 18.7 | Actual36525 | `actual36525.hpp` | Low |
| 18.8 | Actual366 | `actual366.hpp` | Low |
| 18.9 | Business252 | `business252.hpp` | Low |
| 18.10 | Thirty365 | `thirty365.hpp` | Low |
| 18.11 | SimpleDayCounter | `simpledaycounter.hpp` | Low |
| 18.12 | One (trivial) | `one.hpp` | Low |

### 18C — Indexes (40+ missing)

Currently implemented: 16 overnight/IBOR indexes + 5 swap indexes.

| # | Index Group | Examples | Target | Complexity |
|---|-----------|---------|--------|------------|
| 18.13 | BMA Index | `bmaindex.hpp` | `ql_jax/indexes/bma.py` | Low |
| 18.14 | Inflation indexes | AUCPI, EUHICP, FRHICP, UKHICP, UKRPI, USCPI, ZACPI | `ql_jax/indexes/inflation.py` | Low each |
| 18.15 | Equity index | `equityindex.hpp` | `ql_jax/indexes/equity.py` | Medium |
| 18.16 | Additional IBOR: BBSW, BIBOR, BKBM, CDI, CDOR, PRIBOR, ROBOR, TRLIBOR, WIBOR, ZIBOR, JIBAR, SHIBOR, THBFIX, DKKLIBOR, SEKLIBOR, NZDLIBOR | various | update `ql_jax/indexes/ibor.py` | Low each |
| 18.17 | Additional RFR: DESTR, SWESTR, KOFR, NZOCR | various | update `ql_jax/indexes/ibor.py` | Low each |
| 18.18 | Additional swap indexes | various markets | update `ql_jax/indexes/swap.py` | Low each |
| 18.19 | IndexManager | `indexmanager.hpp` | `ql_jax/indexes/manager.py` | Low |
| 18.20 | Region definitions | `region.hpp` | `ql_jax/indexes/region.py` | Low |
| 18.21 | Custom index | `custom.hpp` | `ql_jax/indexes/custom.py` | Low |

### 18D — Quotes

| # | Quote Type | QuantLib Equivalent | Target | Complexity |
|---|-----------|-------------------|--------|------------|
| 18.22 | CompositeQuote | `compositequote.hpp` | `ql_jax/quotes/composite.py` | Low |
| 18.23 | DerivedQuote | `derivedquote.hpp` | `ql_jax/quotes/derived.py` | Low |
| 18.24 | ForwardValueQuote | `forwardvaluequote.hpp` | `ql_jax/quotes/forward_value.py` | Low |
| 18.25 | ImpliedStdDevQuote | `impliedstddevquote.hpp` | `ql_jax/quotes/implied_vol.py` | Low |

### 18E — Currencies

| # | Currency Group | QuantLib Files | Target | Complexity |
|---|-------------|---------------|--------|------------|
| 18.26 | Full currency definitions (40+) | `africa.hpp`, `america.hpp`, `asia.hpp`, `europe.hpp`, `oceania.hpp`, `crypto.hpp` | expand `ql_jax/currencies/__init__.py` | Low |

**Tests:** `tests/test_calendars.py`, `tests/test_daycounters.py`, `tests/test_indexes_full.py`

---

## Phase 19: Missing Examples

Currently: 3 examples (equity_option.py, bonds.py, cds.py). Need 17 more:

| # | Example | Description | Complexity |
|---|---------|-------------|------------|
| 19.1 | `bermudan_swaption.py` | Bermudan swaption with short-rate models | Medium |
| 19.2 | `fitted_bond_curve.py` | Yield curve fitting (Nelson-Siegel, Svensson) | Medium |
| 19.3 | `fra.py` | Forward rate agreement pricing | Low |
| 19.4 | `repo.py` | Repo instrument | Low |
| 19.5 | `callable_bonds.py` | Callable bond pricing with Hull-White | Medium |
| 19.6 | `convertible_bonds.py` | Convertible bond pricing | High |
| 19.7 | `discrete_hedging.py` | Discrete hedging simulation | Medium |
| 19.8 | `basket_losses.py` | Portfolio credit loss | Medium |
| 19.9 | `market_models.py` | LMM calibration and pricing | High |
| 19.10 | `gaussian1d_models.py` | Gaussian short-rate examples | Medium |
| 19.11 | `multicurve_bootstrap.py` | Multi-curve framework | Medium |
| 19.12 | `global_optimizer.py` | Global optimization for calibration | Medium |
| 19.13 | `multidim_integral.py` | Multi-dimensional integration | Low |
| 19.14 | `replication.py` | Static replication strategies | Medium |
| 19.15 | `cva_irs.py` | CVA for interest-rate swaps | High |
| 19.16 | `asian_option.py` | Asian option pricing (arithmetic/geometric) | Low |
| 19.17 | `variance_swap.py` | Variance swap pricing | Medium |
| 19.18 | `heston_calibration.py` | Heston model calibration demo | Medium |
| 19.19 | `portfolio_greeks.py` | AD Greeks for a portfolio (vmap showcase) | Medium |
| 19.20 | `gpu_benchmark.py` | GPU vs CPU performance comparison | Medium |

**Tests:** Each example should be runnable as an integration test.

---

## Phase 20: Missing Tests

Currently: 36 test files, ~383 tests. Target: 150+ test files. Key missing test coverage areas:

| # | Test Area | Target File(s) | Coverage Gap |
|---|----------|----------------|-------------|
| 20.1 | Lookback/Cliquet/Basket options | `tests/test_lookback.py`, `tests/test_basket.py` | New instruments |
| 20.2 | Exotic engines (chooser, compound, quanto, Margrabe, extensible) | `tests/test_exotic_engines.py` | New engines |
| 20.3 | FD framework (meshers, operators, schemes) | `tests/test_fd_framework.py` | New methods |
| 20.4 | FD engines (Heston FD, SABR FD, barrier FD) | `tests/test_fd_engines.py` | New engines |
| 20.5 | Advanced tree engines (trinomial, barrier lattice) | `tests/test_tree_engines.py` | New engines |
| 20.6 | Gaussian1d / GSR / MarkovFunctional models | `tests/test_gaussian1d.py` | New models |
| 20.7 | Market models (SMM, evolvers, pathwise) | `tests/test_market_models.py` | New models |
| 20.8 | CMS coupons and digital coupons | `tests/test_cms_coupons.py` | New cashflows |
| 20.9 | Advanced vol surfaces (local vol, SABR cube, optionlet stripping) | `tests/test_vol_surfaces.py` | New term structures |
| 20.10 | All 48 calendars | `tests/time/test_all_calendars.py` | Missing calendars |
| 20.11 | All day counters | `tests/time/test_all_daycounters.py` | Missing dc variants |
| 20.12 | Advanced interpolation (SABR, 2D, Chebyshev) | `tests/math/test_interpolation_2d.py` | New math |
| 20.13 | Additional optimization (DiffEvol, SA) | `tests/math/test_optimization.py` | New math |
| 20.14 | Inflation instruments (CPI swap, YoY swap, ZC swap) | `tests/test_inflation_instruments.py` | New instruments |
| 20.15 | Process tests (Bates, Merton76, GBM, OU) | `tests/test_processes.py` | New processes |
| 20.16 | ISDA CDS engine | `tests/test_isda_cds.py` | New engine |
| 20.17 | Convertible bonds | `tests/test_convertible.py` | New instrument |
| 20.18 | Double barrier options | `tests/test_double_barrier.py` | New instrument |
| 20.19 | American analytic approximations (BAW, BjS) | `tests/test_american_approx.py` | New engines |
| 20.20 | Variance swaps | `tests/test_variance_swap.py` | New instrument |

---

## Priority & Dependency Matrix

### Implementation Order (recommended)

```
Phase  9 (Instruments)      ─────┐
Phase 10 (Processes)        ─────┤
Phase 16 (Math)             ─────┼── Can be done in parallel
Phase 18 (Calendars/Idx)    ─────┘
           │
           ▼
Phase 12 (Models)           ─────┐
Phase 15 (Term Structures)  ─────┼── Depend on math + processes
Phase 17 (Cashflows)        ─────┘
           │
           ▼
Phase 13 (FD Framework)     ─────┐
Phase 14 (Lattice Framework)─────┼── Depend on models + instruments
Phase 11 (Engines)          ─────┘
           │
           ▼
Phase 19 (Examples)         ─────┐
Phase 20 (Tests)            ─────┼── Final integration
           └─────────────────────┘
```

### Estimated Effort by Phase

| Phase | Description | New Files | Complexity | Priority |
|-------|------------|-----------|------------|----------|
| 9 | Instruments | ~20 | Medium | P1 |
| 10 | Processes | ~16 | Medium | P1 |
| 11 | Engines | ~50 | High | P1 |
| 12 | Models | ~20 | High | P1 |
| 13 | FD Framework | ~14 | Very High | P2 |
| 14 | Lattice Framework | ~4 | Medium | P2 |
| 15 | Term Structures | ~25 | High | P1 |
| 16 | Math | ~30 | Medium | P2 |
| 17 | Cashflows | ~15 | Medium | P2 |
| 18 | Calendars/Indexes | ~10 | Low | P3 |
| 19 | Examples | ~17 | Medium | P3 |
| 20 | Tests | ~120 | Medium | P1 |

### Total Scope

- **~340 new files** to reach near-parity with QuantLib C++
- Current coverage: **~15-20%** of QuantLib's full functionality
- **Priority 1** items (Phases 9-12, 15, 20) would bring coverage to **~50-60%**
- **Priority 2** items (Phases 13-14, 16-17) would bring coverage to **~80-85%**
- **Priority 3** items (Phases 18-19) complete the remaining **~15-20%**

---

## Notes

1. **ql/experimental/** (30+ subdirectories) is explicitly out of scope per project.md §3.2
2. Market models (Phase 12C) represent the largest single gap — 125 C++ files vs 1 Python file — but are also the most specialized and least-used component
3. The FD framework (Phase 13) is the most architecturally complex gap, requiring a complete mesher/operator/scheme infrastructure
4. Many "Low complexity" items (calendars, indexes, day counters) are tedious but straightforward — good candidates for batch implementation
5. All new implementations should include JAX-specific features: `jax.grad` differentiability, `jax.vmap` vectorization, `jax.jit` compilation compatibility
