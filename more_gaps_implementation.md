# ql-jax Gap Analysis & Implementation Plan

## Executive Summary

Systematic comparison of QuantLib C++ (`/mnt/c/finance/QuantLib/ql/`) against `ql-jax` (`/home/jude/ql-jax/ql_jax/`).

| Module | QuantLib C++ (.hpp) | ql-jax (.py) | Coverage |
|--------|-------------------|-------------|----------|
| instruments | 88 | 30 | ~55% (key types present, many specialty missing) |
| pricingengines | 170 | 28 | ~25% (core engines present, most variants missing) |
| processes | 22 | 15 | ~75% (good coverage) |
| models | 159 | 22 | ~15% (short-rate/equity OK, market models skeletal) |
| termstructures | 132 | 28 | ~35% (core present, many curve types missing) |
| cashflows | 36 | 17 | ~65% (good coverage) |
| methods | 147 | 12 | ~10% (basic FD/MC, many operators missing) |
| math | 177 | 45 | ~40% (core present, many utilities missing) |
| time | 73 | 5 | ~40% (16 calendars vs 47, few day counters) |
| indexes | 67 | 5 | ~35% (20 ibor, no swap index instances) |
| quotes | 11 | 1 | ~10% (only SimpleQuote) |

**Current state**: 230 source files, 595 passing tests.

---

## Phase 1: Missing Instrument Types (Priority: HIGH)

### 1.1 Missing Interest Rate Instruments
| QuantLib | Status | Notes |
|----------|--------|-------|
| `BMASwap` | ✅ EXISTS | In `asset_swap.py` |
| `FloatFloatSwap` | ✅ EXISTS | In `asset_swap.py` |
| `NonStandardSwap` | ✅ EXISTS | In `asset_swap.py` |
| `OvernightIndexedSwap` | ✅ EXISTS | In `swap.py` |
| `ZeroCouponSwap` | ❌ MISSING | Simple zero-coupon swap |
| `MultipleResetsSwap` | ❌ MISSING | Sub-periods/multiple resets per period |
| `FloatFloatSwaption` | ❌ MISSING | Swaption on float-float swap |
| `NonStandardSwaption` | ❌ MISSING | Swaption on non-standard swap |
| `CPICapFloor` | ❌ MISSING | CPI inflation cap/floor instrument |
| `InflationCapFloor` | ❌ MISSING | Standalone inflation cap/floor |
| `Futures` / `OvernightIndexFuture` | ❌ MISSING | Interest rate futures |
| `PerpetualFutures` | ❌ MISSING | Perpetual futures |

### 1.2 Missing Exotic Instruments
| QuantLib | Status | Notes |
|----------|--------|-------|
| `SoftBarrierOption` | ✅ EXISTS | In `partial_barrier.py` |
| `BondForward` | ✅ EXISTS | In `forward.py` |
| `FXForward` | ✅ EXISTS | In `forward.py` |
| `Stock` | ✅ EXISTS | In `forward.py` |
| `CompositeInstrument` | ✅ EXISTS | In `forward.py` |
| `EquityTotalReturnSwap` | ✅ EXISTS | In `asset_swap.py` |
| `VanillaStorageOption` | ❌ MISSING | Gas/energy storage option |
| `VanillaSwingOption` | ❌ MISSING | Energy swing option |
| `StickyRatchet` | ❌ MISSING | Sticky/ratchet cap |

### 1.3 Missing Bond Subtypes
| QuantLib | Status | Notes |
|----------|--------|-------|
| `FixedRateBond` | ✅ EXISTS | `make_fixed_rate_bond()` |
| `ZeroCouponBond` | ✅ EXISTS | `make_zero_coupon_bond()` |
| `FloatingRateBond` | ✅ EXISTS | `make_floating_rate_bond()` |
| `AmortizingFixedRateBond` | ✅ EXISTS | `make_amortizing_fixed_rate_bond()` |
| `ConvertibleBond` | ✅ EXISTS | `convertible_bond.py` |
| `AmortizingFloatingRateBond` | ❌ MISSING | Amortizing floater |
| `AmortizingCMSRateBond` | ❌ MISSING | Amortizing CMS rate bond |
| `CMSRateBond` | ❌ MISSING | CMS-linked bond |
| `CPIBond` | ❌ MISSING | Inflation-linked bond |
| `BTP` | ❌ MISSING | Italian government bond |

**Files to create/modify**:
- `ql_jax/instruments/futures.py` — IR futures + overnight index futures + perpetual futures
- `ql_jax/instruments/inflation_capfloor.py` — CPICapFloor, InflationCapFloor
- `ql_jax/instruments/swap.py` — add ZeroCouponSwap, MultipleResetsSwap
- `ql_jax/instruments/swaption.py` — add FloatFloatSwaption, NonStandardSwaption
- `ql_jax/instruments/energy.py` — VanillaStorageOption, VanillaSwingOption
- `ql_jax/instruments/bond.py` — add make_cpi_bond, make_amortizing_floating_rate_bond, make_cms_rate_bond

**Tests**: `tests/test_instruments_phase1.py` (~30 tests)

---

## Phase 2: Missing Pricing Engines — Vanilla/American Options (Priority: HIGH)

### 2.1 Analytic American Engines
| QuantLib | Status | Notes |
|----------|--------|-------|
| `AnalyticEuropeanEngine` | ✅ EXISTS | `black_scholes_price()` |
| `AnalyticHestonEngine` | ✅ EXISTS | `heston_price()` |
| `BinomialEngine` | ✅ EXISTS | `binomial_price()` |
| `MCAmericanEngine` | ✅ EXISTS | `mc_american_bs()` |
| `MCEuropeanEngine` | ✅ EXISTS | `mc_european_bs()` |
| `FDBlackScholesVanillaEngine` | ✅ EXISTS | `fd_black_scholes_price()` |
| `FDHestonVanillaEngine` | ✅ EXISTS | `fd_heston_price()` |
| `BaroneAdesiWhaleyEngine` | ❌ MISSING | Analytic American approximation |
| `BjerksundStenslandEngine` | ❌ MISSING | Analytic American approximation |
| `JuQuadraticEngine` | ❌ MISSING | Quadratic American approximation |
| `QdFpAmericanEngine` | ❌ MISSING | Fixed-point iteration American |
| `QdPlusAmericanEngine` | ❌ MISSING | QD+ American |
| `IntegralEngine` | ❌ MISSING | European by numerical integration |
| `JumpDiffusionEngine` | ❌ MISSING | Merton jump-diffusion pricing |

### 2.2 Analytic Heston Variants
| QuantLib | Status | Notes |
|----------|--------|-------|
| `AnalyticPtdHestonEngine` | ❌ MISSING | Piecewise time-dependent Heston |
| `AnalyticPdfHestonEngine` | ❌ MISSING | PDF-based Heston |
| `CosHestonEngine` | ❌ MISSING | COS method for Heston |
| `ExponentialFittingHestonEngine` | ❌ MISSING | Exponential fitting |
| `HestonExpansionEngine` | ❌ MISSING | Expansion approximation |
| `BatesEngine` | ❌ MISSING | Bates model pricing |
| `AnalyticGJRGARCHEngine` | ❌ MISSING | GJRGARCH analytic |

### 2.3 Hybrid & Specialty Vanilla
| QuantLib | Status | Notes |
|----------|--------|-------|
| `AnalyticBSMHullWhiteEngine` | ❌ MISSING | BSM + Hull-White hybrid |
| `AnalyticHestonHullWhiteEngine` | ❌ MISSING | Heston + HW hybrid |
| `MCHestonHullWhiteEngine` | ❌ MISSING | MC Heston + HW |
| `AnalyticEuropeanVasicekEngine` | ❌ MISSING | European + Vasicek rates |
| `AnalyticCEVEngine` | ❌ MISSING | CEV model European |
| `AnalyticH1HWEngine` | ❌ MISSING | H1-HW model |
| `CashDividendEuropeanEngine` | ❌ MISSING | Escrowed dividend model |
| `AnalyticDividendEuropeanEngine` | ❌ MISSING | Discrete dividends |

### 2.4 FD Vanilla Variants
| QuantLib | Status | Notes |
|----------|--------|-------|
| `FDBatesVanillaEngine` | ❌ MISSING | FD Bates |
| `FDBlackScholesShoutEngine` | ❌ MISSING | FD shout option |
| `FDCEVVanillaEngine` | ❌ MISSING | FD CEV |
| `FDCIRVanillaEngine` | ❌ MISSING | FD CIR |
| `FDHestonHullWhiteVanillaEngine` | ❌ MISSING | FD Heston-HW hybrid |
| `FDSABRVanillaEngine` | ❌ MISSING | FD SABR |
| `FDSimpleBSSwingEngine` | ❌ MISSING | FD swing option |

### 2.5 MC Vanilla Variants
| QuantLib | Status | Notes |
|----------|--------|-------|
| `MCEuropeanHestonEngine` | ✅ EXISTS | `mc_european_heston()` |
| `MCDigitalEngine` | ❌ MISSING | MC digital option |
| `MCEuropeanGJRGARCHEngine` | ❌ MISSING | MC GJRGARCH |

**Files to create**:
- `ql_jax/engines/analytic/american.py` — BAW, BjS, JuQuadratic, QD+, QDFp
- `ql_jax/engines/analytic/heston_variants.py` — PtdHeston, PdfHeston, CosHeston, ExpFitting, BatesEngine
- `ql_jax/engines/analytic/hybrid.py` — BSM-HW, Heston-HW, Vasicek, CEV, dividend engines
- `ql_jax/engines/fd/bates.py` — FD Bates
- `ql_jax/engines/fd/cev.py` — FD CEV
- `ql_jax/engines/fd/sabr.py` — FD SABR
- `ql_jax/engines/analytic/jump_diffusion.py` — Merton jump-diffusion

**Tests**: `tests/test_american_engines.py`, `tests/test_heston_engines.py` (~40 tests)

---

## Phase 3: Missing Pricing Engines — Barrier & Path-Dependent (Priority: HIGH)

### 3.1 Barrier Engine Gaps
| QuantLib | Status | Notes |
|----------|--------|-------|
| `AnalyticBarrierEngine` | ✅ EXISTS | `barrier_price()` |
| `AnalyticDoubleBarrierEngine` | ✅ EXISTS | `double_barrier_price()` |
| `FDBlackScholesBarrierEngine` | ✅ EXISTS | `fd_barrier_price()` |
| `MCBarrierEngine` | ✅ EXISTS | `mc_barrier_bs()` |
| `AnalyticBinaryBarrierEngine` | ❌ MISSING | Cash-or-nothing barrier |
| `AnalyticDoubleBarrierBinaryEngine` | ❌ MISSING | Double barrier binary |
| `AnalyticPartialTimeBarrierEngine` | ❌ MISSING | Partial-time barrier analytic |
| `AnalyticSoftBarrierEngine` | ❌ MISSING | Soft barrier analytic |
| `AnalyticTwoAssetBarrierEngine` | ❌ MISSING | Two-asset barrier analytic |
| `BinomialBarrierEngine` | ❌ MISSING | Tree-based barrier |
| `FDBlackScholesRebateEngine` | ❌ MISSING | FD rebate |
| `FDHestonBarrierEngine` | ❌ MISSING | 2D FD Heston barrier |
| `FDHestonDoubleBarrierEngine` | ❌ MISSING | 2D FD Heston double barrier |
| `FDHestonRebateEngine` | ❌ MISSING | FD Heston rebate |

### 3.2 Asian Engine Gaps
| QuantLib | Status | Notes |
|----------|--------|-------|
| `AnalyticContGeomAvPrice` | ✅ EXISTS | `geometric_asian_price()` |
| `TurnbullWakemanAsianEngine` | ✅ EXISTS | `turnbull_wakeman_price()` |
| `MCDiscrArithAvPrice` | ✅ EXISTS | `mc_asian_arithmetic_bs()` |
| `AnalyticDiscrGeomAvPrice` | ❌ MISSING | Discrete geometric avg price |
| `AnalyticDiscrGeomAvStrike` | ❌ MISSING | Geometric avg strike |
| `ChoiAsianEngine` | ❌ MISSING | Choi semi-analytic Asian |
| `ContArithAsianLevyEngine` | ❌ MISSING | Levy continuous arithmetic |
| `FDBlackScholesAsianEngine` | ❌ MISSING | FD Asian |
| `MCDiscrArithAvPriceHeston` | ❌ MISSING | MC Asian under Heston |
| `MCDiscrArithAvStrike` | ❌ MISSING | MC avg strike |
| `MCDiscrGeomAvPrice` | ❌ MISSING | MC geometric average |
| `MCDiscrGeomAvPriceHeston` | ❌ MISSING | MC geometric Heston |

### 3.3 Basket & Spread Engine Gaps
| QuantLib | Status | Notes |
|----------|--------|-------|
| `KirkEngine` | ✅ EXISTS | `kirk_spread_price()` |
| `BjerksundStenslandSpreadEngine` | ✅ EXISTS | `bjerksund_stensland_spread()` |
| `ChoiBasketEngine` | ❌ MISSING | Choi basket |
| `DengLiZhouBasketEngine` | ❌ MISSING | DLZ basket |
| `FD2DBlackScholesVanillaEngine` | ❌ MISSING | 2D FD basket/spread |
| `FDNDimBlackScholesVanillaEngine` | ❌ MISSING | N-D FD |
| `MCAmericanBasketEngine` | ❌ MISSING | MC American basket |
| `MCEuropeanBasketEngine` | ❌ MISSING | MC European basket |
| `OperatorSplittingSpreadEngine` | ❌ MISSING | OS spread |
| `SingleFactorBSMBasketEngine` | ❌ MISSING | Moment-matching basket |
| `SpreadBlackScholesVanillaEngine` | ❌ MISSING | FD spread |
| `StulzEngine` | ❌ MISSING | Stulz basket analytic |

### 3.4 Lookback Engine Gaps
| QuantLib | Status | Notes |
|----------|--------|-------|
| `AnalyticContinuousFixedLookback` | ✅ EXISTS | `fixed_lookback_price()` |
| `AnalyticContinuousFloatingLookback` | ✅ EXISTS | `floating_lookback_price()` |
| `MCLookbackEngine` | ✅ EXISTS | `mc_lookback_price()` |
| `AnalyticContPartialFixedLookback` | ❌ MISSING | Partial fixed lookback |
| `AnalyticContPartialFloatingLookback` | ❌ MISSING | Partial floating lookback |

### 3.5 Exotic Engine Completeness
| QuantLib | Status | Notes |
|----------|--------|-------|
| `AnalyticEuropeanMargrabeEngine` | ✅ (assumed in analytic/) | Margrabe exchange |
| `AnalyticAmericanMargrabeEngine` | ❌ MISSING | American Margrabe |
| `AnalyticComplexChooserEngine` | ✅ EXISTS | In compound.py |
| `AnalyticCompoundOptionEngine` | ✅ EXISTS | `compound_option_price()` |
| `AnalyticSimpleChooserEngine` | ✅ EXISTS | In compound.py area |
| `AnalyticHolderExtensibleEngine` | ❌ MISSING | Needs engine |
| `AnalyticWriterExtensibleEngine` | ❌ MISSING | Needs engine |
| `AnalyticTwoAssetCorrelationEngine` | ❌ MISSING | Two-asset correlation |

**Files to create**:
- `ql_jax/engines/analytic/barrier_binary.py` — Binary barrier engines
- `ql_jax/engines/analytic/partial_barrier.py` — Partial time barrier
- `ql_jax/engines/analytic/asian_variants.py` — Discrete geometric, Choi, Levy
- `ql_jax/engines/analytic/basket.py` — Stulz, DLZ, Choi, moment-matching
- `ql_jax/engines/analytic/lookback_partial.py` — Partial lookback
- `ql_jax/engines/analytic/exotic_engines.py` — American Margrabe, extensible, two-asset correlation
- `ql_jax/engines/fd/heston_barrier.py` — FD Heston barrier, double barrier, rebate
- `ql_jax/engines/fd/asian.py` — FD Asian
- `ql_jax/engines/mc/basket.py` — MC European/American basket
- `ql_jax/engines/mc/asian_variants.py` — MC Asian strike, geometric, Heston

**Tests**: `tests/test_barrier_engines.py`, `tests/test_asian_basket_engines.py` (~40 tests)

---

## Phase 4: Missing Pricing Engines — Interest Rate & Credit (Priority: HIGH)

### 4.1 Bond Engines
| QuantLib | Status | Notes |
|----------|--------|-------|
| `DiscountingBondEngine` | ✅ EXISTS | `discounting_bond_npv()` |
| `BondFunctions` | ✅ EXISTS | `BondFunctions` class |
| `BinomialConvertibleEngine` | ✅ EXISTS | `bsm_convertible_tree()` |
| `RiskyBondEngine` | ❌ MISSING | Credit-risky bond pricing |

### 4.2 Cap/Floor Engines
| QuantLib | Status | Notes |
|----------|--------|-------|
| `BlackCapFloorEngine` | ✅ EXISTS | `black_capfloor_price()` |
| `AnalyticCapFloorEngine` | ❌ MISSING | Short-rate analytics |
| `BachelierCapFloorEngine` | ❌ MISSING | Normal vol cap/floor |
| `Gaussian1dCapFloorEngine` | ❌ MISSING | GSR/Gaussian1d cap/floor |
| `MCHullWhiteEngine` (capfloor) | ❌ MISSING | MC HW cap/floor |
| `TreeCapFloorEngine` | ❌ MISSING | Lattice cap/floor |

### 4.3 Credit Engines
| QuantLib | Status | Notes |
|----------|--------|-------|
| `MidpointCdsEngine` | ✅ EXISTS | `midpoint_cds_npv()` |
| `IntegralCdsEngine` | ❌ MISSING | Integral CDS |
| `IsdaCdsEngine` | ❌ MISSING | ISDA standard CDS |

### 4.4 Swaption Engines
| QuantLib | Status | Notes |
|----------|--------|-------|
| `BlackSwaptionEngine` | ✅ EXISTS | `black_swaption_price()` |
| `HW Swaption` (Jamshidian) | ✅ EXISTS | `hw_swaption_price()` |
| `TreeSwaptionEngine` | ❌ MISSING | Trinomial tree swaption |
| `G2SwaptionEngine` | ❌ MISSING | G2++ swaption |
| `FDG2SwaptionEngine` | ❌ MISSING | FD G2 swaption |
| `FDHullWhiteSwaptionEngine` | ❌ MISSING | FD HW swaption |
| `JamshidianSwaptionEngine` | ❌ MISSING | Separate Jamshidian engine |
| `Gaussian1dSwaptionEngine` | ❌ MISSING | GSR swaption |
| `Gaussian1dNonstandardSwaptionEngine` | ❌ MISSING | GSR non-standard swaption |
| `BasketGeneratingEngine` | ❌ MISSING | Basket generating for Bermudan |

### 4.5 Swap & Forward Engines
| QuantLib | Status | Notes |
|----------|--------|-------|
| `DiscountingSwapEngine` | ✅ EXISTS | `discounting_swap_npv()` |
| `TreeSwapEngine` | ❌ MISSING | Lattice swap pricing |
| `CVASwapEngine` | ❌ MISSING | CVA-adjusted swap |
| `DiscountingFXForwardEngine` | ❌ MISSING | FX forward pricing |
| `ForwardPerformanceEngine` | ❌ MISSING | Forward performance |
| `MCForwardEuropeanBSEngine` | ❌ MISSING | MC forward start |
| `MCForwardEuropeanHestonEngine` | ❌ MISSING | MC forward start Heston |
| `ReplicatingVarianceSwapEngine` | ✅ EXISTS | `mc_variance_swap_price()` |

### 4.6 Other Engines
| QuantLib | Status | Notes |
|----------|--------|-------|
| `AnalyticCliquetEngine` | ✅ EXISTS | `cliquet_price()` |
| `AnalyticPerformanceEngine` | ❌ MISSING | Performance option |
| `MCPerformanceEngine` | ❌ MISSING | MC performance |
| `InflationCapFloorEngines` | ✅ EXISTS | `black_yoy_capfloor_price()` |
| `QuantoEngine` | ✅ EXISTS | `quanto_vanilla_price()` |

**Files to create**:
- `ql_jax/engines/capfloor/analytic.py` — Analytic, Bachelier, Gaussian1d, tree
- `ql_jax/engines/credit/isda.py` — ISDA CDS engine
- `ql_jax/engines/credit/integral.py` — Integral CDS engine
- `ql_jax/engines/swaption/tree.py` — Tree swaption
- `ql_jax/engines/swaption/g2.py` — G2 swaption
- `ql_jax/engines/swaption/gaussian1d.py` — GSR swaption
- `ql_jax/engines/swaption/fd.py` — FD HW/G2 swaption
- `ql_jax/engines/swap/tree.py` — Tree swap
- `ql_jax/engines/swap/cva.py` — CVA swap
- `ql_jax/engines/forward/fx.py` — FX forward, forward performance
- `ql_jax/engines/bond/risky.py` — Risky bond
- `ql_jax/engines/mc/forward_start.py` — MC forward start engines

**Tests**: `tests/test_ir_engines.py` (~30 tests)

---

## Phase 5: Finite Difference Framework Completion (Priority: MEDIUM-HIGH)

### 5.1 Missing FD Operators
QuantLib has ~40 FD operator classes. ql-jax has basic BS and Heston operators.

**Missing operators**:
- `FdmBatesOp` — Bates model operator
- `FdmCEVOp` — CEV model operator
- `FdmCIROp` — CIR model operator
- `FdmG2Op` — G2++ two-factor operator
- `FdmHullWhiteOp` — Hull-White operator
- `FdmHestonHullWhiteOp` — Hybrid Heston-HW operator
- `FdmBlackScholesFwdOp` — Forward BS (Fokker-Planck)
- `FdmHestonFwdOp` — Forward Heston
- `FdmLocalVolFwdOp` — Forward local vol
- `FdmSABROp` — SABR operator
- `FdmOrnsteinUhlenbeckOp` — OU operator
- `FdmSquareRootFwdOp` — Square root forward
- `FdmWienerOp` — Wiener process operator
- `NinePointLinearOp` — 2D nine-point stencil
- `SecondOrderMixedDerivativeOp` — Cross-derivative

### 5.2 Missing FD Schemes
| QuantLib | Status | Notes |
|----------|--------|-------|
| `CrankNicolsonScheme` | ✅ EXISTS | In `schemes.py` |
| `ExplicitEulerScheme` | ✅ EXISTS | In `schemes.py` |
| `ImplicitEulerScheme` | ✅ EXISTS | In `schemes.py` |
| `CraigSneydScheme` | ❌ MISSING | ADI scheme |
| `DouglasScheme` | ❌ MISSING | Douglas ADI |
| `HundsdorferScheme` | ❌ MISSING | Hundsdorfer ADI |
| `ModifiedCraigSneydScheme` | ❌ MISSING | Modified CS |
| `MethodOfLinesScheme` | ❌ MISSING | MOL scheme |
| `TRBDF2Scheme` | ❌ MISSING | TR-BDF2 |

### 5.3 Missing FD Meshers
| QuantLib | Status | Notes |
|----------|--------|-------|
| `Concentrating1dMesher` | ✅ EXISTS | In `meshers.py` |
| `Uniform1dMesher` | ✅ EXISTS | In `meshers.py` |
| `FdmBlackScholesMesher` | ✅ EXISTS | In `meshers.py` |
| `FdmHestonVarianceMesher` | ✅ EXISTS | In `meshers.py` |
| `FdmCEV1dMesher` | ❌ MISSING | CEV mesher |
| `ExponentialJump1dMesher` | ❌ MISSING | Jump mesher |
| `FdmBlackScholesMultiStrikeMesher` | ❌ MISSING | Multi-strike mesher |
| `FdmSimpleProcess1dMesher` | ❌ MISSING | Generic process mesher |
| `Predefined1dMesher` | ❌ MISSING | User-specified grid |

### 5.4 Missing FD Step Conditions & Boundary Conditions
- `FdmAmericanStepCondition` — ✅ EXISTS
- `FdmBermudanStepCondition` — ❌ MISSING
- `FdmSimpleSwingCondition` — ❌ MISSING
- `FdmSimpleStorageCondition` — ❌ MISSING
- `FdmArithmeticAverageCondition` — ❌ MISSING
- `FdmSnapshotCondition` — ❌ MISSING
- `FdmTimeDependentDirichletBoundary` — ❌ MISSING
- `FdmDiscountDirichletBoundary` — ❌ MISSING

### 5.5 Missing FD Solvers
- `Fdm1dSolver` — ❌ MISSING (generic 1D)
- `Fdm2dSolver` — ❌ MISSING (generic 2D)
- `Fdm3dSolver` — ❌ MISSING (generic 3D)
- `FdmNdimSolver` — ❌ MISSING (generic N-D)
- `FdmBatesSolver` — ❌ MISSING
- `FdmCIRSolver` — ❌ MISSING
- `FdmG2Solver` — ❌ MISSING
- `FdmHullWhiteSolver` — ❌ MISSING
- `FdmHestonHullWhiteSolver` — ❌ MISSING

### 5.6 Risk-Neutral Density Calculators
- `BSMRndCalculator` — ❌ MISSING
- `GBSMRndCalculator` — ❌ MISSING
- `CEVRndCalculator` — ❌ MISSING
- `HestonRndCalculator` — ❌ MISSING
- `LocalVolRndCalculator` — ❌ MISSING
- `SquareRootProcessRndCalculator` — ❌ MISSING

**Files to create**:
- `ql_jax/methods/finitedifferences/operators_2d.py` — 2D operators (nine-point, mixed derivative)
- `ql_jax/methods/finitedifferences/model_operators.py` — Bates, CEV, CIR, G2, HW, SABR operators
- `ql_jax/methods/finitedifferences/adi_schemes.py` — ADI schemes (Craig-Sneyd, Douglas, Hundsdorfer)
- `ql_jax/methods/finitedifferences/meshers_extended.py` — CEV, exponential jump, multi-strike meshers
- `ql_jax/methods/finitedifferences/step_conditions_ext.py` — Bermudan, swing, storage, snapshot conditions
- `ql_jax/methods/finitedifferences/generic_solvers.py` — Generic 1D/2D/3D/ND solvers
- `ql_jax/methods/finitedifferences/rnd_calculators.py` — Risk-neutral density calculators

**Tests**: `tests/test_fd_framework.py` (~35 tests)

---

## Phase 6: Term Structure Gaps (Priority: MEDIUM-HIGH)

### 6.1 Yield Term Structures
| QuantLib | Status | Notes |
|----------|--------|-------|
| `FlatForward` | ✅ EXISTS | |
| `DiscountCurve` | ✅ EXISTS | |
| `ZeroCurve` | ✅ EXISTS | |
| `ForwardCurve` | ✅ EXISTS | |
| `PiecewiseYieldCurve` | ✅ EXISTS | |
| `FittedBondDiscountCurve` | ✅ EXISTS | |
| `ImpliedTermStructure` | ✅ EXISTS | |
| `ForwardSpreadedTermStructure` | ✅ EXISTS | |
| `ZeroSpreadedTermStructure` | ✅ EXISTS | |
| `CompositeZeroYieldStructure` | ✅ EXISTS | |
| `UltimateForwardTermStructure` | ✅ EXISTS | |
| `PiecewiseZeroSpreadedTermStructure` | ✅ EXISTS | |
| `InterpolatedSimpleZeroCurve` | ❌ MISSING | Simple zero rate interpolation |
| `SpreadDiscountCurve` | ❌ MISSING | Spread on discount curve |
| `PiecewiseForwardSpreadedTermStructure` | ❌ MISSING | Forward spread piecewise |
| `PiecewiseSpreadYieldCurve` | ❌ MISSING | Piecewise spread bootstrapping |
| `QuantoTermStructure` | ❌ MISSING | Quanto-adjusted yield curve |
| `MultiCurve` | ❌ MISSING | Multi-curve framework |

### 6.2 Additional Rate Helpers
| QuantLib | Status | Notes |
|----------|--------|-------|
| `DepositRateHelper` | ✅ EXISTS | |
| `FraRateHelper` | ✅ EXISTS | |
| `SwapRateHelper` | ✅ EXISTS | |
| `OISRateHelper` | ✅ EXISTS | |
| `FuturesRateHelper` | ✅ EXISTS | |
| `BondHelper` / `FixedRateBondHelper` | ✅ EXISTS | |
| `MultipleResetsSwapHelper` | ❌ MISSING | |
| `OvernightIndexFutureRateHelper` | ❌ MISSING | |

### 6.3 Volatility Term Structures
| QuantLib | Status | Notes |
|----------|--------|-------|
| `BlackConstantVol` | ✅ EXISTS | |
| `BlackVarianceCurve` | ✅ EXISTS | |
| `BlackVarianceSurface` | ✅ EXISTS | |
| `LocalVolSurface` | ✅ EXISTS | |
| `LocalConstantVol` | ✅ EXISTS | |
| `ImpliedVolTermStructure` | ✅ EXISTS | |
| `HestonBlackVolSurface` | ✅ EXISTS | |
| `AndreasenHugeVolatilityAdapter` | ✅ EXISTS | |
| `BlackVolSurfaceDelta` | ❌ MISSING | Delta-parameterized vol surface |
| `FixedLocalVolSurface` | ❌ MISSING | Fixed local vol from grid |
| `GridModelLocalVolSurface` | ❌ MISSING | Grid model local vol |
| `LocalVolCurve` | ❌ MISSING | 1D local vol |
| `NoExceptLocalVolSurface` | ❌ MISSING | Exception-safe local vol |
| `PiecewiseBlackVarianceSurface` | ❌ MISSING | Piecewise black variance |
| `BlackVolTimeExtrapolation` | ❌ MISSING | Time extrapolation for vol |

### 6.4 Swaption Volatility
| QuantLib | Status | Notes |
|----------|--------|-------|
| `SwaptionConstantVol` | ✅ EXISTS | |
| `SwaptionVolMatrix` | ✅ EXISTS | |
| `SwaptionVolCube` | ✅ EXISTS | |
| `InterpolatedSwaptionVolatilityCube` | ❌ MISSING | Interpolated cube |
| `SABRSwaptionVolatilityCube` | ❌ MISSING | SABR cube |
| `ZABRSwaptionVolatilityCube` | ❌ MISSING | ZABR cube |
| `Gaussian1dSwaptionVolatility` | ❌ MISSING | GSR-based swaption vol |
| `CMSMarket` / `CMSMarketCalibration` | ❌ MISSING | CMS market calibration |

### 6.5 Cap/Floor Volatility
| QuantLib | Status | Notes |
|----------|--------|-------|
| `ConstantCapFloorTermVol` | ✅ EXISTS | |
| `CapFloorTermVolCurve` | ✅ EXISTS | |
| `CapFloorTermVolSurface` | ✅ EXISTS | |
| `ConstantOptionletVol` | ✅ EXISTS | |
| `OptionletStripper1` | ❌ MISSING | Optionlet stripping method 1 |
| `OptionletStripper2` | ❌ MISSING | Optionlet stripping method 2 |
| `StrippedOptionlet` | ❌ MISSING | Stripped optionlet data |
| `StrippedOptionletAdapter` | ❌ MISSING | Adapter |
| `CapletVarianceCurve` | ❌ MISSING | Caplet variance curve |
| `SpreadedOptionletVol` | ✅ EXISTS | |

### 6.6 Inflation Volatility
| QuantLib | Status | Notes |
|----------|--------|-------|
| `CPIVolatilityStructure` | ❌ MISSING | CPI vol structure |
| `ConstantCPIVolatility` | ❌ MISSING | Constant CPI vol |
| `YoYInflationOptionletVolatilityStructure` | ❌ MISSING | YoY optionlet vol |

### 6.7 Credit Term Structures
| QuantLib | Status | Notes |
|----------|--------|-------|
| `FlatHazardRate` | ✅ EXISTS | |
| `InterpolatedHazardRateCurve` | ✅ EXISTS | |
| `InterpolatedSurvivalProbabilityCurve` | ✅ EXISTS | |
| `PiecewiseDefaultCurve` | ✅ EXISTS | |
| `InterpolatedDefaultDensityCurve` | ❌ MISSING | Interpolated default density |
| `DefaultDensityStructure` | ❌ MISSING | Base class |

### 6.8 Smile Sections
| QuantLib | Status | Notes |
|----------|--------|-------|
| `FlatSmileSection` | ✅ EXISTS | |
| `InterpolatedSmileSection` | ✅ EXISTS | |
| `SABRSmileSection` | ✅ EXISTS | |
| `AtmSmileSection` | ❌ MISSING | ATM smile section |
| `AtmAdjustedSmileSection` | ❌ MISSING | Adjusted ATM |
| `KahaleSmileSection` | ❌ MISSING | Kahale arbitrage-free |
| `Gaussian1dSmileSection` | ❌ MISSING | GSR smile section |
| `SABRInterpolatedSmileSection` | ❌ MISSING | SABR interpolated |
| `ZABRSmileSection` | ❌ MISSING | ZABR smile |
| `ZABRInterpolatedSmileSection` | ❌ MISSING | ZABR interpolated |

**Files to create/modify**:
- `ql_jax/termstructures/yield_/quanto.py` — Quanto term structure
- `ql_jax/termstructures/yield_/spread_curves.py` — SpreadDiscountCurve, piecewise spread curves
- `ql_jax/termstructures/yield_/rate_helpers.py` — Add MultipleResetsSwapHelper, OvernightFutureHelper
- `ql_jax/termstructures/volatility/equityfx_extended.py` — Delta surface, piecewise, local vol curve
- `ql_jax/termstructures/volatility/swaption_cube.py` — Interpolated/SABR/ZABR cubes
- `ql_jax/termstructures/volatility/optionlet_extended.py` — Optionlet strippers 1/2, stripped optionlet
- `ql_jax/termstructures/volatility/inflation_vol.py` — CPI vol, YoY optionlet vol
- `ql_jax/termstructures/volatility/smile_section_ext.py` — Kahale, ATM, Gaussian1d, ZABR smile sections
- `ql_jax/termstructures/credit/default_density.py` — Default density structures

**Tests**: `tests/test_termstructures_phase6.py` (~35 tests)

---

## Phase 7: Model Gaps (Priority: MEDIUM)

### 7.1 Short-Rate Models
| QuantLib | Status | Notes |
|----------|--------|-------|
| `Vasicek` | ✅ EXISTS | |
| `HullWhite` | ✅ EXISTS | |
| `BlackKarasinski` | ✅ EXISTS | |
| `CoxIngersollRoss` | ✅ EXISTS | |
| `ExtendedCoxIngersollRoss` | ✅ EXISTS | |
| `G2` | ✅ EXISTS | |
| `GSR` | ✅ EXISTS | |
| `Gaussian1dModel` | ✅ EXISTS | |
| `MarkovFunctional` | ✅ EXISTS | |
| **All short-rate models implemented** | ✅ | |

### 7.2 Equity Models
| QuantLib | Status | Notes |
|----------|--------|-------|
| `HestonModel` | ✅ EXISTS | |
| `BatesModel` | ✅ EXISTS | |
| `GJRGARCHModel` | ✅ EXISTS | |
| `HestonSLVFDMModel` | ✅ EXISTS | |
| `PiecewiseTimeDependentHestonModel` | ✅ EXISTS | |
| `HestonModelHelper` | ✅ EXISTS | |
| `HestonSLVMCModel` | ❌ MISSING | MC-based SLV calibration |

### 7.3 Calibration
| QuantLib | Status | Notes |
|----------|--------|-------|
| `CalibrationHelper` | ✅ EXISTS | |
| `SwaptionHelper` | ✅ EXISTS | |
| `CapHelper` | ✅ EXISTS | |

### 7.4 Volatility Models
| QuantLib | Status | Notes |
|----------|--------|-------|
| `GARCH` | ✅ EXISTS | |
| `ConstantEstimator` | ❌ MISSING | Historical vol constant |
| `GarmanKlass` | ❌ MISSING | Garman-Klass estimator |
| `SimpleLocalEstimator` | ❌ MISSING | Simple local vol estimator |

### 7.5 Market Models (LIBOR Market Model)
QuantLib has **~100+ files** across market models. ql-jax has a skeletal implementation.

**Existing in ql-jax**:
- `LMMCurveState`, `CoterminalCurveState`
- `LMMConfig`, correlation models, drift computation
- Basic evolvers, vol models

**Missing from QuantLib market models**:
- Full `AccountingEngine` / `PathwiseAccountingEngine`
- `BrownianGenerator` implementations (Sobol, SobolBridge)
- Callability models (upper bound engines, LS calibrator)
- `Discounter` / `PathwiseDiscounter`
- `EvolutionDescription`
- Full product hierarchy (MultiStep, OneStep, Pathwise products)
- `ProxyGreekEngine`
- `SwapForwardMappings`
- Many correlation models
- Full evolver implementations (LogNormal, Forward, Predictor-Corrector)

**Files to create**:
- `ql_jax/models/equity/heston_slv_mc.py` — MC SLV model
- `ql_jax/models/volatility/estimators.py` — ConstantEstimator, GarmanKlass, SimpleLocal
- `ql_jax/models/marketmodels/accounting.py` — Accounting engines
- `ql_jax/models/marketmodels/brownian_generators.py` — Brownian generators
- `ql_jax/models/marketmodels/products.py` — Product hierarchy
- `ql_jax/models/marketmodels/evolvers_full.py` — Extended evolvers

**Tests**: `tests/test_models_phase7.py` (~25 tests)

---

## Phase 8: Math Module Gaps (Priority: MEDIUM)

### 8.1 Core Math Utilities
| QuantLib | Status | Notes |
|----------|--------|-------|
| `Array` | ✅ (jnp.array) | JAX native |
| `Matrix` | ✅ (jnp.array) | JAX native |
| `factorial` | ❌ MISSING | Factorial function |
| `modifiedBessel` | ❌ MISSING | Modified Bessel functions |
| `incompleteBeta/Gamma` | ❌ MISSING | Special functions (use scipy?) |
| `errorFunction` | ❌ MISSING | Error function (jax.scipy) |
| `BSpline` | ❌ MISSING | B-spline basis |
| `BernsteinPolynomial` | ❌ MISSING | Bernstein polynomial |
| `FastFourierTransform` | ❌ MISSING | FFT wrapper |
| `RichardsonExtrapolation` | ❌ MISSING | Richardson extrapolation |
| `GeneralLinearLeastSquares` | ❌ MISSING | General linear regression |
| `LinearLeastSquaresRegression` | ❌ MISSING | Weighted linear regression |
| `KernelFunctions` | ❌ MISSING | Kernel functions |
| `PascalTriangle` | ❌ MISSING | Combinatorial utility |
| `PrimeNumbers` | ❌ MISSING | Prime number generation |
| `NumericalDifferentiation` | ❌ MISSING | Numerical derivatives |
| `TransformedGrid` | ❌ MISSING | Grid transformation |

### 8.2 Distributions
| QuantLib | Status | Notes |
|----------|--------|-------|
| `NormalDistribution` | ✅ EXISTS | |
| `BinomialDistribution` | ✅ EXISTS | |
| `BivariateNormalDistribution` | ✅ EXISTS | |
| `ChiSquaredDistribution` | ✅ EXISTS | |
| `GammaDistribution` | ✅ EXISTS | |
| `PoissonDistribution` | ✅ EXISTS | |
| `StudentTDistribution` | ✅ EXISTS | |
| `BivariateStudentTDistribution` | ❌ MISSING | Bivariate Student-t |

### 8.3 Integrals
| QuantLib | Status | Notes |
|----------|--------|-------|
| `SimpsonIntegral` | ✅ EXISTS | |
| `TrapezoidIntegral` | ✅ EXISTS | |
| `GaussianQuadratures` | ✅ EXISTS | |
| `SegmentIntegral` | ❌ MISSING | Segment integral |
| `GaussLobattoIntegral` | ❌ MISSING | Gauss-Lobatto |
| `KronrodIntegral` | ❌ MISSING | Gauss-Kronrod |
| `TanhSinhIntegral` | ❌ MISSING | Tanh-sinh quadrature |
| `ExpSinhIntegral` | ❌ MISSING | Exp-sinh quadrature |
| `FilonIntegral` | ❌ MISSING | Filon (oscillatory) |
| `DiscreteIntegrals` | ❌ MISSING | Discrete integration |
| `ExponentialIntegrals` | ❌ MISSING | Exponential integrals |
| `TwoDimensionalIntegral` | ❌ MISSING | 2D integration |
| `MomentBasedGaussianPolynomial` | ❌ MISSING | Moment-based quadrature |
| `GaussLaguerreCosinePolynomial` | ❌ MISSING | Laguerre cosine |

### 8.4 Interpolations
All major interpolation types implemented. Missing:
| QuantLib | Status | Notes |
|----------|--------|-------|
| `BicubicSplineInterpolation` | ❌ MISSING | 2D bicubic spline |
| `BilinearInterpolation` | ❌ MISSING | 2D bilinear |
| `MultiCubicSpline` | ❌ MISSING | N-D cubic spline |

### 8.5 Optimization
| QuantLib | Status | Notes |
|----------|--------|-------|
| `LevenbergMarquardt` | ✅ EXISTS | |
| `Simplex` | ✅ EXISTS | |
| `BFGS` | ✅ EXISTS | |
| `ConjugateGradient` | ✅ EXISTS | |
| `DifferentialEvolution` | ✅ EXISTS | |
| `SimulatedAnnealing` | ✅ EXISTS | |
| `SteepestDescent` | ✅ EXISTS | |
| `ProjectedConstraint` | ❌ MISSING | Projected constraints |
| `ProjectedCostFunction` | ❌ MISSING | Projected cost function |
| `Projection` | ❌ MISSING | Parameter projection |
| `SphereCylinder` | ❌ MISSING | Sphere-cylinder constraint |

### 8.6 Random Number Generation
| QuantLib | Status | Notes |
|----------|--------|-------|
| `MersenneTwister` | ✅ EXISTS (JAX PRNG) | |
| `SobolRSG` | ✅ EXISTS | |
| `FaureRSG` | ✅ EXISTS | |
| `SobolBrownianBridgeRSG` | ✅ EXISTS | |
| `InverseCumulativeRNG` | ✅ EXISTS | |
| `HaltonRSG` | ❌ MISSING | Halton quasi-random |
| `LatticeRSG` | ❌ MISSING | Lattice rules |
| `RandomizedLDS` | ❌ MISSING | Randomized low-discrepancy |
| `KnuthUniformRNG` | ❌ MISSING | Knuth generator |
| `RanluxUniformRNG` | ❌ MISSING | RANLUX generator |
| `Xoshiro256StarStarRNG` | ❌ MISSING | Xoshiro256** |
| `Burley2020SobolRSG` | ❌ MISSING | Burley scrambled Sobol |
| `StochasticCollocationInvCDF` | ❌ MISSING | Stochastic collocation |

### 8.7 Solvers
All 1D root-finding solvers implemented (Brent, Bisection, Newton, Secant, Ridder, Halley, FalsePosition, NewtonSafe). **Complete**.

### 8.8 Statistics
| QuantLib | Status | Notes |
|----------|--------|-------|
| `GeneralStatistics` | ✅ EXISTS | |
| `RiskStatistics` | ✅ EXISTS | |
| `IncrementalStatistics` | ❌ MISSING | Online/streaming statistics |
| `ConvergenceStatistics` | ❌ MISSING | Convergence tracking |
| `DiscrepancyStatistics` | ❌ MISSING | Low-discrepancy stats |
| `SequenceStatistics` | ❌ MISSING | Multi-dimensional stats |
| `Histogram` | ❌ MISSING | Histogram class |

**Files to create**:
- `ql_jax/math/special.py` — Bessel, incomplete beta/gamma, error functions, factorial
- `ql_jax/math/bspline.py` — B-spline, Bernstein polynomial
- `ql_jax/math/fft.py` — FFT wrapper
- `ql_jax/math/richardson.py` — Richardson extrapolation
- `ql_jax/math/regression.py` — General linear least squares, weighted regression
- `ql_jax/math/distributions/bivariate_student_t.py` — Bivariate Student-t
- `ql_jax/math/integrals/advanced.py` — Lobatto, Kronrod, TanhSinh, Filon, 2D, discrete
- `ql_jax/math/interpolations/interp2d_ext.py` — Bicubic, bilinear
- `ql_jax/math/optimization/projection.py` — Projected constraints, projections
- `ql_jax/math/random/halton.py` — Halton sequences
- `ql_jax/math/random/randomized_lds.py` — Randomized LDS
- `ql_jax/math/statistics/advanced.py` — Incremental, convergence, discrepancy, sequence stats, histogram

**Tests**: `tests/test_math_phase8.py` (~30 tests)

---

## Phase 9: Monte Carlo Framework (Priority: MEDIUM)

### 9.1 Missing MC Infrastructure
| QuantLib | Status | Notes |
|----------|--------|-------|
| `BrownianBridge` | ✅ EXISTS | |
| `LongstaffSchwartzPathPricer` | ✅ EXISTS | |
| `Path` / `MultiPath` | ✅ EXISTS | |
| `PathGenerator` / `MultiPathGenerator` | ✅ EXISTS | |
| `MonteCarloModel` | ❌ MISSING | Generic MC framework class |
| `MCTraits` | ❌ MISSING | MC traits/type system |
| `EarlyExercisePathPricer` | ❌ MISSING | Generic early exercise |
| `ExerciseStrategy` | ❌ MISSING | Exercise strategy interface |
| `ParametricExercise` | ❌ MISSING | Parametric exercise |
| `LSMBasisSystem` | ❌ MISSING | LSM basis functions |
| `GenericLSRegression` | ❌ MISSING | Generic LS regression |
| `NodeData` | ❌ MISSING | Tree node data |
| `PathPricer` | ❌ MISSING | Abstract path pricer |

**Files to create**:
- `ql_jax/methods/montecarlo/framework.py` — MonteCarloModel, MCTraits, PathPricer
- `ql_jax/methods/montecarlo/early_exercise.py` — EarlyExercise, ExerciseStrategy, LSMBasis
- `ql_jax/methods/montecarlo/generic_ls.py` — Generic LS regression

**Tests**: `tests/test_mc_framework.py` (~15 tests)

---

## Phase 10: Lattice Framework Gaps (Priority: MEDIUM)

### 10.1 Missing Lattice Components
| QuantLib | Status | Notes |
|----------|--------|-------|
| `BinomialTree` | ✅ EXISTS | |
| `TrinomialTree` | ✅ EXISTS | |
| `BSMLattice` | ✅ EXISTS | |
| `Lattice` | ❌ MISSING | Abstract lattice base |
| `Lattice1D` | ❌ MISSING | 1D lattice base |
| `Lattice2D` | ❌ MISSING | 2D lattice |
| `TFLattice` | ❌ MISSING | Two-factor lattice |
| `Tree` | ❌ MISSING | Abstract tree base |
| `FiniteDifferenceModel` (lattice) | ❌ MISSING | FD model for lattices |

**Files to create**:
- `ql_jax/methods/lattices/__init__.py` — Package init
- `ql_jax/methods/lattices/lattice.py` — Lattice, Lattice1D, Lattice2D
- `ql_jax/methods/lattices/two_factor.py` — TFLattice, two-factor trees

**Tests**: `tests/test_lattice_framework.py` (~10 tests)

---

## Phase 11: Time Module Gaps (Priority: MEDIUM)

### 11.1 Missing Calendars
ql-jax has 26 calendars. QuantLib has 47.

**Missing calendars** (21):
`Argentina`, `Austria`, `Chile`, `CzechRepublic`, `Finland`, `Hungary`, `Iceland`, `Indonesia`, `Israel`, `NewZealand`, `Poland`, `Romania`, `Russia`, `SaudiArabia`, `Slovakia`, `Taiwan`, `Thailand`, `Turkey`, `Ukraine`, `Botswana`

### 11.2 Missing Day Counters
ql-jax uses a `DayCountConvention` enum mapping to calculation functions.

| QuantLib | Status | Notes |
|----------|--------|-------|
| `Actual360` | ✅ EXISTS | |
| `Actual365Fixed` | ✅ EXISTS | |
| `ActualActual` | ✅ EXISTS | |
| `Thirty360` | ✅ EXISTS | |
| `Business252` | ✅ EXISTS | |
| `Actual364` | ❌ MISSING | 364-day year |
| `Actual36525` | ❌ MISSING | 365.25-day year |
| `Actual366` | ❌ MISSING | 366-day year |
| `Thirty365` | ❌ MISSING | 30/365 convention |
| `One` | ❌ MISSING | No accrual |
| `SimpleDayCounter` | ❌ MISSING | Simple day counter |
| `YearFractionToDate` | ❌ MISSING | Inverse year fraction |

### 11.3 Missing Schedule/Date Utilities
| QuantLib | Status | Notes |
|----------|--------|-------|
| `Date` | ✅ EXISTS | |
| `Schedule` | ✅ EXISTS | |
| `Calendar` | ✅ EXISTS | |
| `DayCounter` | ✅ EXISTS | |
| `Period` | ❌ MISSING | Period class (months, years, etc.) |
| `IMM` | ❌ MISSING | IMM date utilities |
| `ASX` | ❌ MISSING | ASX date utilities |
| `ECB` | ❌ MISSING | ECB meeting dates |
| `Frequency` | Partial | Enum values exist |
| `DateGeneration` | Partial | Basic rules exist |
| `BusinessDayConvention` | Partial | |

**Files to create/modify**:
- `ql_jax/time/calendars.py` — Add 21 missing calendars
- `ql_jax/time/daycounter.py` — Add Actual364, Actual36525, Actual366, Thirty365, One
- `ql_jax/time/period.py` — Period class with arithmetic
- `ql_jax/time/imm.py` — IMM date utilities
- `ql_jax/time/ecb.py` — ECB dates

**Tests**: `tests/test_time_phase11.py` (~25 tests)

---

## Phase 12: Index Gaps (Priority: MEDIUM)

### 12.1 Missing IBOR Index Factory Functions
ql-jax has 20 IBOR/overnight indexes. QuantLib has ~42.

**Missing**:
`AUDLibor`, `BBSW`, `Bibor`, `BKBM`, `CADLibor`, `CDI`, `CDOR`, `DESTR`, `DKKLibor`, `Jibar`, `KOFR`, `Mosprime`, `NZDLibor`, `NZOCR`, `Pribor`, `Robor`, `SEKLibor`, `Shibor`, `SWESTR`, `THBFix`, `Tonar`, `TRLibor`, `Wibor`, `Zibor`

### 12.2 Missing Swap Index Factory Functions
QuantLib has 6 swap index implementations. ql-jax has the `SwapIndex` class but no factory functions.

**Missing**: `EuriborSwapIsdaFixA`, `UsdLiborSwapIsdaFixAm`, etc.

### 12.3 Missing Inflation Index Factory Functions
ql-jax has generic `ZeroInflationIndex` and `YoYInflationIndex`. QuantLib has 7 specific inflation indexes.

**Missing**: `AUCPI`, `EUHICP`, `FRHICP`, `UKHICP`, `UKRPI`, `USCPI`, `ZACPI`

### 12.4 Missing Index Infrastructure
| QuantLib | Status | Notes |
|----------|--------|-------|
| `IndexManager` | ❌ MISSING | Global index fixing store |
| `Region` | ❌ MISSING | Geographic region for inflation |

**Files to create/modify**:
- `ql_jax/indexes/ibor.py` — Add ~24 missing IBOR/overnight index factories
- `ql_jax/indexes/swap.py` — Add swap index factory functions
- `ql_jax/indexes/inflation.py` — Add specific inflation index factories
- `ql_jax/indexes/manager.py` — IndexManager for storing fixings

**Tests**: `tests/test_indexes_phase12.py` (~20 tests)

---

## Phase 13: Quotes Module (Priority: LOW)

### 13.1 Missing Quote Types
| QuantLib | Status | Notes |
|----------|--------|-------|
| `SimpleQuote` | ✅ EXISTS | |
| `CompositeQuote` | ❌ MISSING | f(quote1, quote2) |
| `DerivedQuote` | ❌ MISSING | f(quote) |
| `DeltaVolQuote` | ❌ MISSING | Delta-vol quote |
| `EurodollarFuturesQuote` | ❌ MISSING | ED futures quote |
| `ForwardSwapQuote` | ❌ MISSING | Forward swap rate |
| `ForwardValueQuote` | ❌ MISSING | Forward value |
| `FuturesConvAdjustmentQuote` | ❌ MISSING | Convexity adjustment |
| `ImpliedStdDevQuote` | ❌ MISSING | Implied std dev |
| `LastFixingQuote` | ❌ MISSING | Last index fixing |

**Files to create**:
- `ql_jax/quotes/composite.py` — CompositeQuote, DerivedQuote
- `ql_jax/quotes/market.py` — DeltaVolQuote, ForwardSwapQuote, ForwardValueQuote, etc.

**Tests**: `tests/test_quotes_phase13.py` (~15 tests)

---

## Phase 14: Cashflow Completeness (Priority: LOW)

### 14.1 Missing Cashflow Types
| QuantLib | Status | Notes |
|----------|--------|-------|
| `FixedRateCoupon` | ✅ EXISTS | |
| `FloatingRateCoupon` / `IborCoupon` | ✅ EXISTS | |
| `OvernightIndexedCoupon` | ✅ EXISTS | |
| `CappedFlooredCoupon` | ✅ EXISTS | |
| `CMSCoupon` | ✅ EXISTS | |
| `DigitalCoupon` / `DigitalIborCoupon` | ✅ EXISTS | |
| `CPICoupon` / `YoYInflationCoupon` | ✅ EXISTS | |
| `AverageBMACoupon` | ✅ EXISTS | |
| `RangeAccrualCoupon` | ✅ EXISTS | |
| `SubPeriodsCoupon` (MultipleResets) | ✅ EXISTS | |
| `EquityCashFlow` | ✅ EXISTS | |
| `Duration` | ✅ EXISTS | |
| `BlackIborCouponPricer` | ✅ EXISTS | |
| `AnalyticHaganPricer` | ✅ EXISTS | |
| `LinearTSRPricer` | ✅ EXISTS | |
| `BlackOvernightIndexedCouponPricer` | ❌ MISSING | Black overnight pricer |
| `CashFlowVectors` | ❌ MISSING | Leg-building utilities |
| `ConundrumPricer` | ❌ MISSING | CMS conundrum pricer |
| `CPICouponPricer` | ❌ MISSING | Dedicated CPI pricer |
| `RateAveraging` | ❌ MISSING | Rate averaging methods |

**Files to create/modify**:
- `ql_jax/cashflows/overnight_pricer.py` — Add Black overnight pricer
- `ql_jax/cashflows/vectors.py` — CashFlowVectors (leg builders)
- `ql_jax/cashflows/cms.py` — Add ConundrumPricer

**Tests**: `tests/test_cashflows_phase14.py` (~15 tests)

---

## Phase 15: Processes Completeness (Priority: LOW)

### 15.1 Missing Processes
| QuantLib | Status | Notes |
|----------|--------|-------|
| `BlackScholesProcess` | ✅ EXISTS | |
| `GeneralizedBSProcess` | ✅ EXISTS | |
| `HestonProcess` | ✅ EXISTS | |
| `BatesProcess` | ✅ EXISTS | |
| `HestonSLVProcess` | ✅ EXISTS | |
| `GJRGARCHProcess` | ✅ EXISTS | |
| `HullWhiteProcess` | ✅ EXISTS | |
| `G2Process` | ✅ EXISTS | |
| `CoxIngersollRossProcess` | ✅ EXISTS | |
| `Merton76Process` | ✅ EXISTS | |
| `OrnsteinUhlenbeckProcess` | ✅ EXISTS | |
| `SquareRootProcess` | ✅ EXISTS | |
| `GeometricBrownianMotionProcess` | ✅ EXISTS | |
| `StochasticProcessArray` | ✅ EXISTS | |
| `EulerDiscretization` | ✅ EXISTS | |
| `EndEulerDiscretization` | ✅ EXISTS | |
| `ForwardMeasureProcess` | ❌ MISSING | Forward measure |
| `GSRProcess` / `GSRProcessCore` | ❌ MISSING | GSR process |
| `HybridHestonHullWhiteProcess` | ❌ MISSING | Hybrid Heston-HW |
| `JointStochasticProcess` | ❌ MISSING | Joint process |
| `MfStateProcess` | ❌ MISSING | Markov-functional state |

**Files to create**:
- `ql_jax/processes/gsr.py` — GSR process
- `ql_jax/processes/hybrid.py` — HybridHestonHullWhite, JointStochasticProcess
- `ql_jax/processes/forward_measure.py` — ForwardMeasureProcess
- `ql_jax/processes/markov_functional_state.py` — MfStateProcess

**Tests**: `tests/test_processes_phase15.py` (~10 tests)

---

## Phase 16: Currencies Module (Priority: LOW)

### 16.1 Missing Currency Support
ql-jax has `currencies/__init__.py` but no currency definitions. QuantLib has full currency catalogs.

**Missing**:
- Africa currencies
- Americas currencies (USD, CAD, BRL, MXN, ARS, etc.)
- Asia currencies (JPY, CNY, HKD, SGD, KRW, INR, etc.)
- Europe currencies (EUR, GBP, CHF, SEK, NOK, DKK, PLN, etc.)
- Oceania currencies (AUD, NZD)
- Crypto currencies
- `ExchangeRateManager`

**Files to create**:
- `ql_jax/currencies/america.py`
- `ql_jax/currencies/europe.py`
- `ql_jax/currencies/asia.py`
- `ql_jax/currencies/africa.py`
- `ql_jax/currencies/oceania.py`
- `ql_jax/currencies/exchange_rate.py`

**Tests**: `tests/test_currencies_phase16.py` (~10 tests)

---

## Phase 17: Patterns & Utilities (Priority: LOW)

### 17.1 Missing Patterns
| QuantLib | Status | Notes |
|----------|--------|-------|
| `Observable` | ✅ EXISTS | |
| `LazyObject` | ✅ EXISTS | |
| `Visitor` | ✅ EXISTS | |
| `Singleton` | ❌ MISSING | Singleton pattern |
| `CuriouslyRecurring` | N/A | CRTP, not needed in Python |

### 17.2 Missing Utilities
- `Settings` — ❌ MISSING (evaluation date, etc.)
- `SavedSettings` — ❌ MISSING (context manager for settings)
- `tracing` / `dataformatters` — ❌ MISSING

**Files to create**:
- `ql_jax/settings.py` — Global settings (evaluation date)
- `ql_jax/patterns/singleton.py` — Singleton pattern

---

## Phase 18: Extended Tests & Examples (Priority: MEDIUM)

### 18.1 New Test Files for Each Phase
After each phase, create dedicated tests:
- `tests/test_instruments_phase1.py` (~30 tests)
- `tests/test_american_engines.py` (~40 tests)
- `tests/test_barrier_engines.py` (~40 tests)
- `tests/test_ir_engines.py` (~30 tests)
- `tests/test_fd_framework.py` (~35 tests)
- `tests/test_termstructures_phase6.py` (~35 tests)
- `tests/test_models_phase7.py` (~25 tests)
- `tests/test_math_phase8.py` (~30 tests)
- `tests/test_mc_framework.py` (~15 tests)
- `tests/test_lattice_framework.py` (~10 tests)
- `tests/test_time_phase11.py` (~25 tests)
- `tests/test_indexes_phase12.py` (~20 tests)
- `tests/test_quotes_phase13.py` (~15 tests)
- `tests/test_cashflows_phase14.py` (~15 tests)
- `tests/test_processes_phase15.py` (~10 tests)
- `tests/test_currencies_phase16.py` (~10 tests)

**Total new tests: ~385**

### 18.2 New Examples
- `examples/american_options.py` — BAW, BjS, QD+ comparison
- `examples/heston_variants.py` — All Heston engine comparison
- `examples/multi_curve_bootstrap.py` — OIS + IBOR dual curve
- `examples/cds_isda_pricing.py` — ISDA CDS pricing
- `examples/bermudan_swaption.py` — Bermudan swaption pricing
- `examples/basket_pricing.py` — Multi-asset basket/spread options
- `examples/lmm_calibration.py` — LIBOR Market Model
- `examples/fd_2d_pricing.py` — 2D finite differences

---

## Implementation Priority Summary

| Priority | Phases | Est. New Files | Est. New Tests |
|----------|--------|---------------|----------------|
| **HIGH** | 1-4 | ~25 files | ~140 tests |
| **MEDIUM-HIGH** | 5-6 | ~16 files | ~70 tests |
| **MEDIUM** | 7-12 | ~20 files | ~105 tests |
| **LOW** | 13-17 | ~15 files | ~50 tests |

**Grand Total**: ~76 new files, ~385 new tests, targeting ~980 total tests.

---

## Notes

1. **Market Models (LMM)**: QuantLib has ~100+ files for LIBOR Market Models. Full port would be massive. Recommend implementing core features only (accounting engine, basic products, Sobol generators).

2. **Experimental**: QuantLib's `experimental/` directory (~263 hpp, ~158 cpp) is excluded from this analysis. It contains: Asian basket, arithmetic average OIS, auto-correlation vol, catbonds, commodity, convertible bonds (extended), credit models, FD Heston, forward options, GARCH, hybrid models, inflation, Kirk, math extensions, MC variance swap, MQ3, noarbSABR, processes, risk stats, short-rate models, stub, swaption vol cube, template, variance gamma, vol model. Could be added as a future Phase 19+.

3. **JAX Considerations**: Some QuantLib patterns (Observer/Observable, lazy evaluation, handle/link) are C++ idioms. Python/JAX replacements are simpler. Focus on mathematical/financial content, not design pattern parity.

4. **Thread Safety / Settings**: QuantLib uses `Settings::instance().evaluationDate()`. ql-jax should implement a simple `Settings` class for evaluation date management.
