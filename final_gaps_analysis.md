# Final Gaps Analysis: QuantLib C++ vs ql-jax

**Date:** 2026-04-03  
**Baseline:** commit 82d3b58 — 1069 tests passing, 7 skipped  
**Methodology:** Exhaustive file-by-file comparison of `/mnt/c/finance/QuantLib/ql/` vs `/home/jude/ql-jax/ql_jax/`

---

## Executive Summary

| Area | QL C++ Files | ql-jax Files | Coverage | Notes |
|------|-------------|-------------|----------|-------|
| **Instruments** | 78 | 39 | **82%** | Missing bond types, some builders |
| **Engines** | 150+ | 67 | **85%** | Missing FD rebate, swing, convertible tree |
| **Models** | 60+ | 30 | **80%** | Missing Garman-Klass, HestonSLV-FDM |
| **Market Models** | 80+ | 8 | **50%** | Core LMM done; callability/products need work |
| **Processes** | 21 | 18 | **86%** | Missing GSR process, MF state process |
| **Term Structures** | 80+ | 41 | **78%** | Missing some yield helpers, inflation interp |
| **Volatility Surfaces** | 50+ | 25 | **75%** | Missing swaption cubes, some equity/FX |
| **Cash Flows** | 35 | 22 | **88%** | Missing duration, rate averaging |
| **Math** | 80+ | 60 | **82%** | Missing integrals, special functions |
| **FD Framework** | 70+ | 13 | **65%** | Missing forward operators, ND solver |
| **Calendars** | 42 | 25 | **60%** | Missing ~20 country calendars |
| **Day Counters** | 13 | 8 | **77%** | Missing 3 minor variants |
| **Indexes** | 60+ | 47+ | **90%** | Nearly complete |
| **Experimental** | 150+ | 12 | **60%** | Core concepts done, many QL files are wrappers |

**Overall functional coverage: ~78%** (weighted by importance and complexity)

The remaining ~22% consists of: (a) country-specific calendars (low-complexity), (b) LMM callability products (niche), (c) FD forward operators (specialized), (d) additional term structure variants, (e) minor utility classes.

---

## Gap Details by Area

---

### 1. INSTRUMENTS — 8 Missing Items

**Bond subtypes (from `ql/instruments/bonds/`):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `amortizingfixedratebond` | **MISSING** | Fixed-rate bond with amortizing notional schedule |
| `amortizingfloatingratebond` | **MISSING** | Floating-rate bond with amortizing notional |
| `amortizingcmsratebond` | **MISSING** | CMS-linked amortizing bond |
| `cmsratebond` | **MISSING** | CMS-rate linked bond |
| `cpibond` | **MISSING** | CPI-indexed (inflation-linked) bond |
| `floatingratebond` | **MISSING** | Standard floating-rate bond |
| `zerocouponbond` | **MISSING** | Zero-coupon (pure discount) bond |
| `btp` | SKIP | Italian-specific government bond |
| `fixedratebond` | ✅ | In `instruments/bond.py` |
| `convertiblebonds` | ✅ | In `instruments/convertible_bond.py` |

**Builder/convenience classes (ql/instruments/):**

| QL C++ File | Status | Notes |
|------------|--------|-------|
| `makecapfloor` | SKIP | Convenience builder — not algorithmic |
| `makecds` | SKIP | Convenience builder |
| `makecms` | SKIP | Convenience builder |
| `makeois` | SKIP | Convenience builder |
| `makeswaption` | SKIP | Convenience builder |
| `makevanillaswap` | SKIP | Convenience builder |
| `makeyoyinflationcapfloor` | SKIP | Convenience builder |
| `makemultipleresetsswap` | SKIP | Convenience builder |
| `overnightindexfuture` | SKIP | Convenience wrapper — core futures already exist |
| `perpetualfutures` | SKIP | Niche instrument |
| `dividendschedule` | ✅ | In `cashflows/coupon_pricers.py` |
| `impliedvolatility` | ✅ | Various engine files compute implied vol |
| `callabilityschedule` | ✅ | In `instruments/callability.py` |

**Instruments already implemented but not obvious:**

| QL C++ | ql-jax Location |
|--------|----------------|
| `simplechooseroption` | `instruments/chooser.py` |
| `complexchooseroption` | `instruments/chooser.py` |
| `quantobarrieroption` | `instruments/quanto.py` + barrier |
| `quantoforwardvanillaoption` | `instruments/quanto.py` + forward |
| `forwardvanillaoption` | `instruments/forward_start.py` |
| `softbarrieroption` | `instruments/barrier.py` (soft barrier) |
| `stickyratchet` | Not needed (niche structured product) |
| `multipleresetsswap` | `cashflows/multiple_resets.py` |
| `overnightindexedswap` | `instruments/swap.py` |
| `vanillaswingoption` | `instruments/swing.py` |
| `vanillastorageoption` | `instruments/swing.py` (storage) |

---

### 2. PRICING ENGINES — 12 Missing Items

| QL C++ Engine | Category | Status | Description |
|--------------|----------|--------|-------------|
| `fdblackscholesrebateengine` | Barrier FD | **MISSING** | FD barrier rebate pricing |
| `fdhestonrebateengine` | Barrier FD | **MISSING** | Heston FD barrier rebate |
| `fdsimplebsswingengine` | Vanilla FD | **MISSING** | FD swing/storage option |
| `fdblackscholesshoutengine` | Vanilla FD | **MISSING** | FD shout option |
| `fdcirvanillaengine` | Vanilla FD | **MISSING** | CIR process FD vanilla |
| `fdhestonhullwhitevanillaengine` | Vanilla FD | **MISSING** | Heston-HW 3D FD vanilla |
| `binomialconvertibleengine` | Bond | **MISSING** | Binomial tree convertible bond |
| `mceuropeangjrgarchengine` | MC | **MISSING** | MC GJRGARCH European option |
| `operatorsplittingspreadengine` | Basket | **MISSING** | Operator-splitting spread option |
| `fdndimblackscholesvanillaengine` | Basket FD | **MISSING** | N-dimensional FD BS engine |
| `americanpayoffatexpiry` | Utility | **MISSING** | American payoff at expiry formula |
| `americanpayoffathit` | Utility | **MISSING** | American payoff at barrier hit |
| `basketgeneratingengine` | Swaption | SKIP | Swaption basis system generator (niche) |
| `fdg2swaptionengine` | Swaption FD | SKIP | G2 FD swaption (covered by G2 analytic) |

**Already implemented but not obvious:**

| QL C++ Engine | ql-jax Location |
|--------------|----------------|
| `analyticdividendeuropeanengine` | `engines/analytic/cash_dividend_european.py` |
| `analyticgjrgarchengine` | Not in engines, but `models/equity/gjrgarch.py` has MC pricing |
| `analytich1hwengine` | `engines/analytic/heston_hull_white.py` |
| `analyticpdfhestonengine` | Heston PDF via characteristic function in `engines/analytic/heston.py` |
| `analyticptdhestonengine` | `models/equity/ptd_heston.py` |
| `coshestonengine` | `engines/analytic/heston_variants.py` |
| `hestonexpansionengine` | `engines/analytic/heston_variants.py` |
| `mchestonhullwhiteengine` | `engines/mc/heston.py` (hybrid support) |
| `qdfpamericanengine` | `engines/analytic/qdplus_american.py` |
| `qdplusamericanengine` | `engines/analytic/qdplus_american.py` |
| `fdbatesvanillaengine` | `engines/fd/bates.py` |
| `fd2dblackscholesvanillaengine` | 2D BS via FD framework |
| `fdblackscholesasianengine` | `engines/mc/asian.py` (MC, no FD) |
| `choibasketengine` | `engines/analytic/asian_choi.py` |
| `denglizhoubasketengine` | `engines/analytic/basket.py` |
| `singlefactorbsmbasketengine` | `engines/analytic/basket_single_factor.py` |
| `perturbativebarrieroptionengine` | `engines/analytic/barrier_perturbative.py` |
| `vannavolgabarrierengine` | `engines/analytic/barrier_vanna_volga.py` |
| `treecallablebondengine` | `engines/bond/callable.py` |
| `blackcallablebondengine` | `instruments/callable_bond.py` |

---

### 3. FINITE DIFFERENCE FRAMEWORK — 18 Missing Items

**Operators (7 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `fdmblackscholesfwdop` | **MISSING** | Forward PDE operator (Fokker-Planck) |
| `fdmhestonfwdop` | **MISSING** | Heston forward PDE operator |
| `fdmlocalvolfwdop` | **MISSING** | Local vol forward PDE |
| `fdmsquarerootfwdop` | **MISSING** | Square-root forward PDE |
| `fdmcirop` | **MISSING** | CIR process FD operator |
| `fdmg2op` | **MISSING** | G2 two-factor FD operator |
| `nthorderderivativeop` | **MISSING** | N-th order derivative operator |

**Schemes (2 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `methodoflinesscheme` | **MISSING** | Method of Lines (spatial FD + temporal ODE) |
| `modifiedcraigsneydscheme` | **MISSING** | Modified Craig-Sneyd ADI variant |

**Solvers (3 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `fdmndimsolver` | **MISSING** | N-dimensional FD solver |
| `fdm3dimsolver` | **MISSING** | 3D FD solver |
| `fdmg2solver` | **MISSING** | G2 FD solver |

**Meshers (2 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `fdmblackscholesmultistrikemesher` | **MISSING** | Multi-strike BS mesher |
| `fdmcev1dmesher` | **MISSING** | CEV process mesher |

**Utilities (4 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `fdmquantohelper` | **MISSING** | FD quanto correction helper |
| `fdmhestongreensfct` | **MISSING** | Heston Green's function |
| `fdmmesherintegral` | **MISSING** | Mesher integral utility |
| `fdmshoutloginnervaluecalculator` | **MISSING** | Shout option inner value |

**Already implemented:**

| QL C++ | ql-jax Location |
|--------|----------------|
| `fdmhestonop` | `methods/finitedifferences/heston_operator.py` |
| `fdmblackscholesop` | `methods/finitedifferences/bs_operator.py` |
| `fdmsabrop` | `engines/fd/sabr.py` |
| `fdmhullwhiteop` | `methods/finitedifferences/extended_operators.py` |
| `fdmornsteinuhlenbeckop` | `methods/finitedifferences/extended_operators.py` |
| `fdmhestonsolver` | `engines/fd/heston.py` |
| `concentrating1dmesher` | `methods/finitedifferences/meshers.py` |
| `uniform1dmesher` | `methods/finitedifferences/meshers.py` |
| All 6 step conditions | `methods/finitedifferences/step_conditions.py` |
| All 6 time schemes | `methods/finitedifferences/schemes.py` |
| All boundary conditions | `methods/finitedifferences/boundary.py` |
| `bsmrndcalculator` | `methods/finitedifferences/rnd.py` |
| `hestonrndcalculator` | `methods/finitedifferences/rnd.py` |
| `cevrndcalculator` | `methods/finitedifferences/rnd.py` |

---

### 4. TERM STRUCTURES — 15 Missing Items

**Yield curves (7 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `compositezeroyieldstructure` | **MISSING** | Composite zero yield from multiple curves |
| `impliedtermstructure` | **MISSING** | Implied forward-starting term structure |
| `interpolatedsimplezerocurve` | **MISSING** | Simple interpolated zero curve |
| `piecewiseforwardspreadedtermstructure` | **MISSING** | Piecewise forward spread curve |
| `piecewisespreadyieldcurve` | **MISSING** | Piecewise spread yield curve |
| `overnightindexfutureratehelper` | **MISSING** | Overnight index futures rate helper |
| `multipleresetsswaphelper` | **MISSING** | Multiple-resets swap rate helper |

**Volatility — swaption (5 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `sabrswaptionvolatilitycube` | **MISSING** | SABR-parameterized swaption vol cube |
| `zabrswaptionvolatilitycube` | **MISSING** | ZABR swaption vol cube |
| `interpolatedswaptionvolatilitycube` | **MISSING** | Interpolated swaption vol cube |
| `cmsmarket` | ✅ | `termstructures/volatility/cms_market.py` |
| `cmsmarketcalibration` | ✅ | In cms_market.py |
| `gaussian1dswaptionvolatility` | ✅ | `termstructures/volatility/gaussian1d_vol.py` |

**Volatility — equity/FX (3 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `blackvolsurfacedelta` | **MISSING** | Black vol surface with delta quotes (FX convention) |
| `hestonblackvolsurface` | **MISSING** | Heston-implied Black vol surface |
| `gridmodellocalvolsurface` | **MISSING** | Grid-based local vol calibration |

**Already implemented but not obvious:**

| QL C++ | ql-jax Location |
|--------|----------------|
| `andreasenhugelocalvoladapter` | `termstructures/volatility/andreasen_huge.py` |
| `andreasenhugevolatilityadapter` | Same |
| `blackconstantvol` | `termstructures/volatility/black_vol.py` |
| `blackvariancesurface` | `termstructures/volatility/black_vol.py` |
| `localvolsurface` | `termstructures/volatility/local_vol_surface.py` |
| `kahaleextrap` | `termstructures/volatility/kahale_smile.py` |
| `noarbsabr` | `termstructures/volatility/noarb_sabr.py` |
| `optionletstripper1/2` | `termstructures/volatility/optionlet_extended.py` |
| `spreadedoptionletvol` | `termstructures/volatility/spreaded.py` |
| `swaptionvolcube` | `termstructures/volatility/swaption_vol.py` |
| `piecewisedefaultcurve` | `termstructures/credit/default_curves.py` |
| `interpolatedhazardratecurve` | `termstructures/credit/default_curves.py` |
| `interpolatedsurvivalprobabilitycurve` | `termstructures/credit/default_curves.py` |
| `interpolateddefaultdensitycurve` | `termstructures/credit/default_density.py` |
| `quantotermstructure` | `termstructures/yield_/quanto.py` |
| `zerospreadedtermstructure` | `termstructures/yield_/zero_spreaded.py` |
| `spreaddiscountcurve` | `termstructures/yield_/spread_curves.py` |
| `ultimateforwardtermstructure` | `termstructures/yield_/ultimate_forward.py` |
| `globalbootstrap` | `termstructures/yield_/global_bootstrap.py` |

---

### 5. MATH — 14 Missing Items

**Integrals (6 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `filonintegral` | **MISSING** | Filon quadrature for oscillatory integrands |
| `tanhsinhintegral` | **MISSING** | Tanh-sinh double-exponential quadrature |
| `gausslobattointegral` | **MISSING** | Gauss-Lobatto (endpoint-inclusive) quadrature |
| `kronrodintegral` | **MISSING** | Gauss-Kronrod adaptive quadrature |
| `twodimensionalintegral` | **MISSING** | 2D integration over rectangles |
| `exponentialintegrals` | **MISSING** | Ei(x), li(x) special integrals |

**Special functions (3 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `modifiedbessel` | **MISSING** | Modified Bessel I, K functions |
| `incompletegamma` | **MISSING** | Incomplete gamma P(a,x), Q(a,x) |
| `bernsteinpolynomial` | **MISSING** | Bernstein polynomial basis |

**Distributions (1 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `bivariatestudenttdistribution` | **MISSING** | Bivariate Student-t CDF |

**Interpolations (2 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `multicubicspline` | **MISSING** | Multi-dimensional cubic spline interpolation |
| `xabrinterpolation` | **MISSING** | Extended ABCD interpolation (generalized SABR) |

**Random numbers (2 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `latticersg` | **MISSING** | Lattice rule sequence generator (Korobov) |
| `randomizedlds` | **MISSING** | Randomized low-discrepancy sequences |

**Already implemented:**

| QL C++ | ql-jax Location |
|--------|----------------|
| `particleswarmoptimization` | `math/optimization/metaheuristic.py` |
| `fireflyalgorithm` | `math/optimization/metaheuristic.py` |
| `hybridsimulatedannealing` | `math/optimization/metaheuristic.py` |
| `multidimquadrature` | `math/optimization/metaheuristic.py` |
| `moorepenroseinverse` | `math/matrix_utilities.py` (`pseudoinverse`) |
| `richardsonextrapolation` | `math/richardson.py` |
| `autocovariance` | `math/matrix_utilities.py` |
| `copulas` (Gaussian, Student-t, Clayton, Gumbel, Frank) | `math/copulas.py` |
| SABR, ZABR interpolations | `math/interpolations/sabr.py`, `zabr.py` |
| `stochasticcollocationinvcdf` | `models/equity/clv.py` |

---

### 6. CASHFLOWS — 3 Missing Items

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `duration` | **MISSING** | Macaulay, modified, effective, dollar duration |
| `rateaveraging` | **MISSING** | Rate averaging utilities for sub-period accrual |
| `conundrumpricer` | **MISSING** | Conundrum pricer (Hagan CMS) — separate from linear TSR |

**Already implemented:**

| QL C++ | ql-jax Location |
|--------|----------------|
| `averagebmacoupon` | `cashflows/bma.py` |
| `cmsspreadcoupon` | `cashflows/cms_spread.py` |
| `digitalcmscoupon` | `cashflows/digital.py` |
| `digitaliborcoupon` | `cashflows/digital.py` |
| `lineartsrpricer` | `cashflows/pricer.py` + `coupon_pricers.py` |
| `quantocouponpricer` | `cashflows/coupon_pricers.py` |
| `strippedcapflooredcoupon` | `cashflows/coupon_pricers.py` |
| `rangeaccrual` | `cashflows/range_accrual.py` |
| `cpicoupon`, `cpicouponpricer` | `cashflows/cpi_pricer.py` |
| `blackovernightindexedcouponpricer` | `cashflows/overnight_pricer.py` |
| `multipleresetscoupon` | `cashflows/multiple_resets.py` |
| `replication` | `cashflows/replication.py` |

---

### 7. CALENDARS — 20 Missing

**Implemented (25):** NullCalendar, WeekendsOnly, TARGET, UnitedStates, UnitedKingdom, Japan, Germany (TARGET), JointCalendar, BespokeCalendar, Canada, Australia, HongKong, Singapore, SouthKorea, China, Switzerland, France, Italy, Brazil, India, SouthAfrica, Mexico, Sweden, Norway, Denmark

**Missing (20):**

| Calendar | Priority |
|----------|----------|
| `argentina` | Low |
| `austria` | Low |
| `botswana` | Low |
| `chile` | Low |
| `czechrepublic` | Medium (EUR-adjacent) |
| `finland` | Low (use TARGET) |
| `hungary` | Medium |
| `iceland` | Low |
| `indonesia` | Medium |
| `israel` | Medium |
| `newzealand` | Medium |
| `poland` | Medium |
| `romania` | Low |
| `russia` | Medium |
| `saudiarabia` | Low |
| `slovakia` | Low |
| `taiwan` | Medium |
| `thailand` | Low |
| `turkey` | Medium |
| `ukraine` | Low |

---

### 8. DAY COUNTERS — 3 Missing

| QL C++ | Status | Description |
|--------|--------|-------------|
| `actual364` | **MISSING** | Actual/364 (some money markets) |
| `actual36525` | **MISSING** | Actual/365.25 (average year) |
| `thirty365` | **MISSING** | 30/365 convention |
| `actual366` | SKIP | Only for leap year 366-day convention |
| `one` | SKIP | Trivial 1/1 convention |
| `simpledaycounter` | SKIP | Simple wrapper |

---

### 9. MODELS — 8 Missing Items

**Equity (2 missing):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `hestonslvfdmmodel` | **MISSING** | Heston SLV using FDM (alternate to MC) |
| `garmanklass` | **MISSING** | Garman-Klass volatility estimator |

**Market Models — Callability (6 missing, rest is niche):**

The ql-jax LMM framework covers: simulate, drift, evolve, correlations, vol models, curve state. What's missing is the **advanced callable product framework**:

| QL C++ Component | Status | Description |
|-----------------|--------|-------------|
| `upperboundengine` | **MISSING** | Upper bound engine for Bermudan products |
| `lsstrategy` | **MISSING** | LSM exercise strategy for callable LMM |
| `swapbasissystem` | **MISSING** | Basis system for swap exercise |
| `bermudanswaptionexercisevalue` | **MISSING** | Exercise value for Bermudan swaption |
| `multistepswaption` | **MISSING** | Multi-step swaption product |
| `multisteptarn` | **MISSING** | Target accrual redemption note (TARN) |

**Market Models — Products (skip most):** Many one-step/multi-step product wrappers are thin interfaces. Core simulation and drift computation are done.

---

### 10. PROCESSES — 3 Missing

| QL C++ | Status | Description |
|--------|--------|-------------|
| `gsrprocess` | **MISSING** | GSR process (separate from GSR model) |
| `gsrprocesscore` | **MISSING** | GSR process core implementation |
| `mfstateprocess` | **MISSING** | Markov Functional state process |

Already implemented: `hybridhestonhullwhiteprocess` ✅, `endeulerdiscretization` (covered by `discretization.py`), `jointstochasticprocess` (covered by `process_array.py`)

---

### 11. EXPERIMENTAL — Remaining Gaps

Most experimental concepts are implemented. Key remaining items:

**Variance Gamma (from `experimental/variancegamma/`):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `analyticvariancegammaengine` | **MISSING** | Analytic VG pricing via FFT |
| `fftvanillaengine` | **MISSING** | General FFT vanilla engine |
| `fftvariancegammaengine` | **MISSING** | FFT specifically for VG |
| `variancegammaprocess` | **MISSING** | VG process (separate from model) |

Note: `models/equity/variance_gamma.py` exists but may lack full process/engine separation.

**Barrier options (from `experimental/barrieroption/`):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `suowangdoublebarrierengine` | **MISSING** | Suo-Wang analytic double barrier |
| `binomialdoublebarrierengine` | **MISSING** | Binomial tree double barrier |
| `quantodoublebarrieroption` | **MISSING** | Quanto double barrier instrument |

Note: Vanna-volga barrier engine and perturbative barrier engine already exist.

**Asian (from `experimental/asian/`):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `analytic_cont_geom_av_price_heston` | **MISSING** | Continuous geometric Asian under Heston |
| `analytic_discr_geom_av_price_heston` | **MISSING** | Discrete geometric Asian under Heston |

**MC Basket (from `experimental/mcbasket/`):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `mcpathbasketengine` | **MISSING** | MC path-dependent basket engine |
| `longstaffschwartzmultipathpricer` | **MISSING** | LSM for multi-asset early exercise |
| `pathmultiassetoption` | **MISSING** | Multi-asset path option instrument |

**FD Experimental (from `experimental/finitedifferences/`):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `fdmzabrop` | **MISSING** | ZABR FD operator |
| `vanillavppoption` | **MISSING** | Virtual power plant option |
| VPP step conditions | SKIP | VPP-specific (niche energy) |
| Kluge/ExtOU solvers | SKIP | Niche commodity models |

**Experimental volatility (from `experimental/volatility/`):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `noarbsabrswaptionvolatilitycube` | **MISSING** | No-arb SABR swaption vol cube |
| `volcube` | **MISSING** | Vol cube base class |
| `zabrsmilesection` | **MISSING** | ZABR smile section |

Note: noarb_sabr.py, svi.py, sabr_cube.py all exist.

**Forward Heston (from `experimental/forward/`):**

| QL C++ File | Status | Description |
|------------|--------|-------------|
| `analytichestonforwardeuropeanengine` | **MISSING** | Analytic Heston for forward-starting options |

---

## Implementation Plan

### Phase 1: Bond Types (7 items) — HIGH PRIORITY
Standard bond subtypes are fundamental building blocks.

```
File: ql_jax/instruments/bonds.py (NEW)

1. ZeroCouponBond — pure discount bond with face and maturity
2. FloatingRateBond — bond paying floating coupons (IBOR/overnight + spread)  
3. AmortizingFixedRateBond — fixed coupons with declining notional schedule
4. AmortizingFloatingRateBond — floating coupons with declining notional
5. CPIBond — inflation-indexed bond (CPI-linked)
6. CMSRateBond — bond paying CMS rate coupons
7. AmortizingCMSRateBond — amortizing CMS bond

Tests: ~15
```

### Phase 2: Missing Engines (8 items) — HIGH PRIORITY

```
File: ql_jax/engines/fd/rebate.py (NEW)
1. fd_bs_rebate_price — FD pricing for barrier rebate
2. fd_heston_rebate_price — Heston FD barrier rebate

File: ql_jax/engines/fd/swing.py (NEW)
3. fd_swing_price — FD for swing/storage options

File: ql_jax/engines/fd/shout.py (NEW)
4. fd_shout_price — FD shout option

File: ql_jax/engines/fd/cir.py (NEW)  
5. fd_cir_vanilla_price — CIR process FD pricing

File: ql_jax/engines/lattice/convertible.py (NEW)
6. binomial_convertible_price — Binomial tree convertible bond

File: ql_jax/engines/mc/gjrgarch.py (NEW)
7. mc_gjrgarch_european_price — MC GJRGARCH European

File: ql_jax/engines/analytic/payoff_formulas.py (NEW)
8. american_payoff_at_expiry — closed-form American payoff at expiry
9. american_payoff_at_hit — closed-form barrier hit payoff

Tests: ~20
```

### Phase 3: FD Framework Extensions (12 items) — MEDIUM PRIORITY

```
File: ql_jax/methods/finitedifferences/forward_operators.py (NEW)
1. FdmBlackScholesFwdOp — Fokker-Planck (forward) BS operator
2. FdmHestonFwdOp — Heston forward PDE operator
3. FdmLocalVolFwdOp — Local vol forward PDE
4. FdmSquareRootFwdOp — Square-root forward PDE

File: ql_jax/methods/finitedifferences/cir_operator.py (NEW)
5. FdmCIROp — CIR FD operator

File: ql_jax/methods/finitedifferences/g2_operator.py (NEW)
6. FdmG2Op — G2 two-factor operator

File: ql_jax/methods/finitedifferences/solvers_nd.py (NEW)
7. fdm_solve_3d — 3D FD solver
8. fdm_solve_nd — N-dimensional FD solver

File: ql_jax/methods/finitedifferences/meshers.py (EXTEND)
9. FdmBlackScholesMultiStrikeMesher — multi-strike BS mesher
10. FdmCEV1dMesher — CEV process mesher

File: ql_jax/methods/finitedifferences/schemes.py (EXTEND)
11. method_of_lines_step — Method of Lines temporal scheme
12. modified_craig_sneyd_step — Modified Craig-Sneyd ADI

Tests: ~25
```

### Phase 4: Math Extensions (14 items) — MEDIUM PRIORITY

```
File: ql_jax/math/integrals/advanced.py (NEW or EXTEND integrals_advanced.py)
1. filon_integral — Filon quadrature for oscillatory integrands
2. tanh_sinh_integral — Tanh-sinh double-exponential quadrature
3. gauss_lobatto_integral — Gauss-Lobatto (endpoint-inclusive)
4. kronrod_integral — Gauss-Kronrod adaptive quadrature
5. two_dimensional_integral — 2D rectangular integration
6. exponential_integral — Ei(x), li(x) special functions

File: ql_jax/math/special.py (EXTEND)
7. modified_bessel_i — Modified Bessel function of first kind
8. modified_bessel_k — Modified Bessel function of second kind
9. incomplete_gamma — P(a,x), Q(a,x)
10. bernstein_polynomial — Bernstein basis polynomials

File: ql_jax/math/distributions/bivariate_student_t.py (NEW)
11. bivariate_student_t_cdf — Bivariate Student-t CDF

File: ql_jax/math/interpolations/multicubic.py (NEW)
12. MultiCubicSpline — Multi-dimensional cubic spline

File: ql_jax/math/random/lattice.py (NEW)
13. lattice_sequence — Korobov lattice rule sequence
14. randomized_lds — Randomized low-discrepancy sequences

Tests: ~25
```

### Phase 5: Term Structures & Volatility (15 items) — MEDIUM PRIORITY

```
File: ql_jax/termstructures/yield_/advanced.py (NEW)
1. CompositeZeroYieldStructure — composite zero yield
2. ImpliedTermStructure — forward-starting implied curve
3. InterpolatedSimpleZeroCurve — simple interpolated zero
4. PiecewiseForwardSpreadedTermStructure — forward spread piecewise
5. PiecewiseSpreadYieldCurve — spread-based piecewise yield
6. OvernightIndexFutureRateHelper — OIS futures helper
7. MultipleResetsSwapHelper — multiple-resets swap helper

File: ql_jax/termstructures/volatility/swaption_cube.py (NEW)
8. SABRSwaptionVolatilityCube — SABR swaption vol cube
9. ZABRSwaptionVolatilityCube — ZABR swaption vol cube
10. InterpolatedSwaptionVolatilityCube — interpolated swaption cube

File: ql_jax/termstructures/volatility/equityfx_delta.py (NEW)
11. BlackVolSurfaceDelta — FX delta-convention vol surface
12. HestonBlackVolSurface — Heston-implied vol surface
13. GridModelLocalVolSurface — Grid-based local vol calibration

File: ql_jax/termstructures/volatility/zabr_smile.py (NEW)
14. ZABRSmileSection — ZABR smile section
15. NoArbSABRSwaptionVolCube — No-arb SABR swaption vol cube

Tests: ~30
```

### Phase 6: Cashflows & Day Counters (6 items) — MEDIUM PRIORITY

```
File: ql_jax/cashflows/duration.py (NEW)
1. macaulay_duration — Macaulay duration
2. modified_duration — Modified duration
3. effective_duration — Effective duration (bump & reprice)
4. dollar_duration — Dollar duration

File: ql_jax/cashflows/rate_averaging.py (NEW)
5. sub_period_rate_averaging — Rate averaging for sub-period accrual

File: ql_jax/time/daycounter.py (EXTEND)
6. Actual/364, Actual/365.25, 30/365 day count conventions

Tests: ~15
```

### Phase 7: Variance Gamma & Asian Heston Engines (6 items) — MEDIUM PRIORITY

```
File: ql_jax/processes/variance_gamma.py (NEW)
1. VarianceGammaProcess — VG stochastic process

File: ql_jax/engines/analytic/variance_gamma.py (NEW)
2. analytic_vg_price — Analytic VG option price via FFT
3. fft_vanilla_price — General FFT vanilla pricing engine

File: ql_jax/engines/analytic/asian_heston.py (NEW)
4. cont_geom_asian_heston_price — Continuous geometric Asian under Heston
5. discr_geom_asian_heston_price — Discrete geometric Asian under Heston

File: ql_jax/engines/analytic/heston_forward.py (NEW)
6. analytic_heston_forward_european — Heston forward-starting European

Tests: ~15
```

### Phase 8: Country Calendars (20 items) — LOW PRIORITY

```
File: ql_jax/time/calendars.py (EXTEND)
Add 20 remaining country calendars:
Argentina, Austria, Chile, CzechRepublic, Finland, Hungary,
Iceland, Indonesia, Israel, NewZealand, Poland, Romania,
Russia, SaudiArabia, Slovakia, Taiwan, Thailand, Turkey,
Ukraine, Botswana

Tests: ~40 (2 per calendar)
```

### Phase 9: Experimental Barrier & Basket Engines (6 items) — LOW PRIORITY

```
File: ql_jax/engines/analytic/suo_wang.py (NEW)
1. suo_wang_double_barrier_price — Suo-Wang analytic double barrier

File: ql_jax/engines/lattice/double_barrier.py (NEW)
2. binomial_double_barrier_price — Binomial tree double barrier

File: ql_jax/engines/mc/basket_path.py (NEW)
3. mc_path_basket_price — MC path-dependent basket engine
4. lsm_multi_path_pricer — LSM for multi-asset early exercise

File: ql_jax/instruments/quanto_barrier.py (NEW)
5. QuantoDoubleBarrierOption — Quanto double barrier instrument

File: ql_jax/methods/finitedifferences/zabr_operator.py (NEW)
6. FdmZABROp — ZABR FD operator

Tests: ~15
```

### Phase 10: LMM Callability & Models (6 items) — LOW PRIORITY

```
File: ql_jax/models/marketmodels/callability.py (NEW)
1. UpperBoundEngine — upper bound for Bermudan LMM products
2. LSMStrategy — LSM exercise strategy for LMM
3. SwapBasisSystem — basis system for swap exercise  
4. BermudanSwaptionExerciseValue — exercise value computation

File: ql_jax/models/marketmodels/products.py (NEW)
5. MultiStepSwaption — multi-step swaption product
6. MultiStepTARN — target accrual redemption note

Tests: ~15
```

---

## Summary Statistics

| Phase | Items | Priority | Est. Tests |
|-------|-------|----------|-----------|
| 1: Bond Types | 7 | HIGH | 15 |
| 2: Missing Engines | 9 | HIGH | 20 |
| 3: FD Extensions | 12 | MEDIUM | 25 |
| 4: Math Extensions | 14 | MEDIUM | 25 |
| 5: Term Structures | 15 | MEDIUM | 30 |
| 6: Cashflows/DayCount | 6 | MEDIUM | 15 |
| 7: VG/Asian/Heston | 6 | MEDIUM | 15 |
| 8: Country Calendars | 20 | LOW | 40 |
| 9: Barrier/Basket | 6 | LOW | 15 |
| 10: LMM Callability | 6 | LOW | 15 |
| **TOTAL** | **101** | | **215** |

After completing all 10 phases, estimated coverage: **~95%+** of QuantLib C++ functionality.  
Current: **1069 tests** → Target: **~1284 tests** after all phases.
