# ql-jax Validation Implementation Plan

Validate ql-jax against the original QuantLib C++ library by writing equivalent JAX scripts
that reproduce the same numerical outputs from three reference codebases.

**Target directory**: `validation/` (new directory in ql-jax)

**Tolerance**: All NPV/price outputs must match within `1e-6` (absolute) or `1e-4` (relative)
for analytic methods. Monte Carlo results should match within statistical confidence bounds.

**Convention**: Each validation script should:
1. Print a comparison table: `| Metric | QuantLib Reference | ql-jax Result | Diff |`
2. Assert tolerances via `numpy.testing.assert_allclose` or equivalent
3. Be runnable standalone: `python validation/<script>.py`

---

## Phase 1: QuantLib C++ Examples (`~/QuantLib/Examples/`)

20 programs. Grouped by implementation readiness based on ql-jax's current capabilities.

### Tier 1 — Ready Now (ql-jax coverage ≥ 90%)

These use instruments and models already well-supported in ql-jax.

| # | C++ Example | Description | ql-jax Modules Needed | Key Outputs to Match | Priority |
|---|-------------|-------------|-----------------------|----------------------|----------|
| 1.1 | **FRA** | Forward rate agreements from bootstrapped Euribor curve; FRA fair rates, NPV at entry | `term_structures`, `instruments.fra` | FRA forward rates, fair rates (NPV ≈ 0 at market) | P0 |
| 1.2 | **Repo** | Bond forward pricing with repo rate discounting | `instruments.bonds`, `term_structures` | Forward delivery price, spot-to-forward relationship | P0 |
| 1.3 | **Bonds** | Bond pricing + yield curve bootstrapping from ZC deposits, fixed-rate bonds, swaps | `instruments.bonds`, `term_structures.yield_curve`, `term_structures.piecewise` | Clean/dirty prices, YTM, Z-spread, forward rates | P0 |
| 1.4 | **CDS** | CDS valuation with hazard rate bootstrapping from market spreads; survival probabilities | `instruments.credit`, `term_structures.credit` | Calibrated hazard rates, survival probs (97%–94% for 1–2Y), CDS NPV | P0 |
| 1.5 | **EquityOption** | European/American/Bermudan pricing across analytic, Heston, tree, MC, FD methods | `instruments.options`, `models.heston`, `engines.*` | Option prices across ~15 methods (~1.5–2.0 put range) | P0 |
| 1.6 | **DiscreteHedging** | Replication error analysis via MC; P&L distribution vs Derman-Kamal formula | `processes.black_scholes`, `instruments.options` | P&L mean/std/skew/kurtosis, theoretical comparison | P1 |
| 1.7 | **FittedBondCurve** | 4 curve fitting methods (exp splines, polynomial, Nelson-Siegel, Svensson) on 15 bonds | `term_structures.fitted_bond_curve` | Fitted discount curves, iteration counts, par rate recovery | P1 |
| 1.8 | **Replication** | Static replication of barrier option using portfolio of vanillas at discrete barrier levels | `instruments.options`, `instruments.barrier` | Replicated barrier value vs analytical benchmark | P1 |
| 1.9 | **MultidimIntegral** | Multi-dimensional numerical integration (Gaussian quadrature, trapezoid) | `math.integration` | Exact vs numerical integration convergence | P2 |
| 1.10 | **GlobalOptimizer** | Global optimization (Firefly, PSO, SA, DE) on Ackley/Rosenbrock/Easom/Eggholder | `math.optimization` | Convergence to known minima, iteration counts | P2 |

### Tier 2 — Partially Ready (ql-jax coverage 50–90%)

Some engine variants or model features may need minor additions.

| # | C++ Example | Description | ql-jax Modules Needed | Gaps to Fill | Priority |
|---|-------------|-------------|-----------------------|--------------|----------|
| 1.11 | **CallableBonds** | Callable fixed-rate bonds via Hull-White tree; sensitivity to mean reversion/vol | `instruments.callable_bonds`, `models.hull_white` | Verify tree engine completeness | P1 |
| 1.12 | **BermudanSwaption** | Calibrate BK/G2/HW/CIR to swaption vols; price Bermudan swaptions via trees | `instruments.swaption`, `models.short_rate` | Verify all 4 model calibrations work end-to-end | P1 |
| 1.13 | **Gaussian1dModels** | GSR + Markov Functional models for Bermudan swaptions; calibration to market | `models.gaussian1d`, `instruments.swaption` | Verify Markov Functional model completeness | P1 |
| 1.14 | **MulticurveBootstrapping** | OIS discounting + EURIBOR 6M forecasting; multi-curve framework | `term_structures.piecewise`, `term_structures.yield_curve` | Multi-curve bootstrap (may need dual-curve plumbing) | P1 |
| 1.15 | **AsianOption** | Discrete/continuous arithmetic/geometric Asian options; MC, PDE, Levy, TW engines | `instruments.asian`, `engines.mc`, `engines.fd` | Verify all Asian engine variants | P1 |

### Tier 3 — Requires Significant Work (ql-jax coverage < 50%)

These need new models, copulas, or complex framework additions.

| # | C++ Example | Description | Missing Infrastructure | Priority |
|---|-------------|-------------|------------------------|----------|
| 1.16 | **ConvertibleBonds** | Convertible bonds with call/put via binomial trees (7 tree types) | Convertible bond binomial pricing (partial impl) | P2 |
| 1.17 | **CVAIRS** | IRS with counterparty credit adjustment (CVA) | CVA engine, multi-curve + credit exposure | P2 |
| 1.18 | **BasketLosses** | Credit portfolio loss via Gaussian/T-copula latent models | **Copulas (all types missing)**, portfolio loss models | P3 |
| 1.19 | **LatentModel** | Latent variable credit models with copula specifications | **Copulas**, latent variable framework | P3 |
| 1.20 | **MarketModels** | LIBOR Market Model: Bermudan callable swaps, pathwise Greeks, Longstaff-Schwartz | LMM pathwise Greeks, upper bounds via regression | P2 |

### Phase 1 Summary

| Tier | Count | Effort | Timeline Estimate |
|------|-------|--------|-------------------|
| Tier 1 (Ready) | 10 | ~1–2 days each | Implement first |
| Tier 2 (Partial) | 5 | ~2–4 days each | After Tier 1 |
| Tier 3 (Gaps) | 5 | ~5–10 days each | Defer or skip |

---

## Phase 2: QuantLib-SWIG Python Examples (`~/QuantLib-SWIG/Python/examples/`)

15 scripts. These are the easiest to port since they're already in Python — we can often
translate line-by-line from QuantLib-SWIG API to ql-jax API.

### Tier 1 — Ready Now

| # | Script | Description | ql-jax Modules Needed | Key Outputs to Match | Priority |
|---|--------|-------------|-----------------------|----------------------|----------|
| 2.1 | **cashflows.py** | Cashflow extraction from 5Y OIS swap; fixed/floating leg enumeration | `instruments.swap`, `cashflows` | Cashflow table: dates, amounts, rates, accrual periods | P0 |
| 2.2 | **european-option.py** | European option across analytic/Heston/integral/FD/binomial/MC | `instruments.options`, `models.heston`, `engines.*` | Option prices across methods (~7.5 call range) | P0 |
| 2.3 | **cds.py** | CDS calibration to spreads (3M–2Y); survival probability computation | `instruments.credit`, `term_structures.credit` | Hazard rates, survival probs (97.04%–94.18%), fair spreads | P0 |
| 2.4 | **bonds.py** | Bond curve bootstrapping (ZC deposits, FRB, swap rates); multi-curve | `instruments.bonds`, `term_structures` | Zero rates, discount factors, bond clean prices | P0 |
| 2.5 | **swap.py** | IRS valuation from deposit/FRA/swap/futures data; curve bootstrapping | `instruments.swap`, `term_structures.piecewise` | Swap NPV, fair rate, cashflow streams | P0 |
| 2.6 | **capsfloors.py** | 10Y cap pricing with Black model (54.7% vol), 3M USD LIBOR | `instruments.capfloor`, `engines.capfloor` | Cap NPV via Black pricing | P1 |

### Tier 2 — Partially Ready

| # | Script | Description | ql-jax Modules Needed | Gaps to Fill | Priority |
|---|--------|-------------|-----------------------|--------------|----------|
| 2.7 | **american-option.py** | American put via BAW/Bjerksund-Stensland/FD/7 binomial trees | `instruments.options`, `engines.binomial` | Verify all 7 binomial tree types match | P1 |
| 2.8 | **bermudan-swaption.py** | Bermudan swaption calibration (HW, BK, Markov Functional) | `instruments.swaption`, `models.short_rate` | Calibration errors in bp, implied vols | P1 |
| 2.9 | **callablebonds.py** | Callable bond via HW tree; parameter sensitivity analysis | `instruments.callable_bonds`, `models.hull_white` | Clean prices under varying HW params (~95–100) | P1 |
| 2.10 | **gaussian1d-models.py** | GSR/Markov Functional for Bermudan payer swaptions | `models.gaussian1d` | Calibrated vols, swaption price matrix | P1 |
| 2.11 | **global-bootstrap.py** | Global curve bootstrap with deposits/FRAs/swaps; handles negative rates | `term_structures.global_bootstrap` | Bootstrapped zero curve, forward rates | P1 |

### Tier 3 — Requires Significant Work

| # | Script | Description | Missing Infrastructure | Priority |
|---|--------|-------------|------------------------|----------|
| 2.12 | **basket-option.py** | Multi-asset basket (max/min/avg payoff) via MC; American basket LSM | American basket MC with LSM regression | P2 |
| 2.13 | **isda-engine.py** | ISDA CDS with conventional markup; recovery sensitivity; Markit comparison | Full ISDA CDS mechanics with upfront premium | P2 |
| 2.14 | **slv.py** | Heston SLV model; 3D leverage function surface visualization | SLV leverage surface calibration | P2 |
| 2.15 | **swing.py** | Swing option with BS + extended OU jump process; FD pricing | Swing FD engine, extended OU with jumps | P2 |

### Phase 2 Summary

| Tier | Count | Effort | Timeline Estimate |
|------|-------|--------|-------------------|
| Tier 1 (Ready) | 6 | ~0.5–1 day each | Implement first |
| Tier 2 (Partial) | 5 | ~1–3 days each | After Tier 1 |
| Tier 3 (Gaps) | 4 | ~3–7 days each | Defer or skip |

---

## Phase 3: quantlib-risk-py Benchmarks (`~/quantlib-risk-py/benchmarks/`)

21 benchmark scripts (+ 2 runners). These focus on **Greeks computation** (FD vs AAD vs JIT),
making them the most directly relevant to ql-jax's JAX autodiff advantage.

### 3A — First-Order Greek Benchmarks

| # | Script | Description | ql-jax Modules | Key Outputs | Priority |
|---|--------|-------------|----------------|-------------|----------|
| 3.1 | **european_option_benchmarks.py** | European call Greeks (delta, div-rho, vega, rho) via FD/AAD/JIT | `instruments.options`, `jax.grad` | 4 Greeks; timing: FD vs `jax.grad` vs `jax.jit(jax.grad)` | P0 |
| 3.2 | **american_option_benchmarks.py** | American put Greeks via BAW/BjS/FD-PDE/QD+ × FD/AAD/JIT | `instruments.options.american` | Greeks per engine; JIT eligibility check | P0 |
| 3.3 | **basket_option_benchmarks.py** | 2-asset basket Greeks (5 sensitivities) via MC; FD vs AAD | `instruments.basket`, `engines.mc` | 5 Greeks (∂S₁, ∂S₂, ∂σ₁, ∂σ₂, ∂r); timing | P1 |
| 3.4 | **swing_option_benchmarks.py** | Swing option Greeks via PDE; FD vs AAD (no JIT) | `instruments.swing`, `engines.fd` | ∂S, ∂σ, ∂r; timing comparison | P2 |

### 3B — Monte Carlo Scenario Risk Benchmarks (100-scenario batch)

| # | Script | Description | ql-jax Modules | Key Outputs | Priority |
|---|--------|-------------|----------------|-------------|----------|
| 3.5 | **cds_benchmarks.py** | CDS 100-scenario Greeks (6 inputs); FD/AAD replay/re-record | `instruments.credit`, `jax.vmap` | Per-scenario Greeks; `jax.vmap(jax.grad)` timing | P0 |
| 3.6 | **monte_carlo_irs_benchmarks.py** | Vanilla IRS 100-scenario Greeks (17 curve inputs) | `instruments.swap`, `term_structures` | 17 curve Greeks per scenario; batch timing | P1 |
| 3.7 | **ois_bootstrapped_IRS_benchmarks.py** | OIS-bootstrapped SOFR IRS (9 inputs); FD/AAD/JIT | `term_structures.ois`, `instruments.swap` | 9 OIS rate Greeks; batch timing | P1 |
| 3.8 | **isda_cds_benchmarks.py** | ISDA CDS 100-scenario Greeks (20 inputs: 6 deposits + 14 swaps) | `instruments.credit.isda` | 20 Greeks per scenario; no JIT (branching) | P2 |
| 3.9 | **monte_carlo_bond_benchmarks.py** | Callable bond 100-scenario Greeks (3 HW params); no JIT (tree branching) | `instruments.callable_bonds`, `models.hull_white` | 3 HW Greeks per scenario | P2 |
| 3.10 | **risky_bond_benchmarks.py** | Risky bond (OIS + CDS curves, 14 inputs); JIT eligible | `instruments.bonds.risky`, `term_structures` | 14 Greeks (9 OIS + 4 CDS + 1 recovery) | P2 |

### 3C — Jacobian & Sensitivity Transformation Benchmarks

| # | Script | Description | ql-jax Modules | Key Outputs | Priority |
|---|--------|-------------|----------------|-------------|----------|
| 3.11 | **hazard_rate_jacobian_benchmarks.py** | ∂NPV/∂(hazard rate) via direct AAD + Jacobian J=∂h/∂s (4×4) | `jax.jacrev`, `term_structures.credit` | Hazard-rate Greeks; Jacobian J; J·K≈I validation | P1 |
| 3.12 | **cds_spread_jacobian_benchmarks.py** | ∂NPV/∂(CDS spread) via reverse Jacobian K=∂s/∂h; K×J≈I check | `jax.jacrev`, `term_structures.credit` | CDS-spread Greeks; bootstrap Jacobian identity | P1 |
| 3.13 | **zero_rate_jacobian_benchmarks.py** | ∂NPV/∂(zero rate) via ZeroCurve AAD + bootstrap Jacobian (9×9) | `jax.jacrev`, `term_structures.yield_curve` | Zero-rate Greeks; Jacobian J; J^T solve | P1 |
| 3.14 | **mm_rate_jacobian_benchmarks.py** | ∂NPV/∂(par rate) via PiecewiseLinearZero + reverse Jacobian | `jax.jacrev`, `term_structures` | Par-rate Greeks; reverse Jacobian K; K×J≈I | P1 |

### 3D — Second-Order (Hessian) Benchmarks

| # | Script | Description | ql-jax Modules | Key Outputs | Priority |
|---|--------|-------------|----------------|-------------|----------|
| 3.15 | **second_order_european.py** | 4×4 Hessian (gamma/vanna/volga); validated vs analytic 2nd-order Greeks | `jax.hessian`, `instruments.options` | Full 4×4 Hessian; FD vs `jax.hessian` timing | P0 |
| 3.16 | **second_order_irs.py** | 17×17 Hessian of 5Y payer IRS | `jax.hessian`, `instruments.swap` | Full 17×17 Hessian; timing comparison | P1 |
| 3.17 | **second_order_cds.py** | 6×6 Hessian of 2Y CDS | `jax.hessian`, `instruments.credit` | Full 6×6 Hessian; FD-over-AAD vs pure FD | P1 |
| 3.18 | **second_order_ir_cap.py** | 18×18 Hessian of 10Y IR cap (17 curve + 1 vol) | `jax.hessian`, `instruments.capfloor` | 18×18 Hessian; rate-rate/rate-vol/vol-vol blocks | P1 |
| 3.19 | **second_order_risky_bond.py** | 14×14 Hessian risky bond (9 OIS + 4 CDS + 1 recovery) | `jax.hessian`, `instruments.bonds.risky` | 14×14 Hessian; cross-curve blocks | P2 |

### Phase 3 Summary

| Sub-phase | Count | Effort | Timeline Estimate |
|-----------|-------|--------|-------------------|
| 3A First-Order Greeks | 4 | ~0.5–2 days each | Implement first |
| 3B Scenario Risk | 6 | ~1–3 days each | After 3A |
| 3C Jacobians | 4 | ~1–2 days each | After 3A |
| 3D Hessians | 5 | ~0.5–2 days each | After 3A |

---

## Suggested Execution Order

Prioritized by: (a) validates core functionality first, (b) builds on preceding work,
(c) maximizes coverage per effort.

### Sprint 1 — Core Validation (P0 items)

Validates the bread-and-butter instruments and demonstrates JAX autodiff advantage.

| Order | Item | Source | What It Proves |
|-------|------|--------|----------------|
| 1 | 2.1 cashflows | SWIG | Basic swap cashflow mechanics work |
| 2 | 1.1 FRA | C++ | Simple IR instrument + curve bootstrap |
| 3 | 1.2 Repo | C++ | Bond forward pricing |
| 4 | 2.4 bonds / 1.3 Bonds | SWIG/C++ | Bond pricing + yield curve bootstrap |
| 5 | 2.5 swap | SWIG | IRS valuation + curve construction |
| 6 | 2.3 cds / 1.4 CDS | SWIG/C++ | CDS pricing + hazard rate bootstrap |
| 7 | 2.2 european-option / 1.5 EquityOption | SWIG/C++ | Multi-method option pricing |
| 8 | 3.1 european_option_benchmarks | risk-py | First-order Greeks: `jax.grad` vs FD |
| 9 | 3.2 american_option_benchmarks | risk-py | American Greeks across engines |
| 10 | 3.5 cds_benchmarks | risk-py | Scenario batch Greeks: `jax.vmap(jax.grad)` |
| 11 | 3.15 second_order_european | risk-py | `jax.hessian` vs analytic 2nd-order Greeks |

### Sprint 2 — Extended Coverage (P1 items)

Validates more complex instruments, calibration, and Jacobian transforms.

| Order | Item | Source | What It Proves |
|-------|------|--------|----------------|
| 12 | 1.6 DiscreteHedging | C++ | MC path generation + hedging |
| 13 | 1.7 FittedBondCurve | C++ | 4 curve fitting methods |
| 14 | 1.8 Replication | C++ | Barrier replication via portfolio |
| 15 | 2.6 capsfloors | SWIG | Cap/floor Black pricing |
| 16 | 2.7 american-option | SWIG | All 7+ binomial trees |
| 17 | 1.11 CallableBonds / 2.9 | C++/SWIG | HW tree callable bond pricing |
| 18 | 1.12 BermudanSwaption / 2.8 | C++/SWIG | Short-rate model calibration |
| 19 | 1.13 Gaussian1dModels / 2.10 | C++/SWIG | GSR + Markov Functional |
| 20 | 1.14 MulticurveBootstrapping | C++ | Dual-curve (OIS + EURIBOR) |
| 21 | 1.15 AsianOption | C++ | Asian option engine variants |
| 22 | 2.11 global-bootstrap | SWIG | Global curve bootstrap |
| 23 | 3.3 basket_option_benchmarks | risk-py | Multi-asset MC Greeks |
| 24 | 3.6 monte_carlo_irs | risk-py | 17-input IRS scenario risk |
| 25 | 3.7 ois_bootstrapped_IRS | risk-py | OIS curve Greeks |
| 26 | 3.11–3.14 Jacobians (all 4) | risk-py | `jax.jacrev` Jacobian transforms |
| 27 | 3.16 second_order_irs | risk-py | 17×17 IRS Hessian |
| 28 | 3.17 second_order_cds | risk-py | 6×6 CDS Hessian |
| 29 | 3.18 second_order_ir_cap | risk-py | 18×18 cap Hessian |

### Sprint 3 — Advanced Topics (P2 items)

| Order | Item | Source | What It Proves |
|-------|------|--------|----------------|
| 30 | 1.9 MultidimIntegral | C++ | Numerical integration accuracy |
| 31 | 1.10 GlobalOptimizer | C++ | Optimization framework |
| 32 | 1.16 ConvertibleBonds | C++ | Binomial convertible pricing |
| 33 | 1.17 CVAIRS | C++ | Counterparty value adjustment |
| 34 | 1.20 MarketModels | C++ | LMM + Longstaff-Schwartz |
| 35 | 2.12 basket-option | SWIG | American basket LSM |
| 36 | 2.13 isda-engine | SWIG | ISDA CDS mechanics |
| 37 | 2.14 slv | SWIG | Heston SLV leverage surface |
| 38 | 2.15 swing | SWIG | Swing option FD pricing |
| 39 | 3.4 swing_option_benchmarks | risk-py | Swing Greeks |
| 40 | 3.8 isda_cds_benchmarks | risk-py | ISDA CDS 20-input batch |
| 41 | 3.9 monte_carlo_bond | risk-py | Callable bond HW batch |
| 42 | 3.10 risky_bond_benchmarks | risk-py | Risky bond 14-input batch |
| 43 | 3.19 second_order_risky_bond | risk-py | 14×14 cross-curve Hessian |

### Sprint 4 — Credit Portfolio (P3 items)

| Order | Item | Source | What It Proves |
|-------|------|--------|----------------|
| 44 | 1.18 BasketLosses | C++ | Copula credit portfolio losses |
| 45 | 1.19 LatentModel | C++ | Latent variable credit models |

---

## Validation Script Template

Each script in `validation/` should follow this pattern:

```python
"""Validation: <Example Name>
Source: ~/QuantLib/Examples/<Dir>/<File>.cpp  (or .py)
"""
import jax
import jax.numpy as jnp
from ql_jax import ...

# --- Reference values from running the original QuantLib example ---
REFERENCE = {
    "metric_name": <value from C++ or Python output>,
    ...
}

def main():
    # 1. Set up market data (same as original)
    # 2. Build instruments using ql-jax API
    # 3. Price and compute Greeks
    # 4. Compare against reference values

    results = {}
    results["metric_name"] = ...

    # Print comparison table
    print(f"{'Metric':<30} {'Reference':>15} {'ql-jax':>15} {'Diff':>12}")
    print("-" * 72)
    for key in REFERENCE:
        ref = REFERENCE[key]
        val = results[key]
        diff = abs(val - ref)
        print(f"{key:<30} {ref:>15.8f} {val:>15.8f} {diff:>12.2e}")

    # Assert tolerances
    for key in REFERENCE:
        np.testing.assert_allclose(
            results[key], REFERENCE[key],
            rtol=1e-4, atol=1e-6,
            err_msg=f"Mismatch in {key}"
        )
    print("\n✓ All values match within tolerance.")

if __name__ == "__main__":
    main()
```

---

## Reference Value Collection Strategy

Before writing each validation script, collect reference values by:

1. **C++ Examples**: Build and run the QuantLib example:
   ```bash
   cd ~/QuantLib/build && make <ExampleName> && ./Examples/<ExampleName>/<ExampleName>
   ```
2. **SWIG Python Examples**: Run directly:
   ```bash
   cd ~/QuantLib-SWIG/Python/examples && python <script>.py
   ```
3. **quantlib-risk-py Benchmarks**: Run the benchmark:
   ```bash
   cd ~/quantlib-risk-py && python benchmarks/<script>.py
   ```

Capture stdout and extract numerical values into `REFERENCE` dicts.

---

## JAX-Specific Validation Points

Beyond matching QuantLib values, the validation should demonstrate ql-jax advantages:

| Feature | How to Validate | Relevant Scripts |
|---------|-----------------|------------------|
| `jax.grad` (reverse-mode AD) | Compare Greeks to FD; verify accuracy + speedup | 3.1–3.4 |
| `jax.vmap` (batch scenarios) | Run 100-scenario risk; compare to loop-based | 3.5–3.10 |
| `jax.jacrev` / `jax.jacfwd` | Compute full Jacobians; verify J·K≈I identities | 3.11–3.14 |
| `jax.hessian` | Compute Hessians; verify vs analytic 2nd-order | 3.15–3.19 |
| `jax.jit` (compilation) | Time JIT vs non-JIT for eligible instruments | 3.1, 3.2, 3.5–3.7 |
| GPU acceleration | Compare CPU vs GPU timing (use `benchmarks/` infra) | All scripts |

---

## Metrics & Reporting

### Per-Script Report
- Pass / Fail (tolerance check)
- Max absolute error across all outputs
- Max relative error across all outputs
- Runtime (ql-jax vs QuantLib reference if available)

### Aggregate Dashboard
- Total scripts: 45 (20 C++ + 15 SWIG + 10 unique benchmarks)
- Pass rate by phase
- Coverage matrix: instrument × model × engine

### CI Integration
Add a workflow `.github/workflows/validation.yml` that runs all validation scripts
and produces a summary table as a build artifact.

---

## Known Gaps Requiring Implementation Before Validation

| Gap | Blocking Scripts | Effort |
|-----|------------------|--------|
| Copula framework (Gaussian, T-copula) | 1.18, 1.19 | High |
| Convertible bond binomial pricing | 1.16 | Medium |
| CVA / counterparty adjustment engine | 1.17 | Medium |
| ISDA CDS upfront premium handling | 2.13, 3.8 | Medium |
| SLV leverage surface calibration | 2.14 | Medium |
| Swing option FD engine (full) | 2.15, 3.4 | Medium |
| Extended OU with jumps | 2.15, 3.4 | Low–Medium |
| American basket MC with LSM | 2.12 | Medium |
| Multi-curve dual-bootstrap | 1.14, 1.20 | Medium |

These gaps do NOT block Sprint 1. Items in Sprint 1 use instruments and models already
fully implemented in ql-jax.
