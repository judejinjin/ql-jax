# Validation Gaps: Last Two Items

Two validation items remain unimplemented out of 45 in `validation_plan.md`.

---

## Gap 1: Repo (Item 1.2 — Sprint 1, P0)

**Description:** Bond forward pricing with repo rate discounting.

### What Exists
- `ql_jax/instruments/forward.py` — `BondForward` dataclass with fields: `forward_date`, `bond_maturity`, `coupon_rate`, `face_value`, `forward_price`. No pricing logic.
- `ql_jax/engines/bond/discounting.py` — Full bond discounting engine with clean/dirty prices, accrued interest, `BondFunctions`.
- `ql_jax/cashflows/analytics.py` — `npv()`, `accrued_amount()`, `_cf_amount()`, `_cf_date()`.

### What's Missing
- **Bond forward pricing engine** — a function that computes:
  - Spot income: PV of coupons paid between settlement and delivery, discounted on the income curve.
  - Dirty forward price: `(DirtySpot - SpotIncome) / DF(delivery)`.
  - Clean forward price: dirty forward minus accrued interest at delivery.
  - NPV: `position * (DirtyForward - Strike) * DF(delivery)`.

### QuantLib C++ Reference
- `ql/instruments/bondforward.hpp/.cpp` — `BondForward : Forward`
- Key methods: `forwardPrice()`, `cleanForwardPrice()`, `spotIncome()`, `spotValue()`.
- Formula: `P_DirtyFwd = (P_DirtySpot - SpotIncome) / DF(deliveryDate)`

### Implementation Plan
1. Add `bond_forward_price()` function to `ql_jax/engines/bond/discounting.py`:
   - Inputs: `bond`, `delivery_date`, `discount_curve`, `income_discount_curve` (optional).
   - Computes spot income (PV of coupons between settlement and delivery).
   - Returns dirty forward price, clean forward price, and NPV vs a strike.
2. Create `validation/repo.py` with tests:
   - **Forward = spot compounding**: verify `dirty_fwd ≈ dirty_spot * exp(r * T)` for zero-coupon bond.
   - **Spot-to-forward with coupons**: verify spot income reduces forward price for coupon bond.
   - **Clean vs dirty**: verify `clean_fwd = dirty_fwd - AI(delivery)`.
   - **AD Greeks**: `jax.grad` of forward price w.r.t. repo rate, bond yield.
   - **Batch pricing**: `jax.vmap` across repo rates.

---

## Gap 2: CVAIRS (Item 1.17 — Sprint 3, P2)

**Description:** IRS with counterparty credit adjustment (CVA), reproducing Brigo-Masetti 2005 Table 2.

### What Exists
- `ql_jax/engines/swap/cva.py` — `cva_swap_engine()`, `dva_swap_engine()`, `bilateral_cva()` using simple exposure-based CVA formula:
  `CVA = (1-R) * Σ[DF(t_i) * EE(t_i) * ΔQ(t_i)]`.
- `ql_jax/engines/swap/discounting.py` — Full swap discounting engine.
- `ql_jax/engines/swaption/black.py` — Black swaption engine.

### What's Missing
- **Swaption-based CVA engine** (Brigo-Masetti approach): The C++ `CounterpartyAdjSwapEngine` computes CVA as:
  ```
  CVA_NPV = baseNPV - (1-R_cpty) * Σ[swaptionlet_NPV_i * P_default(t_{i-1}, t_i)]
           + (1-R_invst) * Σ[putSwaptionlet_NPV_i * P_own_default(t_{i-1}, t_i)]
  ```
  Each swaptionlet prices the exposure at a coupon date as an ATM swaption on the remaining swap.

### QuantLib C++ Reference
- `ql/pricingengines/swap/cvaswapengine.hpp/.cpp` — `CounterpartyAdjSwapEngine`.
- `Examples/CVAIRS/CVAIRS.cpp` — Brigo-Masetti Table 2 reproduction with 3 credit risk levels.
- Inputs: discount curve, Black vol, counterparty hazard curve, recovery rate.

### Implementation Plan
1. Add `cva_swap_npv()` function to `ql_jax/engines/swap/cva.py`:
   - Inputs: swap fair rate, tenor, payment frequency, notional, discount function, Black vol, hazard rate function, recovery rate, (optional) investor hazard function.
   - For each payment date: price an ATM swaption on the remaining swap tenor, multiply by default probability in that period.
   - Sum to get CVA adjustment. Return base NPV - (1-R) * CVA_call + (1-R_inv) * CVA_put.
   - Also return `cva_fair_rate()` — the CVA-adjusted fair swap rate.
2. Create `validation/cvairs.py` with tests:
   - **CVA increases with hazard rate**: verify monotonicity across 3 credit levels.
   - **CVA vanishes for zero default intensity**: verify CVA → 0.
   - **AD Greeks**: `jax.grad` of CVA w.r.t. Black vol, recovery rate, hazard rate.
   - **Batch pricing**: `jax.vmap` across credit scenarios.
