# RetentionIQ — Decisions Log (D1 + D2)

**Date:** 2026-05-07
**Owners:** Giulia (D1) + Juan (D2)

---

## Source data

Two CSVs share the same parent in Drive (and are symlinked into `data/` here):

| File | Path in repo | Purpose |
|---|---|---|
| `orders_raw.csv` | `data/raw/orders_raw.csv` (~210 MB) | Order-grain Daasity export. Used for ad-hoc inspection in `src/data/load.py` + `clean.py`. **Not** the source of the feature table. |
| `customer_features.csv` | `data/features/customer_features.csv` (~98 MB) | **Canonical** customer-grain table, pre-aggregated by the team's upstream pipeline. Source of `feature_table.parquet`. |

Both files are gitignored.

### Inspection summary

| Item | `orders_raw.csv` | `customer_features.csv` |
|---|---|---|
| Rows | 976,659 (after dropping nulls + non-paid: 883,915) | 751,956 |
| Date range | 2024-01-01 → 2026-03-14 | first_order 2024-01-01, last_order 2026-03-15 |
| Customers (unique) | 801,039 raw / 750,819 clean | 751,956 |
| Total revenue (clean) | $155,874,642 | (sum of `total_revenue`) |
| AOV (clean) | $176.35 | $270.56 (mean of `avg_order_value`) |
| Repeat buyers | 105,763 (~13.2%) | 367,372 with `frequency >= 1` (49%) — **note: customer_features.csv computes frequency differently** |

*Mismatch in repeat-buyer count is expected*: my raw-CSV count uses orders post-dedupe; theirs uses post-upstream-cleaning. Use the canonical CSV's numbers when reporting to the team.

---

## Decisions (Step 1b)

### Grain
- Raw CSV: **order grain** (1 row per `order_id`, verified).
- Canonical CSV: **customer grain** (1 row per `customer_id`).

### Date range
**2024-01-01 → 2026-03-15.** ~804 days of history.

### Snapshot date
**2026-03-15** (max `last_order_date` in canonical CSV).

### Currency / multi-store
**Single-store, single-currency (assumed USD).** No `currency` column in either source. Flagged for Franca.

### Churn window
**299 days.** Team-agreed (originally derived from inter-purchase P90 on raw orders). Not relitigating today.

> Old `notebooks/01_explore_daasity_schemas.ipynb` mentions 318 days as "P90"; my recompute on the symlinked CSV gives 299. The team aligned on **299** for this iteration.

### Churn rule
> A customer is **churned** if `days_since_last_order > 299` **AND** they had ≥1 paid order before the snapshot.

The customer_features.csv only contains customers with `total_orders >= 1`, so the second clause is structurally satisfied.

---

## Filter rules

Locked upstream for the canonical CSV. For ad-hoc work on `orders_raw.csv` via `src/data/clean.py`:

1. Drop rows with null `customer_id`, `order_id`, or `order_date`.
2. Keep only `financial_status ∈ {paid, partially_refunded}`.
3. Drop `amount_charged <= 0`.
4. Normalize column names to snake_case.

---

## Open questions / parking lot

- [ ] Confirm currency assumption with Franca / data source.
- [ ] Validate `monetary_value` definition with M1 — is it the BG/NBD-Gamma-Gamma "average value of repeat purchases" (excluding the first order)?
- [ ] `ORDER_TAGS` (raw CSV only) is rich (campaign attribution, fulfillment routing). Out of scope today; flag to M2/Kim as a feature source for v2.

---

## Day's split (recap)

| Owner | Step 2 | Step 3 | Step 4 |
|---|---|---|---|
| Giulia (D1) | **Lead** | Review | `/predict/clv` stub |
| Juan (D2) | Review | **Lead** | `/predict/churn` stub |
