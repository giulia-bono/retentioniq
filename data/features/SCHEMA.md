# `feature_table.parquet` — schema

**Path:** `data/features/feature_table.parquet`
**Rows:** 751,956 (one per customer)
**Snapshot date:** 2026-03-15 (max `last_order_date` in the source)
**Churn window:** 299 days (team-agreed; see `docs/decisions.md`)
**Source:** `data/features/customer_features.csv` → `src/features/build.py::build_features`

> Modeling owners (M1, M2): this is your input. Don't re-derive from `orders_raw.csv` — call `build_features` so column conventions stay consistent with `lifetimes` (BG/NBD) expectations.

---

## Pipeline

```
data/features/customer_features.csv          (canonical, upstream)
        │
        ▼
src.features.build.load_customer_features()   (snake_case columns, parse dates)
        │
        ▼
src.features.build.build_features(_, 299)     (adds is_churned, validates)
        │
        ▼
data/features/feature_table.parquet           (this file)
```

---

## Columns

| # | Column | Dtype | Source | Description |
|---|---|---|---|---|
| 1 | `customer_id` | string | upstream | Hashed Shopify customer ID. Primary key, unique. |
| 2 | `frequency` | int64 | upstream | **BG/NBD frequency** = `total_orders − 1`. Repeat-purchase count. |
| 3 | `recency` | int64 (days) | upstream | **BG/NBD recency** = `last_order_date − first_order_date`. *Time since the customer's birth, measured at last order.* Not snapshot-relative. |
| 4 | `t` | int64 (days) | upstream | **BG/NBD T** = `snapshot − first_order_date`. Customer's observation window. |
| 5 | `days_since_last_order` | int64 (days) | upstream | Snapshot-relative recency. **This is what the churn label uses.** |
| 6 | `monetary_value` | float64 (USD) | upstream | **Gamma-Gamma input.** Average value of *repeat* purchases. 0 for one-time buyers (BG/NBD convention). |
| 7 | `avg_order_value` | float64 (USD) | upstream | `total_revenue / total_orders`. Includes the first purchase, unlike `monetary_value`. |
| 8 | `total_orders` | int64 | upstream | All orders for this customer (post upstream cleaning). |
| 9 | `total_revenue` | float64 (USD) | upstream | Sum of charges across paid + partially-refunded orders. |
| 10 | `avg_days_between_orders` | float64 | upstream | Mean inter-purchase gap. 0 for one-time buyers. |
| 11 | `std_days_between_orders` | float64 | upstream | Std dev of inter-purchase gaps. |
| 12 | `refund_count` | int64 | upstream | Number of refunded orders. |
| 13 | `total_refund_amount` | float64 (USD) | upstream | $ refunded. Can be negative for adjustments. |
| 14 | `refund_rate` | float64 | upstream | `refund_count / total_orders`. |
| 15 | `first_order_date` | datetime64[ns] | upstream | First order timestamp. |
| 16 | `last_order_date` | datetime64[ns] | upstream | Most recent order timestamp. |
| 17 | `is_churned` | int8 (0/1) | this module | **XGBoost target.** 1 if `days_since_last_order > 299`, else 0. |

---

## Distribution snapshot (run 2026-05-07)

| Stat | total_orders | recency | t | days_since_last_order | monetary_value | avg_order_value | refund_rate |
|---|---|---|---|---|---|---|---|
| mean | 1.59 | 61.8 | 367 | 305 | 77.4 | 270.6 | 0.12 |
| std | 1.50 | 130.9 | 223 | 215 | 104.3 | 116.2 | 0.36 |
| 50% | 1 | 0 | 344 | 269 | 0 | 256.8 | 0 |
| max | 421 | 804 | 804 | 804 | 2075.4 | 4121.6 | 13 |

**Class balance:** 367,372 churned / 384,584 active → **48.9% / 51.1%**. Healthy — no resampling needed.

---

## Validation rules (asserted in `build_features`)

1. `customer_id` unique
2. `frequency >= 0` and integer
3. `recency <= t` (BG/NBD invariant)
4. `days_since_last_order >= 0`
5. `is_churned ∈ {0, 1}`, no NaN
6. `monetary_value >= 0`

If any of these trip, surface to D1/D2 and fix upstream. Don't paper over in modeling code.

---

## Known caveats

- **51% of customers have `frequency = 0`** (one-time buyers). BG/NBD models them, but `monetary_value` is 0 for those rows — Gamma-Gamma needs `frequency > 0 AND monetary_value > 0`. Filter before fitting.
- `recency` (BG/NBD) ≠ `days_since_last_order` (churn). Both are exposed; pick the right one for your model.
- All amounts assumed USD. Single-store dataset.
- `total_refund_amount` can be slightly negative (chargeback adjustments). Don't silently `.abs()` it.

---

## Reproducing

```python
from src.features.build import load_customer_features, build_features

cust = load_customer_features()                  # data/features/customer_features.csv
features = build_features(cust, churn_window_days=299)
features.to_parquet("data/features/feature_table.parquet", index=False)
```
