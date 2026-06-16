# Looker extract — demographic features (gender, state/region, country)

**Purpose:** follow-up to `behavioral_features_looker_query.md`. The
product-type/price/basket-size extract took the M2 churn XGBoost temporal
model from AUROC 0.6862 → 0.6979 (still short of the 0.72 target). This
extract adds the deferred gender/location fields to see if they close more
of the gap.

**Pulled via:** the official Looker Python SDK (`looker-sdk`), same pattern
as the behavioral extract. See `src/ingestion/looker_client.py`.

---

## 1. Schema check (already done)

Enumerated `customers.*` dimensions on the `order_line_revenue` explore
(`miracle_orders` model) and validated population at order-line grain
(50k-row sample, `order_line_revenue.customer_id` not null,
`order_date` in `2024-01-01 to 2026-03-15`):

| Field | Null rate | Notes |
|---|---|---|
| `customers.gender` | 0% | `Female` (64.5%), `Male` (18.0%), `Unknown` (17.5%) |
| `customers.gender_accuracy` | ~16% (NaN when gender = `Unknown`) | confidence score 0–1 from inferred-gender service; median 0.99 |
| `customers.country_code` | 0.2% | 98.6% `US`, then `CA`, `AU`, `GB`, `MX`, ... |
| `customers.state_code` | ~9.6% (US rows only) | top: FL, TX, CA, NY, PA, OH, IL, NC, GA |

`customers.city` / `customers.zipcode` also exist but are high-cardinality
and not planned for v1 — state/region is the practical geo grain.

---

## 2. Query definition (new module: `src/ingestion/demographic_features.py`)

**LookML model:** `miracle_orders`
**Explore:** `order_line_revenue`
**Grain:** one row per order line (same as the behavioral extract); deduped
to customer grain in the aggregation step since gender/location are
per-customer attributes, not per-order.

### Fields

```python
FIELDS = [
    "order_line_revenue.customer_id",
    "order_line_revenue.order_date",
    "customers.gender",
    "customers.gender_accuracy",
    "customers.state_code",
    "customers.country_code",
]
```

### Filters

```python
FILTERS = {
    "order_line_revenue.order_date": "2024-01-01 to 2026-03-15",
    "order_status.financial_status": "paid, partially_refunded",
    "order_line_revenue.customer_id": "-NULL",
}
```

(Same filters as the behavioral extract — matches `docs/decisions.md` and
the canonical `customer_features.csv` filter rule.)

### Column rename (Looker CSV exports use field labels as headers)

```python
COLUMN_RENAME = {
    "Order Line Revenue Customer ID": "customer_id",
    "Order Line Revenue Order Date": "order_date",
    "Customer Info Customer Gender": "gender",
    "Customer Info Customer Gender Accuracy": "gender_accuracy",
    "Customer Info State Code": "state_code",
    "Customer Info Country Code": "country_code",
}
```

### Run

```bash
python -m src.ingestion.demographic_features
```

Writes order-line-grain results to `data/raw/order_line_demographics.csv`
(gitignored).

---

## 3. Aggregate to customer grain (new module: `src/features/demographics.py`)

Gender/state/country are constant per customer (sourced from the `customers`
table, joined onto every order line for that customer), so aggregation is a
dedup, not a mode/mean:

```python
demo = lines.drop_duplicates("customer_id")[
    ["customer_id", "gender", "gender_accuracy", "state_code", "country_code"]
]
```

Derived columns:

- **`gender_clean`** — `gender` as-is, except rows where
  `gender_accuracy < 0.6` are remapped to `"unknown"` (low-confidence
  inferences are noise, not signal).
- **`region`** — collapse `state_code` to US Census region
  (Northeast/Midwest/South/West) + `"other"` for non-US / null, to keep
  one-hot cardinality manageable (4-5 categories instead of ~50 states).
- **`is_us`** — `country_code == "US"` as int8 (98.6% US makes raw
  `country_code` one-hots mostly redundant; this is the only bit of signal
  worth keeping from it).

Fill rules: `gender_clean` → `"unknown"`, `region` → `"other"`,
`is_us` → 1 (mode) for any customer with no demographic row at all.

Writes `data/features/customer_demographic_features.csv` (customer grain).

---

## 4. Join + modeling notes for M2

Same join pattern as `behavioral.py::join_behavioral_features` — left join
on `customer_id`, fill missing with `"unknown"`/`"other"`.

In `notebooks/04_churn_xgboost_temporal.ipynb`, add a section 5d alongside
5c: load `order_line_demographics.csv`, filter to `order_date <= split_date`
(leakage guard — in practice gender/location won't change pre- vs post-split,
but keep the filter for consistency with the rest of the pipeline), dedupe to
customer grain, derive `gender_clean` / `region` / `is_us`, one-hot encode
`gender_clean` (3 cols) and `region` (5 cols), append to `FEATURE_COLS`.

Expectation-setting: `gender` and coarse `region` are weak churn predictors
on their own (per the original Looker field-audit reasoning) — this extract
is a cheap way to test that, not a likely large jump. If the AUROC delta is
within CV noise (±0.003), it's not worth the added model complexity /
one-hot columns and should be dropped rather than kept "just in case."
