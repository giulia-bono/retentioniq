# Looker extract — behavioral features (product type, price, basket size)

**Purpose:** add behavioral signal to the M2 churn XGBoost temporal model.
Result: test AUROC 0.6862 → **0.6979**, 5-fold CV 0.6792 → **0.6902** (see
`data/models/churn_xgboost_temporal_metrics.json`).

**Scope:** the highest-expected-signal fields confirmed available in the
Looker `order_line_revenue` explore. Demographic/geo fields (`gender`,
`state`) are deferred to a follow-up extract.

**Pulled via:** the official Looker Python SDK (`looker-sdk`), not a manual
Explore export. See `src/ingestion/looker_client.py` and
`src/ingestion/behavioral_features.py`.

---

## 1. Credentials

Add to `.env` (gitignored — never commit):

```
LOOKER_BASE_URL=https://your-instance.looker.com:19999
LOOKER_CLIENT_ID=...
LOOKER_CLIENT_SECRET=...
LOOKER_VERIFY_SSL=true
```

`src/ingestion/looker_client.get_sdk()` reads these and returns an
authenticated `Looker40SDK` client.

---

## 2. Query definition (`src/ingestion/behavioral_features.py`)

**LookML model:** `miracle_orders` (Miracle Sheets Shopify data, see
`data/README.md`)

**Explore:** `order_line_revenue`

**Grain:** one row per order line.

### Fields

| Field | Notes |
|---|---|
| `order_line_revenue.customer_id` | join key onto `feature_table.parquet` |
| `order_line_revenue.order_id` | needed to dedupe / weight by order, not line |
| `order_line_revenue.order_date` | for date filter / leakage guard |
| `sku_attributes.product_type` | only populated product-level dimension for this brand (~60%); "Signature Collection", "Combined Collection", "Extra Luxe Collection", "Towel", "Detergent Sheets", "Shower Steamers", ... |
| `order_line_revenue.product_price` | unit price — used to derive `price_bucket` |
| `order_line_revenue.num_sku_per_order` | SKUs per order (basket size) |

**Confirmed against the live explore schema (do not re-attempt these):**
`custom_sku_attributes.category` / `color_family` / `price_bucket` don't
exist in this instance. `sku_attributes.category`, `sub_category`, `class`,
and every `sku_attributes_bsd.*` field (color, size, pattern, material,
hierarchy category/class) are **100% null** for Miracle — those
merchandising dimensions were never populated for this brand.
`financial_status` lives on `order_status`, not `order_line_revenue`.

### Filters

- `order_line_revenue.order_date`: `2024-01-01 to 2026-03-15`
  (matches the snapshot date locked in `docs/decisions.md`)
- `order_status.financial_status`: `paid, partially_refunded`
  (matches the canonical filter rule used for `customer_features.csv`)
- `order_line_revenue.customer_id`: `-NULL`

### Run

```bash
python -m src.ingestion.behavioral_features
```

Looker CSV exports use field **labels** as column headers (e.g. "Order Line
Revenue Customer ID"), so the script renames them to stable snake_case
columns (`COLUMN_RENAME`). Writes order-line-grain results to
`data/raw/order_line_attributes.csv` (gitignored, same as the other raw
CSVs).

---

## 3. Aggregate to customer grain (`src/features/behavioral.py`)

```bash
python -m src.features.behavioral
```

- Non-product line items (`Premium Plus Shipping`, `Rush Shipping`,
  `Gift Card`, `Fondue Cashback - UrlBased`) are dropped before aggregation
  so they don't distort `avg_product_price` / basket size / `top_product_type`.
- `product_type` is normalized (`.str.strip().str.title()`) before taking the
  mode, so case variants (e.g. "Detergent sheets" vs "Detergent Sheets")
  collapse into one value.
- `top_product_type`: most frequently purchased product type per customer
  (mode); "unknown" if no order lines survive the filters above (~52% of
  customers — `product_type` is only ~60% populated overall).
- `avg_product_price`: mean unit price across the customer's order lines.
- `price_bucket`: quartile of `avg_product_price` across customers
  (budget/mid/premium/luxury). No pre-built price bucket exists upstream for
  this brand, so it's derived here.
- `avg_num_sku_per_order`: mean basket size.

Writes `data/features/customer_behavioral_features.csv` (customer grain).

---

## 4. Join onto `feature_table.parquet`

```python
import pandas as pd
from src.features.behavioral import join_behavioral_features

features = pd.read_parquet("data/features/feature_table.parquet")
behavioral = pd.read_csv(
    "data/features/customer_behavioral_features.csv", dtype={"customer_id": "string"}
)

features_v2 = join_behavioral_features(features, behavioral)
features_v2.to_parquet("data/features/feature_table_v2.parquet", index=False)
```

---

## 5. Modeling notes for M2

In `notebooks/04_churn_xgboost_temporal.ipynb`, the behavioral features are
recomputed inline (section 5c) from `data/raw/order_line_attributes.csv`,
filtered to `order_date <= split_date` — same leakage guard as the rest of
the notebook's pre-split aggregation. `top_product_type` and `price_bucket`
are one-hot encoded (`ptype_*`, `pricebucket_*`); `avg_product_price` and
`avg_num_sku_per_order` are included as numeric features directly.

Result (test set):

| Metric | Baseline (43 features) | + behavioral (55 features) | Δ |
|---|---|---|---|
| Test AUROC | 0.6862 | 0.6979 | +0.0117 |
| 5-fold CV AUROC | 0.6792 ± 0.0037 | 0.6902 ± 0.0034 | +0.0110 |
| Test AUPRC | 0.9688 | 0.9708 | +0.0020 |

By SHAP mean(|value|), `avg_product_price` ranks 3rd overall and
`avg_num_sku_per_order` ranks 5th — the numeric behavioral features carry
essentially all of the gain; the one-hot `ptype_*` / `pricebucket_*` columns
didn't individually crack the top 10.

Still short of the 0.72 target. Next highest-leverage candidate: the
deferred gender/location extract, or a richer category signal (the 52%
"unknown" `top_product_type` bucket dilutes that dimension).
