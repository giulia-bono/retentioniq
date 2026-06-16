"""Customer-grain behavioral features from Looker order-line attributes.

Input: ``data/raw/order_line_attributes.csv`` (order-line grain, produced by
``src/ingestion/behavioral_features.py``). Columns: ``customer_id``,
``order_id``, ``order_date``, ``product_type``, ``product_price``,
``num_sku_per_order``.

Output: ``data/features/customer_behavioral_features.csv`` (customer grain),
joinable onto ``feature_table.parquet`` via ``customer_id``.

Note: this Looker instance has no merchandising category/color/price-bucket
dimensions populated for Miracle (see module docstring in
``src/ingestion/behavioral_features.py``). ``product_type`` (collection name,
~60% populated) is used as the category signal, and ``price_bucket`` is
derived here from quartiles of ``avg_product_price`` across customers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ORDER_LINE_PATH = "data/raw/order_line_attributes.csv"
OUTPUT_PATH = "data/features/customer_behavioral_features.csv"

CATEGORICAL_FILL = "unknown"
PRICE_BUCKET_LABELS = ["budget", "mid", "premium", "luxury"]

# Non-product line items (shipping fees, gift cards, cashback adjustments) —
# excluded entirely so they don't distort avg_product_price / basket size /
# top_product_type.
NON_PRODUCT_TYPES = {
    "Premium Plus Shipping",
    "Rush Shipping",
    "Gift Card",
    "Fondue Cashback - UrlBased",
}


def _mode_or_na(s: pd.Series):
    m = s.mode()
    return m.iloc[0] if not m.empty else pd.NA


def load_order_line_attributes(path: str | Path = ORDER_LINE_PATH) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"customer_id": "string"},
        parse_dates=["order_date"],
    )
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def build_behavioral_features(lines: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order-line attributes to one row per customer.

    - ``top_product_type``: most frequently purchased collection/product type
      per customer (mode).
    - ``avg_product_price``: mean unit price across the customer's order lines.
    - ``price_bucket``: quartile of ``avg_product_price`` across customers
      (budget/mid/premium/luxury).
    - ``avg_num_sku_per_order``: mean basket size across the customer's orders.

    Non-product line items (shipping, gift cards, cashback — see
    ``NON_PRODUCT_TYPES``) are dropped first and "Detergent sheets" /
    "Detergent Sheets" are normalized to a single casing before the mode is
    taken.
    """
    lines = lines[~lines["product_type"].isin(NON_PRODUCT_TYPES)].copy()
    lines["product_type"] = lines["product_type"].str.strip().str.title()

    behavioral = lines.groupby("customer_id").agg(
        top_product_type=("product_type", _mode_or_na),
        avg_product_price=("product_price", "mean"),
        avg_num_sku_per_order=("num_sku_per_order", "mean"),
    ).reset_index()

    behavioral["price_bucket"] = pd.qcut(
        behavioral["avg_product_price"], q=4, labels=PRICE_BUCKET_LABELS
    )

    behavioral["top_product_type"] = behavioral["top_product_type"].fillna(CATEGORICAL_FILL)
    behavioral["avg_num_sku_per_order"] = behavioral["avg_num_sku_per_order"].fillna(0)

    return behavioral


def join_behavioral_features(
    features: pd.DataFrame, behavioral: pd.DataFrame
) -> pd.DataFrame:
    """Left-join behavioral features onto the main feature table."""
    out = features.merge(behavioral, on="customer_id", how="left")

    out["top_product_type"] = out["top_product_type"].fillna(CATEGORICAL_FILL)
    out["price_bucket"] = out["price_bucket"].astype("object").fillna(CATEGORICAL_FILL)
    out["avg_num_sku_per_order"] = out["avg_num_sku_per_order"].fillna(0)
    out["avg_product_price"] = out["avg_product_price"].fillna(0)

    return out


if __name__ == "__main__":
    lines = load_order_line_attributes()
    behavioral = build_behavioral_features(lines)
    behavioral.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(behavioral):,} customer rows to {OUTPUT_PATH}")
