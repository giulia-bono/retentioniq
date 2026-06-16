"""Pull order-line-grain behavioral attributes from Looker.

Companion to ``sql/features/behavioral_features_looker_query.md``. Runs an
inline query against the ``order_line_revenue`` explore (``miracle_orders``
model — Miracle Sheets Shopify data, see ``data/README.md``) and writes the
result to ``data/raw/order_line_attributes.csv`` for the customer-grain
aggregation step (see ``src/features/behavioral.py``).

Field notes (confirmed against the live explore schema, for the Miracle
brand's data specifically):
  - ``custom_sku_attributes.category`` / ``color_family`` / ``price_bucket``
    don't exist in this instance. Worse: ``sku_attributes.category``,
    ``sub_category``, ``class``, and every ``sku_attributes_bsd.*``
    field (color, size, pattern, material, hierarchy category/class) are
    **100% null** for Miracle — those merchandising dimensions were never
    populated for this brand.
  - The only populated product-level dimension is
    ``sku_attributes.product_type`` (~60% populated; "Signature Collection",
    "Combined Collection", "Extra Luxe Collection", "Towel", ...) — used here
    as the category/collection signal.
  - There is no pre-built price bucket; we pull the raw ``product_price`` and
    bucket it in ``src/features/behavioral.py``.
  - ``financial_status`` lives on ``order_status``, not ``order_line_revenue``.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from looker_sdk import models40 as models

from src.ingestion.looker_client import get_sdk

LOOKML_MODEL = "miracle_orders"
EXPLORE = "order_line_revenue"

FIELDS = [
    "order_line_revenue.customer_id",
    "order_line_revenue.order_id",
    "order_line_revenue.order_date",
    "sku_attributes.product_type",
    "order_line_revenue.product_price",
    "order_line_revenue.num_sku_per_order",
]

FILTERS = {
    "order_line_revenue.order_date": "2024-01-01 to 2026-03-15",
    "order_status.financial_status": "paid, partially_refunded",
    "order_line_revenue.customer_id": "-NULL",
}

# Looker CSV exports use field labels as headers; rename to stable snake_case
# columns for the aggregation step in src/features/behavioral.py.
COLUMN_RENAME = {
    "Order Line Revenue Customer ID": "customer_id",
    "Order Line Revenue Order ID": "order_id",
    "Order Line Revenue Order Date": "order_date",
    "SKU Attributes Product Type": "product_type",
    "Order Line Revenue Variant Price": "product_price",
    "Order Line Revenue Number of SKUs per Order": "num_sku_per_order",
}

OUTPUT_PATH = "data/raw/order_line_attributes.csv"


def fetch_order_line_attributes(output_path: str | Path = OUTPUT_PATH) -> pd.DataFrame:
    """Run the behavioral-features query and write the result to CSV."""
    sdk = get_sdk()

    query = sdk.create_query(
        body=models.WriteQuery(
            model=LOOKML_MODEL,
            view=EXPLORE,
            fields=FIELDS,
            filters=FILTERS,
            limit="-1",  # no row limit, server-paginated
        )
    )

    csv_data = sdk.run_query(query_id=query.id, result_format="csv")

    df = pd.read_csv(io.StringIO(csv_data))
    df = df.rename(columns=COLUMN_RENAME)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Wrote {len(df):,} rows to {output_path}")
    return df


if __name__ == "__main__":
    fetch_order_line_attributes()
