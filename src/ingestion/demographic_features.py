"""Pull order-line-grain demographic attributes from Looker.

Companion to ``sql/features/demographic_features_looker_query.md``. Runs an
inline query against the ``order_line_revenue`` explore (``miracle_orders``
model) and writes the result to ``data/raw/order_line_demographics.csv`` for
the customer-grain dedup step (see ``src/features/demographics.py``).

Field notes (confirmed against the live explore schema, 50k-row sample):
  - ``customers.gender``: 0% null. Female 64.5%, Male 18.0%, Unknown 17.5%.
  - ``customers.gender_accuracy``: confidence score 0-1 from an inferred-
    gender service; NaN when gender = "Unknown".
  - ``customers.country_code``: 0.2% null, 98.6% "US".
  - ``customers.state_code``: ~9.6% null (US rows only).
  - ``financial_status`` lives on ``order_status``, not ``order_line_revenue``
    (same as the behavioral extract).
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
    "order_line_revenue.order_date",
    "customers.gender",
    "customers.gender_accuracy",
    "customers.state_code",
    "customers.country_code",
]

FILTERS = {
    "order_line_revenue.order_date": "2024-01-01 to 2026-03-15",
    "order_status.financial_status": "paid, partially_refunded",
    "order_line_revenue.customer_id": "-NULL",
}

# Looker CSV exports use field labels as headers; rename to stable snake_case
# columns for the aggregation step in src/features/demographics.py.
COLUMN_RENAME = {
    "Order Line Revenue Customer ID": "customer_id",
    "Order Line Revenue Order Date": "order_date",
    "Customer Info Customer Gender": "gender",
    "Customer Info Customer Gender Accuracy": "gender_accuracy",
    "Customer Info State Code": "state_code",
    "Customer Info Country Code": "country_code",
}

OUTPUT_PATH = "data/raw/order_line_demographics.csv"


def fetch_order_line_demographics(output_path: str | Path = OUTPUT_PATH) -> pd.DataFrame:
    """Run the demographic-features query and write the result to CSV."""
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
    fetch_order_line_demographics()
