"""Raw CSV loader for RetentionIQ.

Single entrypoint: ``load_raw(path)`` returns a pandas DataFrame with
snake_case columns, parsed dates, and consistent dtypes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = {
    "customer_id",
    "order_id",
    "order_date",
    "amount_charged",
    "product_amount",
    "financial_status",
}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw orders CSV.

    Parses ``order_date`` as datetime, lowercases column names, casts
    money columns to float64, and trims whitespace from string IDs.
    """
    df = pd.read_csv(
        path,
        parse_dates=["ORDER_DATE"],
        dtype={
            "CUSTOMER_ID": "string",
            "ORDER_ID": "string",
            "ORDER_TAGS": "string",
            "FINANCIAL_STATUS": "string",
            "AMOUNT_CHARGED": "float64",
            "PRODUCT_AMOUNT": "float64",
        },
    )
    df.columns = [c.strip().lower() for c in df.columns]

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"raw CSV is missing expected columns: {sorted(missing)}")

    for col in ("customer_id", "order_id", "financial_status"):
        df[col] = df[col].str.strip()

    return df
