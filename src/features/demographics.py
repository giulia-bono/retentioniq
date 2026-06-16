"""Customer-grain demographic features from Looker order-line attributes.

Input: ``data/raw/order_line_demographics.csv`` (order-line grain, produced
by ``src/ingestion/demographic_features.py``). Columns: ``customer_id``,
``order_date``, ``gender``, ``gender_accuracy``, ``state_code``,
``country_code``.

Output: ``data/features/customer_demographic_features.csv`` (customer
grain), joinable onto ``feature_table.parquet`` via ``customer_id``.

Gender/state/country are sourced from the ``customers`` table and are
constant per customer (joined onto every order line for that customer), so
aggregation is a dedup, not a mode/mean.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ORDER_LINE_PATH = "data/raw/order_line_demographics.csv"
OUTPUT_PATH = "data/features/customer_demographic_features.csv"

GENDER_ACCURACY_THRESHOLD = 0.6
CATEGORICAL_FILL = "unknown"

# US Census Bureau regions.
NORTHEAST = {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"}
MIDWEST = {"IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"}
SOUTH = {
    "DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV",
    "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX",
}
WEST = {"AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"}

STATE_TO_REGION = (
    {s: "northeast" for s in NORTHEAST}
    | {s: "midwest" for s in MIDWEST}
    | {s: "south" for s in SOUTH}
    | {s: "west" for s in WEST}
)


def load_order_line_demographics(path: str | Path = ORDER_LINE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"customer_id": "string"}, parse_dates=["order_date"])
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def build_demographic_features(lines: pd.DataFrame) -> pd.DataFrame:
    """Dedup order-line-grain demographic rows to one row per customer.

    - ``gender_clean``: ``gender`` as-is, except rows where
      ``gender_accuracy < GENDER_ACCURACY_THRESHOLD`` are remapped to
      "unknown" (low-confidence inferences are noise, not signal).
    - ``region``: ``state_code`` collapsed to US Census region
      (northeast/midwest/south/west), "other" for non-US / unmapped states.
    - ``is_us``: 1 if ``country_code == "US"`` else 0.
    """
    demo = lines.drop_duplicates("customer_id").copy()

    low_confidence = demo["gender_accuracy"] < GENDER_ACCURACY_THRESHOLD
    demo["gender_clean"] = demo["gender"].where(~low_confidence, CATEGORICAL_FILL)
    demo["gender_clean"] = demo["gender_clean"].fillna(CATEGORICAL_FILL).str.lower()

    demo["region"] = demo["state_code"].map(STATE_TO_REGION).fillna("other")

    demo["is_us"] = (demo["country_code"] == "US").astype("int8")

    return demo[["customer_id", "gender_clean", "region", "is_us"]]


def join_demographic_features(
    features: pd.DataFrame, demographics: pd.DataFrame
) -> pd.DataFrame:
    """Left-join demographic features onto the main feature table."""
    out = features.merge(demographics, on="customer_id", how="left")

    out["gender_clean"] = out["gender_clean"].fillna(CATEGORICAL_FILL)
    out["region"] = out["region"].fillna("other")
    out["is_us"] = out["is_us"].fillna(1).astype("int8")

    return out


if __name__ == "__main__":
    lines = load_order_line_demographics()
    demographics = build_demographic_features(lines)
    demographics.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(demographics):,} customer rows to {OUTPUT_PATH}")
    print()
    print("gender_clean distribution:")
    print(demographics["gender_clean"].value_counts())
    print()
    print("region distribution:")
    print(demographics["region"].value_counts())
    print()
    print("is_us distribution:")
    print(demographics["is_us"].value_counts())
