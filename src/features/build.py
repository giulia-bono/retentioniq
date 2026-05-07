"""Feature table assembly for RetentionIQ.

Source of truth is ``data/features/customer_features.csv`` (produced upstream
by Daasity / the team's pipeline; one row per customer, BG/NBD-friendly
column set including ``RECENCY`` (first→last) and ``DAYS_SINCE_LAST_ORDER``).

This module:
  1. Loads that CSV
  2. Snake_cases column names
  3. Adds the ``is_churned`` label using the snapshot-relative
     ``days_since_last_order`` and the agreed 299-day churn window
  4. Validates the table before downstream model training reads it

Churn rule (locked in ``docs/decisions.md``):
    is_churned = (days_since_last_order > 299).astype(int)

The customer-level CSV only contains customers with >= 1 order, so the
"and at least 1 prior order" clause is structurally satisfied.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CUSTOMER_FEATURES_PATH = "data/features/customer_features.csv"

OUTPUT_COLUMNS = [
    "customer_id",
    "frequency",
    "recency",
    "t",
    "days_since_last_order",
    "monetary_value",
    "avg_order_value",
    "total_orders",
    "total_revenue",
    "avg_days_between_orders",
    "std_days_between_orders",
    "refund_count",
    "total_refund_amount",
    "refund_rate",
    "first_order_date",
    "last_order_date",
    "is_churned",
]


def load_customer_features(path: str | Path = CUSTOMER_FEATURES_PATH) -> pd.DataFrame:
    """Load the canonical customer-grain CSV and snake_case the columns."""
    df = pd.read_csv(
        path,
        parse_dates=["FIRST_ORDER_DATE", "LAST_ORDER_DATE"],
        dtype={"CUSTOMER_ID": "string"},
    )
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def build_features(
    customer_features: pd.DataFrame, churn_window_days: int
) -> pd.DataFrame:
    """Return the modeling-ready feature table.

    Inputs:
        customer_features: output of ``load_customer_features``
        churn_window_days: silence threshold (snapshot-relative). 299 today.

    Output column groups:
        BG/NBD inputs:        frequency, recency, t
        Gamma-Gamma input:    monetary_value
        Churn target:         is_churned (0/1)
        Sanity / extras:      total_orders, total_revenue, avg_order_value,
                              avg/std_days_between_orders, refund_*
    """
    if churn_window_days <= 0:
        raise ValueError("churn_window_days must be positive")

    out = customer_features.copy()
    out["is_churned"] = (out["days_since_last_order"] > churn_window_days).astype("int8")

    out = out[OUTPUT_COLUMNS]
    _validate(out)
    return out


def _validate(df: pd.DataFrame) -> None:
    assert df["customer_id"].is_unique, "duplicate customer_id rows"
    assert (df["frequency"] >= 0).all(), "frequency must be >= 0"
    assert df["frequency"].dtype.kind in "iu", "frequency must be integer"
    assert (df["recency"] <= df["t"]).all(), "recency > t (BG/NBD invariant)"
    assert (df["days_since_last_order"] >= 0).all(), "negative days_since_last_order"
    assert df["is_churned"].isin((0, 1)).all(), "is_churned must be 0/1"
    assert df["is_churned"].notna().all(), "is_churned has NaN"
    # monetary_value can legitimately be 0 for one-time buyers (BG/NBD convention)
    assert (df["monetary_value"] >= 0).all(), "negative monetary_value"


def class_balance(df: pd.DataFrame) -> dict:
    n = len(df)
    n_churned = int(df["is_churned"].sum())
    pct = n_churned / n if n else 0.0
    return {
        "n": n,
        "n_churned": n_churned,
        "n_active": n - n_churned,
        "pct_churned": pct,
        "extreme": pct < 0.05 or pct > 0.95,
    }
