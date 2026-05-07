"""Cleaning + customer-table assembly for RetentionIQ.

Pipeline (called from notebooks or scripts):

    df  = load_raw("data/raw/orders_raw.csv")
    df  = drop_invalid_orders(df)
    df  = aggregate_to_order_grain(df)            # no-op for current source
    cst = build_customer_table(df, snapshot_date) # one row per customer

Filter rules are locked in ``docs/decisions.md`` — change them there first.
"""
from __future__ import annotations

import pandas as pd

VALID_FINANCIAL_STATUSES = ("paid", "partially_refunded")


def drop_invalid_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that can't be modeled.

    Rules (see ``docs/decisions.md``):
      1. null ``customer_id`` / ``order_id`` / ``order_date``
      2. ``financial_status`` not in {paid, partially_refunded}
      3. ``amount_charged`` <= 0
    """
    out = df.dropna(subset=["customer_id", "order_id", "order_date"])
    out = out[out["financial_status"].isin(VALID_FINANCIAL_STATUSES)]
    out = out[out["amount_charged"] > 0]
    return out.reset_index(drop=True)


def aggregate_to_order_grain(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse order-line rows into one row per ``order_id``.

    Source today is already order-grain (one row per ``order_id``), so this
    is a defensive no-op that *also* asserts the invariant. Kept as a hook
    in case Daasity exports change.
    """
    if df["order_id"].is_unique:
        return df

    agg = (
        df.groupby("order_id", as_index=False)
        .agg(
            customer_id=("customer_id", "first"),
            order_date=("order_date", "min"),
            amount_charged=("amount_charged", "sum"),
            product_amount=("product_amount", "sum"),
            financial_status=("financial_status", "first"),
        )
    )
    return agg


def build_customer_table(
    orders: pd.DataFrame, snapshot_date: pd.Timestamp
) -> pd.DataFrame:
    """One row per customer, ready for feature engineering.

    Columns:
      - customer_id
      - first_order_date, last_order_date
      - n_orders, total_revenue, avg_order_value
      - tenure_days   = snapshot_date - first_order_date
      - recency_days  = snapshot_date - last_order_date
      - frequency     = n_orders - 1   (BG/NBD: repeat purchases only)
    """
    snapshot_date = pd.Timestamp(snapshot_date).normalize()

    agg = (
        orders.groupby("customer_id", as_index=False)
        .agg(
            first_order_date=("order_date", "min"),
            last_order_date=("order_date", "max"),
            n_orders=("order_id", "nunique"),
            total_revenue=("amount_charged", "sum"),
        )
    )
    agg["avg_order_value"] = agg["total_revenue"] / agg["n_orders"]
    agg["tenure_days"] = (snapshot_date - agg["first_order_date"].dt.normalize()).dt.days
    agg["recency_days"] = (snapshot_date - agg["last_order_date"].dt.normalize()).dt.days
    agg["frequency"] = agg["n_orders"] - 1
    return agg
