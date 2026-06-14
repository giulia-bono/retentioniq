"""
RetentionIQ - decision log (rung 2 of the maturity ladder)

"Record the intelligence." Every recommendation the system makes is logged,
and later the outcome is attached. That record IS the proprietary asset: it's
what lets the model stop being junior and start learning which action works
for which behavioural fingerprint.

Why SQLite: the professor's constraint is "no CRM, we have Shopify". So
RetentionIQ itself becomes the system of record for retention decisions. A
file-based SQLite table is dependency-free and swappable for Postgres later.

The flywheel this enables:
    rung 2  record decision + outcome        <- this file
    rung 3  learn which action lifts repeat  (supervised, on this table)
    rung 4  optimise action choice           (contextual bandit / RL, reward = uplift)
    rung 5  agent owns the loop end to end    (agentic process owner)

Holdout note: each decision gets an `assignment` of "treatment" or "holdout".
Withholding the action for a random control slice is what lets you measure
TRUE uplift (not just before/after), which is also the reward signal for rung 4.
"""
from __future__ import annotations

import sqlite3
import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_DB = "data/decisions/retentioniq_decisions.db"


class DecisionLog:
    def __init__(self, db_path: str = DEFAULT_DB, holdout_rate: float = 0.10):
        self.db_path = db_path
        self.holdout_rate = holdout_rate
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id      TEXT PRIMARY KEY,
                customer_id      TEXT NOT NULL,
                decided_at       TEXT NOT NULL,
                churn_prob       REAL,
                predicted_clv    REAL,
                category         TEXT,
                action           TEXT,
                intensity        TEXT,
                strategy         TEXT,
                why              TEXT,           -- json list
                value_at_stake   REAL,
                est_action_cost  REAL,
                assignment       TEXT,           -- treatment | holdout
                -- outcome, filled in later from Shopify orders
                action_taken     INTEGER,        -- 0/1, did marketing actually act
                acted_at         TEXT,
                repurchased      INTEGER,        -- 0/1 within the window
                revenue_recovered REAL,
                outcome_at       TEXT
            )
            """
        )
        self._conn.commit()

    # -- rung 2: write the decision ---------------------------------------
    def record(self, rec: dict, assignment: Optional[str] = None) -> str:
        """rec = Recommendation.to_dict(). Returns the decision_id."""
        decision_id = str(uuid.uuid4())
        if assignment is None:
            assignment = "holdout" if random.random() < self.holdout_rate else "treatment"
        self._conn.execute(
            """INSERT INTO decisions
               (decision_id, customer_id, decided_at, churn_prob, predicted_clv,
                category, action, intensity, strategy, why,
                value_at_stake, est_action_cost, assignment)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                decision_id, rec["customer_id"], datetime.now(timezone.utc).isoformat(),
                rec.get("churn_prob"), rec.get("predicted_clv"),
                rec.get("category"), rec.get("action"), rec.get("intensity"),
                rec.get("strategy"), json.dumps(rec.get("why", [])),
                rec.get("value_at_stake"), rec.get("est_action_cost"), assignment,
            ),
        )
        self._conn.commit()
        return decision_id

    # -- did marketing act on it? -----------------------------------------
    def mark_action_taken(self, decision_id: str, taken: bool = True):
        self._conn.execute(
            "UPDATE decisions SET action_taken=?, acted_at=? WHERE decision_id=?",
            (1 if taken else 0, datetime.now(timezone.utc).isoformat(), decision_id),
        )
        self._conn.commit()

    # -- attach the outcome (sourced from Shopify orders) ------------------
    def attach_outcome(self, decision_id: str, repurchased: bool, revenue_recovered: float = 0.0):
        self._conn.execute(
            """UPDATE decisions
               SET repurchased=?, revenue_recovered=?, outcome_at=?
               WHERE decision_id=?""",
            (1 if repurchased else 0, revenue_recovered,
             datetime.now(timezone.utc).isoformat(), decision_id),
        )
        self._conn.commit()

    # -- rung 3/4: pull the learning set ----------------------------------
    def training_frame(self):
        """Closed decisions (outcome known) for learning which action works."""
        cur = self._conn.execute(
            "SELECT * FROM decisions WHERE outcome_at IS NOT NULL"
        )
        return [dict(row) for row in cur.fetchall()]

    def uplift_snapshot(self):
        """Crude treatment-vs-holdout repurchase rates per category.
        This is the reward signal the bandit/RL step will optimise."""
        cur = self._conn.execute(
            """SELECT category, assignment,
                      AVG(repurchased) AS repurchase_rate, COUNT(*) AS n
               FROM decisions WHERE outcome_at IS NOT NULL
               GROUP BY category, assignment"""
        )
        return [dict(row) for row in cur.fetchall()]

    def close(self):
        self._conn.close()


if __name__ == "__main__":
    # demo: record a decision, mark it acted, attach an outcome
    from recommender import recommend

    log = DecisionLog(db_path="data/decisions/demo.db")
    rec = recommend({
        "customer_id": "C1", "churn_prob": 0.78, "predicted_clv": 140.0,
        "features": {"refund_rate": 0.25, "recency_per_gap": 2.1, "frequency": 4},
        "shap_drivers": [("refund_rate", 0.9), ("recency_per_gap", 0.6)],
    }).to_dict()

    did = log.record(rec)
    print("recorded decision:", did, "->", rec["category"])
    log.mark_action_taken(did, True)
    log.attach_outcome(did, repurchased=True, revenue_recovered=140.0)
    print("uplift snapshot:", log.uplift_snapshot())
    log.close()
