"""
RetentionIQ - recommender (rung 1 of the maturity ladder)

Turns a churn score + SHAP drivers + predicted CLV into a CATEGORISED,
strategy-linked recommended action, with the "why" attached and the
value-at-stake math.

Professor's notes this implements:
  - don't group actions, split them by CATEGORY (discount / loyalty / ...)
  - each category connects to a BUSINESS STRATEGY
  - the output combines: the WHY (SHAP) + the targeted RECOMMENDATION + the STRATEGY
  - intensity is value-gated (high CLV gets a human touch, low CLV gets automation)

What this is NOT: a root-cause diagnoser. The data is behavioural, so the
recommendation is matched to the behavioural fingerprint we can see, not to a
stated reason like "late delivery". Refund signals are the one cause-ish hint.

All thresholds below are STARTING HEURISTICS to calibrate on real data, not
validated cutoffs. Tune them against the decision log once outcomes accrue.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# --------------------------------------------------------------------------
# Action categories  (each maps to a business strategy lever)
# --------------------------------------------------------------------------
SERVICE_RECOVERY = "service_recovery"
TIMING_NUDGE = "timing_nudge"
WINBACK = "winback"
LOYALTY = "loyalty"
DISCOUNT = "discount"
FIRST_REPEAT = "first_repeat"
HOLD = "hold"

STRATEGY = {
    SERVICE_RECOVERY: "Fix the cause - protect high-value relationships",
    TIMING_NUDGE:     "Re-engage early and cheap, before the window closes",
    WINBACK:          "Last recoverable moment - margin-aware reactivation",
    LOYALTY:          "Reward and deepen - grow LTV of healthy customers",
    DISCOUNT:         "Price-led reactivation - use only on price-sensitive signals",
    FIRST_REPEAT:     "Convert one-time buyers to repeat - the 88% problem",
    HOLD:             "Protect budget - do not spend on noisy or lost customers",
}

# --------------------------------------------------------------------------
# Tunable thresholds  (placeholders - calibrate on your data / churn window)
# --------------------------------------------------------------------------
CHURN_WINDOW_DAYS = 299      # team-agreed window (note: nb01 uses 318)
REFUND_RATE_HIGH = 0.15      # share of orders refunded that flags dissatisfaction
RECENCY_DRIFT = 1.0          # recency_per_gap > 1 means past their normal rhythm
RECENCY_OVERDUE = 1.6        # well past rhythm, approaching the window
GAP_CV_ERRATIC = 1.0         # high variability = unpredictable buyer
P_ALIVE_LOW = 0.20           # BG/NBD says probably already dropped out
AOV_DROP = 0.80              # current AOV below 80% of their historical average

# CLV tiers (median 12m CLV ~ $14.72, P90 ~ $118 in the Miracle data)
CLV_HIGH = 120.0
CLV_MID = 30.0

# Rough action-cost assumptions per category+intensity (the BRAND sets these)
ACTION_COST = {
    (SERVICE_RECOVERY, "personal"): 35.0,
    (SERVICE_RECOVERY, "semi"): 12.0,
    (WINBACK, "personal"): 25.0,
    (WINBACK, "semi"): 8.0,
    (DISCOUNT, "semi"): 10.0,
    (DISCOUNT, "automated"): 6.0,
    (TIMING_NUDGE, "automated"): 0.5,
    (LOYALTY, "automated"): 3.0,
    (FIRST_REPEAT, "automated"): 1.0,
    (HOLD, "none"): 0.0,
}

# Human-readable reasons for the drivers SHAP surfaces
DRIVER_REASON = {
    "refund_rate": "returning a high share of orders",
    "refund_count": "multiple refunds on record",
    "recency_per_gap": "gone quiet well past their usual reorder rhythm",
    "days_since_last_order": "long silence since the last order",
    "recency": "no recent activity",
    "value_density": "spending less per active day",
    "avg_order_value": "smaller baskets than before",
    "gap_cv": "an erratic, hard-to-predict buying pattern",
    "frequency": "few or no repeat orders",
    "bgnbd_p_alive": "a low modelled probability of still being active",
}


@dataclass
class Recommendation:
    customer_id: str
    churn_prob: float
    predicted_clv: float
    category: str
    action: str
    intensity: str              # personal / semi / automated / none
    strategy: str
    why: list = field(default_factory=list)
    value_at_stake: float = 0.0
    est_action_cost: float = 0.0
    net_if_recovered: float = 0.0
    priority: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _intensity_for_clv(clv: float) -> str:
    if clv >= CLV_HIGH:
        return "personal"
    if clv >= CLV_MID:
        return "semi"
    return "automated"


def _top_up_drivers(shap_drivers, k: int = 3):
    """Return the feature names pushing churn UP, strongest first.
    shap_drivers: list of (feature_name, shap_value). Positive = raises churn."""
    if not shap_drivers:
        return []
    ups = [(f, v) for f, v in shap_drivers if v > 0]
    ups.sort(key=lambda x: abs(x[1]), reverse=True)
    return [f for f, _ in ups[:k]]


def recommend(customer: dict) -> Recommendation:
    """customer expects:
        customer_id, churn_prob, predicted_clv,
        features: dict of raw feature values,
        shap_drivers: optional list of (feature, shap_value)
    """
    cid = customer["customer_id"]
    churn = float(customer.get("churn_prob", 0.0))
    clv = float(customer.get("predicted_clv", 0.0))
    f = customer.get("features", {})
    drivers = customer.get("shap_drivers", [])
    top = _top_up_drivers(drivers)

    intensity = _intensity_for_clv(clv)
    why = [DRIVER_REASON.get(d, d) for d in top] or ["elevated churn score"]

    # ---- routing: priority order of actionable signals -------------------
    refund_rate = float(f.get("refund_rate", 0.0))
    rpg = float(f.get("recency_per_gap", 0.0))
    gap_cv = float(f.get("gap_cv", 0.0))
    p_alive = float(f.get("bgnbd_p_alive", 1.0))
    days_since = float(f.get("days_since_last_order", 0.0))
    frequency = float(f.get("frequency", 0.0))
    aov_ratio = float(f.get("aov_ratio", 1.0))  # current AOV / historical AOV
    is_one_time = bool(f.get("is_one_time", frequency == 0))

    if is_one_time:
        category, action = FIRST_REPEAT, "Send the first-to-second purchase nudge (education + relevant product), timed before the window"
        intensity = "automated"
    elif refund_rate >= REFUND_RATE_HIGH:
        category = SERVICE_RECOVERY
        action = "Reach out about the returns: offer an exchange, fit/sizing help, or a replacement. Do NOT discount."
    elif p_alive <= P_ALIVE_LOW or days_since > CHURN_WINDOW_DAYS:
        if clv >= CLV_HIGH:
            category, action = WINBACK, "Last-chance personal win-back, strongest offer, justified by value at stake"
        else:
            category, action, intensity = HOLD, "Likely already gone and low value, do not spend", "none"
    elif rpg >= RECENCY_OVERDUE:
        category, action = WINBACK, "Timed win-back now, they are overdue and approaching the churn window"
    elif rpg >= RECENCY_DRIFT:
        category, action = TIMING_NUDGE, "Light replenishment / 'time to restock?' nudge, early and cheap"
    elif aov_ratio <= AOV_DROP:
        if clv >= CLV_MID:
            category, action = LOYALTY, "Re-engage with relevant or higher-value products, reward continued loyalty"
        else:
            category, action = DISCOUNT, "Margin-aware price offer, the signal looks price-sensitive"
    elif gap_cv >= GAP_CV_ERRATIC:
        category, action, intensity = HOLD, "Erratic pattern, signal is noisy, low priority", "none"
    elif clv >= CLV_HIGH:
        category, action = LOYALTY, "Healthy and high-value, reward to deepen the relationship"
    else:
        category, action, intensity = HOLD, "No strong actionable signal, monitor", "none"

    # ---- value math ------------------------------------------------------
    value_at_stake = round(clv * churn, 2)
    cost = ACTION_COST.get((category, intensity), ACTION_COST.get((category, "automated"), 0.0))
    net = round(value_at_stake - cost, 2)
    # priority for ranking the at-risk list: expected loss, discounted if likely gone
    priority = round(value_at_stake * max(p_alive, 0.05), 2)

    return Recommendation(
        customer_id=cid, churn_prob=round(churn, 4), predicted_clv=round(clv, 2),
        category=category, action=action, intensity=intensity,
        strategy=STRATEGY[category], why=why,
        value_at_stake=value_at_stake, est_action_cost=cost,
        net_if_recovered=net, priority=priority,
    )


if __name__ == "__main__":
    # demo customers (synthetic - real data lives on Drive)
    demo = [
        {"customer_id": "C1", "churn_prob": 0.78, "predicted_clv": 140.0,
         "features": {"refund_rate": 0.25, "recency_per_gap": 2.1, "frequency": 4},
         "shap_drivers": [("refund_rate", 0.9), ("recency_per_gap", 0.6), ("frequency", -0.1)]},
        {"customer_id": "C2", "churn_prob": 0.55, "predicted_clv": 22.0,
         "features": {"recency_per_gap": 1.2, "frequency": 3, "refund_rate": 0.0},
         "shap_drivers": [("recency_per_gap", 0.7)]},
        {"customer_id": "C3", "churn_prob": 0.9, "predicted_clv": 9.0,
         "features": {"is_one_time": True, "frequency": 0}},
    ]
    for c in demo:
        r = recommend(c)
        print(f"\n{r.customer_id}: {r.category.upper()}  (intensity={r.intensity})")
        print(f"  action  : {r.action}")
        print(f"  strategy: {r.strategy}")
        print(f"  why     : {', '.join(r.why)}")
        print(f"  value   : ${r.value_at_stake} at stake, cost ${r.est_action_cost}, net ${r.net_if_recovered}")
