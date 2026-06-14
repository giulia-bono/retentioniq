# RetentionIQ — Demo Pipeline

How to go from the committed models to a working, clickable demo on real Miracle customers. Two steps: score the customers, then open the app.

---

## What changed — temporal model migration (this round)

The churn model was rebuilt to fix a **label leakage** problem. Short version:

- **The problem.** The old snapshot model defined churn as `days_since_last_order > 299`, which made that feature the label in disguise. It scored ~1.0 AUROC because it was reading the answer, and it could only flag customers who had *already* lapsed.
- **The fix (temporal model).** Churn is now "no purchase in the 299 days *after* a cutoff date." Features are measured at the cutoff, so recency is a real signal, not the label. Honest **AUROC ~0.68 (5-fold CV 0.679 +/- 0.004)**, and `days_since` drops to the 7th most important feature (top driver is `bgnbd_e_purchases_299d`).
- **What moved to temporal.** `score_customers.py` auto-detects and uses the temporal model + `feature_table_temporal.parquet`. Segmentation and evaluation have temporal twins: `05_segmentation_temporal.ipynb`, `06_evaluation_temporal.ipynb`.
- **New framing: value at stake, not raw churn.** With the temporal model, raw churn is highest for one-time buyers (CLV ~0), so chasing churn means chasing people not worth saving. The demo now ranks by **value at stake (CLV x churn)**. Header shows total value at stake (~$550k) and actionable value (~$333k across win-back / service-recovery / loyalty / timing-nudge). One-time buyers (88% of the base) are a separate volume play (first-repeat nudge).

---

## What's in here

| File | What it is |
|---|---|
| `score_customers.py` | Loads the committed models + the feature table, writes a scored customer table. Auto-detects the temporal model if present, else falls back to the snapshot. |
| `retentioniq_demo_app.html` | The clickable demo. Reads the scored table, falls back to synthetic data if it's not there yet. |
| `src/actions/recommender.py` | The rules recommender (category + strategy + value per customer). |
| `src/actions/decision_log.py` | Records each recommendation and its outcome (the learning loop). |

**Notebooks (temporal pipeline):**
- `04_churn_xgboost_temporal.ipynb` — trains the temporal model, exports `feature_table_temporal.parquet`.
- `05_segmentation_temporal.ipynb` — KMeans on the temporal table, guard-rule k selection, profile-driven segment names.
- `06_evaluation_temporal.ipynb` — honest evaluation + the before/after leakage story (live).

---

## Quick start

The app works **right now** with synthetic data, just open it (step 2). To run it on **real** Miracle customers, do step 1 first.

### Step 0 — Regenerate the temporal feature table (only if missing)

`feature_table_temporal.parquet` is gitignored (large + PII). To rebuild it, open `04_churn_xgboost_temporal.ipynb`, run cells 1-13, then the export cell:

```python
df[["customer_id", "is_churned"] + FEATURE_COLS].to_parquet(
    "../data/features/feature_table_temporal.parquet", index=False)
```

(You only need `orders_raw.csv` + `bgnbd_temporal_params.json` for this; orders on the Drive and the json on running the notebooks.)

### Step 1 — Score the customers

```bash
pip install pandas pyarrow xgboost lifetimes

python score_customers.py
```

It auto-detects the temporal model (`data/models/churn_xgboost_temporal.json`) and the temporal feature table, and writes:
- `data/scored/customer_scores.parquet` — full scored table (~395,612 customers)
- `data/scored/customer_scores_sample.json` — top 200 by value at stake, for the app

It prints the population totals (value at stake / actionable / one-timers) and the top 5 so you can sanity-check immediately.

> Windows note: paths use backslashes. Run from the repo root, then `copy data\scored\customer_scores_sample.json app\`.

### Step 2 — Run the app

Put `customer_scores_sample.json` in the **same folder** as `retentioniq_demo_app.html` (i.e. `app/`), then serve that folder:

```bash
cd app
python -m http.server 3000
```

Open **http://localhost:3000/retentioniq_demo_app.html**

- If the JSON is there, the top badge says **Live · scored Miracle customers**.
- If not, the badge says **Demo data · synthetic** and the app still runs.

---

## What the demo shows

1. **At-risk list**, ranked by **value at stake** (CLV x churn), actionable plays first.
2. **Click a customer**: churn probability, the behavioural drivers (the why), the recommended action with its strategy, and the value at stake with action cost and net-if-recovered.
3. **Record a decision** as treatment or holdout. The "what works" table and maturity bar update. This is the learning loop: recommend, record, learn, improve.

Outcomes in the demo are simulated on click so the loop fills in live. In production they come from Shopify orders.

---

## Honest notes (so the story holds up)

- **The model is the temporal one now.** Present the leakage story as the headline: the old setup leaked (days_since alone ~1.0 AUROC against the snapshot label), the temporal model fixes it (honest 0.68, days_since no longer the top driver). 0.68 with no leakage beats 0.99 that reads the answer.
- **AUPRC ~0.97 is not as strong as it looks.** The temporal churn base rate is ~94%, so AUPRC sits high by construction. Lead with AUROC.
- **Don't oversell the confusion matrix.** At the operating threshold the model flags almost everyone (precision ~base rate, recall ~1.0). The value is in the *ranking* (value at stake), not a hard cut.
- **CLV is 0 for one-time buyers.** BG/NBD + Gamma-Gamma only speaks for repeat buyers. That's 88% of the base showing 0, by design.
- **"Why" is behavioural rules, not SHAP.** The per-customer why says which signal is firing (overdue, refunds, high value), aligned with the recommended action and readable. SHAP lives at the *model* level in `06_evaluation_temporal` (top drivers), where it shows the leak is gone. Per-customer SHAP was tested and collapses to one dominant feature, so the rule layer is the better per-customer explanation.

---

## How it fits the maturity ladder

- Rung 1 (now): rules recommender + decision log.
- Rung 2: learn from the recorded decisions.
- Rung 3: optimise the action choice (contextual bandit, reward = recovered revenue).
- Rung 4: agentic, runs the loop on its own.

The decision log is the foundation the whole ladder stands on.
