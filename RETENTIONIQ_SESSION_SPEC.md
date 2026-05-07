# RetentionIQ — Today's Session Spec (D1 + D2)

**Owners:** Giulia (D1) + Juan (D2)
**Mode:** Pair programming, in person, shared screen
**Goal:** Ship the data layer, feature table, and FastAPI stubs by EOD so M1, M2, R1, and F1 are unblocked tomorrow.

> **How to use this with Claude Code:** Drop this file at the root of your repo. Run `claude` from the repo root. Tell Claude: *"Read RETENTIONIQ_SESSION_SPEC.md and let's start with Step 1."* Each step has explicit acceptance criteria — don't move on until they're all green.

---

## Context Claude Code needs

You're building **RetentionIQ**, a predictive customer-intelligence product for Shopify brands, as a master's final project. The pipeline is:

`Daasity UOS data → cleaned customer table → BG/NBD + churn features → CLV + churn models → FastAPI endpoints → Streamlit frontend`

You and Juan own the first three stages of that pipeline plus the FastAPI stubs. The full team has 8 people — the rest are blocked until your handoff.

**Key prior decisions (do not relitigate):**
- Models: BG/NBD + Gamma-Gamma for CLV; XGBoost for churn (logistic + RF as benchmarks)
- Explainability: SHAP for per-customer churn drivers
- Storage: Snowflake in prod, local CSV today
- Snapshot-based labeling, not streaming

---

## Repo layout you should have (or create today)

```
retentioniq/
├── data/
│   ├── raw/                  # miracle_sheets.csv lives here
│   ├── interim/              # cleaned customer table
│   └── features/             # BG/NBD inputs + churn label
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_features.ipynb
├── src/
│   ├── data/
│   │   ├── load.py
│   │   └── clean.py
│   ├── features/
│   │   └── build.py
│   └── api/
│       ├── main.py           # FastAPI app
│       └── schemas.py        # pydantic models
├── tests/
├── RETENTIONIQ_SESSION_SPEC.md   # this file
├── pyproject.toml
└── README.md
```

If the repo doesn't exist yet, ask Claude Code to scaffold it before Step 2.

---

## Step 1 — Sync & alignment (15 min, no code)

**Purpose:** lock the decisions that everything else depends on.

### 1a. Inspect the CSV
Have Claude Code run a quick inspection script on `data/raw/miracle_sheets.csv`:

```python
import pandas as pd
df = pd.read_csv("data/raw/miracle_sheets.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Dtypes:\n", df.dtypes)
print("Date range:", df["order_date"].min(), "→", df["order_date"].max())
print("Unique customers:", df["customer_id"].nunique())
print("Sample:\n", df.head(3))
```

Adjust column names (`customer_id`, `order_date`) if Daasity uses different ones — confirm with the actual schema.

### 1b. Answer these explicitly (write into `docs/decisions.md`):

- **Grain:** order / order-line / customer
- **Date range:** `____` to `____`
- **Unique customers:** `____` (and how many are repeat buyers)
- **Currency / multi-store:** single or multi
- **Snapshot date:** `____` (default: max `order_date` in dataset)
- **Churn window:** `____ days` (P90 of inter-purchase time; fall back to P80 if dataset is small)
- **Churn rule:** *A customer is churned if `(snapshot_date - last_order_date) > [window]` AND they had at least 1 order before the snapshot.*

### 1c. Day's work split

| Owner | Step 2 | Step 3 | Step 4 |
|---|---|---|---|
| Giulia (D1) | **Lead** | Review | `/predict/clv` stub |
| Juan (D2) | Review | **Lead** | `/predict/churn` stub |

### Acceptance criteria for Step 1 ✅
- [ ] `docs/decisions.md` exists with all values filled in
- [ ] Both of you can state the churn rule out loud without looking
- [ ] Repo structure exists (or scaffolding ticket logged)

---

## Step 2 — Data layer (60 min, Giulia leads)

**Purpose:** turn the raw CSV into a clean, customer-grain table that Step 3 can consume.

### 2a. `src/data/load.py`
Single function: `load_raw(path: str) -> pd.DataFrame`. Parses dates, normalizes column names to snake_case, casts dtypes.

### 2b. `src/data/clean.py`
Functions:
- `drop_invalid_orders(df)` — remove rows with null `customer_id`, `order_id`, or `order_date`; remove orders with non-positive totals.
- `aggregate_to_order_grain(df)` — if input is order-line, sum to one row per `order_id`.
- `build_customer_table(df, snapshot_date)` — output one row per customer with:
  - `customer_id`
  - `first_order_date`, `last_order_date`
  - `n_orders`, `total_revenue`, `avg_order_value`
  - `tenure_days` = snapshot_date − first_order_date
  - `recency_days` = snapshot_date − last_order_date
  - `frequency` = n_orders − 1 (BG/NBD convention: repeat purchases only)

### 2c. Notebook 01 (`notebooks/01_eda.ipynb`)
Should call into `src/data/` rather than re-implementing logic. Include:
1. Schema overview (Section 1c above)
2. Orders-per-customer histogram
3. Inter-purchase time distribution + P90 line (this **proves** the churn window choice)
4. Revenue Pareto (top 20% → X% of revenue)
5. Survival curve for repurchase

### Acceptance criteria for Step 2 ✅
- [ ] `data/interim/customers.parquet` exists, one row per customer
- [ ] No nulls in `customer_id`, `first_order_date`, `last_order_date`, `n_orders`, `total_revenue`
- [ ] `frequency` is non-negative integer; `recency_days >= 0`
- [ ] Notebook 01 runs top-to-bottom with no errors
- [ ] Pareto chart is in the notebook (M2/Kim needs this for GTM)

---

## Step 3 — Feature engineering (60 min, Juan leads)

**Purpose:** produce the modeling-ready feature table with churn label.

### 3a. `src/features/build.py`

```python
def build_features(customers: pd.DataFrame, churn_window_days: int) -> pd.DataFrame:
    """
    Returns one row per customer with:
      - frequency, recency_days, tenure_days, avg_order_value  (BG/NBD inputs)
      - total_revenue, n_orders                                 (Gamma-Gamma + sanity)
      - is_churned (0/1)                                        (target for XGBoost)
    """
```

### 3b. Validation
Add a quick `assert` block / test:
- `frequency >= 0` everywhere
- `recency_days <= tenure_days` everywhere (mathematical sanity)
- `is_churned` is 0/1, no NaN
- Class balance is reported (e.g., `35% churned`) — flag if it's < 5% or > 95%

### 3c. Save to `data/features/feature_table.parquet`
Also write a `data/features/SCHEMA.md` describing every column, units, and expected ranges. M1 and M2 will read this — make it good.

### Acceptance criteria for Step 3 ✅
- [ ] `feature_table.parquet` exists
- [ ] All validation asserts pass
- [ ] `SCHEMA.md` documents each column
- [ ] You've posted the path + class balance in #team channel
- [ ] Notebook 02 demonstrates the feature distributions (one figure per BG/NBD input)

---

## Step 4 — FastAPI stubs (45 min, split work)

**Purpose:** unblock Eric + Davidson (F1) so frontend can wire endpoints in parallel. Real model loading lands later when M1/M2 hand off `.pkl` files.

### 4a. `src/api/schemas.py` (do this together first, 5 min)

```python
from pydantic import BaseModel
from typing import List

class CLVRequest(BaseModel):
    customer_id: str
    horizon_days: int = 365

class CLVResponse(BaseModel):
    customer_id: str
    predicted_clv: float
    horizon_days: int

class ChurnDriver(BaseModel):
    feature: str
    shap_value: float

class ChurnRequest(BaseModel):
    customer_id: str

class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    top_drivers: List[ChurnDriver]
```

### 4b. `src/api/main.py`

Two endpoints, both stubbed (return deterministic mock data based on `customer_id` hash so F1 sees stable values):

- **Giulia:** `POST /predict/clv` — returns `CLVResponse` with mock `predicted_clv`
- **Juan:** `POST /predict/churn` — returns `ChurnResponse` with mock probability + 3 mock drivers

Add `GET /health` returning `{"status": "ok"}` together.

### 4c. Run locally

```bash
uvicorn src.api.main:app --reload --port 8000
```

Hit both endpoints with `curl` to confirm they return valid responses. Save the `curl` commands in `docs/api_examples.md` so F1 can copy-paste.

### Acceptance criteria for Step 4 ✅
- [ ] Server starts without errors
- [ ] All three endpoints return 200 with valid pydantic responses
- [ ] `docs/api_examples.md` has working `curl` examples
- [ ] OpenAPI docs visible at `http://localhost:8000/docs`

---

## Step 5 — Handoff (15 min, together)

Post in #team:

```
✅ D1+D2 EOD update

• Feature table: data/features/feature_table.parquet
  Schema: data/features/SCHEMA.md
  Class balance: __% churned
  → @Tanzeel @Kim — you're unblocked for modeling

• FastAPI stubs running locally on :8000
  Endpoints: POST /predict/clv, POST /predict/churn, GET /health
  Examples: docs/api_examples.md
  → @Eric @Davidson — you can start wiring the frontend against these

• Decisions doc: docs/decisions.md (churn window, snapshot date, etc.)

Blockers: [none / list them]
```

### Acceptance criteria for Step 5 ✅
- [ ] Message posted
- [ ] All four `@`-mentioned people have acknowledged
- [ ] Franca (P1) is aware of any blockers

---

## Working agreements with Claude Code today

1. **One step at a time.** Finish acceptance criteria before moving on. If Claude tries to jump ahead, pull it back.
2. **Commit after every step.** Conventional commits: `feat(data): build customer table`, `feat(api): stub clv endpoint`, etc.
3. **Tests where it matters.** Don't write a test suite for the stubs, but DO write the validation asserts in Step 3 — those catch the bugs that would silently corrupt M1/M2's models.
4. **If something's unclear, stop and write it in `docs/decisions.md`** before continuing. Future-you and the rest of the team will need it.
5. **Time-box.** If a step blows past 1.5x its budget, flag to Franca rather than absorb it silently.

---

## If you finish early

Stretch goals, in priority order:
1. Add a `pytest` test for `build_customer_table` covering the recency/frequency math.
2. Dockerfile for the FastAPI service so F1 can run it without your machine on.
3. Pre-commit hooks (`black`, `ruff`) so the rest of the team doesn't push ugly code.

Don't touch the actual model training today — that's M1/M2's lane and you'll step on their work.
