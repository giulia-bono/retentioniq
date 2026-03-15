# Claude Code Guidelines — Shopify Analytics Data Product PoC

## Project Overview

**What:** A data product (PoC) built on top of Shopify API data from the Miracle Sheets brand. The product must solve a real user problem that someone would pay for.

**Who:** 8-person master's group (AI & Data Science). Delivery: end of June 2026.

**Data source:** Full Shopify API access for Miracle Sheets (orders, customers, products, inventory, marketing, analytics, etc.)

---

## 1. Product Concept

### The Problem

DTC (Direct-to-Consumer) Shopify brands like Miracle Sheets lack actionable, predictive intelligence about their customers. Shopify's native analytics are descriptive and backward-looking. Brand operators need to know **who will churn, who will buy again, what to offer them, and when** — but today they export CSVs and guess.

### The Product: **RetentionIQ** — Predictive Customer Intelligence for Shopify Brands

A pluggable analytics layer that connects to any Shopify store and delivers:

1. **Customer Lifetime Value (CLV) prediction** — forecast revenue per customer over 6/12 months
2. **Churn risk scoring** — flag customers likely to never return, with explainability
3. **Next-purchase timing** — predict when a customer will buy again (BG/NBD or similar)
4. **Smart segmentation** — auto-cluster customers into actionable cohorts (high-value loyalists, one-and-done, deal hunters, etc.)
5. **Reactivation recommendations** — suggest discount depth, channel, and timing per segment

### Why Someone Would Pay

- Shopify brands spend 5–10x more acquiring new customers than retaining existing ones.
- Retention tools like Klaviyo handle email but don't predict behavior.
- RetentionIQ sits between raw Shopify data and marketing execution, filling the intelligence gap.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                  │
│          Streamlit Dashboard (or React frontend)        │
│   Segments │ CLV Rankings │ Churn Alerts │ Reco Engine  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│                     APPLICATION LAYER                   │
│                  FastAPI Backend (Python)                │
│   /predict/clv │ /predict/churn │ /segments │ /recos    │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│                      MODEL LAYER                        │
│  CLV Model │ Churn Classifier │ BG/NBD │ Segmentation   │
│            (scikit-learn / lifetimes / XGBoost)          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│                       DATA LAYER                        │
│     Feature Store (Parquet / DuckDB / PostgreSQL)       │
│         ↑ ETL pipeline from raw Shopify data            │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│                    INGESTION LAYER                       │
│             Shopify Admin API (REST + GraphQL)           │
│  Orders │ Customers │ Products │ Inventory │ Marketing   │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Shopify API Data Extraction

### 3.1 Authentication

```python
# Use Shopify's Admin API with a private app token
# Store credentials in .env — NEVER commit them
SHOPIFY_STORE_URL=https://miracle-sheets.myshopify.com
SHOPIFY_ACCESS_TOKEN=shpat_xxxxxxxxxxxxx
SHOPIFY_API_VERSION=2024-10
```

### 3.2 Key Endpoints to Extract

| Entity | API Endpoint | Key Fields | Priority |
|--------|-------------|------------|----------|
| Orders | `/admin/api/{version}/orders.json` | id, created_at, total_price, line_items, customer, discount_codes, financial_status, fulfillment_status, refunds | P0 |
| Customers | `/admin/api/{version}/customers.json` | id, email, created_at, orders_count, total_spent, tags, addresses, default_address | P0 |
| Products | `/admin/api/{version}/products.json` | id, title, product_type, variants, tags, created_at | P1 |
| Inventory | `/admin/api/{version}/inventory_levels.json` | inventory_item_id, available, location_id | P2 |
| Marketing Events | `/admin/api/{version}/marketing_events.json` | channel, started_at, budget, utm_parameters | P2 |
| Abandoned Checkouts | `/admin/api/{version}/checkouts.json` | token, created_at, customer, line_items, abandoned_checkout_url | P1 |

### 3.3 Extraction Script Structure

```
src/
  ingestion/
    shopify_client.py        # Authenticated API client with rate-limit handling
    extract_orders.py        # Paginated order extraction (handle 250/page limit)
    extract_customers.py     # Customer data pull
    extract_products.py      # Product catalog
    extract_checkouts.py     # Abandoned checkouts
    run_full_extraction.py   # Orchestrator — runs all extractions, saves to raw/
```

### 3.4 Critical Implementation Notes

- **Rate limits:** Shopify allows 40 requests/second (REST) with a leaky bucket. Implement exponential backoff. Check `X-Shopify-Shop-Api-Call-Limit` header.
- **Pagination:** Use `link` header for cursor-based pagination. Never rely on page numbers.
- **Date filtering:** Use `created_at_min` / `updated_at_min` for incremental pulls.
- **Data volume:** Miracle Sheets likely has 10k–500k orders. Plan for full historical pull + daily incremental.
- **PII handling:** Customer emails and addresses are PII. Hash emails for modeling; keep raw data in a restricted layer. Document GDPR/privacy approach.

---

## 4. Data Pipeline & Feature Engineering

### 4.1 Raw → Clean → Features

```
data/
  raw/                  # JSON dumps from Shopify API (gitignored)
  clean/                # Deduplicated, typed, validated Parquet files
  features/             # Model-ready feature tables
  models/               # Serialized trained models
```

### 4.2 Core Feature Table: `customer_features`

Build one row per customer with these feature groups:

**Recency / Frequency / Monetary (RFM)**
- `days_since_last_order` — recency
- `total_orders` — frequency
- `total_revenue` — monetary
- `avg_order_value`
- `days_between_orders_avg` / `days_between_orders_std`
- `tenure_days` — days since first order

**Behavioral**
- `unique_products_bought`
- `unique_product_types_bought`
- `has_used_discount` (bool)
- `discount_rate` — % of orders with a discount code
- `avg_discount_pct`
- `refund_count` / `refund_rate`
- `abandoned_checkout_count`

**Temporal**
- `first_order_date` / `last_order_date`
- `is_weekend_buyer` — >50% orders on weekends
- `preferred_hour_of_day`
- `order_trend` — slope of order frequency over time (accelerating vs decelerating)

**Product Affinity**
- `top_product_type` — most purchased category
- `product_diversity_index` — Shannon entropy of product type distribution
- `repeat_product_rate` — % of orders containing a previously bought product

### 4.3 Pipeline Implementation

```python
# Use DuckDB for fast local analytics or Pandas for simplicity
# Parquet as interchange format for all intermediate tables

# Pipeline order:
# 1. Load raw JSON → normalize nested structures (line_items, refunds)
# 2. Dedup on order_id / customer_id
# 3. Build order-level table (one row per order, flattened)
# 4. Aggregate to customer-level features
# 5. Add labels (churn = no order in last 90 days, CLV = next-12-month revenue)
# 6. Save to features/customer_features.parquet
```

---

## 5. Models

### 5.1 Customer Lifetime Value (CLV)

**Approach A — Probabilistic (recommended for PoC):**
Use the `lifetimes` library (BG/NBD + Gamma-Gamma model). Requires only frequency, recency, T (tenure), and monetary value. Well-suited for contractual/non-contractual settings.

```python
from lifetimes import BetaGeoFitter, GammaGammaFitter

bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(data['frequency'], data['recency'], data['T'])

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(data['frequency'], data['monetary_value'])

clv = ggf.customer_lifetime_value(
    bgf, data['frequency'], data['recency'], data['T'],
    data['monetary_value'], time=12, freq='D'
)
```

**Approach B — ML-based (stretch goal):**
XGBoost regressor on the full feature set, predicting next-12-month revenue. Requires a proper train/test split on time (no data leakage).

### 5.2 Churn Prediction

**Definition:** A customer is "churned" if they have not placed an order in the last 90 days (calibrate this threshold using Miracle Sheets' repurchase cycle — sheets are durable, so 120–180 days might be more appropriate).

**Model:** XGBoost or LightGBM binary classifier.

```python
# Label engineering (no leakage)
# Split timeline: use data up to T_split for features, data after T_split for labels
# T_split = max_date - 90 days (or chosen churn window)

# Features: all customer_features computed using only orders before T_split
# Label: 1 if customer had 0 orders in [T_split, T_split + 90 days], else 0
```

**Key outputs:**
- Churn probability per customer (0–1)
- SHAP values for explainability (why is this customer at risk?)
- Feature importance ranking

### 5.3 Next-Purchase Timing

BG/NBD model from `lifetimes` already provides `conditional_expected_number_of_purchases_up_to_time()` — use this to estimate when a customer is likely to buy next.

### 5.4 Customer Segmentation

**Approach:** K-Means or HDBSCAN on scaled RFM + behavioral features.

Target 4–6 segments, e.g.:
- **Loyalists** — high frequency, high CLV, low churn risk
- **At-Risk VIPs** — high historical spend but declining frequency
- **One-and-Done** — single order, high churn probability
- **Deal Seekers** — only buy with discounts, low full-price conversion
- **New Potentials** — recent first purchase, profile similar to loyalists
- **Dormant** — no activity in 6+ months but not formally churned

### 5.5 Model Evaluation

| Model | Metric | Target |
|-------|--------|--------|
| CLV (probabilistic) | MAE on held-out 6-month revenue | < 30% of mean CLV |
| CLV (ML) | RMSE, R² on time-split test set | R² > 0.4 |
| Churn | AUC-ROC, Precision@top-20% | AUC > 0.75 |
| Segmentation | Silhouette score + business interpretability | Silhouette > 0.3 |

---

## 6. API Layer

### FastAPI Backend

```
src/
  api/
    main.py              # FastAPI app entry point
    routers/
      customers.py       # GET /customers/{id}/profile — full customer intelligence card
      predictions.py     # GET /predict/clv, /predict/churn
      segments.py        # GET /segments, /segments/{id}/customers
      recommendations.py # GET /recos/{customer_id} — reactivation suggestions
    services/
      model_service.py   # Load and serve trained models
      feature_service.py # Compute features on-the-fly or from cache
```

### Key Endpoints

```
GET  /api/v1/dashboard/summary
     → total customers, avg CLV, churn rate, segment distribution

GET  /api/v1/customers?segment=at_risk&sort=clv_desc&limit=50
     → paginated customer list with predictions

GET  /api/v1/customers/{customer_id}
     → full profile: CLV, churn score, segment, order history, SHAP explanation

GET  /api/v1/segments
     → list of segments with size, avg CLV, avg churn rate

GET  /api/v1/predictions/batch
     → trigger batch re-prediction for all customers

POST /api/v1/recommendations/{customer_id}
     → AI-generated reactivation recommendation (discount %, channel, timing)
```

---

## 7. Frontend (Streamlit PoC)

### Pages

1. **Executive Dashboard** — KPIs (total customers, avg CLV, churn rate), segment pie chart, CLV distribution histogram, revenue trend
2. **Customer Explorer** — searchable/filterable table of all customers with CLV, churn risk, segment badges. Click to drill into individual profile.
3. **Customer Profile** — single customer deep dive: order timeline, CLV prediction, churn probability gauge, SHAP waterfall chart, recommended action
4. **Segment Analysis** — one page per segment: defining characteristics, top customers, transition matrix (who moved between segments over time)
5. **Reactivation Playbook** — for each at-risk segment, suggest email campaigns, discount offers, timing. Show estimated revenue recovery.

### Streamlit Structure

```
app/
  streamlit_app.py       # Main entry (st.set_page_config, navigation)
  pages/
    1_dashboard.py
    2_customer_explorer.py
    3_customer_profile.py
    4_segments.py
    5_reactivation.py
  components/
    charts.py            # Plotly/Altair chart builders
    kpi_cards.py         # Metric card components
    filters.py           # Sidebar filters
```

---

## 8. Project Structure (Full)

```
retentioniq/
│
├── CLAUDE_CODE_GUIDELINES.md    # This file
├── README.md                     # Project overview + setup instructions
├── pyproject.toml                # Dependencies (use poetry or uv)
├── .env.example                  # Template for secrets
├── .gitignore
│
├── data/
│   ├── raw/                      # Shopify API dumps (gitignored)
│   ├── clean/                    # Processed Parquet files
│   ├── features/                 # Feature tables
│   └── models/                   # Serialized models (.pkl, .joblib)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clv_modeling.ipynb
│   ├── 04_churn_modeling.ipynb
│   ├── 05_segmentation.ipynb
│   └── 06_evaluation.ipynb
│
├── src/
│   ├── ingestion/                # Shopify API extraction scripts
│   ├── pipeline/                 # ETL: raw → clean → features
│   ├── models/                   # Model training and prediction code
│   ├── api/                      # FastAPI backend
│   └── utils/                    # Helpers (logging, config, etc.)
│
├── app/                          # Streamlit frontend
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_api.py
│
└── docs/
    ├── data_dictionary.md        # Every field, its source, and transformations
    ├── model_cards.md            # Model documentation (per ML best practices)
    └── architecture.md           # System design decisions
```

---

## 9. Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Language | Python 3.11+ | Team expertise, ML ecosystem |
| Data extraction | `requests` + custom Shopify client | Full control over pagination/rate limits |
| Data processing | Pandas + DuckDB | Fast local analytics, SQL-friendly |
| Feature store | Parquet files (PoC) → PostgreSQL (prod) | Simple, portable, fast reads |
| CLV modeling | `lifetimes` library | Gold standard for BG/NBD + Gamma-Gamma |
| Churn modeling | XGBoost / LightGBM | Best tabular performance |
| Explainability | SHAP | Model-agnostic feature explanations |
| Segmentation | scikit-learn (KMeans, HDBSCAN) | Well-understood, easy to tune |
| API | FastAPI | Async, auto-docs, type-safe |
| Frontend | Streamlit | Fast PoC iteration, Python-native |
| Charts | Plotly | Interactive, beautiful, Streamlit-integrated |
| Package mgmt | `uv` or `poetry` | Reproducible environments |
| Testing | pytest | Standard Python testing |

---

## 10. Team Work Split (8 People)

| Role | People | Scope |
|------|--------|-------|
| Data Engineering | 2 | Shopify extraction, ETL pipeline, feature engineering, data quality |
| ML Engineering | 2 | CLV model, churn model, segmentation, evaluation, SHAP |
| Backend | 1 | FastAPI endpoints, model serving, caching |
| Frontend | 1 | Streamlit dashboard, charts, UX |
| Product / Analysis | 1 | Business logic, segment definitions, reactivation strategies, stakeholder docs |
| Integration / DevOps | 1 | CI/CD, Docker, testing, deployment, documentation |

---

## 11. Milestones

| Week | Dates (approx.) | Deliverable |
|------|-----------------|-------------|
| 1–2 | Mid-March → End March | Shopify data extraction complete. Raw data in Parquet. EDA notebook done. |
| 3–4 | April 1–14 | Feature engineering pipeline. Clean customer_features table. |
| 5–6 | April 15–28 | CLV + Churn models trained and evaluated. Segmentation done. |
| 7–8 | April 29 – May 12 | FastAPI backend serving predictions. Integration tests passing. |
| 9–10 | May 13–26 | Streamlit dashboard functional. End-to-end demo working. |
| 11–12 | May 27 – June 9 | Polish: explainability, edge cases, documentation, model cards. |
| 13 | June 10–16 | Final testing, presentation prep. |
| 14 | June 17–end | **Delivery and presentation.** |

---

## 12. Claude Code — Specific Instructions

When building any component of this project, follow these rules:

### General
- Always create files in the project structure defined in Section 8.
- Use type hints everywhere in Python code.
- Add docstrings to every function and class.
- Follow PEP 8. Use `ruff` for linting.
- Every module should have a corresponding test file in `tests/`.

### Data Ingestion
- Build a `ShopifyClient` class that handles authentication, rate limiting (respect `X-Shopify-Shop-Api-Call-Limit`), pagination (cursor-based via `link` header), and retries with exponential backoff.
- Each extraction script should be idempotent — running it twice should not duplicate data.
- Save raw API responses as JSON lines (`.jsonl`) in `data/raw/` with timestamps in filenames.
- Log extraction progress (records fetched, pages processed, errors).

### Data Pipeline
- Use Pandas for transformations, DuckDB for aggregations when performance matters.
- Never modify raw data in place. Always read from `raw/`, write to `clean/` or `features/`.
- The feature engineering pipeline must be a single runnable script: `python -m src.pipeline.build_features`.
- Include data validation checks (null rates, value ranges, cardinality).

### Models
- Always split data by time, never randomly, to prevent leakage.
- Save all models with `joblib.dump()` including metadata (training date, features used, hyperparams, metrics).
- Create a `ModelCard` dataclass to document each model.
- Use SHAP for all tree-based models. Save SHAP values alongside predictions.

### API
- Use Pydantic models for all request/response schemas.
- Add `/health` and `/ready` endpoints.
- Return consistent error responses with proper HTTP status codes.
- Add CORS middleware for frontend integration.

### Frontend
- Use `st.cache_data` for data loading and `st.cache_resource` for model loading.
- All charts should be built with Plotly for interactivity.
- Use `st.metric()` for KPI cards.
- Add loading spinners for any computation that takes >1 second.

### Security & Privacy
- Never commit `.env`, raw data files, or API tokens.
- Hash customer emails with SHA-256 before using in any shareable output.
- The `.gitignore` must exclude: `data/raw/`, `.env`, `*.pkl`, `*.joblib`, `__pycache__/`.

---

## 13. Getting Started (First Session)

When you start working on this project with Claude Code, begin with:

```bash
# 1. Initialize the project
mkdir -p retentioniq && cd retentioniq
git init

# 2. Create the directory structure
mkdir -p data/{raw,clean,features,models}
mkdir -p src/{ingestion,pipeline,models,api/routers,api/services,utils}
mkdir -p app/{pages,components}
mkdir -p notebooks tests docs

# 3. Create .env.example
echo "SHOPIFY_STORE_URL=https://your-store.myshopify.com" > .env.example
echo "SHOPIFY_ACCESS_TOKEN=shpat_xxxxx" >> .env.example

# 4. Initialize pyproject.toml and install dependencies
# Use uv or poetry
```

Then tell Claude Code: **"Read CLAUDE_CODE_GUIDELINES.md and start building the Shopify data extraction layer following Section 3."**

---

## 14. Definition of Done (for the PoC)

The PoC is complete when:

- [ ] Shopify data is extracted and stored as clean Parquet files
- [ ] Customer feature table has 15+ engineered features
- [ ] CLV model produces per-customer 12-month revenue predictions
- [ ] Churn model classifies customers with AUC > 0.75
- [ ] Segmentation produces 4–6 interpretable customer segments
- [ ] FastAPI serves predictions via documented endpoints
- [ ] Streamlit dashboard shows executive summary, customer explorer, and individual profiles
- [ ] SHAP explanations are available for churn predictions
- [ ] All code has tests with >60% coverage
- [ ] Documentation includes data dictionary, model cards, and architecture diagram
- [ ] Demo can run end-to-end from raw data to dashboard in under 10 minutes
