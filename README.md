# RetentionIQ — Predictive Customer Intelligence for Shopify Brands

A data product PoC that connects to any Shopify store and delivers actionable, predictive intelligence: customer lifetime value forecasts, churn risk scores, next-purchase timing, smart segmentation, and reactivation recommendations.

Built on top of Miracle Sheets Shopify data as part of an 8-person AI & Data Science master's project (delivery: end of June 2026).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                     │
│           Streamlit Dashboard  (app/)                       │
│   Segments | CLV Rankings | Churn Alerts | Reco Engine      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                      APPLICATION LAYER                      │
│               FastAPI Backend  (src/api/)                   │
│   /predict/clv | /predict/churn | /segments | /recos        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                        MODEL LAYER                          │
│   src/models/                                               │
│   CLV (BG/NBD + Gamma-Gamma)  |  Churn (XGBoost)           │
│   Segmentation (KMeans/HDBSCAN)  |  Next-purchase timing   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                        DATA LAYER                           │
│   src/pipeline/   →   data/features/  (Parquet)            │
│   Feature engineering: RFM + behavioral + temporal          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                      INGESTION LAYER                        │
│   src/ingestion/   →   data/raw/  (.jsonl)                  │
│   Shopify Admin API (REST)                                  │
│   Orders | Customers | Products | Checkouts | Marketing     │
└─────────────────────────────────────────────────────────────┘
```

---

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
cd retentioniq

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the data

The CSV datasets are shared via Google Drive (not committed to Git — large files + PII).
See [data/README.md](data/README.md) for the download link and setup instructions.

### 3. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in your Shopify store URL and access token
```

### 4. Run data extraction

```bash
python -m src.ingestion.run_full_extraction
```

Raw API responses are saved as `.jsonl` files in `data/raw/` (gitignored).

### 5. Build features

```bash
python -m src.pipeline.build_features
```

Outputs `data/features/customer_features.parquet`.

### 6. Train models

```bash
python -m src.models.train_all
```

Serialized models are saved to `data/models/` (gitignored — use artifact storage in prod).

### 7. Start the API

```bash
uvicorn src.api.main:app --reload
# Docs at http://localhost:8000/docs
```

### 8. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
.
├── CLAUDE_CODE_GUIDELINES.md   # Full product spec and coding rules
├── README.md
├── requirements.txt
├── .env.example                # Credential template (never commit .env)
├── .gitignore
│
├── data/
│   ├── raw/                    # Shopify API dumps — gitignored (large + PII)
│   ├── clean/                  # Deduplicated, typed Parquet files
│   ├── features/               # Model-ready customer_features table
│   └── models/                 # Serialized trained models — gitignored
│
├── notebooks/
│   ├── 01_explore_daasity_schemas.ipynb
│   ├── 02_model_training_benchmarks.ipynb
│   ├── 03_clv_modeling.ipynb       (planned)
│   ├── 04_churn_modeling.ipynb     (planned)
│   ├── 05_segmentation.ipynb       (planned)
│   └── 06_evaluation.ipynb         (planned)
│
├── sql/
│   └── features/               # SQL queries used for feature generation
│
├── src/
│   ├── ingestion/              # Shopify API extraction scripts
│   ├── pipeline/               # ETL: raw → clean → features
│   ├── models/                 # Training, prediction, model cards
│   ├── api/
│   │   ├── main.py
│   │   ├── routers/            # customers, predictions, segments, recos
│   │   └── services/           # model_service, feature_service
│   └── utils/                  # Logging, config helpers
│
├── app/                        # Streamlit frontend
│   ├── streamlit_app.py
│   ├── pages/                  # Dashboard, Explorer, Profile, Segments, Reactivation
│   └── components/             # Charts, KPI cards, filters
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_api.py
│
└── docs/
    ├── data_dictionary.md      # Every field, source, and transformation
    ├── model_cards.md          # Model documentation
    └── architecture.md         # System design decisions
```

---

## Key Features

| Feature | Technology |
|---------|-----------|
| Customer Lifetime Value | `lifetimes` (BG/NBD + Gamma-Gamma) |
| Churn Prediction | XGBoost + SHAP explainability |
| Next-purchase timing | BG/NBD conditional expectations |
| Segmentation | KMeans / HDBSCAN on RFM features |
| API | FastAPI with Pydantic schemas |
| Dashboard | Streamlit + Plotly |

---

## Milestones

| Dates | Deliverable |
|-------|-------------|
| Mid-March → End March 2026 | Shopify extraction + EDA notebook |
| April 1–14 | Feature engineering pipeline |
| April 15–28 | CLV + Churn models evaluated |
| April 29 – May 12 | FastAPI backend serving predictions |
| May 13–26 | Streamlit dashboard functional |
| May 27 – June 9 | Explainability, edge cases, documentation |
| June 10–16 | Final testing + presentation prep |
| June 17+ | **Delivery and presentation** |

---

## Team

8-person master's group (AI & Data Science):
Data Engineering (×2) · ML Engineering (×2) · Backend (×1) · Frontend (×1) · Product/Analysis (×1) · DevOps/Integration (×1)

---

## Security & Privacy

- Customer emails are hashed with SHA-256 before use in any shareable output.
- Raw data (`data/raw/`) is gitignored — treat it as restricted.
- Never commit `.env`, model binaries, or raw API dumps.
- See CLAUDE_CODE_GUIDELINES.md §12 for full security rules.
