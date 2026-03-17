# Data Directory

The raw and feature CSV files are **not committed to Git** (large files + customer PII).
They are shared via Google Drive and must be downloaded manually before running the notebooks.

---

## Download the Data

**Google Drive folder (accessible to anyone with the link):**
[https://drive.google.com/drive/folders/1vBN1SAAAMDbRpkk1yVbxAOz4EGT2w5SK?usp=sharing](https://drive.google.com/drive/folders/1vBN1SAAAMDbRpkk1yVbxAOz4EGT2w5SK?usp=sharing)

### Files to download

| File | Destination in this repo | Size |
|------|--------------------------|------|
| `orders_raw.csv` | `data/raw/orders_raw.csv` | ~201 MB |
| `customer_features.csv` | `data/features/customer_features.csv` | ~98 MB |

---

## Setup (one-time)

1. Open the Google Drive link above and download both CSV files.
2. Place them in the correct folders:

```
retentioniq/
└── data/
    ├── raw/
    │   └── orders_raw.csv          ← put it here
    └── features/
        └── customer_features.csv   ← put it here
```

3. You're ready to run the notebooks:

```bash
jupyter notebook notebooks/01_explore_daasity_schemas.ipynb
jupyter notebook notebooks/02_model_training_benchmarks.ipynb
```

---

## Notes

- These files are gitignored (`data/raw/`, `data/features/*.csv`) — do **not** force-commit them.
- The data covers Miracle Sheets Shopify orders from Jan 2024 to Mar 2026.
- Treat the data as confidential — do not share outside the team.
