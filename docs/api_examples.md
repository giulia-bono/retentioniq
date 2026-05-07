# RetentionIQ API — examples for F1

**Stub version: 0.1.0-stub.** Real models land when M1/M2 hand off `.pkl` files; the response shapes won't change.

## Run the server

```bash
# from repo root
source .venv/bin/activate
uvicorn src.api.main:app --reload --port 8000
```

OpenAPI docs: http://localhost:8000/docs
OpenAPI schema: http://localhost:8000/openapi.json

---

## `GET /health`

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{"status": "ok"}
```

---

## `POST /predict/clv`  *(Giulia / D1)*

```bash
curl -s -X POST http://127.0.0.1:8000/predict/clv \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "000002094a200abd84e6cd40a850edd5", "horizon_days": 365}'
```

```json
{
  "customer_id": "000002094a200abd84e6cd40a850edd5",
  "predicted_clv": 1431.0,
  "horizon_days": 365
}
```

Notes:
- `horizon_days` is optional, defaults to **365**, accepts 1–3650.
- Stub output is deterministic per `customer_id` (md5 hash → seed). Same ID → same number every call. Useful for snapshot tests in the frontend.
- Real model: BG/NBD purchase frequency × Gamma-Gamma monetary value, projected over `horizon_days`.

---

## `POST /predict/churn`  *(Juan / D2)*

```bash
curl -s -X POST http://127.0.0.1:8000/predict/churn \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "000002094a200abd84e6cd40a850edd5"}'
```

```json
{
  "customer_id": "000002094a200abd84e6cd40a850edd5",
  "churn_probability": 0.8381,
  "top_drivers": [
    {"feature": "recency_days",    "shap_value":  0.48},
    {"feature": "frequency",       "shap_value":  0.96},
    {"feature": "avg_order_value", "shap_value":  0.62}
  ]
}
```

Notes:
- `churn_probability` ∈ [0, 1].
- `top_drivers` always returns 3 in the stub. Real model returns top-N by absolute SHAP value (N TBD by M2). Frontend should not hard-code 3.
- SHAP convention: **positive = pushes toward churn**, negative = pulls away. Confirm with M2 before publishing customer-facing copy.

---

## Error shapes (FastAPI defaults)

Validation failures return `422` with:

```json
{
  "detail": [
    {"type": "missing", "loc": ["body", "customer_id"], "msg": "Field required"}
  ]
}
```

`horizon_days` outside 1–3650 also returns `422`.

---

## Quick Python client (for the F1 frontend / scripts)

```python
import requests
r = requests.post("http://localhost:8000/predict/churn", json={"customer_id": "abc123"})
r.raise_for_status()
print(r.json())
```
