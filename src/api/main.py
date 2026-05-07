"""RetentionIQ FastAPI service — stub endpoints.

Returns deterministic mock data keyed off ``customer_id`` so F1 sees stable
values during frontend development. Real model loading lands when M1/M2
hand off ``.pkl`` files.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Endpoints:
    GET  /health
    POST /predict/clv       (Giulia / D1)
    POST /predict/churn     (Juan / D2)
"""
from __future__ import annotations

import hashlib
from typing import List

from fastapi import FastAPI

from src.api.schemas import (
    CLVRequest,
    CLVResponse,
    ChurnDriver,
    ChurnRequest,
    ChurnResponse,
    HealthResponse,
)

app = FastAPI(
    title="RetentionIQ API",
    version="0.1.0-stub",
    description="Predictive customer-intelligence API for Shopify brands.",
)


def _seed(customer_id: str) -> int:
    """Stable integer derived from customer_id for mock generation."""
    return int(hashlib.md5(customer_id.encode()).hexdigest(), 16)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict/clv", response_model=CLVResponse)
def predict_clv(req: CLVRequest) -> CLVResponse:
    """STUB — replace with BG/NBD + Gamma-Gamma inference once M1 ships ``.pkl``."""
    seed = _seed(req.customer_id)
    base = 50 + (seed % 1500)
    horizon_factor = req.horizon_days / 365
    predicted = round(base * horizon_factor, 2)
    return CLVResponse(
        customer_id=req.customer_id,
        predicted_clv=predicted,
        horizon_days=req.horizon_days,
    )


@app.post("/predict/churn", response_model=ChurnResponse)
def predict_churn(req: ChurnRequest) -> ChurnResponse:
    """STUB — replace with XGBoost + SHAP inference once M2 ships ``.pkl``."""
    seed = _seed(req.customer_id)
    prob = round(((seed % 10000) / 10000), 4)

    drivers: List[ChurnDriver] = [
        ChurnDriver(feature="recency_days", shap_value=round(((seed >> 4) % 200) / 100 - 1, 3)),
        ChurnDriver(feature="frequency", shap_value=round(((seed >> 8) % 200) / 100 - 1, 3)),
        ChurnDriver(feature="avg_order_value", shap_value=round(((seed >> 12) % 200) / 100 - 1, 3)),
    ]
    return ChurnResponse(
        customer_id=req.customer_id,
        churn_probability=prob,
        top_drivers=drivers,
    )
