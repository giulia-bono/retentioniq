"""Pydantic models for the RetentionIQ FastAPI service.

These contracts are the handoff to F1 (Eric + Davidson) — change them
deliberately; the frontend will be wired against these shapes.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class CLVRequest(BaseModel):
    customer_id: str = Field(..., description="Hashed Shopify customer ID")
    horizon_days: int = Field(365, ge=1, le=3650, description="Prediction horizon in days")


class CLVResponse(BaseModel):
    customer_id: str
    predicted_clv: float = Field(..., description="Predicted customer lifetime value in USD over horizon_days")
    horizon_days: int


class ChurnDriver(BaseModel):
    feature: str = Field(..., description="Feature name (matches feature_table.parquet column)")
    shap_value: float = Field(..., description="SHAP contribution; +ve pushes toward churn, -ve away from churn")


class ChurnRequest(BaseModel):
    customer_id: str


class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    top_drivers: List[ChurnDriver]


class HealthResponse(BaseModel):
    status: str = "ok"
