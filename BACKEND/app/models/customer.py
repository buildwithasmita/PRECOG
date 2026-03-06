from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class CustomerInput(BaseModel):
    """Input payload for single-customer response prediction."""

    days_since_last_kyc: int = Field(
        ..., ge=0, le=5000, description="Days since the last KYC refresh"
    )
    days_until_due: int = Field(
        ..., ge=-3650, le=3650, description="Days until KYC refresh due date"
    )
    previous_refresh_count: int = Field(
        ..., ge=0, le=20, description="Number of previous KYC refreshes"
    )
    previous_response_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Historical KYC response rate"
    )
    account_age_months: int = Field(
        ..., ge=1, le=600, description="Customer account age in months"
    )
    card_active: bool = Field(..., description="Whether card is currently active")
    transaction_count_30d: int = Field(
        ..., ge=0, le=10000, description="Transactions in the last 30 days"
    )
    transaction_count_90d: int = Field(
        ..., ge=0, le=30000, description="Transactions in the last 90 days"
    )
    avg_transaction_amount: float = Field(
        ..., ge=0.0, le=1_000_000.0, description="Average transaction amount"
    )
    app_logins_30d: int = Field(
        ..., ge=0, le=500, description="App logins in the last 30 days"
    )
    email_open_rate: float = Field(..., ge=0.0, le=1.0, description="Email open rate")
    sms_response_rate: float = Field(
        ..., ge=0.0, le=1.0, description="SMS response rate"
    )
    risk_category: str = Field(..., description="Risk category: Low, Medium, or High")
    customer_segment: str = Field(
        ..., description="Customer segment: Basic, Gold, or Platinum"
    )
    contact_preference: str = Field(
        ..., description="Preferred channel: email, sms, or call"
    )

    @validator("risk_category")
    def validate_risk_category(cls, value: str) -> str:
        normalized = value.strip().title()
        allowed = {"Low", "Medium", "High"}
        if normalized not in allowed:
            raise ValueError("risk_category must be one of: Low, Medium, High")
        return normalized

    @validator("customer_segment")
    def validate_customer_segment(cls, value: str) -> str:
        normalized = value.strip().title()
        allowed = {"Basic", "Gold", "Platinum"}
        if normalized not in allowed:
            raise ValueError("customer_segment must be one of: Basic, Gold, Platinum")
        return normalized

    @validator("contact_preference")
    def validate_contact_preference(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"email", "sms", "call"}
        if normalized not in allowed:
            raise ValueError("contact_preference must be one of: email, sms, call")
        return normalized

    class Config:
        json_schema_extra = {
            "example": {
                "days_since_last_kyc": 825,
                "days_until_due": -95,
                "previous_refresh_count": 3,
                "previous_response_rate": 0.76,
                "account_age_months": 64,
                "card_active": True,
                "transaction_count_30d": 24,
                "transaction_count_90d": 69,
                "avg_transaction_amount": 14500.5,
                "app_logins_30d": 14,
                "email_open_rate": 0.63,
                "sms_response_rate": 0.48,
                "risk_category": "Medium",
                "customer_segment": "Gold",
                "contact_preference": "email",
            }
        }


class BatchPredictionRequest(BaseModel):
    """Payload for batch prediction requests."""

    customer_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of customer IDs. If omitted, predict for all customers.",
    )
    segment_filter: Optional[str] = Field(
        default=None,
        description="Optional filter for predicted segment: High, Medium, or Low.",
    )

    @validator("customer_ids")
    def validate_customer_ids(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is not None and len(value) == 0:
            raise ValueError("customer_ids cannot be an empty list")
        return value

    @validator("segment_filter")
    def validate_segment_filter(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.strip().title()
        if normalized not in {"High", "Medium", "Low"}:
            raise ValueError("segment_filter must be one of: High, Medium, Low")
        return normalized


class PredictionResponse(BaseModel):
    """Single-customer prediction response payload."""

    customer_id: str = Field(..., description="Unique customer identifier")
    response_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted response probability"
    )
    predicted_segment: str = Field(
        ..., description="Predicted segment: High, Medium, or Low"
    )
    recommended_channels: List[str] = Field(
        ..., description="Recommended outreach channels"
    )
    outreach_cost: float = Field(
        ..., ge=0.0, description="Estimated outreach cost per customer"
    )
    confidence_level: str = Field(..., description="Prediction confidence level")

    @validator("response_probability")
    def round_probability(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("response_probability must be between 0 and 1")
        return round(float(value), 4)

    @validator("predicted_segment")
    def validate_predicted_segment(cls, value: str) -> str:
        normalized = value.strip().title()
        if normalized not in {"High", "Medium", "Low"}:
            raise ValueError("predicted_segment must be one of: High, Medium, Low")
        return normalized

    @validator("recommended_channels")
    def validate_recommended_channels(cls, value: List[str]) -> List[str]:
        allowed = {"email", "sms", "call"}
        normalized = [channel.strip().lower() for channel in value]
        if not normalized:
            raise ValueError("recommended_channels cannot be empty")
        if any(channel not in allowed for channel in normalized):
            raise ValueError("recommended_channels can only include: email, sms, call")
        return normalized

    @validator("confidence_level")
    def validate_confidence_level(cls, value: str) -> str:
        normalized = value.strip().title()
        if normalized not in {"High", "Medium"}:
            raise ValueError("confidence_level must be High or Medium")
        return normalized


class BatchPredictionResponse(BaseModel):
    """Batch response containing customer-level predictions and summary stats."""

    total_customers: int = Field(
        ..., ge=0, description="Total number of customers scored"
    )
    predictions: List[PredictionResponse] = Field(
        ..., description="Predictions for each customer"
    )
    summary: Dict[str, float] = Field(
        ...,
        description="Aggregate metrics: high_count, medium_count, low_count, baseline_cost, optimized_cost, savings",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response generation timestamp"
    )


class DashboardData(BaseModel):
    """Payload for Tableau/PowerBI dashboard consumption."""

    segment_distribution: Dict[str, int] = Field(
        ..., description="Count of customers by segment"
    )
    cost_analysis: Dict[str, float] = Field(
        ..., description="Baseline/optimized/savings cost breakdown"
    )
    top_features: List[Dict[str, float]] = Field(
        ..., description="Top model features and their scores"
    )
    compliance_metrics: Dict[str, float] = Field(
        ..., description="Compliance-oriented KPI bundle"
    )
