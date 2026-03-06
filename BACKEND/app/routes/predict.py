from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.models.customer import (BatchPredictionRequest,
                                 BatchPredictionResponse, CustomerInput,
                                 DashboardData, PredictionResponse)
from app.services.feature_engineering import FeatureEngineer
from app.services.ml_model import KYCResponsePredictor
from app.services.segmentation import CustomerSegmentation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Predictions"])

DATA_PATH = Path("data/raw/customers.csv")
MODEL_PATH = "data/models/kyc_model.joblib"


def get_feature_engineer() -> FeatureEngineer:
    return FeatureEngineer()


def get_ml_model() -> KYCResponsePredictor:
    model = KYCResponsePredictor(model_path=MODEL_PATH)
    model.load_model(MODEL_PATH)
    return model


def get_segmentation() -> CustomerSegmentation:
    return CustomerSegmentation()


def _load_customer_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Customer data file not found at: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def _to_prediction_response(row: pd.Series) -> PredictionResponse:
    prob = float(row["response_probability"])
    confidence = "High" if prob > 0.8 or prob < 0.2 else "Medium"
    return PredictionResponse(
        customer_id=str(row["customer_id"]),
        response_probability=prob,
        predicted_segment=str(row["segment"]),
        recommended_channels=list(row["recommended_channels"]),
        outreach_cost=float(row["outreach_cost"]),
        confidence_level=confidence,
    )


def _run_prediction_pipeline(
    raw_df: pd.DataFrame,
    feature_engineer: FeatureEngineer,
    model: KYCResponsePredictor,
    segmentation: CustomerSegmentation,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    features_df = feature_engineer.engineer_features(raw_df)

    if not feature_engineer.validate_features(features_df):
        raise ValueError("Feature validation failed")

    X = features_df.drop(columns=["responded_to_last_refresh"])
    probabilities = model.predict_proba(X)

    segmented_df = segmentation.segment_customers(
        customer_ids=raw_df["customer_id"].astype(str).tolist(),
        probabilities=probabilities,
    )

    roi = segmentation.calculate_roi(segmented_df, total_customers=len(raw_df))
    return segmented_df, roi, features_df


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer),
    model: KYCResponsePredictor = Depends(get_ml_model),
    segmentation: CustomerSegmentation = Depends(get_segmentation),
) -> BatchPredictionResponse:
    logger.info("Batch prediction request received")

    try:
        raw_df = _load_customer_data()

        if request.customer_ids is not None:
            requested_ids = set(request.customer_ids)
            available_ids = set(raw_df["customer_id"].astype(str))
            missing_ids = sorted(requested_ids - available_ids)
            if missing_ids:
                raise HTTPException(
                    status_code=404,
                    detail=f"Invalid customer IDs: {missing_ids[:20]}",
                )
            raw_df = raw_df[
                raw_df["customer_id"].astype(str).isin(requested_ids)
            ].copy()

        segmented_df, roi, _ = _run_prediction_pipeline(
            raw_df, feature_engineer, model, segmentation
        )

        if request.segment_filter is not None:
            segmented_df = segmented_df[
                segmented_df["segment"] == request.segment_filter
            ].copy()

        predictions = [
            _to_prediction_response(row) for _, row in segmented_df.iterrows()
        ]

        summary = {
            "high_count": float((segmented_df["segment"] == "High").sum()),
            "medium_count": float((segmented_df["segment"] == "Medium").sum()),
            "low_count": float((segmented_df["segment"] == "Low").sum()),
            "baseline_cost": float(roi["baseline_cost"]),
            "optimized_cost": float(segmented_df["outreach_cost"].sum()),
            "savings": float(
                roi["baseline_cost"] - segmented_df["outreach_cost"].sum()
            ),
        }

        return BatchPredictionResponse(
            total_customers=len(segmented_df),
            predictions=predictions,
            summary=summary,
            timestamp=datetime.utcnow(),
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.exception("Data/model file missing")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        logger.exception("Feature engineering/prediction validation error")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during batch prediction")
        raise HTTPException(status_code=500, detail="Batch prediction failed") from exc


@router.post("/predict/single", response_model=PredictionResponse)
async def predict_single(
    customer: CustomerInput,
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer),
    model: KYCResponsePredictor = Depends(get_ml_model),
    segmentation: CustomerSegmentation = Depends(get_segmentation),
) -> PredictionResponse:
    logger.info("Single prediction request received")

    try:
        payload = customer.model_dump()
        payload.update(
            {
                "customer_id": "SINGLE_INPUT",
                "last_kyc_date": "2024-01-01",
                "kyc_refresh_due_date": "2026-01-01",
                "responded_to_last_refresh": False,
            }
        )
        raw_df = pd.DataFrame([payload])

        features_df = feature_engineer.engineer_features(raw_df)
        if not feature_engineer.validate_features(features_df):
            raise ValueError("Feature validation failed")

        X = features_df.drop(columns=["responded_to_last_refresh"])
        prob = float(model.predict_proba(X)[0])

        segmented_df = segmentation.segment_customers(
            ["SINGLE_INPUT"], np.array([prob])
        )
        return _to_prediction_response(segmented_df.iloc[0])
    except FileNotFoundError as exc:
        logger.exception("Model file missing for single prediction")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        logger.exception("Invalid single prediction input/features")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during single prediction")
        raise HTTPException(status_code=500, detail="Single prediction failed") from exc


@router.get("/analytics/dashboard-data", response_model=DashboardData)
async def get_dashboard_data(
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer),
    model: KYCResponsePredictor = Depends(get_ml_model),
    segmentation: CustomerSegmentation = Depends(get_segmentation),
) -> DashboardData:
    logger.info("Dashboard analytics request received")

    try:
        raw_df = _load_customer_data()
        segmented_df, roi, _ = _run_prediction_pipeline(
            raw_df, feature_engineer, model, segmentation
        )

        segment_distribution = {
            "High": int((segmented_df["segment"] == "High").sum()),
            "Medium": int((segmented_df["segment"] == "Medium").sum()),
            "Low": int((segmented_df["segment"] == "Low").sum()),
        }

        feature_importance = model.get_feature_importance().head(10)
        top_features = feature_importance.to_dict(orient="records")

        due_soon_pct = float(
            ((raw_df["days_until_due"] >= 0) & (raw_df["days_until_due"] <= 30)).mean()
            * 100
        )
        overdue_pct = float((raw_df["days_until_due"] < 0).mean() * 100)

        compliance_metrics = {
            "due_soon_percentage": due_soon_pct,
            "overdue_percentage": overdue_pct,
            "expected_response_rate": float(
                segmented_df["response_probability"].mean() * 100
            ),
        }

        return DashboardData(
            segment_distribution=segment_distribution,
            cost_analysis=roi,
            top_features=top_features,
            compliance_metrics=compliance_metrics,
        )
    except FileNotFoundError as exc:
        logger.exception("Data/model file missing for dashboard data")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during dashboard analytics")
        raise HTTPException(
            status_code=500, detail="Dashboard analytics failed"
        ) from exc


@router.get(
    "/analytics/segment/{segment_name}", response_model=List[PredictionResponse]
)
async def get_segment_customers(
    segment_name: str,
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer),
    model: KYCResponsePredictor = Depends(get_ml_model),
    segmentation: CustomerSegmentation = Depends(get_segmentation),
) -> List[PredictionResponse]:
    logger.info("Segment analytics request received for segment=%s", segment_name)

    normalized_segment = segment_name.strip().title()
    if normalized_segment not in {"High", "Medium", "Low"}:
        raise HTTPException(
            status_code=400, detail="segment_name must be High, Medium, or Low"
        )

    try:
        raw_df = _load_customer_data()
        segmented_df, _, _ = _run_prediction_pipeline(
            raw_df, feature_engineer, model, segmentation
        )

        filtered = segmented_df[segmented_df["segment"] == normalized_segment].copy()
        filtered = filtered.sort_values("response_probability", ascending=False).head(
            100
        )

        return [_to_prediction_response(row) for _, row in filtered.iterrows()]
    except FileNotFoundError as exc:
        logger.exception("Data/model file missing for segment analytics")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during segment analytics")
        raise HTTPException(status_code=500, detail="Segment analytics failed") from exc


@router.get("/health")
async def health_check() -> dict[str, bool | str]:
    model = KYCResponsePredictor(model_path=MODEL_PATH)
    model.load_model(MODEL_PATH)
    return {
        "status": "healthy",
        "model_loaded": model.model is not None,
    }
