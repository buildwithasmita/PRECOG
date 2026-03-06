from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.services.feature_engineering import FeatureEngineer
from app.services.ml_model import KYCResponsePredictor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def train_pipeline(data_path: str = "data/raw/customers.csv") -> dict[str, float]:
    """Run full training pipeline from raw data to saved model."""
    csv_path = Path(data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data not found at: {csv_path}")

    logger.info("Loading raw data from %s", csv_path)
    raw_df = pd.read_csv(csv_path)

    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(raw_df)

    if not engineer.validate_features(features_df):
        raise ValueError(
            "Feature validation failed. Fix feature engineering issues before training."
        )

    X = features_df.drop(columns=["responded_to_last_refresh"])
    y = features_df["responded_to_last_refresh"].astype(int)

    predictor = KYCResponsePredictor(model_path="data/models/kyc_model.joblib")
    metrics = predictor.train(X, y)

    logger.info("Training metrics: %s", metrics)
    print("Training metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nTop 10 features by importance:")
    print(predictor.get_feature_importance().head(10).to_string(index=False))

    return metrics


if __name__ == "__main__":
    train_pipeline()
