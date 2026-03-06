from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.feature_engineering import FeatureEngineer
from app.services.ml_model import KYCResponsePredictor
from app.services.segmentation import CustomerSegmentation


def _build_mock_customers(n_rows: int) -> pd.DataFrame:
    """Create deterministic mock raw customer records for service tests."""
    idx = np.arange(n_rows)
    risk_values = np.array(["Low", "Medium", "High"])
    segment_values = np.array(["Basic", "Gold", "Platinum"])
    contact_values = np.array(["email", "sms", "call"])

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i+1}" for i in idx],
            "last_kyc_date": ["2024-01-01"] * n_rows,
            "days_since_last_kyc": 365 + (idx % 730),
            "kyc_refresh_due_date": ["2026-01-01"] * n_rows,
            "days_until_due": -200 + (idx % 230),
            "previous_refresh_count": idx % 6,
            "previous_response_rate": np.clip(0.2 + (idx % 10) * 0.08, 0.0, 1.0),
            "account_age_months": 12 + (idx % 109),
            "card_active": (idx % 5 != 0),
            "transaction_count_30d": idx % 101,
            "transaction_count_90d": idx % 301,
            "avg_transaction_amount": 1000 + (idx % 50) * 800,
            "app_logins_30d": idx % 31,
            "email_open_rate": np.clip(0.1 + (idx % 10) * 0.08, 0.0, 1.0),
            "sms_response_rate": np.clip(0.05 + (idx % 10) * 0.07, 0.0, 1.0),
            "risk_category": risk_values[idx % 3],
            "customer_segment": segment_values[idx % 3],
            "contact_preference": contact_values[idx % 3],
            "responded_to_last_refresh": (idx % 2 == 0),
        }
    )
    return df


@pytest.fixture
def sample_customers() -> pd.DataFrame:
    """Generate sample customer data with all required raw fields."""
    return _build_mock_customers(100)


class TestFeatureEngineering:
    def test_engineer_features_basic(self, sample_customers: pd.DataFrame) -> None:
        engineer = FeatureEngineer()
        features_df = engineer.engineer_features(sample_customers)

        expected_columns = {
            "kyc_urgency_score",
            "kyc_overdue",
            "last_kyc_age_category",
            "engagement_score",
            "transaction_velocity",
            "transaction_recency",
            "high_value_customer",
            "loyal_customer",
            "response_trend",
            "risk_segment_combo",
            "contact_pref_email",
            "contact_pref_sms",
            "contact_pref_call",
            "responded_to_last_refresh",
        }

        missing = expected_columns - set(features_df.columns)
        assert not missing, f"Missing engineered columns: {sorted(missing)}"
        assert not features_df.isnull().any().any(), "Engineered features contain null values"
        assert features_df["engagement_score"].between(0, 100).all(), "engagement_score should be in [0, 100]"

    def test_validate_features(self, sample_customers: pd.DataFrame) -> None:
        engineer = FeatureEngineer()
        features_df = engineer.engineer_features(sample_customers)
        assert engineer.validate_features(features_df) is True, "validate_features should return True for valid data"

    def test_handle_edge_cases(self) -> None:
        edge_df = _build_mock_customers(1)
        edge_df.loc[0, "transaction_count_30d"] = 0
        edge_df.loc[0, "transaction_count_90d"] = 0
        edge_df.loc[0, "days_until_due"] = 0

        engineer = FeatureEngineer()
        features_df = engineer.engineer_features(edge_df)

        numeric = features_df.select_dtypes(include=[np.number]).to_numpy()
        assert np.isfinite(numeric).all(), "Edge-case features should not contain inf/-inf"


class TestSegmentation:
    def test_segment_customers(self) -> None:
        seg = CustomerSegmentation()
        ids = ["C1", "C2", "C3"]
        probs = np.array([0.8, 0.5, 0.2])

        result = seg.segment_customers(ids, probs)

        assert result["segment"].tolist() == ["High", "Medium", "Low"], "Segment assignment is incorrect"
        assert result["outreach_cost"].tolist() == [5.0, 15.0, 50.0], "Outreach cost assignment is incorrect"

    def test_calculate_roi(self) -> None:
        seg = CustomerSegmentation()
        segmented_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "response_probability": [0.85, 0.75, 0.55],
                "segment": ["High", "High", "Medium"],
                "recommended_channels": [["email"], ["email"], ["email", "sms"]],
                "outreach_cost": [5.0, 5.0, 15.0],
            }
        )

        roi = seg.calculate_roi(segmented_df, total_customers=3)

        assert roi["baseline_cost"] == 45.0, "Baseline cost should be total_customers * 15"
        assert roi["optimized_cost"] == 25.0, "Optimized cost should equal sum of outreach_cost"
        assert roi["savings"] == 20.0, "Savings should be baseline - optimized"
        assert roi["baseline_cost"] > roi["optimized_cost"], "Baseline should be greater than optimized cost"

    def test_get_segment_summary(self) -> None:
        seg = CustomerSegmentation()
        segmented_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3", "C4"],
                "response_probability": [0.9, 0.6, 0.2, 0.75],
                "segment": ["High", "Medium", "Low", "High"],
                "recommended_channels": [
                    ["email"],
                    ["email", "sms"],
                    ["email", "sms", "call"],
                    ["email"],
                ],
                "outreach_cost": [5.0, 15.0, 50.0, 5.0],
            }
        )

        summary = seg.get_segment_summary(segmented_df)

        for key in ["High", "Medium", "Low", "Overall"]:
            assert key in summary, f"Missing segment summary key: {key}"

        for segment_key, metrics in summary.items():
            for metric_name, metric_value in metrics.items():
                assert isinstance(metric_value, (int, float)), (
                    f"Summary metric '{segment_key}.{metric_name}' should be numeric"
                )


class TestMLModel:
    def test_model_loading(self) -> None:
        model_path = Path("data/models/kyc_model.joblib")
        assert model_path.exists(), "Expected trained model file does not exist at data/models/kyc_model.joblib"

        model = KYCResponsePredictor(model_path=str(model_path))
        model.load_model(str(model_path))
        assert model.model is not None, "Model should load successfully"

    def test_predict_proba(self) -> None:
        model_path = Path("data/models/kyc_model.joblib")
        assert model_path.exists(), "Model file missing; train model before running this test"

        model = KYCResponsePredictor(model_path=str(model_path))
        model.load_model(str(model_path))

        raw_df = _build_mock_customers(10)
        engineer = FeatureEngineer()
        features_df = engineer.engineer_features(raw_df)
        X = features_df.drop(columns=["responded_to_last_refresh"])

        probs = model.predict_proba(X)

        assert probs.shape == (10,), f"Expected probability shape (10,), got {probs.shape}"
        assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be in [0, 1]"
