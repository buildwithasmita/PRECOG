from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Transform raw customer records into model-ready features for KYC response prediction."""

    def __init__(self) -> None:
        self._feature_names: List[str] = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features and keep target column for model workflows."""
        logger.info("Starting feature engineering")
        features_df = df.copy()

        # 1) Time-based features: capture urgency and whether KYC is overdue.
        # Guard against days_until_due == -1, which would make denominator zero.
        urgency_denominator = (features_df["days_until_due"] + 1).replace(0, 1)
        features_df["kyc_urgency_score"] = 1.0 / urgency_denominator
        features_df["kyc_overdue"] = (features_df["days_until_due"] < 0).astype(int)
        features_df["last_kyc_age_category"] = np.where(
            features_df["days_since_last_kyc"] < 365,
            "Recent",
            "Old",
        )

        # 2) Composite engagement score: weighted blend of digital interaction signals.
        features_df["engagement_score"] = (
            (features_df["app_logins_30d"] / 30.0) * 0.4
            + features_df["email_open_rate"] * 0.3
            + features_df["sms_response_rate"] * 0.3
        ) * 100.0

        # 3) Transaction activity signals: pace, recency ratio, and high-value flag.
        features_df["transaction_velocity"] = (
            features_df["transaction_count_30d"] / 30.0
        )
        features_df["transaction_recency"] = features_df["transaction_count_30d"] / (
            features_df["transaction_count_90d"] + 1
        )
        features_df["high_value_customer"] = (
            features_df["avg_transaction_amount"] > 10000
        ).astype(int)

        # 4) Historical behavior: loyalty flag and weighted response history trend.
        features_df["loyal_customer"] = (
            (features_df["account_age_months"] > 24)
            & (features_df["previous_response_rate"] > 0.7)
        ).astype(int)
        features_df["response_trend"] = (
            features_df["previous_response_rate"]
            * features_df["previous_refresh_count"]
        )

        # 5) Risk-segment interaction captures joint risk/profile behavior patterns.
        features_df["risk_segment_combo"] = (
            features_df["risk_category"].astype(str)
            + "_"
            + features_df["customer_segment"].astype(str)
        )

        # 6) Encode categoricals with fixed business mappings.
        risk_map = {"Low": 0, "Medium": 1, "High": 2}
        segment_map = {"Basic": 0, "Gold": 1, "Platinum": 2}
        age_map = {"Recent": 0, "Old": 1}

        features_df["risk_category"] = (
            features_df["risk_category"].map(risk_map).fillna(-1).astype(int)
        )
        features_df["customer_segment"] = (
            features_df["customer_segment"].map(segment_map).fillna(-1).astype(int)
        )
        features_df["last_kyc_age_category"] = (
            features_df["last_kyc_age_category"].map(age_map).fillna(-1).astype(int)
        )
        features_df["card_active"] = features_df["card_active"].astype(int)

        contact_ohe = pd.get_dummies(
            features_df["contact_preference"],
            prefix="contact_pref",
            dtype=int,
        )
        expected_contact_cols = [
            "contact_pref_email",
            "contact_pref_sms",
            "contact_pref_call",
        ]
        for col in expected_contact_cols:
            if col not in contact_ohe.columns:
                contact_ohe[col] = 0
        contact_ohe = contact_ohe[expected_contact_cols]
        features_df = pd.concat([features_df, contact_ohe], axis=1)

        # Encode interaction feature to numeric code for gradient boosting compatibility.
        features_df["risk_segment_combo"] = pd.Categorical(
            features_df["risk_segment_combo"]
        ).codes

        # 7) Drop non-feature IDs/dates while keeping target.
        features_df = features_df.drop(
            columns=[
                "customer_id",
                "last_kyc_date",
                "kyc_refresh_due_date",
                "contact_preference",
            ]
        )

        self._feature_names = [
            col for col in features_df.columns if col != "responded_to_last_refresh"
        ]

        logger.info(
            "Feature engineering complete with %d features", len(self._feature_names)
        )
        return features_df

    def get_feature_names(self) -> List[str]:
        """Return final model feature columns (excluding target)."""
        return self._feature_names.copy()

    def validate_features(self, df: pd.DataFrame) -> bool:
        """Validate engineered data quality and expected feature ranges."""
        logger.info("Validating engineered features")

        if df.isnull().any().any():
            logger.error("Validation failed: null values detected")
            return False

        numeric_df = df.select_dtypes(include=[np.number])
        if np.isinf(numeric_df.to_numpy()).any():
            logger.error("Validation failed: infinite values detected")
            return False

        range_checks = [
            ("engagement_score", df["engagement_score"].between(0, 100).all()),
            ("kyc_overdue", df["kyc_overdue"].isin([0, 1]).all()),
            ("high_value_customer", df["high_value_customer"].isin([0, 1]).all()),
            ("loyal_customer", df["loyal_customer"].isin([0, 1]).all()),
            ("risk_category", df["risk_category"].isin([0, 1, 2]).all()),
            ("customer_segment", df["customer_segment"].isin([0, 1, 2]).all()),
            ("card_active", df["card_active"].isin([0, 1]).all()),
            (
                "previous_response_rate",
                df["previous_response_rate"].between(0, 1).all(),
            ),
            ("email_open_rate", df["email_open_rate"].between(0, 1).all()),
            ("sms_response_rate", df["sms_response_rate"].between(0, 1).all()),
        ]

        for feature_name, passed in range_checks:
            if not passed:
                logger.error(
                    "Validation failed: out-of-range values in '%s'", feature_name
                )
                return False

        logger.info("Feature validation passed")
        return True
