from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class KYCResponsePredictor:
    """Train and serve an XGBoost model for KYC refresh response prediction."""

    def __init__(self, model_path: str = "data/models/kyc_model.joblib") -> None:
        self.model_path = Path(model_path)
        self.model: XGBClassifier | None = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
        )
        self.feature_names: list[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Train model, evaluate on holdout split, save artifact, and return metrics."""
        logger.info("Starting model training with %d samples", len(X))

        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        if self.model is None:
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=42,
            )

        self.model.fit(X_train, y_train)
        logger.info("Model training complete")

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = self.evaluate_model(y_test, y_pred, y_prob)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, self.model_path)
        logger.info("Model saved to %s", self.model_path)

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities for input rows."""
        if self.model is None:
            logger.info("Model not in memory, attempting to load from disk")
            self.load_model(str(self.model_path))

        if self.model is None:
            msg = f"Model not trained or found at '{self.model_path}'. Train or load a model first."
            logger.error(msg)
            raise FileNotFoundError(msg)

        probs = self.model.predict_proba(X)[:, 1]
        return probs

    def load_model(self, model_path: str) -> None:
        """Load a model artifact from disk into memory."""
        path = Path(model_path)
        if not path.exists():
            logger.warning("Model file not found at %s", path)
            self.model = None
            return

        artifact: Any = joblib.load(path)

        if isinstance(artifact, dict) and "model" in artifact:
            self.model = artifact["model"]
            self.feature_names = artifact.get("feature_names", [])
        else:
            self.model = artifact
            self.feature_names = []

        self.model_path = path
        logger.info("Model loaded from %s", path)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return sorted feature importance from trained model."""
        if self.model is None:
            logger.info("Model not in memory, attempting to load from disk")
            self.load_model(str(self.model_path))

        if self.model is None:
            msg = "Model not trained or loaded; feature importance unavailable."
            logger.error(msg)
            raise ValueError(msg)

        if not hasattr(self.model, "feature_importances_"):
            msg = "Loaded model does not expose feature_importances_."
            logger.error(msg)
            raise ValueError(msg)

        if not self.feature_names:
            self.feature_names = [
                f"feature_{i}" for i in range(len(self.model.feature_importances_))
            ]

        importance_df = pd.DataFrame(
            {
                "feature_name": self.feature_names,
                "importance_score": self.model.feature_importances_,
            }
        ).sort_values("importance_score", ascending=False)

        importance_df = importance_df.reset_index(drop=True)
        return importance_df

    def evaluate_model(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        y_prob: pd.Series | np.ndarray,
    ) -> dict[str, float]:
        """Compute standard binary classification metrics."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
        }

        logger.info("Evaluation metrics: %s", metrics)
        return metrics
