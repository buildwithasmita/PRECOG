from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.services.data_generator import generate_customer_data
from app.services.feature_engineering import FeatureEngineer
from app.services.ml_model import KYCResponsePredictor
from app.services.segmentation import CustomerSegmentation

RAW_DATA_PATH = Path("data/raw/customers.csv")
MODEL_PATH = "data/models/kyc_model.joblib"
DASHBOARD_DIR = Path("data/dashboard")


def _ensure_data_ready() -> pd.DataFrame:
    """Load customer data and generate synthetic data if missing."""
    if not RAW_DATA_PATH.exists():
        return generate_customer_data(10000)
    return pd.read_csv(RAW_DATA_PATH)


def _build_prediction_frame() -> pd.DataFrame:
    """Create customer-level predictions with segment recommendations."""
    raw_df = _ensure_data_ready()

    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(raw_df)

    if not engineer.validate_features(features_df):
        raise ValueError("Feature validation failed; cannot export dashboard data")

    X = features_df.drop(columns=["responded_to_last_refresh"])

    model = KYCResponsePredictor(model_path=MODEL_PATH)
    model.load_model(MODEL_PATH)
    probabilities = model.predict_proba(X)

    segmentation = CustomerSegmentation()
    segmented_df = segmentation.segment_customers(
        customer_ids=raw_df["customer_id"].astype(str).tolist(),
        probabilities=probabilities,
    )

    prediction_df = segmented_df.merge(
        raw_df[
            [
                "customer_id",
                "days_until_due",
                "risk_category",
                "customer_segment",
                "transaction_count_30d",
                "app_logins_30d",
            ]
        ],
        on="customer_id",
        how="left",
    )

    prediction_df["recommended_channels"] = prediction_df["recommended_channels"].apply(
        lambda channels: ", ".join(channels)
    )
    prediction_df["response_probability"] = prediction_df["response_probability"].round(
        2
    )
    prediction_df["outreach_cost"] = prediction_df["outreach_cost"].round(2)

    return prediction_df


def export_predictions_for_dashboard() -> dict[str, str]:
    """Export predictions, segment summary, and cost analysis CSVs for dashboarding."""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    prediction_df = _build_prediction_frame()

    predictions_export = prediction_df[
        [
            "customer_id",
            "response_probability",
            "segment",
            "recommended_channels",
            "outreach_cost",
            "days_until_due",
            "risk_category",
            "customer_segment",
            "transaction_count_30d",
            "app_logins_30d",
        ]
    ].copy()
    predictions_path = DASHBOARD_DIR / "predictions.csv"
    predictions_export.to_csv(predictions_path, index=False)

    segment_summary = (
        prediction_df.groupby("segment", as_index=False)
        .agg(
            customer_count=("customer_id", "count"),
            avg_response_probability=("response_probability", "mean"),
            total_outreach_cost=("outreach_cost", "sum"),
        )
        .copy()
    )

    channel_map = {
        "High": "email",
        "Medium": "email, sms",
        "Low": "email, sms, call",
    }
    segment_summary["expected_response_rate"] = (
        segment_summary["avg_response_probability"] * 100
    ).round(2)
    segment_summary["avg_response_probability"] = segment_summary[
        "avg_response_probability"
    ].round(2)
    segment_summary["total_outreach_cost"] = segment_summary[
        "total_outreach_cost"
    ].round(2)
    segment_summary["recommended_channels"] = segment_summary["segment"].map(
        channel_map
    )

    ordered_segments = pd.Categorical(
        segment_summary["segment"], categories=["High", "Medium", "Low"], ordered=True
    )
    segment_summary = segment_summary.assign(segment=ordered_segments).sort_values(
        "segment"
    )
    segment_summary["segment"] = segment_summary["segment"].astype(str)

    segment_summary_path = DASHBOARD_DIR / "segment_summary.csv"
    segment_summary.to_csv(segment_summary_path, index=False)

    segmentation = CustomerSegmentation()
    roi = segmentation.calculate_roi(prediction_df, total_customers=len(prediction_df))
    cost_analysis = pd.DataFrame(
        {
            "metric_name": ["Baseline Cost", "Optimized Cost", "Savings", "Savings %"],
            "value": [
                round(roi["baseline_cost"], 2),
                round(roi["optimized_cost"], 2),
                round(roi["savings"], 2),
                round(roi["savings_percentage"], 2),
            ],
        }
    )
    cost_analysis_path = DASHBOARD_DIR / "cost_analysis.csv"
    cost_analysis.to_csv(cost_analysis_path, index=False)

    return {
        "predictions": str(predictions_path),
        "segment_summary": str(segment_summary_path),
        "cost_analysis": str(cost_analysis_path),
    }


def export_feature_importance() -> str:
    """Export ranked model feature importance for dashboard consumption."""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    model = KYCResponsePredictor(model_path=MODEL_PATH)
    model.load_model(MODEL_PATH)

    importance = model.get_feature_importance().copy()
    importance["importance_score"] = importance["importance_score"].round(2)
    importance["rank"] = range(1, len(importance) + 1)

    output_path = DASHBOARD_DIR / "feature_importance.csv"
    importance[["feature_name", "importance_score", "rank"]].to_csv(
        output_path, index=False
    )
    return str(output_path)


def export_compliance_metrics() -> str:
    """Export high-level compliance metrics for dashboard summary cards."""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    df = _ensure_data_ready()
    total_customers = len(df)
    due_in_30_days = int(
        ((df["days_until_due"] >= 0) & (df["days_until_due"] <= 30)).sum()
    )
    overdue = int((df["days_until_due"] < 0).sum())
    compliance_pct = (
        round(((total_customers - overdue) / total_customers) * 100, 2)
        if total_customers
        else 0.0
    )

    compliance_df = pd.DataFrame(
        {
            "metric_name": [
                "Total Customers",
                "Due in 30 days",
                "Overdue",
                "Compliance %",
            ],
            "value": [
                float(total_customers),
                float(due_in_30_days),
                float(overdue),
                compliance_pct,
            ],
        }
    )

    output_path = DASHBOARD_DIR / "compliance.csv"
    compliance_df.to_csv(output_path, index=False)
    return str(output_path)


def generate_all_dashboard_exports() -> dict[str, str]:
    """Generate all dashboard export CSV files and return created paths."""
    prediction_files = export_predictions_for_dashboard()
    feature_path = export_feature_importance()
    compliance_path = export_compliance_metrics()

    files = {
        **prediction_files,
        "feature_importance": feature_path,
        "compliance": compliance_path,
    }

    print("Dashboard export summary:")
    for name, path in files.items():
        print(f"  {name}: {path}")

    return files


if __name__ == "__main__":
    files = generate_all_dashboard_exports()
    print("Dashboard data exported!")
    for name, path in files.items():
        print(f"  {name}: {path}")
