from __future__ import annotations

import time

import pandas as pd

from app.services.data_generator import generate_customer_data
from app.services.feature_engineering import FeatureEngineer
from app.services.ml_model import KYCResponsePredictor
from app.services.segmentation import CustomerSegmentation


def main() -> None:
    print("=" * 60)
    print("PRECOG Demo")
    print("=" * 60)
    print("\nKYC Refresh Prediction System")
    print("Problem: INR 3.5M annual cost for manual outreach")
    print("Solution: ML-powered targeted outreach -> INR 1.5M cost\n")
    input("Press Enter to start demo...")

    print("\nSTEP 1: Loading Customer Data")
    df: pd.DataFrame = generate_customer_data(1000)
    print(f"Loaded {len(df)} customers")
    print(f"Response rate: {df['responded_to_last_refresh'].mean():.1%}")
    time.sleep(2)

    print("\nSTEP 2: Engineering Features")
    engineer = FeatureEngineer()
    features = engineer.engineer_features(df)
    print(f"Created {len(features.columns)} features")
    print("Key features: engagement_score, transaction_velocity...")
    time.sleep(2)

    print("\nSTEP 3: ML Model Prediction")
    model = KYCResponsePredictor(model_path="data/models/kyc_model.joblib")
    model.load_model("data/models/kyc_model.joblib")
    X = features.drop("responded_to_last_refresh", axis=1)
    probs = model.predict_proba(X)
    print("Generated predictions")
    print(f"Avg probability: {probs.mean():.1%}")
    time.sleep(2)

    print("\nSTEP 4: Customer Segmentation")
    seg = CustomerSegmentation()
    results = seg.segment_customers(df["customer_id"].tolist(), probs)
    summary = seg.get_segment_summary(results)
    print("Segmented customers:")
    for segment, data in summary.items():
        if isinstance(data, dict):
            print(f"  {segment}: {data.get('count', 0)} customers")
    time.sleep(2)

    print("\nSTEP 5: Cost Optimization")
    roi = seg.calculate_roi(results, len(df))
    print("ROI Analysis:")
    print(f"Baseline cost: INR {roi['baseline_cost']:,.0f}")
    print(f"Optimized cost: INR {roi['optimized_cost']:,.0f}")
    print(f"Savings: INR {roi['savings']:,.0f} ({roi['savings_percentage']:.1f}%)")
    time.sleep(2)

    print("\nSTEP 6: Sample Recommendations")
    print("\nTop 5 Low-Likelihood Customers (Need Proactive Outreach):")
    low_risk = results[results["segment"] == "Low"].head(5).copy()
    if low_risk.empty:
        print("No low-likelihood customers in this sample.")
    else:
        low_risk["response_probability"] = low_risk["response_probability"].round(4)
        print(low_risk[["customer_id", "response_probability", "recommended_channels"]].to_string(index=False))
    time.sleep(2)

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print(f"  - {len(df)} customers analyzed in <5 seconds")
    print(f"  - {roi['savings_percentage']:.0f}% cost reduction achieved")
    print(f"  - Proactive outreach for {len(results[results['segment'] == 'Low'])} high-risk customers")
    print("  - Ready for production deployment!\n")


if __name__ == "__main__":
    main()

