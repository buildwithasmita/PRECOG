from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def generate_customer_data(
    n_customers: int = 10000, random_seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic KYC refresh customer data with realistic response correlations."""
    np.random.seed(random_seed)

    today = pd.Timestamp(date.today())

    customer_ids = [f"CUST{i:06d}" for i in range(1, n_customers + 1)]

    # Last KYC date between 2-3 years ago (730 to 1095 days)
    days_ago = np.random.randint(730, 1096, size=n_customers)
    last_kyc_date = today - pd.to_timedelta(days_ago, unit="D")

    days_since_last_kyc = (today - last_kyc_date).days.astype(int)
    kyc_refresh_due_date = last_kyc_date + pd.to_timedelta(730, unit="D")
    days_until_due = (kyc_refresh_due_date - today).days.astype(int)

    previous_refresh_count = np.random.randint(0, 6, size=n_customers)
    account_age_months = np.random.randint(12, 121, size=n_customers)
    card_active = np.random.choice([True, False], size=n_customers, p=[0.80, 0.20])

    app_logins_30d = np.clip(np.random.poisson(lam=10, size=n_customers), 0, 30)

    # Email/SMS behavior uses beta distributions to keep values in [0, 1].
    email_open_rate = np.random.beta(a=2.3, b=2.0, size=n_customers)
    sms_response_rate = np.random.beta(a=1.8, b=2.5, size=n_customers)

    risk_category = np.random.choice(
        ["Low", "Medium", "High"], size=n_customers, p=[0.60, 0.30, 0.10]
    )
    customer_segment = np.random.choice(
        ["Basic", "Gold", "Platinum"], size=n_customers, p=[0.50, 0.30, 0.20]
    )
    contact_preference = np.random.choice(
        ["email", "sms", "call"], size=n_customers, p=[0.60, 0.25, 0.15]
    )

    # Transaction activity: inactive cards tend to have lower activity.
    tx30_base = np.random.normal(loc=20, scale=10, size=n_customers)
    tx30_penalty = np.where(
        card_active, 0, np.random.normal(loc=10, scale=4, size=n_customers)
    )
    transaction_count_30d = np.clip(np.round(tx30_base - tx30_penalty), 0, 100).astype(
        int
    )

    tx90_noise = np.random.normal(loc=0, scale=20, size=n_customers)
    transaction_count_90d = np.clip(
        np.round(transaction_count_30d * 3 + tx90_noise), 0, 300
    ).astype(int)

    # Log-normal amount distribution clipped to the business range.
    avg_transaction_amount = np.clip(
        np.random.lognormal(mean=8.9, sigma=0.8, size=n_customers), 1000, 50000
    )

    # Historical response rate is influenced by engagement and card activity.
    prev_rate_base = (
        0.25
        + 0.25 * (app_logins_30d / 30)
        + 0.25 * email_open_rate
        + 0.20 * sms_response_rate
        + 0.10 * card_active.astype(float)
    )
    previous_response_rate = np.clip(
        prev_rate_base + np.random.normal(0, 0.1, size=n_customers), 0.0, 1.0
    )

    # Build a latent score and convert to response probability.
    # A sharper signal with small label noise helps produce a trainable synthetic target.
    latent_score = (
        -0.35
        + 2.8 * (previous_response_rate - 0.5)
        + 1.7 * (app_logins_30d / 30.0 - 0.33)
        + 1.5 * (email_open_rate - 0.45)
        + 0.9 * (sms_response_rate - 0.40)
        + 0.7 * card_active.astype(float)
        + 0.5 * (transaction_count_30d / 100.0)
    )
    latent_score -= np.where(risk_category == "High", 1.2, 0.0)
    latent_score -= np.where(risk_category == "Medium", 0.5, 0.0)
    response_prob = 1.0 / (1.0 + np.exp(-latent_score))

    # Required business correlations for realistic ML patterns:
    # 1) High app usage + high email engagement -> ~80% response.
    high_engagement = (app_logins_30d >= 12) & (email_open_rate >= 0.60)
    response_prob = np.where(
        high_engagement, np.maximum(response_prob, 0.80), response_prob
    )

    # 2) Low transactions + inactive card -> ~20% response.
    low_activity_inactive = (transaction_count_30d <= 8) & (~card_active)
    response_prob = np.where(
        low_activity_inactive, np.minimum(response_prob, 0.20), response_prob
    )

    # 3) High historical response -> ~90% response.
    historically_responsive = previous_response_rate >= 0.80
    response_prob = np.where(
        historically_responsive, np.maximum(response_prob, 0.90), response_prob
    )

    # 4) Low app usage + low email engagement -> ~10% response.
    disengaged = (app_logins_30d <= 3) & (email_open_rate <= 0.20)
    response_prob = np.where(disengaged, np.minimum(response_prob, 0.10), response_prob)

    # Calibrate to ~50% overall response rate for realistic KYC campaigns.
    response_prob = np.clip(response_prob, 0.02, 0.98)
    shift_to_target = 0.50 - response_prob.mean()
    response_prob = np.clip(response_prob + shift_to_target, 0.02, 0.98)

    # Convert probabilities to labels with controlled noise (~12% flips).
    responded_to_last_refresh = response_prob >= 0.5
    noise_mask = np.random.rand(n_customers) < 0.12
    responded_to_last_refresh[noise_mask] = ~responded_to_last_refresh[noise_mask]

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "last_kyc_date": last_kyc_date.date,
            "days_since_last_kyc": days_since_last_kyc,
            "kyc_refresh_due_date": kyc_refresh_due_date.date,
            "days_until_due": days_until_due,
            "previous_refresh_count": previous_refresh_count,
            "previous_response_rate": previous_response_rate.round(4),
            "account_age_months": account_age_months,
            "card_active": card_active,
            "transaction_count_30d": transaction_count_30d,
            "transaction_count_90d": transaction_count_90d,
            "avg_transaction_amount": np.round(avg_transaction_amount, 2),
            "app_logins_30d": app_logins_30d,
            "email_open_rate": email_open_rate.round(4),
            "sms_response_rate": sms_response_rate.round(4),
            "risk_category": risk_category,
            "customer_segment": customer_segment,
            "contact_preference": contact_preference,
            "responded_to_last_refresh": responded_to_last_refresh,
        }
    )

    output_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "customers.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    response_rate = df["responded_to_last_refresh"].mean() * 100
    print(f"Generated {len(df):,} customers -> {output_path}")
    print(f"Overall response rate: {response_rate:.2f}%")
    print("Segment distribution:")
    print(
        (df["customer_segment"].value_counts(normalize=True) * 100).round(2).to_string()
    )

    return df


if __name__ == "__main__":
    generate_customer_data()
