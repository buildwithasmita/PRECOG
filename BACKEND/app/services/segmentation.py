from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


class CustomerSegmentation:
    """Segment customers by response probability and compute outreach economics."""

    HIGH_THRESHOLD = 0.7
    MEDIUM_THRESHOLD = 0.3

    COST_EMAIL = 5.0
    COST_EMAIL_SMS = 15.0
    COST_EMAIL_SMS_CALL = 50.0

    def segment_customers(
        self, customer_ids: List[str], probabilities: np.ndarray
    ) -> pd.DataFrame:
        """Assign segment, channel recommendations, and outreach cost per customer."""
        if len(customer_ids) != len(probabilities):
            raise ValueError("customer_ids and probabilities must have the same length")

        probs = np.asarray(probabilities, dtype=float)

        segments = np.where(
            probs >= self.HIGH_THRESHOLD,
            "High",
            np.where(probs >= self.MEDIUM_THRESHOLD, "Medium", "Low"),
        )

        channels_map = {
            "High": ["email"],
            "Medium": ["email", "sms"],
            "Low": ["email", "sms", "call"],
        }
        cost_map = {
            "High": self.COST_EMAIL,
            "Medium": self.COST_EMAIL_SMS,
            "Low": self.COST_EMAIL_SMS_CALL,
        }

        segmented_df = pd.DataFrame(
            {
                "customer_id": customer_ids,
                "response_probability": probs,
                "segment": segments,
            }
        )
        segmented_df["recommended_channels"] = segmented_df["segment"].map(channels_map)
        segmented_df["outreach_cost"] = (
            segmented_df["segment"].map(cost_map).astype(float)
        )

        return segmented_df

    def calculate_roi(
        self, segmented_df: pd.DataFrame, total_customers: int
    ) -> Dict[str, float]:
        """Compare baseline outreach spend to optimized segment-based spend."""
        baseline_cost = float(total_customers * self.COST_EMAIL_SMS)
        optimized_cost = float(segmented_df["outreach_cost"].sum())
        savings = baseline_cost - optimized_cost
        savings_percentage = (
            (savings / baseline_cost * 100.0) if baseline_cost > 0 else 0.0
        )

        return {
            "baseline_cost": baseline_cost,
            "optimized_cost": optimized_cost,
            "savings": savings,
            "savings_percentage": savings_percentage,
        }

    def get_segment_summary(
        self, segmented_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Return count/probability/cost breakdown for each segment plus overall stats."""
        summary: Dict[str, Dict[str, float]] = {}

        for segment in ["High", "Medium", "Low"]:
            subset = segmented_df[segmented_df["segment"] == segment]
            summary[segment] = {
                "count": int(len(subset)),
                "avg_probability": (
                    float(subset["response_probability"].mean()) if len(subset) else 0.0
                ),
                "total_cost": float(subset["outreach_cost"].sum()),
            }

        summary["Overall"] = {
            "total_cost": float(segmented_df["outreach_cost"].sum()),
            "expected_response_rate": (
                float(segmented_df["response_probability"].mean())
                if len(segmented_df)
                else 0.0
            ),
        }

        return summary

    def recommend_outreach_strategy(self, segment: str) -> Dict[str, object]:
        """Provide actionable channel/timing/tone strategy for a given segment."""
        strategy_map: Dict[str, Dict[str, object]] = {
            "High": {
                "channels": ["email"],
                "timing": "Within 3 days",
                "message_tone": "Standard",
                "follow_up": False,
                "cost_per_customer": self.COST_EMAIL,
            },
            "Medium": {
                "channels": ["email", "sms"],
                "timing": "Immediate",
                "message_tone": "Urgent",
                "follow_up": True,
                "cost_per_customer": self.COST_EMAIL_SMS,
            },
            "Low": {
                "channels": ["email", "sms", "call"],
                "timing": "Within 1 day",
                "message_tone": "Personal",
                "follow_up": True,
                "cost_per_customer": self.COST_EMAIL_SMS_CALL,
            },
        }

        normalized = segment.strip().title()
        if normalized not in strategy_map:
            raise ValueError("segment must be one of: High, Medium, Low")

        return strategy_map[normalized]
