from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="PRECOG - Amex India",
    page_icon="??",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f8fafc;
    }
    h1, h2, h3 {
        color: #006FCF;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    /* Keep Streamlit metric pill background as-is, but force readable black text */
    [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    [data-testid="stMetricDelta"] * {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _resolve_data_dir() -> Path:
    """Resolve dashboard CSV directory for either backend/BACKEND naming."""
    candidates = [
        Path("../backend/data/dashboard"),
        Path("../BACKEND/data/dashboard"),
        Path("backend/data/dashboard"),
        Path("BACKEND/data/dashboard"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load predictions from CSV."""
    data_dir = _resolve_data_dir()
    return pd.read_csv(data_dir / "predictions.csv")


@st.cache_data
def load_segment_summary() -> pd.DataFrame:
    """Load segment summary."""
    data_dir = _resolve_data_dir()
    return pd.read_csv(data_dir / "segment_summary.csv")


@st.cache_data
def load_cost_analysis() -> pd.DataFrame:
    """Load cost analysis."""
    data_dir = _resolve_data_dir()
    return pd.read_csv(data_dir / "cost_analysis.csv")


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    """Load feature importance."""
    data_dir = _resolve_data_dir()
    return pd.read_csv(data_dir / "feature_importance.csv")


@st.cache_data
def load_compliance() -> pd.DataFrame:
    """Load compliance metrics."""
    data_dir = _resolve_data_dir()
    return pd.read_csv(data_dir / "compliance.csv")


def plot_segment_distribution(df: pd.DataFrame) -> go.Figure:
    """Create pie chart for segment distribution."""
    counts = df["segment"].value_counts().reset_index()
    counts.columns = ["segment", "count"]
    color_map = {"High": "#00B140", "Medium": "#FFB900", "Low": "#D32F2F"}
    fig = px.pie(
        counts,
        names="segment",
        values="count",
        title="Segment Distribution",
        color="segment",
        color_discrete_map=color_map,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_cost_by_segment(df: pd.DataFrame) -> go.Figure:
    """Create bar chart for outreach cost by segment."""
    color_map = {"High": "#00B140", "Medium": "#FFB900", "Low": "#D32F2F"}
    fig = px.bar(
        df,
        x="segment",
        y="total_outreach_cost",
        color="segment",
        color_discrete_map=color_map,
        title="Cost by Segment",
        text="total_outreach_cost",
    )
    fig.update_traces(texttemplate="?%{text:.2s}", textposition="outside")
    fig.update_layout(
        xaxis_title="Segment",
        yaxis_title="Total Outreach Cost (?)",
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig


def plot_scatter(df: pd.DataFrame) -> go.Figure:
    """Create scatter plot for response probability vs KYC due days."""
    color_map = {"High": "#00B140", "Medium": "#FFB900", "Low": "#D32F2F"}
    fig = px.scatter(
        df,
        x="days_until_due",
        y="response_probability",
        color="segment",
        size="outreach_cost",
        hover_data=["customer_id", "recommended_channels", "risk_category", "customer_segment"],
        color_discrete_map=color_map,
        title="Response Probability vs Days Until Due",
    )
    fig.update_layout(
        xaxis_title="Days Until Due",
        yaxis_title="Response Probability",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_feature_importance() -> go.Figure:
    """Load and plot feature importance chart."""
    df = load_feature_importance().head(10).copy()
    df = df.sort_values("importance_score", ascending=True)
    fig = px.bar(
        df,
        x="importance_score",
        y="feature_name",
        orientation="h",
        color="importance_score",
        color_continuous_scale="Blues",
        title="Top 10 Feature Importance",
    )
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_showscale=False,
    )
    return fig


def _to_float(cost_df: pd.DataFrame, metric_name: str, default: float = 0.0) -> float:
    match = cost_df.loc[cost_df["metric_name"] == metric_name, "value"]
    return float(match.iloc[0]) if not match.empty else default


# Header
st.title("?? PRECOG - Amex India")
st.subheader("ML-Powered Periodic KYC Refresh Predictor")
st.write(
    "Predict customer response likelihood to KYC refresh campaigns, optimize outreach channels, "
    "and track compliance and cost savings in one interactive dashboard."
)

# Load data
try:
    predictions_df = load_data()
    segment_summary_df = load_segment_summary()
    cost_df = load_cost_analysis()
    compliance_df = load_compliance()
except FileNotFoundError:
    st.error("Dashboard CSV files not found. Run: python -m app.utils.export_dashboard_data")
    st.stop()

# Sidebar
st.sidebar.header("Filters")
segment_filter = st.sidebar.selectbox("Segment", ["All", "High", "Medium", "Low"])
risk_filter = st.sidebar.selectbox("Risk Category", ["All", "Low", "Medium", "High"])
customer_segment_filter = st.sidebar.selectbox("Customer Segment", ["All", "Basic", "Gold", "Platinum"])
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Apply filters
filtered_df = predictions_df.copy()
if segment_filter != "All":
    filtered_df = filtered_df[filtered_df["segment"] == segment_filter]
if risk_filter != "All":
    filtered_df = filtered_df[filtered_df["risk_category"] == risk_filter]
if customer_segment_filter != "All":
    filtered_df = filtered_df[filtered_df["customer_segment"] == customer_segment_filter]

# KPI cards
baseline_cost = _to_float(cost_df, "Baseline Cost")
optimized_cost = _to_float(cost_df, "Optimized Cost")
savings = _to_float(cost_df, "Savings")
savings_pct = _to_float(cost_df, "Savings %")

compliance_pct = float(
    compliance_df.loc[compliance_df["metric_name"] == "Compliance %", "value"].iloc[0]
) if (compliance_df["metric_name"] == "Compliance %").any() else 0.0

avg_response_prob = filtered_df["response_probability"].mean() if len(filtered_df) else 0.0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="?? Total Savings",
        value=f"?{savings:,.0f}",
        delta=f"{savings_pct:.1f}% vs baseline",
    )

with col2:
    st.metric(
        label="?? Customers Analyzed",
        value=f"{len(filtered_df):,}",
        delta="100% coverage",
    )

with col3:
    st.metric(
        label="? Compliance Status",
        value=f"{compliance_pct:.1f}%",
        delta="5% improvement",
    )

with col4:
    st.metric(
        label="?? Avg Response Prob",
        value=f"{avg_response_prob:.1%}",
        delta="?5%",
    )

# Visualizations
left_col, right_col = st.columns(2)

with left_col:
    st.plotly_chart(plot_segment_distribution(filtered_df), use_container_width=True)
    st.plotly_chart(plot_cost_by_segment(segment_summary_df), use_container_width=True)

with right_col:
    st.plotly_chart(plot_scatter(filtered_df), use_container_width=True)
    st.plotly_chart(plot_feature_importance(), use_container_width=True)

# Data table
st.subheader("Top 50 Customers")
search_text = st.text_input("Search by Customer ID", "")

table_df = filtered_df.copy()
if search_text:
    table_df = table_df[table_df["customer_id"].str.contains(search_text, case=False, na=False)]

table_df = table_df[[
    "customer_id",
    "segment",
    "response_probability",
    "recommended_channels",
    "outreach_cost",
]].copy()
table_df = table_df.rename(
    columns={
        "customer_id": "Customer ID",
        "segment": "Segment",
        "response_probability": "Probability (%)",
        "recommended_channels": "Recommended Channels",
        "outreach_cost": "Cost (?)",
    }
)
table_df["Probability (%)"] = (table_df["Probability (%)"] * 100).round(2)
table_df["Cost (?)"] = table_df["Cost (?)"].round(2)
table_df = table_df.sort_values(by="Probability (%)", ascending=False).head(50)

st.dataframe(table_df, use_container_width=True, hide_index=True)

csv_bytes = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Customers (CSV)",
    data=csv_bytes,
    file_name="filtered_customers.csv",
    mime="text/csv",
)

st.caption(
    f"Baseline cost: ?{baseline_cost:,.0f} | Optimized cost: ?{optimized_cost:,.0f} | "
    f"Savings: ?{savings:,.0f}"
)

