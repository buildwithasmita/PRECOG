# PRECOG Dashboard Specification

## Overview
Executive dashboard for American Express KYC analysts showing ML predictions, cost savings, and outreach recommendations.

## Data Sources
1. predictions.csv - Customer-level predictions
2. segment_summary.csv - Segment-level aggregates
3. cost_analysis.csv - ROI calculations
4. feature_importance.csv - Model insights
5. compliance.csv - Compliance metrics

## Dashboard Layout (4 sections)

### SECTION 1: Executive Scorecard (Top Row)
**KPI Cards (4 cards side-by-side):**

Card 1: Total Cost Savings
- Metric: ?2,500,000 (example)
- Subtitle: "50% reduction vs baseline"
- Color: Green
- Icon: ??
- Source: cost_analysis.csv

Card 2: Customers Analyzed
- Metric: 10,000
- Breakdown: High (30%), Med (50%), Low (20%)
- Color: Blue
- Icon: ??
- Source: predictions.csv count

Card 3: Compliance Status
- Metric: 95% (example)
- Subtitle: "On-track for refresh"
- Color: Orange if <90%, Green if >=90%
- Icon: ?
- Source: compliance.csv

Card 4: Avg Response Probability
- Metric: 62% (example)
- Trend: ?5% vs last month
- Color: Blue
- Icon: ??
- Source: predictions.csv avg

### SECTION 2: Segment Analysis (Left Side)
**Visualization 1: Segment Distribution (Pie Chart)**
- Data: segment_summary.csv
- Slices: High, Medium, Low
- Colors: Green, Yellow, Red
- Show: Count + percentage
- Hover: Show avg probability

**Visualization 2: Cost by Segment (Bar Chart)**
- Data: segment_summary.csv
- X-axis: Segment
- Y-axis: Total outreach cost (?)
- Color by segment
- Show: Cost per customer label

### SECTION 3: Customer Deep Dive (Center)
**Visualization 3: Scatter Plot (Response Prob vs Days Until Due)**
- Data: predictions.csv
- X-axis: days_until_due
- Y-axis: response_probability
- Color: segment
- Size: outreach_cost
- Filter: By segment, risk_category
- Hover: customer_id, recommended_channels

**Visualization 4: Top 20 Low-Likelihood Customers (Table)**
- Data: predictions.csv filtered by segment='Low'
- Columns:
  * Customer ID
  * Response Prob (%)
  * Days Until Due
  * Recommended Channels
  * Cost (?)
- Sort: By probability ascending
- Action: Clickable to drill-down

### SECTION 4: Model Insights (Right Side)
**Visualization 5: Feature Importance (Horizontal Bar)**
- Data: feature_importance.csv
- Top 10 features
- X-axis: Importance score
- Y-axis: Feature name
- Color: Gradient blue
- Helps: Explain "why this prediction?"

**Visualization 6: Compliance Funnel**
- Data: compliance.csv
- Show: Total ? Due Soon ? Overdue
- Visual: Funnel chart
- Colors: Green ? Yellow ? Red

## Interactivity
1. Filter by Segment (dropdown)
2. Filter by Risk Category (dropdown)
3. Date range selector (for compliance)
4. Drill-down from pie chart to table

## Color Scheme
- Primary: Amex Blue (#006FCF)
- Success: Green (#00B140)
- Warning: Yellow (#FFB900)
- Danger: Red (#D32F2F)
- Background: White (#FFFFFF)
- Text: Dark Gray (#333333)

## Export Options
- PDF report (for executives)
- Excel data export
- Email scheduled reports

IMPORTANT:
- Mobile-responsive layout
- Load time < 3 seconds
- Auto-refresh every 15 minutes
- Accessible color contrasts
