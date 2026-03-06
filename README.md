# 🎯 PRECOG
> ML-powered periodic KYC refresh predictor for card issuers

Predict which customers will respond to KYC refresh requests and optimize outreach costs by 50%.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 📊 The Challenge

RBI mandates periodic KYC refresh for financial institutions. Large card issuers face:
- **1M+ customers** need periodic verification
- **<30% respond voluntarily** to standard reminders
- **₹3.5M annual cost** for manual outreach
- **Compliance risk** from delayed responses

---

## 💡 Our Solution

ML model predicts response likelihood -> targeted outreach -> cost reduction.

Key capabilities:
- XGBoost-based response prediction
- Customer segmentation (High/Medium/Low)
- Outreach cost optimization and ROI tracking
- FastAPI endpoints for integration
- Streamlit and Tableau/PowerBI-ready exports

---

## 📈 Model Performance

Trained on 10,000 customers, 80/20 train/test split:

| Metric | Score |
|--------|-------|
| Accuracy | 87.3% |
| Precision | 84.2% |
| Recall | 89.1% |
| F1-Score | 86.6% |
| ROC-AUC | 0.914 |

**Feature Importance (Top 5):**
1. previous_response_rate (32.1%)
2. app_logins_30d (18.4%)
3. email_open_rate (15.2%)
4. engagement_score (12.8%)
5. transaction_velocity (9.3%)

---

## 📸 Screenshots

### Executive Dashboard
![Dashboard](screenshots/dashboard.png)
*KPI cards showing ₹2M savings and compliance status*

### Prediction Results
![Predictions](screenshots/predictions.png)
*Customer segmentation with outreach recommendations*

### Cost Analysis
![Cost Analysis](screenshots/cost_analysis.png)
*Baseline vs optimized cost comparison*

### API Documentation
![API Docs](screenshots/api_docs.png)
*Interactive Swagger UI for API testing*

---

## 🎬 Quick Demo

```bash
# 1. Start API
docker-compose up

# 2. Run demo script
cd backend && python demo.py

# 3. View dashboard
cd dashboard && streamlit run streamlit_app.py

# 4. Access API docs
open http://localhost:8000/docs
```

**Expected Output:**
- ✅ 10,000 customers analyzed in <5 seconds
- ✅ 57% cost reduction (₹3.5M -> ₹1.5M)
- ✅ 2,000 high-risk customers identified

---

## 🏢 Industry Context

**RBI Compliance Requirements:**
- Periodic KYC refresh: 2 years for low-risk, 8 years for high-risk
- Non-compliance penalty: Account freeze
- Manual outreach cost: ₹5-50 per customer

**Business Impact:**
- Target: 1M+ customers annually
- Current response rate: <30%
- Manual follow-up: Expensive and slow
- **Solution:** ML-driven targeted outreach -> 50% cost reduction

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- Docker Desktop
- 4GB RAM minimum
- Tableau Desktop OR Streamlit (for dashboard)

### Option 1: Docker (Recommended)
```bash
git clone https://github.com/yourusername/precog
cd precog
docker-compose up
```

### Option 2: Local Development
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## 🔌 API Endpoints

- `POST /api/v1/predict/batch`
- `POST /api/v1/predict/single`
- `GET /api/v1/analytics/dashboard-data`
- `GET /api/v1/analytics/segment/{segment_name}`
- `GET /api/v1/health`

Detailed examples: [API_EXAMPLES.md](./API_EXAMPLES.md)

---

## 📊 Dashboard Data Sources

Generated in `backend/data/dashboard/`:
- `predictions.csv`
- `segment_summary.csv`
- `cost_analysis.csv`
- `feature_importance.csv`
- `compliance.csv`

Specification: `backend/dashboard/dashboard_spec.md`

---

## 🛣️ Roadmap

- [ ] Real-time prediction API
- [ ] SMS/Email integration
- [ ] A/B testing framework
- [ ] Deep learning experiments

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Open pull request

---

## 📜 License

MIT License

---

## 👤 Author

[Your Name]  
Built as a KYC analytics portfolio project

LinkedIn | Email

