# API Usage Examples

## Base URL
```text
http://localhost:8000
```

## 1. Health Check
```bash
curl http://localhost:8000/api/v1/health
```
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/health" -Method Get
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 2. Batch Prediction (All Customers)
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customer_ids": null
  }'
```
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict/batch" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"customer_ids": null}'
```

Response:
```json
{
  "total_customers": 10000,
  "predictions": [...],
  "summary": {
    "high_count": 3000,
    "medium_count": 5000,
    "low_count": 2000,
    "baseline_cost": 150000,
    "optimized_cost": 75000,
    "savings": 75000
  }
}
```

## 3. Single Customer Prediction
```bash
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "days_since_last_kyc": 730,
    "days_until_due": 30,
    "previous_refresh_count": 2,
    "previous_response_rate": 0.5,
    "account_age_months": 48,
    "card_active": true,
    "transaction_count_30d": 25,
    "transaction_count_90d": 80,
    "avg_transaction_amount": 5000,
    "app_logins_30d": 15,
    "email_open_rate": 0.6,
    "sms_response_rate": 0.4,
    "risk_category": "Medium",
    "customer_segment": "Gold",
    "contact_preference": "email"
  }'
```
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict/single" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "days_since_last_kyc": 730,
    "days_until_due": 30,
    "previous_refresh_count": 2,
    "previous_response_rate": 0.5,
    "account_age_months": 48,
    "card_active": true,
    "transaction_count_30d": 25,
    "transaction_count_90d": 80,
    "avg_transaction_amount": 5000,
    "app_logins_30d": 15,
    "email_open_rate": 0.6,
    "sms_response_rate": 0.4,
    "risk_category": "Medium",
    "customer_segment": "Gold",
    "contact_preference": "email"
  }'
```

Response:
```json
{
  "customer_id": "GENERATED",
  "response_probability": 0.6543,
  "predicted_segment": "Medium",
  "recommended_channels": ["email", "sms"],
  "outreach_cost": 15.0,
  "confidence_level": "Medium"
}
```

## 4. Get Dashboard Data
```bash
curl http://localhost:8000/api/v1/analytics/dashboard-data
```
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/analytics/dashboard-data" -Method Get
```

## 5. Filter by Segment
```bash
curl http://localhost:8000/api/v1/analytics/segment/Low
```
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/analytics/segment/Low" -Method Get
```

## Python Examples

### Using requests library
```python
import requests

# Batch prediction
response = requests.post(
    "http://localhost:8000/api/v1/predict/batch",
    json={"customer_ids": None}
)
data = response.json()
print(f"Savings: INR {data['summary']['savings']:,.0f}")
```

### Using httpx (async)
```python
import httpx
import asyncio

async def predict():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/predict/batch",
            json={"customer_ids": None}
        )
        return response.json()

data = asyncio.run(predict())
```
