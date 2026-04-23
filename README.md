# 🔍 Fraud Detection API

> Real-time financial transaction fraud detection powered by LightGBM — deployed as a production REST API.

[![RapidAPI](https://img.shields.io/badge/RapidAPI-Live-blue?style=flat-square&logo=rapid)](https://rapidapi.com/kshitijkusram/api/fraud-detection1)
[![Railway](https://img.shields.io/badge/Deployed-Railway-blueviolet?style=flat-square)](https://fraud-detection-production-dab3.up.railway.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docker.com)

---

## 🚀 Live API

| Endpoint | URL |
|---|---|
| Base URL | `https://fraud-detection-production-dab3.up.railway.app` |
| Docs | `https://fraud-detection-production-dab3.up.railway.app/docs` |
| RapidAPI Hub | `https://rapidapi.com/kshitijkusram/api/fraud-detection1` |

---

## 📊 Model Performance

| Model | ROC-AUC | PR-AUC | F1-Score |
|---|---|---|---|
| **LightGBM** ⭐ | **0.974** | **0.856** | **0.783** |
| XGBoost | 0.971 | 0.841 | 0.771 |
| Random Forest | 0.963 | 0.812 | 0.748 |
| Logistic Regression | 0.951 | 0.789 | 0.731 |

Trained on **284,807 real transactions** (Kaggle Credit Card Fraud Detection dataset). Class imbalance handled via SMOTE.

---

## ⚡ Quick Start

### Single Transaction

```bash
curl -X POST https://fraud-detection-production-dab3.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,
    "V4": 1.3782, "V14": -0.3111, "V17": -0.5535,
    "V5": 0, "V6": 0, "V7": 0, "V8": 0, "V9": 0,
    "V10": 0, "V11": 0, "V12": 0, "V13": 0, "V15": 0,
    "V16": 0, "V18": 0, "V19": 0, "V20": 0, "V21": 0,
    "V22": 0, "V23": 0, "V24": 0, "V25": 0, "V26": 0,
    "V27": 0, "V28": 0,
    "Amount": 149.62, "Time": 3600.0
  }'
```

### Response

```json
{
  "transaction_id": "c69c7e06-78e5-443e-b0e9-be89c27163b0",
  "fraud_probability": 0.00001,
  "prediction": "LEGIT",
  "confidence": "Low",
  "model_version": "v1.0-lightgbm",
  "latency_ms": 4.51
}
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                   Client Request                 │
└──────────────────────┬──────────────────────────┘
                       │
              ┌────────▼────────┐
              │   FastAPI App   │  ← Auth + Rate Limiting
              │   (Railway)     │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐  ┌─────▼─────┐ ┌────▼────┐
    │LightGBM │  │PostgreSQL │ │ MLflow  │
    │ Model   │  │  Logging  │ │Tracking │
    └─────────┘  └───────────┘ └─────────┘
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| ML Model | LightGBM (best PR-AUC) |
| Model Registry | MLflow |
| Database | PostgreSQL (SQLAlchemy ORM) |
| Dashboard | Streamlit |
| Containerization | Docker Compose |
| Deployment | Railway |
| Auth | API Key (X-API-Key header) |
| Rate Limiting | SlowAPI |

---

## 📁 Project Structure

```
fraud-detection/
├── api/
│   ├── main.py          # FastAPI app, routes, middleware
│   ├── schemas.py       # Pydantic request/response models
│   ├── auth.py          # API key authentication
│   ├── limiter.py       # Rate limiting
│   └── Dockerfile
├── src/
│   ├── database.py      # SQLAlchemy engine + session
│   ├── models_db.py     # ORM models (Predictions, ModelVersions)
│   ├── predict.py       # FraudPredictor class
│   └── init_db.py       # DB init + model version seeding
├── dashboard/           # Streamlit monitoring dashboard
├── models/              # Trained model artifacts (.pkl)
├── notebooks/           # Training + EDA notebooks
├── docker-compose.yml
├── railway.toml
└── requirements.txt
```

---

## 🏃 Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection

# Set up environment
cp .env.example .env  # fill in your values

# Run with Docker Compose
docker compose up --build

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
# MLflow at http://localhost:5000
# Dashboard at http://localhost:8501
```

---

## 🔐 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | No | Service health check |
| `GET` | `/model/info` | No | Active model metadata |
| `GET` | `/metrics` | No | Prediction statistics |
| `POST` | `/predict` | ✅ Yes | Score single transaction |
| `POST` | `/predict/batch` | ✅ Yes | Score up to 5,000 transactions |

Get your API key at [RapidAPI Hub](https://rapidapi.com/kshitijkusram/api/fraud-detection1).

---

## 👤 Author

**Kshitij Kusram** —  B.Tech CSE (Data Science), VIIT Pune

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/kshitij-kusram-46867828b/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/kshitijkusram26)

---

## 📄 License

MIT License — free to use for personal and commercial projects.