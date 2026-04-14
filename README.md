# Transaction Fraud Detection

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.11-orange?logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red?logo=streamlit)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![LightGBM](https://img.shields.io/badge/LightGBM-PR--AUC%3A0.88-brightgreen)

> Detect fraudulent credit card transactions in real-time using LightGBM,
> served via a FastAPI REST service, tracked with MLflow, logged to PostgreSQL,
> and monitored through a Streamlit dashboard — all orchestrated with Docker Compose.

---

## Results

| Model | PR-AUC | ROC-AUC | F1 | Threshold |
|---|---|---|---|---|
| **LightGBM** | **0.8803** | 0.9813 | 0.8663 | 0.80 |
| XGBoost | 0.8724 | 0.9787 | 0.8750 | 0.56 |
| Random Forest | 0.8452 | 0.9865 | 0.8324 | 0.82 |
| Logistic Regression | 0.7186 | 0.9723 | 0.3037 | 0.88 |

> Evaluated using PR-AUC — the correct metric for severely imbalanced data (0.17% fraud rate)

---

## Architecture
[Kaggle CSV - 284K transactions]
↓
[Data Pipeline - SMOTE + Feature Engineering]
↓
[Model Training - 4 models compared via MLflow]
↓
[best_model.pkl - LightGBM saved]
↓
[FastAPI - POST /predict endpoint]
↓
[PostgreSQL - every prediction logged]
↓
[Streamlit Dashboard - live analyst UI]

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection

# 2. Copy env
cp .env.example .env

# 3. Start all services
docker-compose up --build

# 4. Download dataset from Kaggle and place at:
#    data/raw/creditcard.csv

# 5. Train models
python -m src.train

# Services:
#   FastAPI docs  → http://localhost:8000/docs
#   MLflow UI     → http://localhost:5000
#   Dashboard     → http://localhost:8501
```

---

## Project Structure
fraud-detection/
├── data/
│   ├── raw/                  # creditcard.csv (download from Kaggle)
│   └── processed/            # Scaled numpy arrays + EDA plots
├── notebooks/
│   └── eda.py                # EDA script — 6 visualisation plots
├── src/
│   ├── data_pipeline.py      # ETL + SMOTE + feature engineering
│   ├── train.py              # Train 4 models + MLflow logging
│   ├── predict.py            # Inference wrapper
│   ├── database.py           # SQLAlchemy engine + session
│   └── models_db.py          # ORM models
├── api/
│   ├── main.py               # FastAPI — 5 endpoints
│   ├── schemas.py            # Pydantic request/response models
│   └── Dockerfile
├── dashboard/
│   ├── app.py                # Streamlit — 4 page dashboard
│   └── Dockerfile
├── sql/
│   └── init.sql              # PostgreSQL schema — 4 tables
├── tests/
│   └── test_pipeline.py      # 15 pytest unit tests
├── docker-compose.yml
├── requirements.txt
└── .env.example

---

## Dataset

**Kaggle Credit Card Fraud Detection**
- 284,807 transactions
- 492 fraud cases (0.172% — severely imbalanced)
- 28 PCA features (V1–V28) + Amount + Time
- Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Place at: `data/raw/creditcard.csv`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML Models | LightGBM, XGBoost, Random Forest, Logistic Regression |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Experiment Tracking | MLflow 2.x |
| REST API | FastAPI + Pydantic v2 |
| Database | PostgreSQL 16 + SQLAlchemy |
| Dashboard | Streamlit |
| Visualisation | Plotly, Seaborn, Matplotlib |
| Containerisation | Docker + Docker Compose |
| Testing | Pytest — 15 unit tests |
| CI/CD | GitHub Actions |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service liveness + model + DB status |
| GET | `/model/info` | Active model metadata from PostgreSQL |
| GET | `/metrics` | Running prediction statistics |
| POST | `/predict` | Score a single transaction |
| POST | `/predict/batch` | Score up to 5,000 transactions |

Full interactive docs: http://localhost:8000/docs

---

## Sample Prediction

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1":-2.31,"V2":1.95,"V3":-1.61,"V4":3.99,
       "V5":-0.52,"V6":-1.42,"V7":-2.92,"V8":0.08,
       "V9":-0.15,"V10":-0.64,"V11":-1.87,"V12":-1.26,
       "V13":-0.07,"V14":-2.17,"V15":0.13,"V16":-1.23,
       "V17":-1.51,"V18":-1.05,"V19":0.11,"V20":0.06,
       "V21":0.06,"V22":-0.07,"V23":-0.07,"V24":-0.03,
       "V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02,
       "Amount":1.00,"Time":406}'
```

**Response:**
```json
{
  "transaction_id": "8f0b02dc-46cc-44ba-b320-71286d41c43a",
  "fraud_probability": 0.97823,
  "prediction": "FRAUD",
  "confidence": "HIGH",
  "model_version": "v1.0-lightgbm",
  "latency_ms": 4.51
}
```

---

## PostgreSQL Schema

```sql
predictions    -- every API call logged here
batch_jobs     -- batch prediction job tracking
model_versions -- registered model metadata
daily_stats    -- aggregated daily fraud statistics
```

---

## Running Tests

```bash
pytest tests/ -v
# 15 passed
```

---

## Key Design Decisions

**Why PR-AUC over accuracy?**
With only 0.17% fraud, a model predicting always LEGIT scores 99.83% accuracy but catches zero fraud. PR-AUC focuses on minority class performance.

**Why SMOTE after split?**
Applying SMOTE before splitting leaks synthetic test samples into training — inflating metrics. Always fit SMOTE on train only.

**Why LightGBM won?**
Fastest training (4s vs 45s for Random Forest) with highest PR-AUC (0.88). Handles class imbalance natively via `is_unbalance=True`.

**Why RobustScaler?**
Fraud transaction amounts have extreme outliers. RobustScaler uses median/IQR instead of mean/std — not affected by outliers.

---

## License

MIT — free to use for portfolio and learning purposes.
