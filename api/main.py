import sys, os, time, uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from loguru import logger
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database import get_db, ping_db
from src.models_db import Prediction, ModelVersion
from src.predict import get_predictor, FraudPredictor
from api.schemas import (
    TransactionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    MetricsResponse,
    ErrorResponse,
)
from api.auth import verify_api_key          # 🔐 NEW
from api.limiter import limiter              # 🚦 NEW


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Fraud Detection API...")
    if not ping_db():
        logger.warning("DB not reachable")
    else:
        from src.init_db import init_db
        init_db()
    try:
        app.state.predictor = get_predictor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        app.state.predictor = None
    app.state.stats = {"total": 0, "fraud": 0, "legit": 0}
    yield
    logger.info("Shutting down API...")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="""
## Transaction Fraud Detection

Predict whether a financial transaction is fraudulent using ML models
trained on the IEEE-CIS dataset (XGBoost / LightGBM).

### Authentication
All prediction endpoints require an **X-API-Key** header.
Contact the provider to get your key.

### Endpoints
- **POST /predict** — score a single transaction
- **POST /predict/batch** — score up to 5,000 transactions
- **GET /health** — service liveness check
- **GET /model/info** — active model metadata
- **GET /metrics** — prediction statistics
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# ── Rate limiter state ────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS — lock this down in production ──────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # set ALLOWED_ORIGINS in .env for prod
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Latency header middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    ms       = round((time.time() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(ms)
    return response

# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500},
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return {"message": "Fraud Detection API v1.0 — visit /docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health(request: Request):
    model_ready = request.app.state.predictor is not None
    db_ready    = ping_db()
    status      = "healthy" if (model_ready and db_ready) else "degraded"
    return HealthResponse(
        status       = status,
        model_loaded = model_ready,
        db_connected = db_ready,
        api_version  = "1.0.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info(db: Session = Depends(get_db)):
    mv = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    if not mv:
        raise HTTPException(status_code=404, detail="No active model found")
    return ModelInfoResponse(
        version   = mv.version,
        pr_auc    = float(mv.pr_auc)    if mv.pr_auc    else None,
        roc_auc   = float(mv.roc_auc)   if mv.roc_auc   else None,
        f1_score  = float(mv.f1_score)  if mv.f1_score  else None,
        threshold = float(mv.threshold) if mv.threshold else 0.5,
        notes     = mv.notes,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
def metrics(request: Request, db: Session = Depends(get_db)):
    stats    = request.app.state.stats
    db_total = db.query(Prediction).count()
    db_fraud = db.query(Prediction).filter(Prediction.prediction == "FRAUD").count()
    fraud_rate = round(db_fraud / db_total, 5) if db_total > 0 else 0.0
    return MetricsResponse(
        session_total = stats["total"],
        session_fraud = stats["fraud"],
        session_legit = stats["legit"],
        db_total      = db_total,
        db_fraud      = db_fraud,
        fraud_rate_db = fraud_rate,
    )


# 🔐 Protected — requires X-API-Key header
# 🚦 Rate limited — 60 requests/minute per IP
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("60/minute")
def predict(
    body: TransactionRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    predictor: FraudPredictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — run training first",
        )

    t_start = time.time()
    result  = predictor.predict_one(body.model_dump())
    latency = round((time.time() - t_start) * 1000, 2)

    transaction_id = str(uuid.uuid4())
    mv             = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    model_version  = mv.version if mv else "unknown"

    try:
        row = Prediction(
            transaction_id    = transaction_id,
            amount            = body.Amount,
            time_seconds      = body.Time,
            v1=body.V1, v2=body.V2, v3=body.V3,
            v4=body.V4, v14=body.V14, v17=body.V17,
            fraud_probability = result["fraud_probability"],
            prediction        = result["prediction"],
            confidence        = result["confidence"],
            model_version     = model_version,
            latency_ms        = latency,
        )
        db.add(row)
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to log prediction to DB: {e}")
        db.rollback()

    request.app.state.stats["total"] += 1
    if result["prediction"] == "FRAUD":
        request.app.state.stats["fraud"] += 1
    else:
        request.app.state.stats["legit"] += 1

    return PredictionResponse(
        transaction_id    = transaction_id,
        fraud_probability = result["fraud_probability"],
        prediction        = result["prediction"],
        confidence        = result["confidence"],
        model_version     = model_version,
        latency_ms        = latency,
    )


# 🔐 Protected — requires X-API-Key header
# 🚦 Rate limited — 10 batch requests/minute per IP
@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")
def predict_batch(
    body: BatchPredictionRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    predictor: FraudPredictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t_start = time.time()
    records = [t.model_dump() for t in body.transactions]
    results = predictor.predict_batch(records)
    latency = round((time.time() - t_start) * 1000, 2)

    mv            = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    model_version = mv.version if mv else "unknown"
    fraud_count   = sum(1 for r in results if r["prediction"] == "FRAUD")

    try:
        db.bulk_insert_mappings(Prediction, [
            {
                "transaction_id":    str(uuid.uuid4()),
                "amount":            records[i]["Amount"],
                "time_seconds":      records[i].get("Time"),
                "fraud_probability": results[i]["fraud_probability"],
                "prediction":        results[i]["prediction"],
                "confidence":        results[i]["confidence"],
                "model_version":     model_version,
                "latency_ms":        round(latency / len(results), 2),
            }
            for i in range(len(results))
        ])
        db.commit()
    except Exception as e:
        logger.warning(f"Batch DB log failed: {e}")
        db.rollback()

    return BatchPredictionResponse(
        total         = len(results),
        fraud_count   = fraud_count,
        legit_count   = len(results) - fraud_count,
        fraud_rate    = round(fraud_count / len(results), 4),
        predictions   = results,
        latency_ms    = latency,
        model_version = model_version,
    )