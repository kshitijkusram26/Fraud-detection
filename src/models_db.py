"""
src/models_db.py
────────────────
SQLAlchemy ORM models.
Each class maps to one table in PostgreSQL (defined in sql/init.sql).
"""

from sqlalchemy import (
    Boolean, Column, Date, Integer, Numeric,
    SmallInteger, String, Text, TIMESTAMP
)
from sqlalchemy.sql import func

from src.database import Base


# ── 1. Predictions ───────────────────────────────────────────
class Prediction(Base):
    __tablename__ = "predictions"

    id                = Column(Integer,     primary_key=True, index=True)
    transaction_id    = Column(String(64),  nullable=False, unique=True)
    created_at        = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Key input features
    amount            = Column(Numeric(12, 2), nullable=False)
    time_seconds      = Column(Numeric(12, 2))
    v1                = Column(Numeric(10, 6))
    v2                = Column(Numeric(10, 6))
    v3                = Column(Numeric(10, 6))
    v4                = Column(Numeric(10, 6))
    v14               = Column(Numeric(10, 6))
    v17               = Column(Numeric(10, 6))

    # Model output
    fraud_probability = Column(Numeric(6, 5), nullable=False)
    prediction        = Column(String(16),    nullable=False)
    confidence        = Column(String(16))
    model_version     = Column(String(32))
    latency_ms        = Column(Numeric(8, 2))

    # Set later by analyst
    actual_label      = Column(SmallInteger, default=None)


# ── 2. Batch Jobs ────────────────────────────────────────────
class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id           = Column(Integer,    primary_key=True, index=True)
    job_id       = Column(String(64), nullable=False, unique=True)
    created_at   = Column(TIMESTAMP(timezone=True), server_default=func.now())
    completed_at = Column(TIMESTAMP(timezone=True))
    status       = Column(String(16), nullable=False, default="PENDING")
    total_rows   = Column(Integer)
    fraud_count  = Column(Integer)
    legit_count  = Column(Integer)
    filename     = Column(String(256))


# ── 3. Model Versions ────────────────────────────────────────
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id            = Column(Integer,    primary_key=True, index=True)
    registered_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    version       = Column(String(32), nullable=False)
    run_id        = Column(String(64))
    pr_auc        = Column(Numeric(6, 5))
    roc_auc       = Column(Numeric(6, 5))
    f1_score      = Column(Numeric(6, 5))
    threshold     = Column(Numeric(6, 5))
    is_active     = Column(Boolean, nullable=False, default=False)
    notes         = Column(Text)


# ── 4. Daily Stats ───────────────────────────────────────────
class DailyStat(Base):
    __tablename__ = "daily_stats"

    stat_date      = Column(Date,    primary_key=True)
    total_txns     = Column(Integer, nullable=False, default=0)
    fraud_count    = Column(Integer, nullable=False, default=0)
    avg_amount     = Column(Numeric(12, 2))
    avg_fraud_prob = Column(Numeric(6, 5))
    updated_at     = Column(TIMESTAMP(timezone=True), server_default=func.now())