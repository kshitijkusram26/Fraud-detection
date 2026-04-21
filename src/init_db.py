"""
src/init_db.py
Run once at startup to create tables and seed model version.
"""
import os
from loguru import logger
from src.database import engine, Base
from src.models_db import ModelVersion
from sqlalchemy.orm import sessionmaker

def init_db():
    logger.info("Creating tables if not exist...")
    Base.metadata.create_all(bind=engine)
    logger.info("Tables ready.")

    # Seed model version if none exists
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        existing = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
        if not existing:
            logger.info("Seeding model version...")
            mv = ModelVersion(
                version   = "v1.0-lightgbm",
                pr_auc    = 0.85,
                roc_auc   = 0.97,
                f1_score  = 0.78,
                threshold = 0.5,
                is_active = True,
                notes     = "LightGBM - best PR-AUC on test set",
            )
            db.add(mv)
            db.commit()
            logger.info("Model version seeded.")
        else:
            logger.info(f"Model version already exists: {existing.version}")
    except Exception as e:
        logger.error(f"DB seed failed: {e}")
        db.rollback()
    finally:
        db.close()