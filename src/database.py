import os 
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine,text
from sqlalchemy.orm import declarative_base,sessionmaker,session

load_dotenv()

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db",
)


engine=create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False
)

sessionLocal=sessionmaker(autocommit=False,autoflush=False,bind=engine)
Base= declarative_base()

def get_db() -> Generator[session,None,None]:
    """ Yield a DB session. Use as FastAPI Depends(get_db)."""
    db=sessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_session() -> Generator[session,None,None]:
    db=sessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def ping_db() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        logger.info('Database connection OK')
        return True
    except Exception as exc:
        logger.error(f'Database connection FAILED: {exc}')
        return False


