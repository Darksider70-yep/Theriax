from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
import os
import logging

load_dotenv(".env")

DATABASE_URL = os.getenv("DATABASE_URL")

# Prefer the configured DATABASE_URL, but fall back to a local SQLite file
# when the configured DB is unreachable to allow local development.
LOCAL_SQLITE_URL = "sqlite:///./theriax_local.db"

logger = logging.getLogger("theriax-backend")

engine = None
if DATABASE_URL:
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        # quick smoke test connection
        conn = engine.connect()
        conn.close()
    except Exception as e:
        logger.warning("Could not connect to DATABASE_URL, falling back to SQLite: %s", e)
        engine = create_engine(LOCAL_SQLITE_URL, connect_args={"check_same_thread": False})
else:
    logger.info("DATABASE_URL not set; using local SQLite database for development.")
    engine = create_engine(LOCAL_SQLITE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
