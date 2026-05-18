"""SQLAlchemy engine / session factory."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from quantify.config import get_settings
from quantify.utils.logger import log


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    cfg = get_settings().mysql
    engine = create_engine(
        cfg.url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,
        future=True,
    )
    log.debug(f"MySQL engine created: {cfg.host}:{cfg.port}/{cfg.database}")
    return engine


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, autoflush=False, future=True)


def get_session() -> Session:
    return _session_factory()()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager that commits on success and rolls back on error."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ensure_database_exists() -> None:
    """Create the target database if it does not exist (uses a server-level URL)."""
    cfg = get_settings().mysql
    server_engine = create_engine(cfg.server_url, future=True)
    with server_engine.connect() as conn:
        conn.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{cfg.database}` "
                f"DEFAULT CHARACTER SET {cfg.charset} "
                f"DEFAULT COLLATE {cfg.charset}_unicode_ci"
            )
        )
        conn.commit()
    server_engine.dispose()
    log.info(f"Database `{cfg.database}` ready.")
