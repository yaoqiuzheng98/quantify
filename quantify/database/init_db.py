"""Schema bootstrap helpers."""

from __future__ import annotations

from quantify.database.engine import ensure_database_exists, get_engine
from quantify.database.models import Base
from quantify.utils.logger import log


def init_db(drop_first: bool = False) -> None:
    """Ensure database + create all tables that don't exist."""
    ensure_database_exists()
    engine = get_engine()
    if drop_first:
        log.warning("Dropping all known tables before re-creation...")
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    log.info(f"Schema synced. Tables: {sorted(Base.metadata.tables.keys())}")
