"""Batch UPSERT helper for MySQL via SQLAlchemy 2.0."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import pandas as pd
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session

from quantify.database.engine import session_scope
from quantify.database.models import Base
from quantify.utils.logger import log


def _clean_records(df: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    """Convert a DataFrame to records, restricted to known columns and with NaN -> None."""
    keep = [c for c in df.columns if c in columns]
    sub = df[keep].copy()

    # Replace NaN / NaT with None for SQL compatibility.
    records: list[dict[str, Any]] = []
    for row in sub.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for k, v in row.items():
            if v is None:
                clean[k] = None
            elif isinstance(v, float) and math.isnan(v):
                clean[k] = None
            elif pd.isna(v):
                clean[k] = None
            else:
                clean[k] = v
        records.append(clean)
    return records


def upsert_dataframe(
    model: type[Base],
    df: pd.DataFrame,
    *,
    chunk_size: int = 2000,
    update_keys: Iterable[str] | None = None,
    session: Session | None = None,
) -> int:
    """Insert/Update a DataFrame into the table backing ``model``.

    Uses MySQL ``INSERT ... ON DUPLICATE KEY UPDATE``. The set of columns to
    update is everything besides primary-key columns (or ``update_keys`` if
    provided).

    Returns the number of rows submitted (note: MySQL's affected-rows count
    can double-count UPDATEs, so we return input count for clarity).
    """
    if df is None or df.empty:
        return 0

    table = model.__table__
    pk_cols = {c.name for c in table.primary_key.columns}
    all_cols = [c.name for c in table.columns]

    records = _clean_records(df, all_cols)
    if not records:
        return 0

    # Drop records where any PK column is None - MySQL rejects NULL PKs.
    records = [r for r in records if all(r.get(pk) is not None for pk in pk_cols)]
    if not records:
        return 0

    if update_keys is None:
        update_cols = [c for c in all_cols if c not in pk_cols and c != "updated_at"]
    else:
        update_cols = list(update_keys)

    def _do(sess: Session) -> int:
        total = 0
        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            stmt = mysql_insert(table).values(chunk)
            update_map = {col: stmt.inserted[col] for col in update_cols}
            stmt = stmt.on_duplicate_key_update(**update_map)
            sess.execute(stmt)
            total += len(chunk)
        return total

    if session is not None:
        n = _do(session)
    else:
        with session_scope() as sess:
            n = _do(sess)
    log.debug(f"Upsert {n} rows into {table.name}")
    return n
