"""Persistence helpers for the LLM-mined factor library."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Any

from sqlalchemy import select

from quantify.database.engine import get_engine, session_scope
from quantify.database.models import FactorLibrary


@dataclass
class FactorRecord:
    """A factor row, decoupled from the SQLAlchemy session lifecycle."""

    name: str
    expression: str
    hypothesis: str | None = None
    category: str | None = None
    universe: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    periods: str | None = None
    ic_mean: float | None = None
    ic_std: float | None = None
    icir: float | None = None
    ic_tstat: float | None = None
    rank_ic_mean: float | None = None
    rank_icir: float | None = None
    quantile_spread: float | None = None
    turnover: float | None = None
    coverage: float | None = None
    status: str = "passed"
    iteration: int | None = None
    metrics_json: str | None = None
    report_path: str | None = None
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@lru_cache(maxsize=1)
def ensure_factor_table() -> None:
    """Create the factor_library table on demand (when used before `db init`)."""
    FactorLibrary.__table__.create(bind=get_engine(), checkfirst=True)


def _to_record(row: FactorLibrary) -> FactorRecord:
    return FactorRecord(
        id=row.id,
        name=row.name,
        expression=row.expression,
        hypothesis=row.hypothesis,
        category=row.category,
        universe=row.universe,
        start_date=row.start_date,
        end_date=row.end_date,
        periods=row.periods,
        ic_mean=row.ic_mean,
        ic_std=row.ic_std,
        icir=row.icir,
        ic_tstat=row.ic_tstat,
        rank_ic_mean=row.rank_ic_mean,
        rank_icir=row.rank_icir,
        quantile_spread=row.quantile_spread,
        turnover=row.turnover,
        coverage=row.coverage,
        status=row.status,
        iteration=row.iteration,
        metrics_json=row.metrics_json,
        report_path=row.report_path,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


_COLUMN_NAMES = {c.name for c in FactorLibrary.__table__.columns}


def list_factors(status: str | None = None) -> list[FactorRecord]:
    ensure_factor_table()
    with session_scope() as sess:
        stmt = select(FactorLibrary).order_by(FactorLibrary.icir.desc().nullslast(), FactorLibrary.id.desc())
        if status is not None:
            stmt = stmt.where(FactorLibrary.status == status)
        rows = sess.execute(stmt).scalars().all()
        return [_to_record(row) for row in rows]


def get_factor_by_name(name: str) -> FactorRecord | None:
    ensure_factor_table()
    with session_scope() as sess:
        row = sess.execute(select(FactorLibrary).where(FactorLibrary.name == name)).scalar_one_or_none()
        return _to_record(row) if row else None


def existing_expressions() -> list[str]:
    """All expressions already stored — used to avoid LLM proposing duplicates."""
    ensure_factor_table()
    with session_scope() as sess:
        return list(sess.execute(select(FactorLibrary.expression)).scalars().all())


def save_factor(record: FactorRecord) -> FactorRecord:
    """Upsert a factor by unique name. Returns the persisted record."""
    ensure_factor_table()
    name = record.name.strip()
    expression = record.expression.strip()
    if not name:
        raise ValueError("因子名称不能为空")
    if not expression:
        raise ValueError("因子表达式不能为空")

    payload: dict[str, Any] = {
        k: v
        for k, v in asdict(record).items()
        if k in _COLUMN_NAMES and k not in {"id", "created_at", "updated_at"}
    }
    payload["name"] = name
    payload["expression"] = expression

    with session_scope() as sess:
        row = sess.execute(select(FactorLibrary).where(FactorLibrary.name == name)).scalar_one_or_none()
        if row is None:
            row = FactorLibrary(**payload)
            sess.add(row)
        else:
            for key, value in payload.items():
                setattr(row, key, value)
        sess.flush()
        return _to_record(row)


def delete_factor(factor_id: int) -> bool:
    ensure_factor_table()
    with session_scope() as sess:
        row = sess.get(FactorLibrary, factor_id)
        if row is None:
            return False
        sess.delete(row)
        return True


def metrics_to_json(metrics: dict[str, Any]) -> str:
    """Serialize a metrics dict to a compact JSON string for storage."""

    def _default(value: Any) -> Any:
        if isinstance(value, date | datetime):
            return value.isoformat()
        return str(value)

    return json.dumps(metrics, ensure_ascii=False, default=_default)
