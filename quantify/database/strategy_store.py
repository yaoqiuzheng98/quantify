"""Persistence helpers for saved backtest strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

from sqlalchemy import select

from quantify.database.engine import get_engine, session_scope
from quantify.database.models import SavedStrategy


@dataclass(frozen=True)
class StrategyRecord:
    id: int
    name: str
    description: str | None
    source: str
    created_at: datetime | None
    updated_at: datetime | None


@lru_cache(maxsize=1)
def ensure_strategy_table() -> None:
    """Create the strategy table when the dashboard is used before `db init`."""
    SavedStrategy.__table__.create(bind=get_engine(), checkfirst=True)


def _to_record(row: SavedStrategy) -> StrategyRecord:
    return StrategyRecord(
        id=row.id,
        name=row.name,
        description=row.description,
        source=row.source,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def list_strategies() -> list[StrategyRecord]:
    ensure_strategy_table()
    with session_scope() as sess:
        rows = (
            sess.execute(
                select(SavedStrategy).order_by(SavedStrategy.updated_at.desc(), SavedStrategy.id.desc())
            )
            .scalars()
            .all()
        )
        return [_to_record(row) for row in rows]


def save_strategy(
    *,
    name: str,
    source: str,
    description: str | None = None,
    strategy_id: int | None = None,
) -> StrategyRecord:
    ensure_strategy_table()
    name = name.strip()
    source = source.strip()
    description = description.strip() if description else None
    if not name:
        raise ValueError("策略名称不能为空")
    if not source:
        raise ValueError("策略代码不能为空")

    with session_scope() as sess:
        row = sess.get(SavedStrategy, strategy_id) if strategy_id is not None else None
        if row is None:
            row = sess.execute(select(SavedStrategy).where(SavedStrategy.name == name)).scalar_one_or_none()
        if row is None:
            row = SavedStrategy(name=name, description=description, source=source)
            sess.add(row)
        else:
            row.name = name
            row.description = description
            row.source = source
        sess.flush()
        return _to_record(row)


def delete_strategy(strategy_id: int) -> bool:
    """按 ID 删除策略,返回是否真的删除了记录(不存在则返回 False)。"""
    ensure_strategy_table()
    with session_scope() as sess:
        row = sess.get(SavedStrategy, strategy_id)
        if row is None:
            return False
        sess.delete(row)
        return True
