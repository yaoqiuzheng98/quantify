"""Index-constituent lookup backed by the ``index_weight`` table.

This powers a JoinQuant-style ``get_index_stocks`` for the backtest engine:
point-in-time membership resolved from the most recent monthly ``index_weight``
snapshot on or before a given date. Codes are returned in **Tushare** format
(``600519.SH``); callers that need JoinQuant format convert via
``to_joinquant_code``.
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

from sqlalchemy import func, select

from quantify.database.engine import session_scope
from quantify.database.models import IndexWeight

from .codes import to_tushare_code


def _as_date(value: object) -> date | None:
    """Coerce ``datetime`` / ``date`` / ``YYYY-MM-DD`` / ``YYYYMMDD`` to ``date``."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Cannot parse date: {value!r}")


@lru_cache(maxsize=2048)
def _constituents_cached(index_code: str, as_of: date | None) -> tuple[str, ...]:
    """Members at the latest ``index_weight`` snapshot on or before ``as_of``.

    Cached because a backtest calls this repeatedly with dates that map to the
    same (roughly monthly) snapshot; constituents are static within a run.
    """
    with session_scope() as sess:
        snapshot_stmt = select(func.max(IndexWeight.trade_date)).where(IndexWeight.index_code == index_code)
        if as_of is not None:
            snapshot_stmt = snapshot_stmt.where(IndexWeight.trade_date <= as_of)
        snapshot = sess.execute(snapshot_stmt).scalar_one_or_none()
        if snapshot is None:
            return ()
        rows = (
            sess.execute(
                select(IndexWeight.con_code)
                .where(IndexWeight.index_code == index_code)
                .where(IndexWeight.trade_date == snapshot)
            )
            .scalars()
            .all()
        )
    return tuple(sorted({code for code in rows if code}))


def index_constituents(index_code: str, as_of: object = None) -> list[str]:
    """Point-in-time index members (Tushare codes) as of ``as_of``.

    ``as_of`` may be a ``date``/``datetime``/string; ``None`` returns the most
    recent snapshot available. Returns an empty list if the index is unknown.
    """
    return list(_constituents_cached(to_tushare_code(index_code), _as_date(as_of)))


def index_constituents_union(index_code: str, start: object, end: object = None) -> list[str]:
    """Union of members (Tushare codes) active across ``[start, end]``.

    Includes the snapshot active at ``start`` (the latest one on or before it),
    so the universe also covers names that drop out of the index mid-window —
    avoiding survivorship bias when this is used to preload backtest data.
    """
    code = to_tushare_code(index_code)
    start_d = _as_date(start)
    end_d = _as_date(end)
    with session_scope() as sess:
        start_snapshot = None
        if start_d is not None:
            start_snapshot = sess.execute(
                select(func.max(IndexWeight.trade_date))
                .where(IndexWeight.index_code == code)
                .where(IndexWeight.trade_date <= start_d)
            ).scalar_one_or_none()
        stmt = select(IndexWeight.con_code).where(IndexWeight.index_code == code).distinct()
        lower = start_snapshot or start_d
        if lower is not None:
            stmt = stmt.where(IndexWeight.trade_date >= lower)
        if end_d is not None:
            stmt = stmt.where(IndexWeight.trade_date <= end_d)
        rows = sess.execute(stmt).scalars().all()
    return sorted({code for code in rows if code})
