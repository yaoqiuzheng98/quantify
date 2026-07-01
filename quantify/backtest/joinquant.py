"""Small JoinQuant compatibility layer for strategy execution."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

import pandas as pd

from quantify.utils.logger import log

from .broker import make_commission, make_slippage
from .codes import to_joinquant_code, to_tushare_code
from .universe import index_constituents


# ---------------------------------------------------------------------------
# JoinQuant-style query() + valuation table for get_fundamentals()
# ---------------------------------------------------------------------------


class _ValuationField:
    """A column descriptor on the JoinQuant ``valuation`` table.

    Holds the JQ field name and the corresponding Tushare ``daily_basic`` column
    name + optional transform (e.g. 万元→亿元).
    """

    # Note: no __slots__ — we need to attach .desc/.asc/.in_ as methods
    jq_name: str
    ts_column: str
    transform: "Callable[[float], float] | None"

    def __init__(self, jq_name: str, ts_column: str, transform: "Callable[[float], float] | None" = None):
        self.jq_name = jq_name
        self.ts_column = ts_column
        self.transform = transform

    def __repr__(self) -> str:
        return f"valuation.{self.jq_name}"


def _wan_to_yi(v: float) -> float:
    """万元 → 亿元 (JoinQuant market_cap 单位是亿元, Tushare total_mv 单位是万元)."""
    return v / 1e4 if v is not None else None


def _wan_gu_to_gu(v: float) -> float:
    """万股 → 股 (JoinQuant capitalization 单位是股, Tushare total_share 单位是万股)."""
    return v * 1e4 if v is not None else None


class _ValuationTable:
    """Mimics JoinQuant's ``valuation`` ORM table object.

    Usage in strategy code (identical to JoinQuant)::

        q = query(valuation.code, valuation.pb_ratio).filter(valuation.code == '000001.XSHE')
        df = get_fundamentals(q, date='2024-01-05')
    """

    code = _ValuationField("code", "ts_code")
    day = _ValuationField("day", "trade_date")
    pe_ratio = _ValuationField("pe_ratio", "pe_ttm")
    pe_ratio_lyr = _ValuationField("pe_ratio_lyr", "pe")
    pb_ratio = _ValuationField("pb_ratio", "pb")
    ps_ratio = _ValuationField("ps_ratio", "ps_ttm")
    pcf_ratio = _ValuationField("pcf_ratio", "dv_ratio")
    turnover_ratio = _ValuationField("turnover_ratio", "turnover_rate")
    market_cap = _ValuationField("market_cap", "total_mv", _wan_to_yi)
    circulating_market_cap = _ValuationField("circulating_market_cap", "circ_mv", _wan_to_yi)
    capitalization = _ValuationField("capitalization", "total_share", _wan_gu_to_gu)
    circulating_cap = _ValuationField("circulating_cap", "float_share", _wan_gu_to_gu)

    # Map JQ field name → _ValuationField for dynamic lookup
    _FIELDS: dict[str, _ValuationField] = {}

    def __init__(self) -> None:
        # Allow `valuation` to be both a class-like (valuation.code) and instance
        pass

    @classmethod
    def _init_fields_map(cls) -> None:
        if not cls._FIELDS:
            for name in dir(cls):
                val = getattr(cls, name)
                if isinstance(val, _ValuationField):
                    cls._FIELDS[name] = val

    @classmethod
    def get_field(cls, jq_name: str) -> _ValuationField | None:
        cls._init_fields_map()
        return cls._FIELDS.get(jq_name)

    @classmethod
    def all_fields(cls) -> dict[str, _ValuationField]:
        cls._init_fields_map()
        return cls._FIELDS


# Singleton instance — strategies use `valuation.code` etc.
valuation = _ValuationTable()


class _FundamentalsQuery:
    """A parsed ``query(valuation.field1, valuation.field2, ...).filter(...).order_by(...).limit(...)`` call.

    We don't use real SQLAlchemy ORM; instead we parse the query object's
    attributes to know which fields to select, which filters to apply, and
    how many rows to return.
    """

    def __init__(self) -> None:
        self.select_fields: list[_ValuationField] = []
        self.filters: list[tuple[str, Any]] = []  # (jq_name, op, value)
        self._order_by: tuple[_ValuationField, str] | None = None  # (field, "asc"/"desc")
        self._limit_n: int | None = None
        # If select_fields is empty, select all
        self.select_all = False


def query(*args: Any) -> _FundamentalsQuery:
    """JoinQuant-style ``query()`` builder for ``get_fundamentals``.

    Supports:
    - ``query(valuation)`` — select all valuation fields
    - ``query(valuation.code, valuation.pb_ratio)`` — select specific fields
    - ``.filter(valuation.code == '000001.XSHE')`` — equality filter on code
    - ``.filter(valuation.code.in_([...]))`` — IN filter on code
    - ``.filter(valuation.market_cap > 1000)`` — comparison filter on numeric fields
    - ``.order_by(valuation.market_cap.desc())`` — order by a field
    - ``.limit(100)`` — limit number of rows

    Returns a ``_FundamentalsQuery`` that ``get_fundamentals`` interprets.
    """
    q = _FundamentalsQuery()

    for arg in args:
        if isinstance(arg, _ValuationTable):
            q.select_all = True
        elif isinstance(arg, _ValuationField):
            q.select_fields.append(arg)

    return q


# Add builder methods to _FundamentalsQuery via monkey-patch so that
# the returned object from query() supports .filter().order_by().limit()
# chaining, matching JoinQuant's API.


def _query_filter(self: _FundamentalsQuery, *conditions: Any) -> _FundamentalsQuery:
    for cond in conditions:
        parsed = _parse_filter_condition(cond)
        if parsed:
            self.filters.append(parsed)
    return self


def _query_order_by(self: _FundamentalsQuery, *args: Any) -> _FundamentalsQuery:
    if args:
        field, direction = _parse_order_arg(args[0])
        if field:
            self._order_by = (field, direction)
    return self


def _query_limit(self: _FundamentalsQuery, n: int) -> _FundamentalsQuery:
    self._limit_n = n
    return self


def _parse_filter_condition(cond: Any) -> tuple[str, Any] | None:
    """Parse a filter condition like ``valuation.code == '000001.XSHE'``.

    We rely on Python's operator overloading: ``_ValuationField.__eq__`` etc.
    return a tuple describing the comparison.
    """
    if isinstance(cond, tuple) and len(cond) == 3:
        return cond  # already parsed (jq_name, op, value)
    return None


def _parse_order_arg(arg: Any) -> tuple[_ValuationField | None, str]:
    """Parse an order_by argument like ``valuation.market_cap.desc()``."""
    if isinstance(arg, tuple) and len(arg) == 2:
        return arg
    return None, "asc"


# Operator overloading on _ValuationField so that ``valuation.code == 'x'``
# produces a parseable tuple instead of a bool.
def _field_eq(self: _ValuationField, other: Any) -> tuple[str, str, Any]:
    return (self.jq_name, "==", other)


def _field_gt(self: _ValuationField, other: Any) -> tuple[str, str, Any]:
    return (self.jq_name, ">", other)


def _field_lt(self: _ValuationField, other: Any) -> tuple[str, str, Any]:
    return (self.jq_name, "<", other)


def _field_ge(self: _ValuationField, other: Any) -> tuple[str, str, Any]:
    return (self.jq_name, ">=", other)


def _field_le(self: _ValuationField, other: Any) -> tuple[str, str, Any]:
    return (self.jq_name, "<=", other)


def _field_in(self: _ValuationField, values: list[str]) -> tuple[str, str, list]:
    return (self.jq_name, "in", values)


def _field_desc(self: _ValuationField) -> tuple[_ValuationField, str]:
    return (self, "desc")


def _field_asc(self: _ValuationField) -> tuple[_ValuationField, str]:
    return (self, "asc")


# Attach operators
_ValuationField.__eq__ = _field_eq  # type: ignore[assignment]
_ValuationField.__gt__ = _field_gt  # type: ignore[assignment]
_ValuationField.__lt__ = _field_lt  # type: ignore[assignment]
_ValuationField.__ge__ = _field_ge  # type: ignore[assignment]
_ValuationField.__le__ = _field_le  # type: ignore[assignment]
_ValuationField.in_ = _field_in  # type: ignore[attr-defined]
_ValuationField.desc = _field_desc  # type: ignore[attr-defined]
_ValuationField.asc = _field_asc  # type: ignore[attr-defined]

# Attach builder methods
_FundamentalsQuery.filter = _query_filter  # type: ignore[attr-defined]
_FundamentalsQuery.order_by = _query_order_by  # type: ignore[attr-defined]
_FundamentalsQuery.limit = _query_limit  # type: ignore[attr-defined]


def _sw_industry_map(ts_codes: list[str], as_of: Any = None) -> dict[str, str]:
    """Return {joinquant_code: sw_l1_name} for *ts_codes* (tushare format).

    Uses the ``index_member_all`` table (SW 2021 classification).  When
    *as_of* is given, only memberships active on that date are considered
    (``in_date <= as_of`` and (``out_date`` is null or ``out_date > as_of``)).
    Otherwise the latest ``is_new='Y'`` membership is used.
    """
    from sqlalchemy import text as sa_text

    from quantify.database.engine import session_scope

    if not ts_codes:
        return {}

    codes_str = "','".join(ts_codes)
    if as_of is not None:
        query = sa_text(
            f"""
            SELECT ts_code, l1_name
            FROM index_member_all
            WHERE ts_code IN ('{codes_str}')
              AND in_date <= :asof
              AND (out_date IS NULL OR out_date > :asof)
            """
        )
        params = {"asof": as_of}
    else:
        query = sa_text(
            f"""
            SELECT ts_code, l1_name
            FROM index_member_all
            WHERE ts_code IN ('{codes_str}')
              AND is_new = 'Y'
              AND out_date IS NULL
            """
        )
        params = {}

    mapping: dict[str, str] = {}
    with session_scope() as sess:
        rows = sess.execute(query, params).fetchall()
    for ts_code, l1_name in rows:
        mapping[to_joinquant_code(ts_code)] = l1_name
    return mapping


@dataclass
class OrderCost:
    open_tax: float = 0.0
    close_tax: float = 0.0
    open_commission: float = 0.0
    close_commission: float = 0.0
    close_today_commission: float = 0.0
    min_commission: float = 0.0


@dataclass
class PriceRelatedSlippage:
    rate: float = 0.0


class JoinQuantCompat:
    """Expose common JoinQuant globals against the local Context object."""

    def __init__(self) -> None:
        self.context: Any | None = None
        # 已注册的调度任务，每项为 (func, freq, day)：
        #   freq="daily"   day 忽略，每个交易日触发
        #   freq="weekly"  day=weekday，第 N 个交易日(每周)，负数表示倒数
        #   freq="monthly" day=monthday，第 N 个交易日(每月)，负数表示倒数
        self.scheduled: list[tuple[Callable, str, int]] = []
        self.options: dict[str, Any] = {}

    @property
    def daily_functions(self) -> list[Callable]:
        """Backward-compatible view of registered daily tasks."""
        return [func for func, freq, _day in self.scheduled if freq == "daily"]

    def bind(self, context: Any) -> None:
        self.context = context

    def _require_context(self) -> Any:
        if self.context is None:
            raise RuntimeError("JoinQuant API called before context is bound")
        return self.context

    def set_option(self, key: str, value: Any) -> None:
        self.options[key] = value

    def set_benchmark(self, security: str) -> None:
        self._require_context().set_benchmark(security)

    def run_daily(self, func: Callable, time: str = "open", **_kwargs: Any) -> None:
        if time not in ("open", "every_bar"):
            raise NotImplementedError("Local engine currently supports run_daily(..., time='open') only")
        self.scheduled.append((func, "daily", 0))

    def run_weekly(self, func: Callable, weekday: int = 1, time: str = "open", **_kwargs: Any) -> None:
        if time not in ("open", "every_bar"):
            raise NotImplementedError("Local engine currently supports time='open' only")
        self.scheduled.append((func, "weekly", weekday))

    def run_monthly(self, func: Callable, monthday: int = 1, time: str = "open", **_kwargs: Any) -> None:
        if time not in ("open", "every_bar"):
            raise NotImplementedError("Local engine currently supports time='open' only")
        self.scheduled.append((func, "monthly", monthday))

    def attribute_history(
        self,
        security: str,
        count: int,
        unit: str = "1d",
        fields: str | list[str] | tuple[str, ...] = "close",
        skip_paused: bool = True,
        df: bool = True,
        **_kwargs: Any,
    ) -> pd.DataFrame | dict[str, list[float]]:
        del skip_paused
        if unit != "1d":
            raise NotImplementedError("Local engine currently supports 1d history only")

        context = self._require_context()
        field_names = [fields] if isinstance(fields, str) else list(fields)
        data = {
            field: context.data.history(to_tushare_code(security), count=count, field=field)
            for field in field_names
        }
        return pd.DataFrame(data) if df else data

    def get_all_securities(self, types: str = "etf", date: Any = None) -> list[str]:
        """JoinQuant-style ``get_all_securities``, returns all securities of the given type.

        Parameters
        ----------
        types:
            Asset class: ``"etf"`` (default).  Only ETF is supported for now.
        date:
            Ignored for now (returns all codes that have data in the local DB).

        Returns
        -------
        list[str]
            Codes in JoinQuant format (``.XSHG`` / ``.XSHE``).
        """
        if types != "etf":
            raise NotImplementedError(f"get_all_securities only supports 'etf', got {types!r}")
        from quantify.database.engine import session_scope
        from sqlalchemy import text as sa_text

        sql = """
            SELECT d.ts_code
            FROM fund_basic b
            JOIN (
                SELECT ts_code,
                       COUNT(*)               AS n,
                       AVG(amount)            AS avg_amt
                FROM fund_daily
                GROUP BY ts_code
            ) d ON d.ts_code = b.ts_code
            WHERE b.fund_type = '股票型'
              AND b.status    = 'L'
              AND d.n         >= 250
              AND d.avg_amt   >= 5000
            ORDER BY d.avg_amt DESC
        """
        with session_scope() as sess:
            rows = sess.execute(sa_text(sql)).fetchall()
        return [to_joinquant_code(r[0]) for r in rows]

    def get_index_stocks(self, index_symbol: str, date: Any = None) -> list[str]:
        """JoinQuant-style index membership, backed by the ``index_weight`` table.

        Returns constituent codes (JoinQuant format ``.XSHG``/``.XSHE``) for
        ``index_symbol`` as of ``date``. When ``date`` is ``None`` the current
        backtest date (``context.current_dt``) is used, matching JoinQuant's
        point-in-time semantics; the latest available snapshot is used as a last
        resort. Resolution falls back to the most recent monthly snapshot on or
        before the requested date.
        """
        as_of = date
        if as_of is None and self.context is not None:
            as_of = getattr(self.context, "current_dt", None)
        return [to_joinquant_code(code) for code in index_constituents(index_symbol, as_of)]

    def get_industry(
        self,
        securities: list[str] | str,
        date: Any = None,
        level: str = "sw_l1",
    ) -> dict[str, dict[str, str]]:
        """JoinQuant-style ``get_industry`` backed by the SW classification table.

        Returns ``{code: {"sw_l1": industry_name, "industry": industry_name}}``
        for each requested security (JoinQuant format).  Only SW L1 is
        supported; *level* is accepted for API compatibility but ignored.
        """
        if isinstance(securities, str):
            securities = [securities]
        ts_codes = [to_tushare_code(s) for s in securities]
        as_of = date
        if as_of is None and self.context is not None:
            as_of = getattr(self.context, "current_dt", None)
        mapping = _sw_industry_map(ts_codes, as_of=as_of)
        result: dict[str, dict[str, str]] = {}
        for jq_code in securities:
            ind = mapping.get(jq_code, "未知")
            result[jq_code] = {"sw_l1": ind, "industry": ind}
        return result

    def get_fundamentals(
        self,
        query_obj: _FundamentalsQuery,
        date: Any = None,
    ) -> pd.DataFrame:
        """JoinQuant-style ``get_fundamentals`` backed by the ``daily_basic`` table.

        Queries valuation data (PE/PB/PS/turnover/market_cap etc.) from MySQL
        ``daily_basic`` for the given date. If *date* is None, uses the current
        backtest date (``context.current_dt``).

        Supports:
        - Field selection: ``query(valuation.code, valuation.pb_ratio)``
        - Filter on code: ``.filter(valuation.code == '000001.XSHE')``
        - Filter on code list: ``.filter(valuation.code.in_([...]))``
        - Filter on numeric fields: ``.filter(valuation.market_cap > 1000)``
        - Order by: ``.order_by(valuation.market_cap.desc())``
        - Limit: ``.limit(100)``
        """
        from sqlalchemy import text as sa_text

        from quantify.database.engine import session_scope

        # Resolve date
        if date is None:
            date = getattr(self._require_context(), "current_dt", None)
        if date is None:
            raise RuntimeError("get_fundamentals: date is required (no backtest context)")

        # Normalize date to YYYY-MM-DD string
        if hasattr(date, "strftime"):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = str(date)

        # Determine which fields to select
        if query_obj.select_all or not query_obj.select_fields:
            fields_map = _ValuationTable.all_fields()
            select_fields = list(fields_map.values())
        else:
            select_fields = query_obj.select_fields

        # Build SQL: select ts_code, trade_date, + all needed columns
        ts_columns = set()
        for f in select_fields:
            ts_columns.add(f.ts_column)
        # Also need ts_code and trade_date for filtering/transform
        ts_columns.add("ts_code")
        ts_columns.add("trade_date")

        col_list = ", ".join(sorted(ts_columns))
        sql = f"SELECT {col_list} FROM daily_basic WHERE trade_date = :dt"
        params: dict[str, Any] = {"dt": date_str}

        # Apply filters
        for jq_name, op, value in query_obj.filters:
            field = _ValuationTable.get_field(jq_name)
            if field is None:
                continue
            if jq_name == "code":
                # Convert JQ codes to Tushare codes for filtering
                if op == "==":
                    ts_code = to_tushare_code(value)
                    sql += f" AND ts_code = :f_{jq_name}"
                    params[f"f_{jq_name}"] = ts_code
                elif op == "in":
                    ts_codes = [to_tushare_code(v) for v in value]
                    placeholders = ", ".join(f":f_{jq_name}_{i}" for i in range(len(ts_codes)))
                    sql += f" AND ts_code IN ({placeholders})"
                    for i, tc in enumerate(ts_codes):
                        params[f"f_{jq_name}_{i}"] = tc
            else:
                # Numeric filter on a valuation field
                ts_col = field.ts_column
                if op in ("==", ">", "<", ">=", "<="):
                    sql += f" AND {ts_col} {op} :f_{jq_name}"
                    params[f"f_{jq_name}"] = value

        # Order by
        if query_obj._order_by:
            field, direction = query_obj._order_by
            sql += f" ORDER BY {field.ts_column} {'DESC' if direction == 'desc' else 'ASC'}"

        # Limit
        if query_obj._limit_n:
            sql += f" LIMIT {int(query_obj._limit_n)}"

        # Execute
        with session_scope() as sess:
            rows = sess.execute(sa_text(sql), params).fetchall()

        if not rows:
            return pd.DataFrame(columns=[f.jq_name for f in select_fields])

        # Build result DataFrame
        result_data: dict[str, list] = {}
        for f in select_fields:
            col_values = []
            for row in rows:
                raw = getattr(row, f.ts_column, None)
                if f.transform and raw is not None:
                    raw = f.transform(raw)
                if f.jq_name == "code":
                    raw = to_joinquant_code(raw) if raw else raw
                col_values.append(raw)
            result_data[f.jq_name] = col_values

        return pd.DataFrame(result_data)

    def order(self, security: str, amount: int):
        return self._require_context().order(security, amount)

    def order_value(self, security: str, value: float):
        return self._require_context().order_value(security, value)

    def order_target_value(self, security: str, value: float):
        return self._require_context().order_target_value(security, value)

    def order_target_percent(self, security: str, percent: float):
        return self._require_context().order_target_percent(security, percent)

    def set_order_cost(self, order_cost: OrderCost, type: str = "stock") -> None:  # noqa: A002
        del type
        rate = max(order_cost.open_commission, order_cost.close_commission)
        broker = self._require_context()._broker  # noqa: SLF001
        broker.set_commission_fn(make_commission(rate=rate, minimum=order_cost.min_commission))
        # 聚宽 OrderCost.close_tax 即卖出印花税率；透传给 broker(仅对股票生效)。
        broker.set_stamp_duty_rate(order_cost.close_tax)

    def set_slippage(self, slippage: PriceRelatedSlippage) -> None:
        self._require_context()._broker.set_slippage_fn(make_slippage(slippage.rate))  # noqa: SLF001

    def namespace(self) -> dict[str, Any]:
        return {
            "log": log,
            "OrderCost": OrderCost,
            "PriceRelatedSlippage": PriceRelatedSlippage,
            "set_option": self.set_option,
            "set_benchmark": self.set_benchmark,
            "set_order_cost": self.set_order_cost,
            "set_slippage": self.set_slippage,
            "run_daily": self.run_daily,
            "run_weekly": self.run_weekly,
            "run_monthly": self.run_monthly,
            "attribute_history": self.attribute_history,
            "get_all_securities": self.get_all_securities,
            "get_index_stocks": self.get_index_stocks,
            "get_industry": self.get_industry,
            "get_fundamentals": self.get_fundamentals,
            "query": query,
            "valuation": valuation,
            "order": self.order,
            "order_value": self.order_value,
            "order_target_value": self.order_target_value,
            "order_target_percent": self.order_target_percent,
        }


def make_jqdata_module(compat: JoinQuantCompat) -> ModuleType:
    module = ModuleType("jqdata")
    namespace = compat.namespace()
    module.__dict__.update(namespace)
    module.__all__ = list(namespace.keys())
    return module
