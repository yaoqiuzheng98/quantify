"""Export MySQL stock data into Qlib's ``.bin`` format and initialize Qlib.

Qlib's on-disk layout (mirrors ``scripts/dump_bin.py`` from the qlib repo) is::

    <provider_uri>/
        calendars/day.txt              # sorted unique trading days, %Y-%m-%d
        instruments/all.txt            # CODE\tSTART\tEND  (tab-separated, uppercase)
        features/<code_lower>/<field_lower>.day.bin

Each ``.bin`` is a little-endian float32 array whose **first element is the
index (into the global calendar) of the instrument's first date**, followed by
the field values reindexed onto the calendar slice ``[min_date, max_date]``
(missing days -> NaN). This is byte-for-byte what Qlib's reader expects.

We store **forward-adjusted** OHLC/vwap (raw * adj_factor / latest_adj_factor)
so factor research and forward returns are split/dividend consistent, plus raw
volume/amount, the adjustment ``factor`` and a few ``daily_basic`` valuation
fields (turn/pe/pb/ps/total_mv/circ_mv).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from quantify.config import get_settings
from quantify.database.engine import session_scope
from quantify.database.models import AdjFactor, DailyBasic, StockBasic, StockDaily
from quantify.utils.logger import log

# Qlib field name -> stored array. These are the columns the LLM may reference
# as ``$open`` / ``$close`` / ``$turn`` etc. in factor expressions.
QLIB_FIELDS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "factor",
    "turn",
    "pe",
    "pb",
    "ps",
    "total_mv",
    "circ_mv",
)

_FREQ = "day"
_BIN_SUFFIX = ".bin"
_INSTRUMENTS_SEP = "\t"
_DATE_FMT = "%Y-%m-%d"


@dataclass
class DumpSummary:
    instruments: int = 0
    calendar_days: int = 0
    fields: list[str] = field(default_factory=list)
    provider_uri: str = ""


def ts_code_to_qlib(ts_code: str) -> str:
    """``600000.SH`` -> ``SH600000`` (Qlib cn instrument convention)."""
    symbol, _, exchange = ts_code.partition(".")
    return f"{exchange.upper()}{symbol}"


def qlib_to_ts_code(instrument: str) -> str:
    """``SH600000`` -> ``600000.SH`` (inverse of :func:`ts_code_to_qlib`)."""
    exchange, symbol = instrument[:2].upper(), instrument[2:]
    return f"{symbol}.{exchange}"


def _provider_uri() -> Path:
    return Path(get_settings().qlib.provider_uri)


def _load_universe(ts_codes: list[str] | None) -> list[str]:
    if ts_codes:
        return list(dict.fromkeys(ts_codes))
    with session_scope() as sess:
        rows = sess.execute(select(StockBasic.ts_code)).scalars().all()
    return list(rows)


def _load_one(
    ts_code: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame | None:
    """Build the forward-adjusted, Qlib-field DataFrame for one stock."""
    with session_scope() as sess:
        daily_stmt = select(
            StockDaily.trade_date,
            StockDaily.open,
            StockDaily.high,
            StockDaily.low,
            StockDaily.close,
            StockDaily.vol,
            StockDaily.amount,
        ).where(StockDaily.ts_code == ts_code)
        adj_stmt = select(AdjFactor.trade_date, AdjFactor.adj_factor).where(AdjFactor.ts_code == ts_code)
        basic_stmt = select(
            DailyBasic.trade_date,
            DailyBasic.turnover_rate,
            DailyBasic.pe,
            DailyBasic.pb,
            DailyBasic.ps,
            DailyBasic.total_mv,
            DailyBasic.circ_mv,
        ).where(DailyBasic.ts_code == ts_code)
        if start_date:
            sd = pd.Timestamp(start_date).date()
            daily_stmt = daily_stmt.where(StockDaily.trade_date >= sd)
            adj_stmt = adj_stmt.where(AdjFactor.trade_date >= sd)
            basic_stmt = basic_stmt.where(DailyBasic.trade_date >= sd)
        if end_date:
            ed = pd.Timestamp(end_date).date()
            daily_stmt = daily_stmt.where(StockDaily.trade_date <= ed)
            adj_stmt = adj_stmt.where(AdjFactor.trade_date <= ed)
            basic_stmt = basic_stmt.where(DailyBasic.trade_date <= ed)

        daily = pd.DataFrame(sess.execute(daily_stmt).all())
        adj = pd.DataFrame(sess.execute(adj_stmt).all())
        basic = pd.DataFrame(sess.execute(basic_stmt).all())

    if daily.empty:
        return None
    daily.columns = ["trade_date", "open", "high", "low", "close", "vol", "amount"]
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values("trade_date").reset_index(drop=True)

    # forward-adjustment ratio: adj_factor / latest_adj_factor
    if not adj.empty:
        adj.columns = ["trade_date", "adj_factor"]
        adj["trade_date"] = pd.to_datetime(adj["trade_date"])
        daily = daily.merge(adj, on="trade_date", how="left")
        daily["adj_factor"] = daily["adj_factor"].ffill().bfill()
        latest = daily["adj_factor"].iloc[-1]
        ratio = daily["adj_factor"] / latest if latest and latest != 0 else 1.0
    else:
        ratio = pd.Series(1.0, index=daily.index)

    out = pd.DataFrame({"trade_date": daily["trade_date"]})
    for col in ("open", "high", "low", "close"):
        out[col] = daily[col].astype(float) * ratio
    out["volume"] = daily["vol"].astype(float)
    out["amount"] = daily["amount"].astype(float)
    # raw vwap (yuan/share) = amount(千元)*1000 / (vol手*100股) = amount*10/vol; then adjust.
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap_raw = (daily["amount"].astype(float) * 10.0) / daily["vol"].astype(float).replace(0, np.nan)
    out["vwap"] = vwap_raw * ratio
    out["factor"] = ratio.astype(float)

    if not basic.empty:
        basic.columns = ["trade_date", "turn", "pe", "pb", "ps", "total_mv", "circ_mv"]
        basic["trade_date"] = pd.to_datetime(basic["trade_date"])
        out = out.merge(basic, on="trade_date", how="left")
    else:
        for col in ("turn", "pe", "pb", "ps", "total_mv", "circ_mv"):
            out[col] = np.nan

    return out


def _write_bins(
    qlib_dir: Path,
    per_code: dict[str, pd.DataFrame],
    calendar: list[pd.Timestamp],
) -> None:
    from qlib.utils import code_to_fname

    features_dir = qlib_dir / "features"
    cal_index = {ts: i for i, ts in enumerate(calendar)}

    for instrument, df in per_code.items():
        df = df.drop_duplicates("trade_date").set_index("trade_date").sort_index()
        start, end = df.index.min(), df.index.max()
        # reindex onto the calendar slice within this instrument's own range
        cal_slice = [ts for ts in calendar if start <= ts <= end]
        df = df.reindex(cal_slice)
        date_index = cal_index[start]

        code_dir = features_dir / code_to_fname(instrument).lower()
        code_dir.mkdir(parents=True, exist_ok=True)
        for fld in QLIB_FIELDS:
            if fld not in df.columns:
                continue
            values = df[fld].to_numpy(dtype="float32")
            payload = np.hstack([np.array([date_index], dtype="float32"), values]).astype("<f4")
            (code_dir / f"{fld.lower()}.{_FREQ}{_BIN_SUFFIX}").write_bytes(payload.tobytes())


def _write_calendar_and_instruments(
    qlib_dir: Path,
    calendar: list[pd.Timestamp],
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> None:
    cal_dir = qlib_dir / "calendars"
    inst_dir = qlib_dir / "instruments"
    cal_dir.mkdir(parents=True, exist_ok=True)
    inst_dir.mkdir(parents=True, exist_ok=True)

    (cal_dir / f"{_FREQ}.txt").write_text(
        "\n".join(ts.strftime(_DATE_FMT) for ts in calendar) + "\n", encoding="utf-8"
    )
    lines = [
        _INSTRUMENTS_SEP.join([code, lo.strftime(_DATE_FMT), hi.strftime(_DATE_FMT)])
        for code, (lo, hi) in sorted(ranges.items())
    ]
    (inst_dir / "all.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dump_qlib_data(
    *,
    ts_codes: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    provider_uri: str | Path | None = None,
) -> DumpSummary:
    """Dump MySQL stock data into Qlib ``.bin`` format.

    Parameters
    ----------
    ts_codes:
        Restrict to these Tushare codes (e.g. ``["600000.SH"]``); ``None`` dumps
        every stock in ``stock_basic``.
    start_date / end_date:
        ``YYYY-MM-DD`` (or ``YYYYMMDD``) bounds; ``None`` means unbounded.
    provider_uri:
        Output directory; defaults to ``settings.qlib.provider_uri``.
    """
    qlib_dir = Path(provider_uri) if provider_uri else _provider_uri()
    qlib_dir.mkdir(parents=True, exist_ok=True)

    codes = _load_universe(ts_codes)
    log.info(f"Dumping {len(codes)} stocks to qlib dir: {qlib_dir}")

    per_code: dict[str, pd.DataFrame] = {}
    all_dates: set[pd.Timestamp] = set()
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}

    for i, ts_code in enumerate(codes, start=1):
        df = _load_one(ts_code, start_date, end_date)
        if df is None or df.empty:
            continue
        instrument = ts_code_to_qlib(ts_code)
        per_code[instrument] = df
        dates = pd.DatetimeIndex(df["trade_date"])
        all_dates.update(dates)
        ranges[instrument] = (dates.min(), dates.max())
        if i % 200 == 0:
            log.info(f"  loaded {i}/{len(codes)} stocks ...")

    if not per_code:
        log.warning("No stock data found to dump. Did you run `quantify fetch stock all`?")
        return DumpSummary(provider_uri=str(qlib_dir))

    calendar = sorted(all_dates)
    _write_bins(qlib_dir, per_code, calendar)
    _write_calendar_and_instruments(qlib_dir, calendar, ranges)

    log.info(f"Qlib dump done: {len(per_code)} instruments, {len(calendar)} calendar days -> {qlib_dir}")
    return DumpSummary(
        instruments=len(per_code),
        calendar_days=len(calendar),
        fields=list(QLIB_FIELDS),
        provider_uri=str(qlib_dir),
    )


_QLIB_INITIALIZED = False


def init_qlib(provider_uri: str | Path | None = None, *, force: bool = False) -> None:
    """Initialize Qlib against the dumped provider directory (idempotent)."""
    global _QLIB_INITIALIZED
    if _QLIB_INITIALIZED and not force:
        return
    import qlib

    settings = get_settings().qlib
    uri = str(provider_uri) if provider_uri else settings.provider_uri
    if not (Path(uri) / "calendars" / f"{_FREQ}.txt").exists():
        raise FileNotFoundError(f"Qlib data not found under {uri!r}. Run `quantify factor dump-data` first.")
    qlib.init(provider_uri=uri, region=settings.region, redirect_logger=False)
    _QLIB_INITIALIZED = True
    log.info(f"Qlib initialized: provider_uri={uri}, region={settings.region}")


def list_instruments(provider_uri: str | Path | None = None) -> list[str]:
    """Read instrument codes (Qlib format) from the dumped ``instruments/all.txt``."""
    uri = Path(provider_uri) if provider_uri else _provider_uri()
    path = uri / "instruments" / "all.txt"
    if not path.exists():
        return []
    codes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.split(_INSTRUMENTS_SEP, 1)[0].strip()
        if token:
            codes.append(token)
    return codes


def calendar_bounds(provider_uri: str | Path | None = None) -> tuple[date | None, date | None]:
    """Return (first, last) trading day available in the dumped calendar."""
    uri = Path(provider_uri) if provider_uri else _provider_uri()
    path = uri / "calendars" / f"{_FREQ}.txt"
    if not path.exists():
        return None, None
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None, None
    return pd.Timestamp(lines[0]).date(), pd.Timestamp(lines[-1]).date()
