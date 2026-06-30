"""Typer CLI entry point.

Subcommands implemented in this milestone:
    quantify db init                  # create database + tables
    quantify db drop                  # drop all known tables (DANGEROUS)
    quantify fetch etf [stage]        # pull ETF data from Tushare
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from quantify.utils.logger import log

app = typer.Typer(help="Quantify CLI", no_args_is_help=True)
db_app = typer.Typer(help="Database management")
fetch_app = typer.Typer(help="Data fetching tasks")
factor_app = typer.Typer(help="LLM factor mining (Qlib + Alphalens)", no_args_is_help=True)
ml_app = typer.Typer(help="ML/DL factor mining (sklearn + XGBoost + PyTorch)", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(fetch_app, name="fetch")
app.add_typer(factor_app, name="factor")
app.add_typer(ml_app, name="ml")


@app.command("dashboard")
def dashboard(
    port: int = typer.Option(8501, "--port", help="Streamlit server port"),
    address: str = typer.Option("localhost", "--address", help="Streamlit server address"),
) -> None:
    """Launch the Streamlit backtest dashboard."""
    import os
    import socket
    import sys

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

    try:
        from streamlit.web import cli as streamlit_cli
    except ModuleNotFoundError:
        typer.echo("Streamlit is not installed. Run `pip install -e '.[web]'` first.", err=True)
        raise typer.Exit(code=1) from None

    actual_port = port
    for candidate_port in range(port, port + 100):
        try:
            with socket.create_server((address, candidate_port), reuse_port=False):
                actual_port = candidate_port
                break
        except OSError:
            continue
    else:
        typer.echo(f"No free port found from {port} to {port + 99}.", err=True)
        raise typer.Exit(code=1)

    if actual_port != port:
        typer.echo(f"Port {port} is busy; using {actual_port} instead.")

    app_path = Path(__file__).resolve().parent / "webapp" / "app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(actual_port),
        "--server.address",
        address,
        "--server.headless",
        "true",
    ]
    streamlit_cli.main()


# ---------------------------------------------------------------------------
# db
# ---------------------------------------------------------------------------
@db_app.command("init")
def db_init() -> None:
    """Create database (if needed) and all tables."""
    from quantify.database.init_db import init_db

    init_db(drop_first=False)


@db_app.command("drop")
def db_drop(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Drop all tables managed by SQLAlchemy metadata."""
    from quantify.database.init_db import init_db

    if not yes:
        typer.confirm("Drop all known tables? This cannot be undone.", abort=True)
    init_db(drop_first=True)


# ---------------------------------------------------------------------------
# fetch etf
# ---------------------------------------------------------------------------
@fetch_app.command("etf")
def fetch_etf(
    stage: str = typer.Argument(
        "all",
        help="Stage to run: all|basic|etf-index-basic|daily|nav|adj|dividend|share|share-size|portfolio|manager",
    ),
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="Incremental update vs full backfill"
    ),
    ts_code: Optional[str] = typer.Option(
        None, "--ts-code", help="Comma-separated ts_codes (default: all in fund_basic)"
    ),
    skip: Optional[str] = typer.Option(
        None, "--skip", help="Comma-separated stages to skip (only used with stage=all)"
    ),
) -> None:
    """Fetch ETF data from Tushare into MySQL."""
    from quantify.fetcher.etf import EtfFetcher

    codes = [c.strip() for c in ts_code.split(",")] if ts_code else None
    skip_set = {s.strip() for s in skip.split(",")} if skip else None

    fetcher = EtfFetcher()

    if stage == "all":
        fetcher.fetch_all(incremental=incremental, ts_codes=codes, skip=skip_set)
        return

    if stage == "basic":
        fetcher.fetch_basic()
        return

    if stage in ("etf-index-basic", "etf_index_basic"):
        fetcher.fetch_etf_index_basic()
        return

    universe = codes if codes else fetcher._load_universe()  # noqa: SLF001
    if not universe:
        log.warning("ETF universe is empty - run `quantify fetch etf basic` first.")
        raise typer.Exit(code=1)

    dispatch = {
        "daily": fetcher.fetch_daily,
        "nav": fetcher.fetch_nav,
        "adj": fetcher.fetch_adj,
        "dividend": fetcher.fetch_dividend,
        "share": fetcher.fetch_share,
        "share-size": fetcher.fetch_share_size,
        "share_size": fetcher.fetch_share_size,
        "portfolio": fetcher.fetch_portfolio,
        "manager": fetcher.fetch_manager,
    }
    if stage not in dispatch:
        raise typer.BadParameter(f"Unknown stage: {stage}")
    dispatch[stage](ts_codes=universe, incremental=incremental)


# ---------------------------------------------------------------------------
# fetch industry
# ---------------------------------------------------------------------------
@fetch_app.command("industry")
def fetch_industry(
    stage: str = typer.Argument(
        "all",
        help="Stage to run: all|trade-cal|sw-classify|sw-member|sw-daily|ci-member|ci-daily",
    ),
    provider: str = typer.Option("sw", "--provider", help="Provider for stage=all: sw|ci|all"),
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="Incremental update vs full backfill for daily stages"
    ),
    index_code: Optional[str] = typer.Option(
        None, "--index-code", help="Comma-separated industry index codes for daily stages"
    ),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="Start date, e.g. 20200101"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="End date, e.g. 20260608"),
    sw_src: str = typer.Option("SW2021", "--sw-src", help="SW classification source, e.g. SW2021"),
    latest: bool = typer.Option(True, "--latest/--all-history", help="Fetch latest industry members only"),
    exchange: str = typer.Option("SSE", "--exchange", help="Exchange for trade-cal stage, e.g. SSE"),
) -> None:
    """Fetch SW/CITIC industry classification, members and daily quotes from Tushare."""
    from quantify.fetcher.industry import IndustryFetcher

    normalized_stage = stage.replace("-", "_").lower()
    codes = [code.strip() for code in index_code.split(",")] if index_code else None
    fetcher = IndustryFetcher()

    if normalized_stage == "all":
        fetcher.fetch_all(
            provider=provider,
            incremental=incremental,
            start_date=start_date,
            end_date=end_date,
            sw_src=sw_src,
        )
        return

    dispatch = {
        "trade_cal": lambda: fetcher.fetch_trade_cal(
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
        ),
        "sw_classify": lambda: fetcher.fetch_sw_classify(src=sw_src),
        "sw_member": lambda: fetcher.fetch_sw_member(src=sw_src, latest=latest),
        "sw_daily": lambda: fetcher.fetch_sw_daily(
            index_codes=codes,
            src=sw_src,
            incremental=incremental,
            start_date=start_date,
            end_date=end_date,
        ),
        "ci_member": lambda: fetcher.fetch_ci_member(latest=latest),
        "ci_daily": lambda: fetcher.fetch_ci_daily(
            index_codes=codes,
            incremental=incremental,
            start_date=start_date,
            end_date=end_date,
        ),
    }
    if normalized_stage not in dispatch:
        raise typer.BadParameter(f"Unknown stage: {stage}")
    dispatch[normalized_stage]()


# ---------------------------------------------------------------------------
# fetch index
# ---------------------------------------------------------------------------
@fetch_app.command("index")
def fetch_index(
    stage: str = typer.Argument(
        "all",
        help="Stage: all|index-basic|index-daily|index-dailybasic|index-weight|moneyflow-ind-dc",
    ),
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="Incremental update vs full backfill"
    ),
    ts_code: Optional[str] = typer.Option(
        None, "--ts-code", help="Comma-separated index codes (daily/weight/dailybasic stages)"
    ),
    market: Optional[str] = typer.Option(
        None, "--market", help="Comma-separated index markets, e.g. SSE,SZSE,CSI,SW"
    ),
    all_index: bool = typer.Option(
        False,
        "--all-index",
        help="Fetch ALL indices in index_basic (default: only ETF-tracked indices)",
    ),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="Start date, e.g. 20200101"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="End date, e.g. 20260608"),
    skip: Optional[str] = typer.Option(
        None, "--skip", help="Comma-separated stages to skip (only used with stage=all)"
    ),
) -> None:
    """Fetch Tushare index-theme datasets into MySQL.

    By default index_daily/index_weight only fetch indices actually tracked by
    ETFs (etf_basic.index_code, a few hundred). Use --all-index to fetch every
    index in index_basic (10k+, mostly irrelevant), or --market to filter.
    """
    from quantify.fetcher.index import IndexFetcher

    normalized = stage.replace("-", "_").lower()
    codes = [c.strip() for c in ts_code.split(",")] if ts_code else None
    markets = [m.strip() for m in market.split(",")] if market else None
    skip_set = {s.strip() for s in skip.split(",")} if skip else None
    etf_only = not all_index
    fetcher = IndexFetcher()

    if normalized == "all":
        fetcher.fetch_all(
            incremental=incremental,
            start_date=start_date,
            end_date=end_date,
            etf_only=etf_only,
            skip=skip_set,
        )
        return

    dispatch = {
        "index_basic": lambda: fetcher.fetch_index_basic(markets=markets),
        "index_daily": lambda: fetcher.fetch_index_daily(
            ts_codes=codes,
            markets=markets,
            etf_only=etf_only,
            incremental=incremental,
            start_date=start_date,
            end_date=end_date,
        ),
        "index_dailybasic": lambda: fetcher.fetch_index_dailybasic(
            ts_codes=codes, incremental=incremental, start_date=start_date, end_date=end_date
        ),
        "index_weight": lambda: fetcher.fetch_index_weight(
            index_codes=codes,
            markets=markets,
            etf_only=etf_only,
            incremental=incremental,
            start_date=start_date,
            end_date=end_date,
        ),
        "moneyflow_ind_dc": lambda: fetcher.fetch_moneyflow_ind_dc(
            incremental=incremental, start_date=start_date, end_date=end_date
        ),
    }
    if normalized not in dispatch:
        raise typer.BadParameter(f"Unknown stage: {stage}")
    dispatch[normalized]()


# ---------------------------------------------------------------------------
# fetch macro (yield curves, global indices, US treasury rates)
# ---------------------------------------------------------------------------
@fetch_app.command("macro")
def fetch_macro(
    stage: str = typer.Argument(
        "all",
        help="Stage: all|yc-cb|index-global|us-tycr|us-trycr",
    ),
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="Incremental update vs full backfill"
    ),
    ts_code: Optional[str] = typer.Option(
        None, "--ts-code", help="Comma-separated codes (index-global stage)"
    ),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="Start date, e.g. 20200101"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="End date, e.g. 20260608"),
    skip: Optional[str] = typer.Option(
        None, "--skip", help="Comma-separated stages to skip (only used with stage=all)"
    ),
) -> None:
    """Fetch macro / cross-asset datasets (yield curves, global indices)."""
    from quantify.fetcher.macro import MacroFetcher

    normalized = stage.replace("-", "_").lower()
    codes = [c.strip() for c in ts_code.split(",")] if ts_code else None
    skip_set = {s.strip() for s in skip.split(",")} if skip else None
    fetcher = MacroFetcher()

    if normalized == "all":
        fetcher.fetch_all(incremental=incremental, start_date=start_date, end_date=end_date, skip=skip_set)
        return

    dispatch = {
        "yc_cb": lambda: fetcher.fetch_yc_cb(
            incremental=incremental, start_date=start_date, end_date=end_date
        ),
        "index_global": lambda: fetcher.fetch_index_global(
            ts_codes=codes, incremental=incremental, start_date=start_date, end_date=end_date
        ),
        "us_tycr": lambda: fetcher.fetch_us_tycr(
            incremental=incremental, start_date=start_date, end_date=end_date
        ),
        "us_trycr": lambda: fetcher.fetch_us_trycr(
            incremental=incremental, start_date=start_date, end_date=end_date
        ),
    }
    if normalized not in dispatch:
        raise typer.BadParameter(f"Unknown stage: {stage}")
    dispatch[normalized]()


# ---------------------------------------------------------------------------
# fetch stock
# ---------------------------------------------------------------------------
@fetch_app.command("stock")
def fetch_stock(
    stage: str = typer.Argument(
        "all",
        help="Stage: all|basic|daily|adj-factor|daily-basic|weekly|monthly|"
        "suspend|namechange|income|balancesheet|cashflow|fina-indicator|"
        "forecast|express|dividend|moneyflow-hsgt|margin|margin-detail|"
        "stk-factor|broker-recommend",
    ),
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="Incremental update vs full backfill"
    ),
    ts_code: Optional[str] = typer.Option(
        None, "--ts-code", help="Comma-separated ts_codes (default: all listed in stock_basic)"
    ),
    skip: Optional[str] = typer.Option(
        None, "--skip", help="Comma-separated stages to skip (only used with stage=all)"
    ),
    include_skipped: bool = typer.Option(
        False,
        "--include-skipped",
        help="Also fetch stages that are skipped by default: weekly, monthly, "
        "margin_detail, stk_factor, broker_recommend",
    ),
) -> None:
    """Fetch A-share stock data from Tushare into MySQL.

    Stage=all runs: basic -> daily -> adj_factor -> daily_basic -> weekly ->
    monthly -> suspend -> namechange -> income -> balancesheet -> cashflow ->
    fina_indicator -> forecast -> express -> dividend -> moneyflow_hsgt ->
    margin -> margin_detail -> stk_factor -> broker_recommend.

    By default margin_detail, stk_factor, broker_recommend,
    weekly, monthly are skipped. Run with ``--include-skipped`` to include them.
    """
    from quantify.fetcher.stock import StockFetcher

    normalized = stage.replace("-", "_").lower()
    codes = [c.strip() for c in ts_code.split(",")] if ts_code else None
    skip_set = {s.strip() for s in skip.split(",")} if skip else None
    fetcher = StockFetcher()

    if normalized == "all":
        fetcher.fetch_all(
            incremental=incremental, ts_codes=codes, skip=skip_set, include_skipped=include_skipped
        )
        return

    if normalized == "basic":
        fetcher.fetch_basic()
        return

    universe = codes if codes else fetcher._load_universe()  # noqa: SLF001
    if not universe:
        log.warning("Stock universe is empty — run `quantify fetch stock basic` first.")
        raise typer.Exit(code=1)

    dispatch: dict = {
        "daily": lambda: fetcher.fetch_daily(ts_codes=universe, incremental=incremental),
        "adj_factor": lambda: fetcher.fetch_adj_factor(ts_codes=universe, incremental=incremental),
        "daily_basic": lambda: fetcher.fetch_daily_basic(ts_codes=universe, incremental=incremental),
        "weekly": lambda: fetcher.fetch_weekly(ts_codes=universe, incremental=incremental),
        "monthly": lambda: fetcher.fetch_monthly(ts_codes=universe, incremental=incremental),
        "suspend": lambda: fetcher.fetch_suspend(ts_codes=universe),
        "namechange": lambda: fetcher.fetch_namechange(ts_codes=universe),
        "income": lambda: fetcher.fetch_income(ts_codes=universe),
        "balancesheet": lambda: fetcher.fetch_balancesheet(ts_codes=universe),
        "cashflow": lambda: fetcher.fetch_cashflow(ts_codes=universe),
        "fina_indicator": lambda: fetcher.fetch_fina_indicator(ts_codes=universe),
        "forecast": lambda: fetcher.fetch_forecast(ts_codes=universe),
        "express": lambda: fetcher.fetch_express(ts_codes=universe),
        "dividend": lambda: fetcher.fetch_dividend(ts_codes=universe),
        "moneyflow_hsgt": lambda: fetcher.fetch_moneyflow_hsgt(incremental=incremental),
        "margin": lambda: fetcher.fetch_margin(incremental=incremental),
        "margin_detail": lambda: fetcher.fetch_margin_detail(incremental=incremental),
        "stk_factor": lambda: fetcher.fetch_stk_factor(ts_codes=universe),
        "broker_recommend": lambda: fetcher.fetch_broker_recommend(ts_codes=universe),
    }
    if normalized not in dispatch:
        raise typer.BadParameter(f"Unknown stage: {stage}")
    dispatch[normalized]()


# ---------------------------------------------------------------------------
# fetch futures
# ---------------------------------------------------------------------------
@fetch_app.command("futures")
def fetch_futures(
    stage: str = typer.Argument(
        "all",
        help="Stage: all|fut-basic|fut-daily|fut-holding|fut-wsr|fut-settle",
    ),
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="Incremental update vs full backfill"
    ),
    skip: Optional[str] = typer.Option(
        None, "--skip", help="Comma-separated stages to skip (only used with stage=all)"
    ),
    include_skipped: bool = typer.Option(
        False,
        "--include-skipped",
        help="Also fetch stages skipped by default: fut_holding, fut_wsr, fut_settle",
    ),
) -> None:
    """Fetch Tushare futures-theme datasets into MySQL.

    Stage=all runs: fut_basic -> fut_daily.
    By default fut_holding, fut_wsr, fut_settle are skipped.
    Run with ``--include-skipped`` to include them.
    """
    from quantify.fetcher.future import FuturesFetcher

    normalized = stage.replace("-", "_").lower()
    skip_set = {s.strip() for s in skip.split(",")} if skip else None
    fetcher = FuturesFetcher()

    if normalized == "all":
        fetcher.fetch_all(incremental=incremental, skip=skip_set, include_skipped=include_skipped)
        return

    symbols = fetcher._load_symbols()  # noqa: SLF001
    if not symbols:
        log.warning("Futures symbols empty — run `quantify fetch futures fut-basic` first.")
        raise typer.Exit(code=1)

    dispatch = {
        "fut_basic": fetcher.fetch_basic,
        "fut_daily": lambda: fetcher.fetch_daily(symbols=symbols, incremental=incremental),
        "fut_holding": lambda: fetcher.fetch_holding(symbols=symbols, incremental=incremental),
        "fut_wsr": lambda: fetcher.fetch_wsr(symbols=symbols, incremental=incremental),
        "fut_settle": lambda: fetcher.fetch_settle(symbols=symbols, incremental=incremental),
    }
    if normalized not in dispatch:
        raise typer.BadParameter(f"Unknown stage: {stage}")
    dispatch[normalized]()


# ---------------------------------------------------------------------------
# fetch fund
# ---------------------------------------------------------------------------
@fetch_app.command("fund")
def fetch_fund(
    stage: str = typer.Argument(
        "all",
        help="Stage: all|company",
    ),
) -> None:
    """Fetch public fund metadata from Tushare into MySQL."""
    from quantify.database.models import FundCompany
    from quantify.database.upsert import upsert_dataframe
    from quantify.tushare_client.client import get_client

    normalized = stage.replace("-", "_").lower()
    client = get_client()

    if normalized in ("all", "company"):
        log.info("Fetching fund_company ...")
        df = client.call("fund_company")
        if df is None or df.empty:
            log.warning("fund_company returned no rows")
            return
        n = upsert_dataframe(FundCompany, df)
        log.info(f"fund_company done. rows={n}")
        return

    raise typer.BadParameter(f"Unknown stage: {stage}")


# ---------------------------------------------------------------------------
# fetch all (full pipeline, one shot)
# ---------------------------------------------------------------------------
@fetch_app.command("all")
def fetch_all_data(
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="Incremental update vs full backfill"
    ),
    exchange: str = typer.Option("SSE", "--exchange", help="Exchange(s) for trade calendar, comma-separated"),
    sw_src: str = typer.Option("SW2021", "--sw-src", help="SW classification source, e.g. SW2021"),
    skip: Optional[str] = typer.Option(
        None,
        "--skip",
        help=(
            "Comma-separated top-level groups to skip: trade_cal|etf|stock|industry|index|macro|futures|fund"
        ),
    ),
    include_skipped: bool = typer.Option(
        False,
        "--include-skipped",
        help="Propagate --include-skipped to stock and futures groups, fetching their "
        "normally-skipped stages (weekly, monthly, stk_factor, margin_detail, broker_recommend, "
        "fut_holding, fut_wsr, fut_settle)",
    ),
) -> None:
    """Fetch EVERYTHING from Tushare into MySQL in dependency order.

    Order: trade calendar -> ETF (basic first) -> stock -> industry (SW + CITIC)
    -> index -> macro -> futures -> fund.
    Use ``--full`` to backfill all history, otherwise incremental.
    """
    from quantify.fetcher.etf import EtfFetcher
    from quantify.fetcher.future import FuturesFetcher
    from quantify.fetcher.index import IndexFetcher
    from quantify.fetcher.industry import IndustryFetcher
    from quantify.fetcher.macro import MacroFetcher
    from quantify.fetcher.stock import StockFetcher

    skip_set = {s.strip().lower() for s in skip.split(",")} if skip else set()

    # 1) Trade calendar (no dependency, used as completeness basis)
    if "trade_cal" not in skip_set:
        industry = IndustryFetcher()
        for exch in (e.strip() for e in exchange.split(",") if e.strip()):
            log.info(f"=== fetch trade_cal ({exch}) ===")
            industry.fetch_trade_cal(exchange=exch)

    # 2) ETF (fetch_all runs basic first, then all per-code stages)
    if "etf" not in skip_set:
        log.info("=== fetch ETF (all stages) ===")
        EtfFetcher().fetch_all(incremental=incremental)

    # 3) Stock (basic first, then time-series, then financials)
    if "stock" not in skip_set:
        log.info("=== fetch stock (all stages) ===")
        StockFetcher().fetch_all(incremental=incremental, include_skipped=include_skipped)

    # 4) Industry (SW + CITIC): classification, members, daily
    if "industry" not in skip_set:
        log.info("=== fetch industry (SW + CITIC) ===")
        IndustryFetcher().fetch_all(provider="all", incremental=incremental, sw_src=sw_src)

    # 5) Index theme: basic, daily, dailybasic, weight, sector money flow
    if "index" not in skip_set:
        log.info("=== fetch index (basic/daily/dailybasic/weight/moneyflow) ===")
        IndexFetcher().fetch_all(incremental=incremental)

    # 6) Macro / cross-asset: yield curves, global indices, US treasury rates
    if "macro" not in skip_set:
        log.info("=== fetch macro (yc_cb/index_global/us_tycr/us_trycr) ===")
        MacroFetcher().fetch_all(incremental=incremental)

    # 7) Futures: basic, daily, and optionally holding/wsr/settle
    if "futures" not in skip_set:
        log.info("=== fetch futures (fut_basic/fut_daily) ===")
        FuturesFetcher().fetch_all(incremental=incremental, include_skipped=include_skipped)

    # 8) Fund company metadata
    if "fund" not in skip_set:
        log.info("=== fetch fund (fund_company) ===")
        from quantify.database.models import FundCompany
        from quantify.database.upsert import upsert_dataframe
        from quantify.tushare_client.client import get_client

        client = get_client()
        df = client.call("fund_company")
        if df is not None and not df.empty:
            upsert_dataframe(FundCompany, df)
            log.info(f"  fund_company: {len(df)} rows")

    log.info("=== fetch all: done ===")


# ---------------------------------------------------------------------------
# fetch skipped — 只拉各数据组默认跳过的阶段
# ---------------------------------------------------------------------------
@fetch_app.command("skipped")
def fetch_skipped(
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="Incremental update vs full backfill"
    ),
) -> None:
    """Sync only the stages skipped by default across all data groups.

    Stock:  weekly, monthly, margin_detail, stk_factor, broker_recommend
    Futures: fut_holding, fut_wsr, fut_settle
    Fund:   fund_company
    """
    from quantify.fetcher.stock import StockFetcher
    from quantify.fetcher.future import FuturesFetcher
    from quantify.database.models import FundCompany
    from quantify.database.upsert import upsert_dataframe
    from quantify.tushare_client.client import get_client
    import pandas as pd

    # --- Stock skipped stages ---
    log.info("=== fetch skipped: stock stages ===")
    stock = StockFetcher()
    universe = stock._load_universe()  # noqa: SLF001
    if not universe:
        log.warning(
            "stock_basic is empty, run `quantify fetch stock basic` first, skipping stock skipped stages"
        )
    else:
        log.info(f"  stock universe: {len(universe)} codes")
        for name, method in [
            ("weekly", stock.fetch_weekly),
            ("monthly", stock.fetch_monthly),
        ]:
            n = method(ts_codes=universe, incremental=incremental)
            log.info(f"  {name}: {n} rows")

        for name, method in [
            ("stk_factor", stock.fetch_stk_factor),
            ("broker_recommend", stock.fetch_broker_recommend),
        ]:
            n = method(ts_codes=universe)
            log.info(f"  {name}: {n} rows")

        n = stock.fetch_margin_detail(incremental=incremental)
        log.info(f"  margin_detail: {n} rows")

    # --- Futures skipped stages ---
    log.info("=== fetch skipped: futures stages ===")
    fut = FuturesFetcher()
    symbols = fut._load_symbols()  # noqa: SLF001
    if not symbols:
        log.warning(
            "fut_basic is empty, run `quantify fetch futures fut-basic` first, "
            "skipping futures skipped stages"
        )
    else:
        log.info(f"  futures symbols: {len(symbols)}")
        for name, method in [
            ("fut_holding", fut.fetch_holding),
            ("fut_wsr", fut.fetch_wsr),
            ("fut_settle", fut.fetch_settle),
        ]:
            n = method(symbols=symbols, incremental=incremental)
            log.info(f"  {name}: {n} rows")

    # --- Fund company ---
    log.info("=== fetch skipped: fund_company ===")
    client = get_client()
    df = client.call("fund_company")
    if df is not None and not df.empty:
        # Tushare setup_date/end_date may be YYYY(4)/YYYYMM(6)/YYYYMMDD(8);
        # pad short forms to 8 chars (→ first of year/month), coerce bad to None.
        for col in ("setup_date", "end_date"):
            if col in df.columns:
                s = df[col].astype(str).str.strip()
                s = s.where(s.str.match(r"^\d{4,8}$"), None)
                s = s.str.ljust(8, "0")
                df[col] = pd.to_datetime(s, format="%Y%m%d", errors="coerce").dt.date
        n = upsert_dataframe(FundCompany, df)
        log.info(f"  fund_company: {n} rows")
    else:
        log.info("  fund_company: empty")

    log.info("=== fetch skipped: done ===")


# ---------------------------------------------------------------------------
# factor mining (LLM + Qlib + Alphalens)
# ---------------------------------------------------------------------------
def _parse_codes(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    return [c.strip() for c in value.split(",") if c.strip()]


@factor_app.command("dump-data")
def factor_dump_data(
    ts_code: Optional[str] = typer.Option(
        None, "--ts-code", help="逗号分隔的股票代码（默认 stock_basic 全市场）"
    ),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="起始日 YYYY-MM-DD/YYYYMMDD"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="结束日 YYYY-MM-DD/YYYYMMDD"),
    provider_uri: Optional[str] = typer.Option(None, "--provider-uri", help="Qlib 输出目录（默认配置）"),
) -> None:
    """把 MySQL 个股日线（前复权）导出为 Qlib .bin 数据。"""
    from quantify.factor.qlib_data import dump_qlib_data

    summary = dump_qlib_data(
        ts_codes=_parse_codes(ts_code),
        start_date=start_date,
        end_date=end_date,
        provider_uri=provider_uri,
    )
    typer.echo(
        f"导出完成：{summary.instruments} 只标的，{summary.calendar_days} 个交易日 -> {summary.provider_uri}"
    )
    typer.echo(f"字段：{', '.join(summary.fields)}")


@factor_app.command("mine")
def factor_mine(
    rounds: int = typer.Option(3, "--rounds", help="单因子挖掘迭代轮数（每轮根据IC反馈改进）"),
    n_factors: int = typer.Option(5, "--n-factors", help="每轮生成的候选因子数"),
    n_compose: int = typer.Option(2, "--n-compose", help="合成因子数量（每个独立生成，带反馈迭代）"),
    universe: Optional[str] = typer.Option(None, "--universe", help="股票池：all / 指数代码(如 000300.SH)"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="评估起始日"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="评估结束日"),
    periods: str = typer.Option("1,5,10", "--periods", help="前瞻收益周期，逗号分隔"),
    quantiles: int = typer.Option(5, "--quantiles", help="分层数"),
    min_ic: float = typer.Option(0.02, "--min-ic", help="|IC| 门槛（仅标记 status=passed，不影响入库）"),
    min_icir: float = typer.Option(
        0.3, "--min-icir", help="|IC_IR| 门槛（仅标记 status=passed，不影响入库）"
    ),
    top_n: int = typer.Option(20, "--top-n", help="策略选股数量"),
    rebalance: int = typer.Option(5, "--rebalance", help="策略调仓频率(交易日)"),
    initial_cash: float = typer.Option(1_000_000, "--initial-cash", help="策略初始资金"),
    max_retries: int = typer.Option(3, "--max-retries", help="回测失败后反馈LLM重试次数"),
    instruction: Optional[str] = typer.Option(None, "--instruction", help="给 LLM 的额外要求"),
) -> None:
    """两阶段闭环：单因子多轮迭代挖掘+回测 → 合成因子挖掘+回测，全部入库并关联策略。"""
    from quantify.factor.evaluator import QualityThresholds
    from quantify.factor.pipeline import MiningConfig, mine_factors

    period_tuple = tuple(int(p) for p in periods.split(",") if p.strip())
    config = MiningConfig(
        rounds=rounds,
        n_factors=n_factors,
        n_compose=n_compose,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        periods=period_tuple or (1, 5, 10),
        quantiles=quantiles,
        primary_period=period_tuple[0] if period_tuple else 1,
        thresholds=QualityThresholds(min_abs_ic=min_ic, min_abs_icir=min_icir),
        extra_instruction=instruction,
        backtest_top_n=top_n,
        backtest_rebalance_days=rebalance,
        backtest_initial_cash=initial_cash,
        backtest_max_retries=max_retries,
    )
    try:
        result = mine_factors(config)
    except Exception as exc:
        typer.echo(f"❌ 挖掘流程出错: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"=== 完成：单因子 {rounds} 轮共评估 {result.n_evaluated} 个(入库 {result.n_passed})，"
        f"合成因子 {len(result.composed)} 个 ==="
    )
    typer.echo("\n单因子入库：")
    for rec in result.saved:
        sid = f"策略#{rec.strategy_id}" if rec.strategy_id else "无策略"
        ic = f"{rec.ic_mean:.4f}" if rec.ic_mean is not None else "NA"
        ir = f"{rec.icir:.4f}" if rec.icir is not None else "NA"
        typer.echo(f"  [{rec.id}] {rec.name}  IC={ic} IR={ir}  {sid}  {rec.expression[:60]}")
    if result.composed:
        typer.echo("\n合成因子入库：")
        for rec in result.composed:
            sid = f"策略#{rec.strategy_id}" if rec.strategy_id else "无策略"
            ic = f"{rec.ic_mean:.4f}" if rec.ic_mean is not None else "NA"
            ir = f"{rec.icir:.4f}" if rec.icir is not None else "NA"
            typer.echo(f"  [{rec.id}] {rec.name}  IC={ic} IR={ir}  {sid}  父={rec.parent_factor_ids}")


@factor_app.command("eval")
def factor_eval(
    expression: str = typer.Argument(..., help="待评估的 Qlib 因子表达式"),
    universe: Optional[str] = typer.Option(None, "--universe", help="股票池：all / 指数代码"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="评估起始日"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="评估结束日"),
    periods: str = typer.Option("1,5,10", "--periods", help="前瞻收益周期"),
    quantiles: int = typer.Option(5, "--quantiles", help="分层数"),
    save: bool = typer.Option(False, "--save", help="通过门槛则存入因子库"),
) -> None:
    """评估单个因子表达式（手动测试用）。"""
    from quantify.factor.evaluator import evaluate_expression, evaluation_window_default

    if not start_date or not end_date:
        ds, de = evaluation_window_default()
        start_date = start_date or ds
        end_date = end_date or de
    period_tuple = tuple(int(p) for p in periods.split(",") if p.strip()) or (1, 5, 10)
    evaluation = evaluate_expression(
        expression,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        periods=period_tuple,
        quantiles=quantiles,
        primary_period=period_tuple[0],
    )
    typer.echo(evaluation.to_feedback_text())
    if save and evaluation.passed:
        from quantify.database.factor_store import FactorRecord, metrics_to_json, save_factor

        rec = save_factor(
            FactorRecord(
                name=f"manual_{abs(hash(expression)) % 1_000_000}",
                expression=expression,
                universe=universe or "all",
                periods=",".join(str(p) for p in period_tuple),
                ic_mean=evaluation.ic_mean,
                ic_std=evaluation.ic_std,
                icir=evaluation.icir,
                ic_tstat=evaluation.ic_tstat,
                rank_ic_mean=evaluation.rank_ic_mean,
                rank_icir=evaluation.rank_icir,
                quantile_spread=evaluation.quantile_spread,
                turnover=evaluation.turnover,
                coverage=evaluation.coverage,
                metrics_json=metrics_to_json(evaluation.to_dict()),
            )
        )
        typer.echo(f"已入库：{rec.name}")


@factor_app.command("list")
def factor_list(
    status: Optional[str] = typer.Option(None, "--status", help="按状态过滤，如 passed"),
) -> None:
    """列出因子库中已保存的因子。"""
    from quantify.database.factor_store import list_factors

    records = list_factors(status=status)
    if not records:
        typer.echo("因子库为空。先运行 `quantify factor mine`。")
        return
    typer.echo(f"共 {len(records)} 个因子：")
    for rec in records:
        ic = f"{rec.ic_mean:.4f}" if rec.ic_mean is not None else "NA"
        ir = f"{rec.icir:.4f}" if rec.icir is not None else "NA"
        typer.echo(f"  [{rec.id}] {rec.name}  IC={ic} IR={ir}  {rec.expression}")


@factor_app.command("compose")
def factor_compose(
    universe: Optional[str] = typer.Option(None, "--universe", help="股票池：all / 指数代码(如 000300.SH)"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="回测起始日"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="回测结束日"),
    max_factors: int = typer.Option(10, "--max-factors", help="参与合成的最大因子数"),
    top_n: int = typer.Option(50, "--top-n", help="每日选股数量"),
    weight: str = typer.Option("icir", "--weight", help="合成方式: equal / ic / icir"),
    min_icir: float = typer.Option(0.3, "--min-icir", help="因子最低 |ICIR| 门槛"),
    rebalance: int = typer.Option(5, "--rebalance", help="调仓频率(交易日)"),
    max_corr: float = typer.Option(0.7, "--max-corr", help="因子间最大允许相关性"),
    export: Optional[str] = typer.Option(None, "--export", help="导出持仓矩阵到 CSV 路径"),
) -> None:
    """从因子库选因子合成组合并做简单回测。"""
    from quantify.factor.compose import ComposeConfig, compose_factors

    config = ComposeConfig(
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        max_factors=max_factors,
        top_n=top_n,
        weight=weight,  # type: ignore[arg-type]
        min_icir=min_icir,
        rebalance_freq=rebalance,
        max_corr=max_corr,
    )
    result = compose_factors(config)
    if not result.selected:
        typer.echo("组合构建失败（因子库为空或无因子满足门槛）。")
        return

    typer.echo("\n=== 组合构建完成 ===")
    typer.echo(f"选定 {len(result.selected)} 个因子：")
    for f in result.selected:
        w = result.weights.get(f.expression, 0)
        typer.echo(f"  {f.name}  权重={w:.3f}  IC={f.ic_mean:.4f} IR={f.icir:.4f}  {f.expression}")

    m = result.metrics
    if m:
        typer.echo("\n--- 组合回测指标 ---")
        typer.echo(f"  交易日数:   {m.get('n_days', 0)}")
        typer.echo(f"  总收益:     {m.get('total_return', 0):.2%}")
        typer.echo(f"  年化收益:   {m.get('ann_return', 0):.2%}")
        typer.echo(f"  年化波动:   {m.get('ann_vol', 0):.2%}")
        typer.echo(f"  夏普比率:   {m.get('sharpe', 0):.3f}")
        typer.echo(f"  最大回撤:   {m.get('max_drawdown', 0):.2%}")
        typer.echo(f"  日胜率:     {m.get('win_rate', 0):.2%}")

    if export and result.holdings is not None:
        result.holdings.to_csv(export)
        typer.echo(f"\n持仓矩阵已导出: {export}")


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------
@app.command("version")
def version() -> None:
    from quantify import __version__

    typer.echo(__version__)


# ---------------------------------------------------------------------------
# ML / DL factor mining
# ---------------------------------------------------------------------------
@ml_app.command("synth")
def ml_synth(
    universe: Optional[str] = typer.Option(None, "--universe", help="股票池：all / 指数代码(如 000300.SH)"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="评估起始日"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="评估结束日"),
    forward_period: int = typer.Option(5, "--forward-period", help="前瞻收益周期(交易日)"),
    test_ratio: float = typer.Option(0.3, "--test-ratio", help="测试集比例(按时间切分)"),
    top_n: int = typer.Option(20, "--top-n", help="选股数量"),
    rebalance: int = typer.Option(5, "--rebalance", help="调仓频率(交易日)"),
    model: str = typer.Option("xgboost", "--model", help="模型: xgboost/lightgbm/ridge/lasso/rf/gbdt"),
    min_icir: float = typer.Option(0.0, "--min-icir", help="因子筛选 |ICIR| 门槛"),
    max_factors: int = typer.Option(20, "--max-factors", help="最多使用因子数"),
) -> None:
    """ML 因子合成：从因子库选因子 → XGBoost/LightGBM/sklearn 预测截面收益 → 向量化回测。"""
    from quantify.ml.factor_synthesis import MLSynthConfig, MLSynthesizer

    config = MLSynthConfig(
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        forward_period=forward_period,
        test_ratio=test_ratio,
        top_n=top_n,
        rebalance_days=rebalance,
        model_type=model,
        min_icir=min_icir,
        max_factors=max_factors,
    )
    try:
        synth = MLSynthesizer(config)
        result = synth.run()
        typer.echo(result.summary())
    except Exception as exc:
        typer.echo(f"❌ ML 合成出错: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@ml_app.command("gp")
def ml_gp(
    universe: Optional[str] = typer.Option(None, "--universe", help="股票池：all / 指数代码(如 000300.SH)"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="评估起始日"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="评估结束日"),
    forward_period: int = typer.Option(5, "--forward-period", help="前瞻收益周期(交易日)"),
    population: int = typer.Option(500, "--population", help="种群大小"),
    generations: int = typer.Option(30, "--generations", help="进化代数"),
    max_depth: int = typer.Option(5, "--max-depth", help="表达式最大深度"),
    top_k: int = typer.Option(10, "--top-k", help="返回前 K 个最优表达式"),
    save: bool = typer.Option(False, "--save", help="将发现的因子保存到 factor_library"),
) -> None:
    """GP 因子发现：遗传规划进化 Qlib 因子表达式，适应度=IC。"""
    from quantify.ml.gp_miner import GPConfig, GPMiner

    config = GPConfig(
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        forward_period=forward_period,
        population=population,
        generations=generations,
        max_depth=max_depth,
        top_k=top_k,
    )
    try:
        miner = GPMiner(config)
        result = miner.run()
        typer.echo(f"\n=== GP 因子发现完成：{len(result.expressions)} 个表达式 ===")
        for i, (expr, tr, te) in enumerate(
            zip(result.expressions, result.fitness, result.test_fitness, strict=False)
        ):
            typer.echo(f"  #{i + 1}: train_IC={tr:.4f} test_IC={te:.4f}  {expr[:100]}")

        if save:
            from quantify.database.factor_store import FactorRecord, save_factor
            from quantify.factor.evaluator import evaluate_expression
            from quantify.factor.pipeline import _normalize_expr, metrics_to_json

            seen = set()
            for i, expr in enumerate(result.expressions):
                norm = _normalize_expr(expr)
                if norm in seen:
                    continue
                seen.add(norm)
                evaluation = evaluate_expression(
                    expr,
                    universe=universe,
                    start_date=start_date,
                    end_date=end_date,
                )
                record = FactorRecord(
                    name=f"gp_factor_{i + 1}",
                    expression=expr,
                    hypothesis="GP evolved factor",
                    category="gp",
                    universe=universe or "all",
                    ic_mean=evaluation.ic_mean,
                    ic_std=evaluation.ic_std,
                    icir=evaluation.icir,
                    rank_ic_mean=evaluation.rank_ic_mean,
                    rank_icir=evaluation.rank_icir,
                    coverage=evaluation.coverage,
                    status="passed" if evaluation.passed else "evaluated",
                    factor_type="single",
                    metrics_json=metrics_to_json(evaluation.to_dict()),
                )
                save_factor(record)
                typer.echo(f"  入库: {record.name} IC={evaluation.ic_mean:.4f}")
    except Exception as exc:
        typer.echo(f"❌ GP 挖掘出错: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@ml_app.command("dl")
def ml_dl(
    universe: Optional[str] = typer.Option(None, "--universe", help="股票池：all / 指数代码(如 000300.SH)"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="评估起始日"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="评估结束日"),
    forward_period: int = typer.Option(5, "--forward-period", help="前瞻收益周期(交易日)"),
    lookback: int = typer.Option(20, "--lookback", help="输入序列长度(历史天数)"),
    model: str = typer.Option("lstm", "--model", help="模型: lstm/transformer"),
    hidden_dim: int = typer.Option(64, "--hidden-dim", help="隐藏层维度"),
    num_layers: int = typer.Option(2, "--num-layers", help="层数"),
    epochs: int = typer.Option(50, "--epochs", help="训练轮数"),
    batch_size: int = typer.Option(256, "--batch-size", help="批大小"),
    lr: float = typer.Option(0.001, "--lr", help="学习率"),
    top_n: int = typer.Option(20, "--top-n", help="选股数量"),
    rebalance: int = typer.Option(5, "--rebalance", help="调仓频率(交易日)"),
    test_ratio: float = typer.Option(0.3, "--test-ratio", help="测试集比例"),
) -> None:
    """DL 端到端选股：LSTM/Transformer 从原始OHLCV预测截面收益。"""
    from quantify.ml.dl_miner import DLConfig, DLMiner

    config = DLConfig(
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        forward_period=forward_period,
        lookback=lookback,
        model_type=model,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        top_n=top_n,
        rebalance_days=rebalance,
        test_ratio=test_ratio,
    )
    try:
        miner = DLMiner(config)
        result = miner.run()
        typer.echo(result.summary())
    except Exception as exc:
        typer.echo(f"❌ DL 训练出错: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@ml_app.command("run")
def ml_run(
    universe: Optional[str] = typer.Option("000300.SH", "--universe", help="股票池"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="评估起始日"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="评估结束日"),
    forward_period: int = typer.Option(5, "--forward-period", help="前瞻收益周期(交易日)"),
    top_n: int = typer.Option(20, "--top-n", help="选股数量"),
    rebalance: int = typer.Option(5, "--rebalance", help="调仓频率(交易日)"),
    test_ratio: float = typer.Option(0.3, "--test-ratio", help="测试集比例"),
    # ML 合成
    ml_model: str = typer.Option(
        "xgboost", "--ml-model", help="ML模型: xgboost/lightgbm/ridge/lasso/rf/gbdt"
    ),
    skip_ml: bool = typer.Option(False, "--skip-ml", help="跳过ML因子合成"),
    # GP 发现
    skip_gp: bool = typer.Option(False, "--skip-gp", help="跳过GP因子发现"),
    gp_population: int = typer.Option(500, "--gp-population", help="GP种群大小"),
    gp_generations: int = typer.Option(30, "--gp-generations", help="GP进化代数"),
    gp_save: bool = typer.Option(False, "--gp-save", help="GP发现的因子入库"),
    # DL 端到端
    skip_dl: bool = typer.Option(False, "--skip-dl", help="跳过DL端到端选股"),
    dl_model: str = typer.Option("lstm", "--dl-model", help="DL模型: lstm/transformer"),
    dl_lookback: int = typer.Option(20, "--dl-lookback", help="DL输入序列长度"),
    dl_epochs: int = typer.Option(30, "--dl-epochs", help="DL训练轮数"),
    dl_hidden: int = typer.Option(64, "--dl-hidden", help="DL隐藏层维度"),
    # 两阶段验证
    skip_validate: bool = typer.Option(False, "--skip-validate", help="跳过两阶段回测验证"),
    validate_model: str = typer.Option(
        "auto", "--validate-model", help="验证哪个模型: auto(取最优)/ml/gp/dl"
    ),
) -> None:
    """一键全流程：ML合成 + GP发现 + DL选股 + 两阶段回测验证。

    默认跑全部4个阶段，可用 --skip-ml/--skip-gp/--skip-dl/--skip-validate 跳过。
    """
    from quantify.ml.factor_synthesis import MLSynthConfig, MLSynthesizer
    from quantify.ml.gp_miner import GPConfig, GPMiner
    from quantify.ml.dl_miner import DLConfig, DLMiner
    from quantify.ml.two_stage import TwoStageBacktest, TwoStageConfig

    results: list[tuple[str, float, object]] = []  # (name, test_ic, vector_bt)
    ml_synth = None  # ref to MLSynthesizer for reusable strategy generation
    gp_expressions: list[str] = []  # GP-discovered expressions to feed into ML

    try:
        # ── Phase 1: GP 因子发现 (run first so results can feed into ML) ──
        if not skip_gp:
            typer.echo("=" * 60)
            typer.echo("Phase 1: GP 因子发现")
            typer.echo("=" * 60)
            gp_cfg = GPConfig(
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                forward_period=forward_period,
                population=gp_population,
                generations=gp_generations,
            )
            miner = GPMiner(gp_cfg)
            gp_result = miner.run()
            typer.echo(f"\nGP 发现 {len(gp_result.expressions)} 个表达式:")
            for i, (expr, tr, te) in enumerate(
                zip(gp_result.expressions, gp_result.fitness, gp_result.test_fitness, strict=False)
            ):
                typer.echo(f"  #{i + 1}: train_IC={tr:.4f} test_IC={te:.4f}  {expr[:100]}")
            gp_expressions = list(gp_result.expressions)

            if gp_save:
                from quantify.database.factor_store import FactorRecord, save_factor
                from quantify.factor.evaluator import evaluate_expression
                from quantify.factor.pipeline import _normalize_expr, metrics_to_json

                seen = set()
                for i, expr in enumerate(gp_result.expressions):
                    norm = _normalize_expr(expr)
                    if norm in seen:
                        continue
                    seen.add(norm)
                    evaluation = evaluate_expression(
                        expr, universe=universe, start_date=start_date, end_date=end_date
                    )
                    record = FactorRecord(
                        name=f"gp_factor_{i + 1}",
                        expression=expr,
                        hypothesis="GP evolved factor",
                        category="gp",
                        universe=universe or "all",
                        ic_mean=evaluation.ic_mean,
                        ic_std=evaluation.ic_std,
                        icir=evaluation.icir,
                        rank_ic_mean=evaluation.rank_ic_mean,
                        rank_icir=evaluation.rank_icir,
                        coverage=evaluation.coverage,
                        status="passed" if evaluation.passed else "evaluated",
                        factor_type="single",
                        metrics_json=metrics_to_json(evaluation.to_dict()),
                    )
                    save_factor(record)
                    typer.echo(f"  入库: {record.name} IC={evaluation.ic_mean:.4f}")

        # ── Phase 2: ML 因子合成 (uses GP-discovered factors as additional features) ──
        if not skip_ml:
            typer.echo("\n" + "=" * 60)
            typer.echo("Phase 2: ML 因子合成" + (" (含 GP 因子)" if gp_expressions else ""))
            typer.echo("=" * 60)
            ml_cfg = MLSynthConfig(
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                forward_period=forward_period,
                top_n=top_n,
                rebalance_days=rebalance,
                test_ratio=test_ratio,
                model_type=ml_model,
            )
            ml_synth = MLSynthesizer(ml_cfg)
            # Inject GP expressions as extra features
            if gp_expressions:
                ml_synth._extra_expressions = gp_expressions
            ml_result = ml_synth.run()
            typer.echo(ml_result.summary())
            results.append(
                (
                    f"ML-{ml_model}",
                    ml_result.test_ic.get("ic_mean", 0),
                    ml_result.test_backtest,
                )
            )

        # ── Phase 3: DL 端到端选股 ──
        if not skip_dl:
            typer.echo("\n" + "=" * 60)
            typer.echo("Phase 3: DL 端到端选股")
            typer.echo("=" * 60)
            dl_cfg = DLConfig(
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                forward_period=forward_period,
                lookback=dl_lookback,
                model_type=dl_model,
                hidden_dim=dl_hidden,
                top_n=top_n,
                rebalance_days=rebalance,
                test_ratio=test_ratio,
                epochs=dl_epochs,
            )
            dl_miner = DLMiner(dl_cfg)
            dl_result = dl_miner.run()
            typer.echo(dl_result.summary())
            results.append(
                (
                    f"DL-{dl_model}",
                    dl_result.test_ic.get("ic_mean", 0),
                    dl_result.test_backtest,
                )
            )

        # ── Phase 4: 两阶段回测验证 ──
        if not skip_validate and results:
            typer.echo("\n" + "=" * 60)
            typer.echo("Phase 4: 两阶段回测验证")
            typer.echo("=" * 60)

            # 选哪个模型去验证
            if validate_model == "auto":
                # 取 test IC 最高的
                results.sort(key=lambda x: abs(x[1]), reverse=True)
                chosen_name, chosen_ic, chosen_bt = results[0]
                typer.echo(f"自动选择最优模型: {chosen_name} (test IC={chosen_ic:.4f})")
            elif validate_model == "ml" and any(r[0].startswith("ML") for r in results):
                chosen_name, chosen_ic, chosen_bt = next(r for r in results if r[0].startswith("ML"))
                typer.echo(f"验证 ML 模型: {chosen_name}")
            elif validate_model == "dl" and any(r[0].startswith("DL") for r in results):
                chosen_name, chosen_ic, chosen_bt = next(r for r in results if r[0].startswith("DL"))
                typer.echo(f"验证 DL 模型: {chosen_name}")
            else:
                chosen_name, chosen_ic, chosen_bt = results[0]
                typer.echo(f"验证模型: {chosen_name}")

            from quantify.factor.evaluator import evaluation_window_default

            default_start, default_end = evaluation_window_default()
            two_stage = TwoStageBacktest(
                TwoStageConfig(
                    universe=universe or "000300.SH",
                    start_date=start_date or default_start,
                    end_date=end_date or default_end,
                )
            )
            event_result = two_stage.validate_vector_result(chosen_bt, rebalance_days=rebalance)
            typer.echo(event_result.summary())

            # 保存可复用策略到 strategy 表（运行时实时计算因子+模型预测，非回放）
            if ml_synth and ml_synth.strategy_source:
                from quantify.database.strategy_store import save_strategy

                strategy_name = f"ml_{chosen_name}_{universe or 'all'}".replace(".", "_").replace("/", "_")
                saved = save_strategy(
                    name=strategy_name,
                    source=ml_synth.strategy_source,
                    description=(
                        f"ML 可复用策略 | 模型={chosen_name} | "
                        f"universe={universe} | test IC={chosen_ic:.4f} | "
                        f"向量化收益={event_result.vectorized_metrics.get('total_return_pct', 0):.2f}% | "
                        f"事件驱动收益={event_result.metrics.get('total_return_pct', 0):.2f}% | "
                        f"模型文件={ml_synth.saved_model_name}"
                    ),
                )
                typer.echo(f"\n可复用策略已入库: #{saved.id} {strategy_name}")
                typer.echo(f"  模型文件: models/{ml_synth.saved_model_name}.*")
                typer.echo("  → 策略在运行时实时计算因子+模型预测，可在 Dashboard 任意区间回测")
            else:
                typer.echo("\n注: 未生成可复用策略（ML 阶段被跳过或策略生成失败）。")

        # ── 汇总 ──
        if results:
            typer.echo("\n" + "=" * 60)
            typer.echo("全流程汇总")
            typer.echo("=" * 60)
            typer.echo(f"{'模型':<20} {'Test IC':>10} {'向量化收益':>12} {'Sharpe':>8}")
            typer.echo("-" * 52)
            for name, ic, bt in results:
                typer.echo(f"{name:<20} {ic:>10.4f} {bt.total_return:>11.2f}% {bt.sharpe:>8.2f}")

    except Exception as exc:
        typer.echo(f"❌ 全流程出错: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@ml_app.command("validate")
def ml_validate(
    universe: Optional[str] = typer.Option("000300.SH", "--universe", help="股票池"),
    start_date: Optional[str] = typer.Option("2020-06-16", "--start-date", help="回测起始日"),
    end_date: Optional[str] = typer.Option("2026-06-16", "--end-date", help="回测结束日"),
    model: str = typer.Option(
        "xgboost", "--model", help="先跑哪个模型: xgboost/lightgbm/ridge/lstm/transformer"
    ),
    top_n: int = typer.Option(20, "--top-n", help="选股数量"),
    rebalance: int = typer.Option(5, "--rebalance", help="调仓频率"),
    forward_period: int = typer.Option(5, "--forward-period", help="前瞻收益周期"),
    lookback: int = typer.Option(20, "--lookback", help="DL lookback (仅 DL 模型)"),
    epochs: int = typer.Option(30, "--epochs", help="DL epochs (仅 DL 模型)"),
) -> None:
    """两阶段回测：先向量化筛选 → 再事件驱动验证（含交易摩擦/T+1/涨跌停）。"""
    from quantify.ml.two_stage import TwoStageBacktest, TwoStageConfig

    try:
        # Step 1: Run ML/DL model to get vectorized backtest result
        if model in ("lstm", "transformer"):
            from quantify.ml.dl_miner import DLConfig, DLMiner

            dl_cfg = DLConfig(
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                forward_period=forward_period,
                lookback=lookback,
                model_type=model,
                top_n=top_n,
                rebalance_days=rebalance,
                epochs=epochs,
            )
            typer.echo(f"=== Stage 1: DL ({model}) 向量化回测 ===")
            miner = DLMiner(dl_cfg)
            dl_result = miner.run()
            typer.echo(dl_result.summary())
            vector_bt = dl_result.test_backtest
        else:
            from quantify.ml.factor_synthesis import MLSynthConfig, MLSynthesizer

            ml_cfg = MLSynthConfig(
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                forward_period=forward_period,
                top_n=top_n,
                rebalance_days=rebalance,
                model_type=model,
            )
            typer.echo(f"=== Stage 1: ML ({model}) 向量化回测 ===")
            synth = MLSynthesizer(ml_cfg)
            ml_result = synth.run()
            typer.echo(ml_result.summary())
            vector_bt = ml_result.test_backtest

        # Step 2: Event-driven validation
        typer.echo("\n=== Stage 2: 事件驱动回测验证 ===")
        two_stage = TwoStageBacktest(
            TwoStageConfig(
                universe=universe or "000300.SH",
                start_date=start_date or "2020-06-16",
                end_date=end_date or "2026-06-16",
            )
        )
        event_result = two_stage.validate_vector_result(vector_bt, rebalance_days=rebalance)
        typer.echo(event_result.summary())

    except Exception as exc:
        typer.echo(f"❌ 两阶段回测出错: {exc}", err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
