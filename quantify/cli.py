"""Typer CLI entry point.

Subcommands implemented in this milestone:
    quantify db init                  # create database + tables
    quantify db drop                  # drop all known tables (DANGEROUS)
    quantify fetch etf [stage]        # pull ETF data from Tushare
"""

from __future__ import annotations

from typing import Optional

import typer

from quantify.utils.logger import log

app = typer.Typer(help="Quantify CLI", no_args_is_help=True)
db_app = typer.Typer(help="Database management")
fetch_app = typer.Typer(help="Data fetching tasks")
app.add_typer(db_app, name="db")
app.add_typer(fetch_app, name="fetch")


@app.command("dashboard")
def dashboard(
    port: int = typer.Option(8501, "--port", help="Streamlit server port"),
    address: str = typer.Option("localhost", "--address", help="Streamlit server address"),
) -> None:
    """Launch the Streamlit backtest dashboard."""
    import os
    import socket
    import sys
    from pathlib import Path

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
        fetcher.fetch_all(incremental=incremental, ts_codes=codes, skip=skip_set, include_skipped=include_skipped)
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

    # --- Stock skipped stages ---
    log.info("=== fetch skipped: stock stages ===")
    stock = StockFetcher()
    universe = stock._load_universe()  # noqa: SLF001
    if not universe:
        log.warning("stock_basic is empty, run `quantify fetch stock basic` first, "
                     "skipping stock skipped stages")
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
        log.warning("fut_basic is empty, run `quantify fetch futures fut-basic` first, "
                     "skipping futures skipped stages")
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
        n = upsert_dataframe(FundCompany, df)
        log.info(f"  fund_company: {n} rows")
    else:
        log.info("  fund_company: empty")

    log.info("=== fetch skipped: done ===")


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------
@app.command("version")
def version() -> None:
    from quantify import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
