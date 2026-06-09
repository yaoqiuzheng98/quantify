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
# fetch all (ETF + industry + trade calendar, one shot)
# ---------------------------------------------------------------------------
@fetch_app.command("all")
def fetch_all_data(
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="Incremental update vs full backfill"
    ),
    exchange: str = typer.Option("SSE", "--exchange", help="Exchange(s) for trade calendar, comma-separated"),
    sw_src: str = typer.Option("SW2021", "--sw-src", help="SW classification source, e.g. SW2021"),
    skip: Optional[str] = typer.Option(
        None, "--skip", help="Comma-separated top-level groups to skip: trade_cal|etf|industry|index"
    ),
) -> None:
    """Fetch EVERYTHING from Tushare into MySQL in dependency order.

    Order: trade calendar -> ETF (basic first) -> industry (SW + CITIC) -> index.
    Use ``--full`` to backfill all history, otherwise incremental.
    """
    from quantify.fetcher.etf import EtfFetcher
    from quantify.fetcher.index import IndexFetcher
    from quantify.fetcher.industry import IndustryFetcher

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

    # 3) Industry (SW + CITIC): classification, members, daily
    if "industry" not in skip_set:
        log.info("=== fetch industry (SW + CITIC) ===")
        IndustryFetcher().fetch_all(provider="all", incremental=incremental, sw_src=sw_src)

    # 4) Index theme: basic, daily, dailybasic, weight, sector money flow
    if "index" not in skip_set:
        log.info("=== fetch index (basic/daily/dailybasic/weight/moneyflow) ===")
        IndexFetcher().fetch_all(incremental=incremental)

    log.info("=== fetch all: done ===")


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------
@app.command("version")
def version() -> None:
    from quantify import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
