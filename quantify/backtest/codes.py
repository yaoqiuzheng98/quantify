"""Security code conversion helpers."""

from __future__ import annotations


def to_tushare_code(code: str) -> str:
    """Convert JoinQuant or Tushare ETF code to Tushare format."""
    code = code.strip().upper()
    if code.endswith(".XSHG"):
        return f"{code[:-5]}.SH"
    if code.endswith(".XSHE"):
        return f"{code[:-5]}.SZ"
    return code


def to_joinquant_code(code: str) -> str:
    """Convert Tushare ETF code to JoinQuant format."""
    code = code.strip().upper()
    if code.endswith(".SH"):
        return f"{code[:-3]}.XSHG"
    if code.endswith(".SZ"):
        return f"{code[:-3]}.XSHE"
    return code


def normalize_codes(codes: list[str]) -> list[str]:
    """Normalize and deduplicate security codes while preserving order."""
    return list(dict.fromkeys(to_tushare_code(code) for code in codes if code.strip()))


def classify_asset(ts_code: str) -> str:
    """Classify a Tushare code as ``"stock"``, ``"etf"``, ``"index"`` or ``"futures"``.

    Based on A-share code conventions (prefix + exchange suffix):

    - ``.SH``: ``5xxxxx`` = fund/ETF, ``000xxx`` = index, otherwise (``6xxxxx``) = stock
    - ``.SZ``: ``15/16/18xxxx`` = fund/ETF, ``399xxx`` = index, otherwise (``00/30xxxx``) = stock
    - ``.BJ``: always stock (Beijing Stock Exchange)
    - ``.SHF/.DCE/.CZC/.CFF``: futures (Shanghai Futures, Dalian, Zhengzhou, CFFEX)

    Defaults to ``"stock"`` for anything unrecognised so a backtest still attempts
    the most common path rather than silently loading nothing.
    """
    code = to_tushare_code(ts_code)
    body, _, suffix = code.partition(".")
    if suffix in ("SHF", "DCE", "CZC", "CFF", "INE"):
        return "futures"
    if suffix == "SH":
        if body.startswith("5"):
            return "etf"
        if body.startswith("000"):
            return "index"
        return "stock"
    if suffix == "SZ":
        if body.startswith(("15", "16", "18")):
            return "etf"
        if body.startswith("399"):
            return "index"
        return "stock"
    if suffix == "BJ":
        return "stock"
    return "stock"
