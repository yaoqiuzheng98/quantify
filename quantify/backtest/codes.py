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
