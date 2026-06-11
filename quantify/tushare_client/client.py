"""Thin wrapper around Tushare Pro with rate limiting + retry."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
import tushare as ts
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from quantify.config import get_settings
from quantify.tushare_client.rate_limiter import RateLimiter
from quantify.utils.logger import log


class TushareClient:
    """Rate-limited, retryable wrapper around the Tushare Pro API."""

    def __init__(self, token: str | None = None, rate_per_min: int | None = None) -> None:
        settings = get_settings().tushare
        self.token = token or settings.token
        if not self.token:
            raise RuntimeError("TUSHARE_TOKEN is empty. Please set it in the .env file or environment.")
        self.rate_per_min = rate_per_min or settings.rate_per_min
        ts.set_token(self.token)
        self._pro = ts.pro_api()
        self._pro._DataApi__token = self.token
        self._pro._DataApi__http_url = settings.http_url
        self._limiter = RateLimiter(max_calls=self.rate_per_min, period=60.0)

    @property
    def pro(self):
        return self._pro

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
    )
    def call(self, api_name: str, **kwargs: Any) -> pd.DataFrame:
        """Invoke a Tushare Pro endpoint by name with kwargs.

        Examples
        --------
        >>> client.call("fund_basic", market="E")
        >>> client.call("fund_daily", trade_date="20250102")
        """
        self._limiter.acquire()
        log.debug(f"tushare call: {api_name} {kwargs}")
        method = getattr(self._pro, api_name, None)
        if method is None:
            # Fallback to generic query
            df = self._pro.query(api_name, **kwargs)
        else:
            df = method(**kwargs)
        if df is None:
            return pd.DataFrame()
        return df


@lru_cache(maxsize=1)
def get_client() -> TushareClient:
    return TushareClient()
