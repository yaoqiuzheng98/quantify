"""Thin wrapper around the Tushare Pro HTTP API with rate limiting + retry.

We post directly to the Tushare Pro endpoint instead of going through the
``tushare`` SDK so that transport-level failures surface as exceptions. The
SDK's ``DataApi.query`` swallows any non-2xx HTTP response and returns an empty
DataFrame, which is indistinguishable from a legitimately empty result (e.g. a
newly listed stock with no history). By calling ``raise_for_status()`` we let
the ``@retry`` decorator back off and retry transient 5xx errors, while a truly
empty payload still returns a clean empty DataFrame that callers can treat as
"no data" without retrying.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
import requests
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
    """Rate-limited, retryable wrapper around the Tushare Pro HTTP API."""

    def __init__(self, token: str | None = None, rate_per_min: int | None = None) -> None:
        settings = get_settings().tushare
        self.token = token or settings.token
        if not self.token:
            raise RuntimeError("TUSHARE_TOKEN is empty. Please set it in the .env file or environment.")
        self.rate_per_min = rate_per_min or settings.rate_per_min
        self.http_url = settings.http_url.rstrip("/")
        self.http_timeout = settings.http_timeout
        self._limiter = RateLimiter(max_calls=self.rate_per_min, period=60.0)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
    )
    def call(self, api_name: str, *, fields: str = "", **kwargs: Any) -> pd.DataFrame:
        """Invoke a Tushare Pro endpoint by name with kwargs.

        Raises on transport errors (non-2xx HTTP) and API-level errors
        (``code != 0``) so the retry decorator can back off. A successful
        response with no rows returns an empty DataFrame.

        Examples
        --------
        >>> client.call("fund_basic", market="E")
        >>> client.call("fund_daily", trade_date="20250102")
        """
        self._limiter.acquire()
        log.debug(f"tushare call: {api_name} {kwargs}")
        payload = {
            "api_name": api_name,
            "token": self.token,
            "params": kwargs,
            "fields": fields,
        }
        res = requests.post(f"{self.http_url}/{api_name}", json=payload, timeout=self.http_timeout)
        res.raise_for_status()
        result = res.json()
        if result.get("code") != 0:
            raise RuntimeError(f"tushare API error for {api_name}: {result.get('msg')}")
        data = result.get("data")
        if not data or not data.get("items"):
            return pd.DataFrame(columns=data["fields"] if data and data.get("fields") else None)
        return pd.DataFrame(data["items"], columns=data["fields"])


@lru_cache(maxsize=1)
def get_client() -> TushareClient:
    return TushareClient()
