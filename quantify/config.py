"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TushareConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TUSHARE_", env_file=".env", extra="ignore")

    token: str = Field(default="", description="Tushare Pro API token")
    rate_per_min: int = Field(default=480, ge=1, description="Max API calls per minute")
    max_workers: int = Field(
        default=2,
        ge=1,
        description="Concurrent fetch threads. The jiaoch.site mirror caps concurrency at 2; "
        "the official API has no concurrency wall, only the per-minute rate limit.",
    )
    http_url: str = Field(default="http://jiaoch.site", description="Tushare API base URL")
    http_timeout: float = Field(default=30.0, gt=0, description="HTTP request timeout in seconds")


class MySQLConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MYSQL_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = "root"
    database: str = "quantify"
    charset: str = "utf8mb4"

    @property
    def url(self) -> str:
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}?charset={self.charset}"
        )

    @property
    def server_url(self) -> str:
        """Connection URL without a target database (used to CREATE DATABASE)."""
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/?charset={self.charset}"


class LogConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOG_", env_file=".env", extra="ignore")

    level: str = "INFO"
    dir: str = "logs"


class LLMConfig(BaseSettings):
    """LLM provider config (DeepSeek by default; any OpenAI-compatible endpoint works)."""

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    api_key: str = Field(default="", description="LLM API key (DeepSeek 控制台获取)")
    base_url: str = Field(default="https://api.deepseek.com", description="OpenAI 兼容 base_url")
    model: str = Field(default="deepseek-chat", description="模型名，如 deepseek-chat / deepseek-reasoner")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="采样温度，因子探索建议偏高")
    max_tokens: int = Field(default=4096, ge=256, description="单次回复最大 token 数")
    timeout: float = Field(default=120.0, gt=0, description="请求超时（秒）")


class QlibConfig(BaseSettings):
    """Qlib data-layer config: where the dumped .bin data lives."""

    model_config = SettingsConfigDict(env_prefix="QLIB_", env_file=".env", extra="ignore")

    provider_uri: str = Field(
        default=str(PROJECT_ROOT / "qlib_data" / "cn_data"),
        description="Qlib .bin 数据目录（dump 输出 / qlib.init 读取）",
    )
    region: str = Field(default="cn", description="Qlib 区域，A 股用 cn")

    @field_validator("provider_uri", mode="before")
    @classmethod
    def _default_provider_uri(cls, value: str | None) -> str:
        # 允许 .env 里留空 QLIB_PROVIDER_URI=，此时回落到项目默认目录。
        if value is None or str(value).strip() == "":
            return str(PROJECT_ROOT / "qlib_data" / "cn_data")
        return str(value).strip()


class Settings:
    """Aggregate settings entry point."""

    def __init__(self) -> None:
        self.tushare = TushareConfig()
        self.mysql = MySQLConfig()
        self.log = LogConfig()
        self.llm = LLMConfig()
        self.qlib = QlibConfig()
        self.project_root = PROJECT_ROOT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
