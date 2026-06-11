"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TushareConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TUSHARE_", env_file=".env", extra="ignore")

    token: str = Field(default="", description="Tushare Pro API token")
    rate_per_min: int = Field(default=480, ge=1, description="Max API calls per minute")
    http_url: str = Field(default="http://jiaoch.site", description="Tushare API base URL")


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


class Settings:
    """Aggregate settings entry point."""

    def __init__(self) -> None:
        self.tushare = TushareConfig()
        self.mysql = MySQLConfig()
        self.log = LogConfig()
        self.project_root = PROJECT_ROOT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
