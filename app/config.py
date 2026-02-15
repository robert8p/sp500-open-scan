from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Security
    # If not set, admin endpoints are left unprotected (NOT recommended for public apps).
    # Render Blueprint sets this automatically via render.yaml.
    admin_token: str = Field("", alias="ADMIN_TOKEN")

    # Data provider
    data_provider: Literal["alpaca"] = Field("alpaca", alias="DATA_PROVIDER")

    # Alpaca
    alpaca_key_id: str | None = Field(None, alias="ALPACA_KEY_ID")
    alpaca_secret_key: str | None = Field(None, alias="ALPACA_SECRET_KEY")
    alpaca_feed: str = Field("iex", alias="ALPACA_FEED")  # often "iex" (free) or "sip" (paid)

    # Paths (relative to repo root)
    # If DATABASE_URL is set (Render Postgres), the app will use it. Otherwise it falls back to DB_PATH (SQLite).
    database_url: str | None = Field(None, alias="DATABASE_URL")
    db_path: str = Field("data/app.db", alias="DB_PATH")
    model_path: str = Field("data/model.joblib", alias="MODEL_PATH")
    ticker_cache_path: str = Field("data/sp500_tickers.csv", alias="TICKER_CACHE_PATH")

    # Scan behavior
    scan_max_tickers: int = Field(500, alias="SCAN_MAX_TICKERS")
    scan_at_minutes_after_open: int = Field(5, alias="SCAN_AT_MINUTES_AFTER_OPEN")
    scan_window_minutes: int = Field(30, alias="SCAN_WINDOW_MINUTES")

    # Training
    model_lookback_days: int = Field(756, alias="MODEL_LOOKBACK_DAYS")  # ~3 years of trading days
    history_days_for_features: int = Field(260, alias="HISTORY_DAYS_FOR_FEATURES")

    # News sentiment
    news_max_articles: int = Field(50, alias="NEWS_MAX_ARTICLES")
    per_symbol_news_articles: int = Field(10, alias="PER_SYMBOL_NEWS_ARTICLES")


def get_settings() -> Settings:
    return Settings()
