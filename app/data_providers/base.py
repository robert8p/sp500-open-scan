from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import pandas as pd
from typing import Protocol, Any


@dataclass(frozen=True)
class Snapshot:
    symbol: str
    open: float | None
    prev_close: float | None
    last_price: float | None
    day_volume: float | None = None
    # Optional; when present should be an ISO-8601 UTC string.
    timestamp_utc: str | None = None


@dataclass(frozen=True)
class NewsArticle:
    headline: str
    summary: str | None
    url: str | None
    created_at_utc: str | None
    symbols: list[str] | None = None


class MarketDataProvider(Protocol):
    name: str

    def get_sp500_tickers(self, max_tickers: int) -> list[str]:
        ...

    def get_daily_history(self, symbols: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
        """Return {symbol: df} with columns: ['open','high','low','close','volume'] indexed by date."""
        ...

    def get_snapshots(self, symbols: list[str]) -> dict[str, Snapshot]:
        """Return latest snapshot (incl today's open + prev close if available)."""
        ...

    def get_intraday_bars(self, symbols: list[str], start_utc: str, end_utc: str, timeframe: str = "1Min") -> dict[str, pd.DataFrame]:
        """Return {symbol: df} of intraday bars between start/end (UTC ISO)."""
        ...

    def get_news(self, symbols: list[str] | None, limit: int) -> list[NewsArticle]:
        """Return latest news articles. If symbols is None/empty, return general market news."""
        ...
