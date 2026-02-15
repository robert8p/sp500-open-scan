from __future__ import annotations

from app.config import Settings
from app.data_providers.alpaca import AlpacaProvider
from app.data_providers.base import MarketDataProvider


def get_provider(settings: Settings) -> MarketDataProvider:
    if settings.data_provider == "alpaca":
        return AlpacaProvider(
            key_id=settings.alpaca_key_id or "",
            secret_key=settings.alpaca_secret_key or "",
            feed=settings.alpaca_feed,
            ticker_cache_path=settings.ticker_cache_path,
        )

    raise ValueError(f"Unsupported provider: {settings.data_provider}")
