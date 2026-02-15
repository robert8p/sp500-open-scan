from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any
import pandas as pd
import requests

from app.data_providers.base import Snapshot, NewsArticle
from app.tickers import load_sp500_tickers


class AlpacaProvider:
    name = "alpaca"

    def __init__(self, key_id: str, secret_key: str, feed: str, ticker_cache_path: str):
        if not key_id or not secret_key:
            raise ValueError("Missing Alpaca credentials. Set ALPACA_KEY_ID and ALPACA_SECRET_KEY.")
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self._feed = feed or "iex"
        self._ticker_cache_path = ticker_cache_path

    def get_sp500_tickers(self, max_tickers: int) -> list[str]:
        return load_sp500_tickers(self._ticker_cache_path, max_tickers)

    def get_snapshots(self, symbols: list[str]) -> dict[str, Snapshot]:
        out: dict[str, Snapshot] = {}
        if not symbols:
            return out

        for chunk in _chunks(symbols, 200):
            url = "https://data.alpaca.markets/v2/stocks/snapshots"
            params = {"symbols": ",".join(chunk), "feed": self._feed}
            r = requests.get(url, headers=self._headers, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            for sym, snap in data.items():
                daily = snap.get("dailyBar") or {}
                prev = snap.get("prevDailyBar") or {}
                latest_trade = snap.get("latestTrade") or {}
                ts = latest_trade.get("t") or daily.get("t") or prev.get("t")
                out[sym] = Snapshot(
                    symbol=sym,
                    open=_to_float(daily.get("o")),
                    prev_close=_to_float(prev.get("c")),
                    last_price=_to_float(latest_trade.get("p")),
                    day_volume=_to_float(daily.get("v")),
                    timestamp_utc=ts,
                )

        return out

    def get_daily_history(self, symbols: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
        """Fetch daily bars for each symbol for roughly the last `lookback_days` trading days."""
        if not symbols:
            return {}

        # Use a generous calendar window to cover weekends/holidays.
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(lookback_days * 2.2))

        frames: dict[str, list[dict[str, Any]]] = {s: [] for s in symbols}

        # Keep each request under typical limits by chunking symbols.
        # We'll fetch in ascending order for stable indicator calculations.
        for chunk in _chunks(symbols, 25):
            url = "https://data.alpaca.markets/v2/stocks/bars"
            params = {
                "symbols": ",".join(chunk),
                "timeframe": "1Day",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "feed": self._feed,
                "adjustment": "all",
                "sort": "asc",
                "limit": 10000,
            }

            next_token = None
            while True:
                p = dict(params)
                if next_token:
                    p["page_token"] = next_token
                r = requests.get(url, headers=self._headers, params=p, timeout=60)
                r.raise_for_status()
                payload = r.json()
                bars = payload.get("bars") or {}
                for sym, items in bars.items():
                    frames.setdefault(sym, []).extend(items)

                next_token = payload.get("next_page_token")
                if not next_token:
                    break

        out: dict[str, pd.DataFrame] = {}
        for sym, items in frames.items():
            if not items:
                continue
            df = pd.DataFrame(items)
            # Expect columns like: t,o,h,l,c,v
            df = df.rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df = df.dropna(subset=["ts"])
            df["date"] = df["ts"].dt.date
            df = df.drop(columns=["ts"])
            df = df.groupby("date", as_index=False).last()
            df = df.set_index("date").sort_index()
            # Keep only last lookback_days rows
            if len(df) > lookback_days:
                df = df.iloc[-lookback_days:]
            out[sym] = df[["open", "high", "low", "close", "volume"]].astype(float)

        return out

    def get_intraday_bars(
        self,
        symbols: list[str],
        start_utc: str,
        end_utc: str,
        timeframe: str = "1Min",
    ) -> dict[str, pd.DataFrame]:
        """Fetch intraday bars (default 1Min) for multiple symbols."""
        if not symbols:
            return {}

        frames: dict[str, list[dict[str, Any]]] = {s: [] for s in symbols}
        for chunk in _chunks(symbols, 50):
            url = "https://data.alpaca.markets/v2/stocks/bars"
            params = {
                "symbols": ",".join(chunk),
                "timeframe": timeframe,
                "start": start_utc,
                "end": end_utc,
                "feed": self._feed,
                "adjustment": "all",
                "sort": "asc",
                "limit": 10000,
            }

            next_token = None
            while True:
                p = dict(params)
                if next_token:
                    p["page_token"] = next_token
                r = requests.get(url, headers=self._headers, params=p, timeout=60)
                r.raise_for_status()
                payload = r.json()
                bars = payload.get("bars") or {}
                for sym, items in bars.items():
                    frames.setdefault(sym, []).extend(items)
                next_token = payload.get("next_page_token")
                if not next_token:
                    break

        out: dict[str, pd.DataFrame] = {}
        for sym, items in frames.items():
            if not items:
                continue
            df = pd.DataFrame(items)
            # t,o,h,l,c,v and sometimes vw,n
            df = df.rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "n_trades"})
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
            for col in ["open","high","low","close","volume","vwap"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            out[sym] = df
        return out

    def get_news(self, symbols: list[str] | None, limit: int) -> list[NewsArticle]:
        """Fetch latest news articles via Alpaca's News API."""
        url = "https://data.alpaca.markets/v1beta1/news"
        params: dict[str, Any] = {
            "sort": "desc",
            "limit": int(limit),
        }
        if symbols:
            params["symbols"] = ",".join(symbols)
        r = requests.get(url, headers=self._headers, params=params, timeout=30)
        # Some plans return 403 for news; surface as empty list.
        if r.status_code == 403:
            return []
        r.raise_for_status()
        payload = r.json() or {}
        articles = payload.get("news") or payload.get("data") or []
        out: list[NewsArticle] = []
        for a in articles:
            out.append(
                NewsArticle(
                    headline=str(a.get("headline") or "").strip(),
                    summary=(str(a.get("summary")) if a.get("summary") is not None else None),
                    url=(str(a.get("url")) if a.get("url") is not None else None),
                    created_at_utc=(str(a.get("created_at")) if a.get("created_at") is not None else None),
                    symbols=a.get("symbols") if isinstance(a.get("symbols"), list) else None,
                )
            )
        return [a for a in out if a.headline]


def _chunks(xs: list[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _to_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None
