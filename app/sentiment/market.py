from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.data_providers.base import MarketDataProvider, NewsArticle


_analyzer = SentimentIntensityAnalyzer()


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _sentiment_from_articles(articles: list[NewsArticle]) -> dict[str, Any]:
    if not articles:
        return {"score": 0.0, "n": 0, "examples": []}

    scores = []
    examples: list[dict[str, str]] = []
    for a in articles:
        text = (a.headline or "") + (". " + a.summary if a.summary else "")
        s = _analyzer.polarity_scores(text)["compound"]
        scores.append(float(s))
        if len(examples) < 5:
            examples.append({"headline": a.headline, "created_at_utc": a.created_at_utc or ""})

    # Robust mean (winsorize)
    arr = np.array(scores, dtype=float)
    if len(arr) >= 5:
        lo, hi = np.quantile(arr, [0.1, 0.9])
        arr = np.clip(arr, lo, hi)

    return {"score": float(arr.mean()), "n": int(len(scores)), "examples": examples}


def compute_market_sentiment(
    provider: MarketDataProvider,
    history: dict[str, pd.DataFrame],
    intraday_5m: dict[str, dict[str, float]] | None = None,
    news_limit: int = 50,
) -> dict[str, Any]:
    """Compute a simple "risk-on" vs "risk-off" composite.

    This is intentionally lightweight and explainable. It blends:
    - Breadth: percent of S&P 500 above their 20D SMA (as-of yesterday)
    - 1D return in SPY and 1D change in VXX (vol proxy) if available
    - 5-minute advance/decline right after the open (if available)
    - Market-news sentiment (if user's Alpaca plan permits News API)
    """

    # 1) Breadth: close > sma20 (as-of last daily bar)
    above = 0
    total = 0
    for sym, df in history.items():
        if sym in {"SPY", "VXX"}:
            continue
        if df is None or len(df) < 25:
            continue
        close = df["close"].astype(float)
        sma20 = close.rolling(20).mean()
        last = close.index[-1]
        if pd.isna(sma20.loc[last]):
            continue
        total += 1
        if float(close.loc[last]) > float(sma20.loc[last]):
            above += 1
    breadth = (above / total) if total else 0.5

    # 2) SPY / VXX (optional but helpful)
    spy_ret_1d = None
    vxx_ret_1d = None
    if "SPY" in history and history["SPY"] is not None and len(history["SPY"]) >= 2:
        h = history["SPY"]["close"].astype(float)
        spy_ret_1d = float(h.iloc[-1] / h.iloc[-2] - 1.0)
    if "VXX" in history and history["VXX"] is not None and len(history["VXX"]) >= 2:
        h = history["VXX"]["close"].astype(float)
        vxx_ret_1d = float(h.iloc[-1] / h.iloc[-2] - 1.0)

    # 3) Open breadth (5m adv/dec)
    adv_dec = None
    if intraday_5m:
        rets = [m.get("ret_5m") for m in intraday_5m.values() if m.get("ret_5m") is not None]
        if rets:
            adv = sum(1 for r in rets if float(r) > 0)
            dec = sum(1 for r in rets if float(r) < 0)
            adv_dec = (adv - dec) / max(1, (adv + dec))

    # 4) Market news sentiment (may be unavailable)
    market_articles = provider.get_news(symbols=None, limit=news_limit)
    news = _sentiment_from_articles(market_articles)

    # Composite in [-1, +1]
    # Breadth maps from [0..1] to [-1..1]
    breadth_component = (breadth - 0.5) * 2.0
    spy_component = _clip((spy_ret_1d or 0.0) * 12.0, -1.0, 1.0)
    vxx_component = _clip(-(vxx_ret_1d or 0.0) * 10.0, -1.0, 1.0)  # higher VXX = risk-off
    advdec_component = _clip((adv_dec or 0.0) * 2.0, -1.0, 1.0)
    news_component = _clip(float(news["score"]) * 1.5, -1.0, 1.0)

    score = (
        0.35 * breadth_component
        + 0.25 * spy_component
        + 0.15 * vxx_component
        + 0.15 * advdec_component
        + 0.10 * news_component
    )
    score = _clip(score, -1.0, 1.0)

    if score >= 0.35:
        interpretation = "risk-on"
    elif score <= -0.35:
        interpretation = "risk-off"
    else:
        interpretation = "neutral"

    return {
        "score": float(score),
        "interpretation": interpretation,
        "components": {
            "breadth_above_sma20": float(breadth),
            "spy_ret_1d": spy_ret_1d,
            "vxx_ret_1d": vxx_ret_1d,
            "adv_dec_5m": adv_dec,
            "news_sentiment": news,
        },
    }
