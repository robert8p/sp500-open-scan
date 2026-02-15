from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import re

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

FALLBACK = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B","JPM","JNJ",
    "V","PG","XOM","UNH","MA","HD","LLY","ABBV","CVX","AVGO",
]


def _normalize_symbol(sym: str) -> str:
    sym = sym.strip().upper()
    # Wikipedia uses BRK.B, BF.B. Most APIs want BRK-B, BF-B
    sym = sym.replace(".", "-")
    return sym


def fetch_sp500_tickers_from_wikipedia() -> list[str]:
    tables = pd.read_html(WIKI_URL)
    # first table usually contains "Symbol"
    df = tables[0]
    if "Symbol" not in df.columns:
        raise ValueError("Wikipedia table missing 'Symbol' column")
    syms = [_normalize_symbol(s) for s in df["Symbol"].astype(str).tolist()]
    syms = [s for s in syms if re.fullmatch(r"[A-Z0-9\-]{1,10}", s)]
    # de-duplicate preserving order
    seen = set()
    out = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def load_sp500_tickers(cache_path: str, max_tickers: int) -> list[str]:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use cache if newer than 7 days
    if path.exists():
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if datetime.now(timezone.utc) - mtime < timedelta(days=7):
            try:
                df = pd.read_csv(path)
                if "symbol" in df.columns and len(df) > 0:
                    return df["symbol"].astype(str).tolist()[:max_tickers]
            except Exception:
                pass

    # Refresh
    try:
        syms = fetch_sp500_tickers_from_wikipedia()
        pd.DataFrame({"symbol": syms}).to_csv(path, index=False)
        return syms[:max_tickers]
    except Exception:
        # Last resort fallback list
        return FALLBACK[:max_tickers]
