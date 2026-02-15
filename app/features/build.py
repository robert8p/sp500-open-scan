from __future__ import annotations

import pandas as pd
import numpy as np
# pandas-ta is no longer installable for Python 3.11+.
# Use the community-maintained fork that supports modern Python versions.
import pandas_ta_classic as ta  # enables df.ta accessor


FEATURE_COLUMNS = [
    "gap",
    "ret1_prev", "ret3_prev", "ret5_prev", "ret10_prev",
    "rsi14_prev",
    "macd_hist_prev",
    "atr14_prev_norm",
    "sma20_dist_prev", "sma50_dist_prev", "sma200_dist_prev",
    "vol_z20_prev",
    "range_prev",

    # Additional trend / volatility / momentum
    "ema20_dist_prev", "ema50_dist_prev",
    "bb_pctb_prev", "bb_width_prev",
    "stochk_prev", "stochd_prev",
    "adx14_prev",
    "cci20_prev",
    "mfi14_prev",
    "obv_norm_prev",

    # Candle shape (previous day)
    "body_prev", "upper_wick_prev", "lower_wick_prev",
]


def build_training_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    """
    daily: index=date, columns open/high/low/close/volume
    returns DataFrame with FEATURE_COLUMNS + label
    label is whether (high/open - 1) >= 0.05 for that day.
    Features are computed as-of the open of that day (mostly previous-day values + today's gap).
    """
    df = daily.copy().sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")

    # Indicators computed from history up to each day.
    # We use the DataFrame accessor (df.ta.*) which is the most stable API.
    rsi14 = df.ta.rsi(length=14)
    atr14 = df.ta.atr(length=14)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    sma20 = df.ta.sma(length=20)
    sma50 = df.ta.sma(length=50)
    sma200 = df.ta.sma(length=200)

    ema20 = df.ta.ema(length=20)
    ema50 = df.ta.ema(length=50)

    bb = df.ta.bbands(length=20, std=2.0)
    stoch = df.ta.stoch(k=14, d=3)
    adx = df.ta.adx(length=14)
    cci20 = df.ta.cci(length=20)
    mfi14 = df.ta.mfi(length=14)
    obv = df.ta.obv()
    ret1 = df["close"].pct_change(1)
    ret3 = df["close"].pct_change(3)
    ret5 = df["close"].pct_change(5)
    ret10 = df["close"].pct_change(10)

    vol_mean20 = df["volume"].rolling(20).mean()
    vol_std20 = df["volume"].rolling(20).std(ddof=0)

    # Candle shape (daily)
    body = (df["close"] - df["open"]) / df["open"]
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]

    # Labels use same-day OHLC
    label = ((df["high"] / df["open"] - 1.0) >= 0.05).astype(int)
    label_2pct = ((df["high"] / df["open"] - 1.0) >= 0.02).astype(int)

    # Gap is known at open: open_today / prev_close - 1
    gap = df["open"] / df["close"].shift(1) - 1.0

    # Previous-day features (shift(1) so day d uses day d-1 indicators)
    feat = pd.DataFrame(index=df.index)
    feat["gap"] = gap

    feat["ret1_prev"] = ret1.shift(1)
    feat["ret3_prev"] = ret3.shift(1)
    feat["ret5_prev"] = ret5.shift(1)
    feat["ret10_prev"] = ret10.shift(1)

    feat["rsi14_prev"] = rsi14.shift(1)

    if macd is not None and "MACDh_12_26_9" in macd.columns:
        feat["macd_hist_prev"] = macd["MACDh_12_26_9"].shift(1)
    else:
        feat["macd_hist_prev"] = np.nan

    feat["atr14_prev_norm"] = (atr14 / df["close"]).shift(1)

    feat["sma20_dist_prev"] = (df["close"] / sma20 - 1.0).shift(1)
    feat["sma50_dist_prev"] = (df["close"] / sma50 - 1.0).shift(1)
    feat["sma200_dist_prev"] = (df["close"] / sma200 - 1.0).shift(1)

    feat["vol_z20_prev"] = ((df["volume"] - vol_mean20) / vol_std20).shift(1)

    feat["range_prev"] = ((df["high"] - df["low"]) / df["open"]).shift(1)

    # Extra indicators (previous-day values)
    feat["ema20_dist_prev"] = (df["close"] / ema20 - 1.0).shift(1)
    feat["ema50_dist_prev"] = (df["close"] / ema50 - 1.0).shift(1)

    if bb is not None and len(getattr(bb, "columns", [])):
        bbp_col = next((c for c in bb.columns if c.startswith("BBP_")), None)
        bbb_col = next((c for c in bb.columns if c.startswith("BBB_")), None)
        feat["bb_pctb_prev"] = bb[bbp_col].shift(1) if bbp_col else np.nan
        feat["bb_width_prev"] = bb[bbb_col].shift(1) if bbb_col else np.nan
    else:
        feat["bb_pctb_prev"] = np.nan
        feat["bb_width_prev"] = np.nan

    if stoch is not None and len(getattr(stoch, "columns", [])):
        k_col = next((c for c in stoch.columns if c.startswith("STOCHk_")), None)
        d_col = next((c for c in stoch.columns if c.startswith("STOCHd_")), None)
        feat["stochk_prev"] = stoch[k_col].shift(1) if k_col else np.nan
        feat["stochd_prev"] = stoch[d_col].shift(1) if d_col else np.nan
    else:
        feat["stochk_prev"] = np.nan
        feat["stochd_prev"] = np.nan

    if adx is not None and len(getattr(adx, "columns", [])):
        adx_col = next((c for c in adx.columns if c.startswith("ADX_")), None)
        feat["adx14_prev"] = adx[adx_col].shift(1) if adx_col else np.nan
    else:
        feat["adx14_prev"] = np.nan

    feat["cci20_prev"] = cci20.shift(1) if cci20 is not None else np.nan
    feat["mfi14_prev"] = mfi14.shift(1) if mfi14 is not None else np.nan

    # OBV normalized by trailing average volume (proxy for accumulation/distribution)
    obv_change20 = obv.diff(20)
    denom = (vol_mean20 * 20).replace(0, np.nan)
    feat["obv_norm_prev"] = (obv_change20 / denom).shift(1)

    # Candle shape (previous day)
    feat["body_prev"] = body.shift(1)
    feat["upper_wick_prev"] = upper_wick.shift(1)
    feat["lower_wick_prev"] = lower_wick.shift(1)

    feat["label"] = label
    feat["label_2pct"] = label_2pct

    # Keep only relevant columns and drop rows with missing values
    keep = FEATURE_COLUMNS + ["label", "label_2pct"]
    feat = feat[keep].replace([np.inf, -np.inf], np.nan).dropna()
    return feat


def build_features_for_today(history_daily: pd.DataFrame, open_today: float, prev_close: float) -> dict[str, float]:
    """Build features for a single symbol 'as-of open' using history up to yesterday + today's open/prev_close."""
    if history_daily is None or history_daily.empty:
        raise ValueError("Missing history_daily")

    df = history_daily.copy().sort_index()

    # last row is assumed to be yesterday (prev day)
    # Compute indicators using full history, then take last value (prev day).
    rsi14 = df.ta.rsi(length=14)
    atr14 = df.ta.atr(length=14)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    sma20 = df.ta.sma(length=20)
    sma50 = df.ta.sma(length=50)
    sma200 = df.ta.sma(length=200)

    ema20 = df.ta.ema(length=20)
    ema50 = df.ta.ema(length=50)

    bb = df.ta.bbands(length=20, std=2.0)
    stoch = df.ta.stoch(k=14, d=3)
    adx = df.ta.adx(length=14)
    cci20 = df.ta.cci(length=20)
    mfi14 = df.ta.mfi(length=14)
    obv = df.ta.obv()

    # Candle shape (previous day)
    body = (df["close"] - df["open"]) / df["open"]
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]

    ret1 = df["close"].pct_change(1)
    ret3 = df["close"].pct_change(3)
    ret5 = df["close"].pct_change(5)
    ret10 = df["close"].pct_change(10)

    vol_mean20 = df["volume"].rolling(20).mean()
    vol_std20 = df["volume"].rolling(20).std(ddof=0)

    last_idx = df.index[-1]

    bbp_col = next((c for c in getattr(bb, "columns", []) if c.startswith("BBP_")), None)
    bbb_col = next((c for c in getattr(bb, "columns", []) if c.startswith("BBB_")), None)
    stochk_col = next((c for c in getattr(stoch, "columns", []) if c.startswith("STOCHk_")), None)
    stochd_col = next((c for c in getattr(stoch, "columns", []) if c.startswith("STOCHd_")), None)
    adx_col = next((c for c in getattr(adx, "columns", []) if c.startswith("ADX_")), None)

    denom20 = (vol_mean20 * 20).replace(0, np.nan)
    obv_norm_series = (obv.diff(20) / denom20) if obv is not None else None

    feat = {
        "gap": float(open_today / prev_close - 1.0),
        "ret1_prev": float(ret1.loc[last_idx]),
        "ret3_prev": float(ret3.loc[last_idx]),
        "ret5_prev": float(ret5.loc[last_idx]),
        "ret10_prev": float(ret10.loc[last_idx]),
        "rsi14_prev": float(rsi14.loc[last_idx]),
        "macd_hist_prev": float(macd.loc[last_idx, "MACDh_12_26_9"]) if macd is not None and "MACDh_12_26_9" in macd.columns else 0.0,
        "atr14_prev_norm": float((atr14.loc[last_idx] / df.loc[last_idx, "close"])),
        "sma20_dist_prev": float((df.loc[last_idx, "close"] / sma20.loc[last_idx] - 1.0)),
        "sma50_dist_prev": float((df.loc[last_idx, "close"] / sma50.loc[last_idx] - 1.0)),
        "sma200_dist_prev": float((df.loc[last_idx, "close"] / sma200.loc[last_idx] - 1.0)),
        "vol_z20_prev": float(((df.loc[last_idx, "volume"] - vol_mean20.loc[last_idx]) / vol_std20.loc[last_idx])),
        "range_prev": float(((df.loc[last_idx, "high"] - df.loc[last_idx, "low"]) / df.loc[last_idx, "open"])),

        "ema20_dist_prev": float((df.loc[last_idx, "close"] / ema20.loc[last_idx] - 1.0)) if ema20 is not None else 0.0,
        "ema50_dist_prev": float((df.loc[last_idx, "close"] / ema50.loc[last_idx] - 1.0)) if ema50 is not None else 0.0,

        "bb_pctb_prev": float(bb[bbp_col].loc[last_idx]) if bb is not None and bbp_col else 0.0,
        "bb_width_prev": float(bb[bbb_col].loc[last_idx]) if bb is not None and bbb_col else 0.0,

        "stochk_prev": float(stoch[stochk_col].loc[last_idx]) if stoch is not None and stochk_col else 0.0,
        "stochd_prev": float(stoch[stochd_col].loc[last_idx]) if stoch is not None and stochd_col else 0.0,

        "adx14_prev": float(adx[adx_col].loc[last_idx]) if adx is not None and adx_col else 0.0,
        "cci20_prev": float(cci20.loc[last_idx]) if cci20 is not None else 0.0,
        "mfi14_prev": float(mfi14.loc[last_idx]) if mfi14 is not None else 0.0,

        "obv_norm_prev": float(obv_norm_series.loc[last_idx]) if obv_norm_series is not None else 0.0,

        "body_prev": float(body.loc[last_idx]),
        "upper_wick_prev": float(upper_wick.loc[last_idx]),
        "lower_wick_prev": float(lower_wick.loc[last_idx]),
    }

    # Replace any non-finite values with 0
    for k, v in list(feat.items()):
        if not np.isfinite(v):
            feat[k] = 0.0

    return feat
