from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from app.config import Settings
from app.data_providers.factory import get_provider
from app.db import ensure_db, insert_scan, insert_scores, get_engine
from app.market_time import should_run_scan
from app.ml.model_io import bundle_exists, load_bundle
from app.ml.train import train_and_save
from app.features.build import build_features_for_today
from app.ml.predict import predict_probabilities, explain_features
from app.sentiment.market import compute_market_sentiment


def _compute_5m_metrics(bars: dict[str, Any], minutes: int = 5) -> dict[str, dict[str, float]]:
    """Reduce per-symbol 1Min bars into the first `minutes` of the session."""
    out: dict[str, dict[str, float]] = {}
    for sym, df in bars.items():
        try:
            if df is None or df.empty:
                continue
            d = df.copy().head(int(minutes))
            # Some feeds return tz-aware index; assume already UTC
            first_open = float(d["open"].iloc[0])
            last_close = float(d["close"].iloc[-1])
            ret_5m = last_close / first_open - 1.0
            hi = float(d["high"].max())
            lo = float(d["low"].min())
            rng = (hi - lo) / first_open if first_open else 0.0
            vol = float(d["volume"].sum()) if "volume" in d.columns else 0.0
            vwap = None
            if "vwap" in d.columns and d["vwap"].notna().any():
                vwap = float(d["vwap"].dropna().iloc[-1])
            out[sym] = {
                "ret_5m": float(ret_5m),
                "range_5m": float(rng),
                "vol_5m": float(vol),
                "vwap_5m": vwap,
            }
        except Exception:
            continue
    return out


def _apply_probability_adjustments(
    p_model: float,
    market_score: float,
    intraday: dict[str, float] | None,
    expected_5m_vol: float | None,
) -> tuple[float, dict[str, float]]:
    """Small, explainable adjustments on top of the calibrated model probability."""
    p = float(p_model)
    adj: dict[str, float] = {}

    # Market regime (risk-on boosts; risk-off dampens)
    market_adj = 1.0 + 0.20 * float(market_score)
    adj["market_multiplier"] = float(market_adj)

    # Early momentum (5m return) – modest effect
    mom_mult = 1.0
    if intraday and intraday.get("ret_5m") is not None:
        r = float(intraday["ret_5m"])
        mom_mult = 1.0 + 0.25 * float(__import__("math").tanh(r * 35.0))
    adj["momentum_multiplier"] = float(mom_mult)

    # Relative volume vs expected (avg_daily/78)
    vol_mult = 1.0
    if intraday and intraday.get("vol_5m") is not None and expected_5m_vol and expected_5m_vol > 0:
        rel = float(intraday["vol_5m"]) / float(expected_5m_vol)
        # cap rel to keep stability
        rel = max(0.0, min(5.0, rel))
        vol_mult = 1.0 + 0.10 * float(__import__("math").tanh((rel - 1.0) * 0.9))
    adj["rel_volume_multiplier"] = float(vol_mult)

    p_adj = p * market_adj * mom_mult * vol_mult
    p_adj = float(max(0.0, min(1.0, p_adj)))
    return p_adj, adj


def run_scan(settings: Settings, force: bool = False) -> dict[str, Any]:
    engine = get_engine(settings)
    ensure_db(engine)

    now_utc = datetime.now(timezone.utc)
    ok, timing = should_run_scan(now_utc, settings.scan_at_minutes_after_open, settings.scan_window_minutes)
    if (not ok) and (not force):
        return {"status": "skipped", "timing": timing}

    provider = get_provider(settings)
    tickers = provider.get_sp500_tickers(max_tickers=settings.scan_max_tickers)

    # Add a couple of broad market symbols to improve sentiment context.
    extra_symbols = ["SPY", "VXX"]

    # Ensure model exists (auto-train once if missing)
    if not bundle_exists(settings.model_path):
        train_and_save()
    bundle = load_bundle(settings.model_path)
    model_version = str(bundle.get("version", "unknown"))

    # Get history for indicators and snapshots for today's open/prev close
    history = provider.get_daily_history(tickers + extra_symbols, lookback_days=settings.history_days_for_features)
    snaps = provider.get_snapshots(tickers + extra_symbols)

    today = now_utc.date()

    feature_rows = []
    row_meta = []

    skipped = 0
    for sym in tickers:
        hist = history.get(sym)
        snap = snaps.get(sym)
        if hist is None or hist.empty or snap is None:
            skipped += 1
            continue

        # Remove today's partial bar if present
        if len(hist) >= 2 and hist.index[-1] == today:
            hist = hist.iloc[:-1]

        if hist.empty:
            skipped += 1
            continue

        open_today = snap.open
        prev_close = snap.prev_close

        # Fall back to last known close if snapshot missing prev_close
        if prev_close is None:
            try:
                prev_close = float(hist["close"].iloc[-1])
            except Exception:
                prev_close = None

        if open_today is None or prev_close is None:
            skipped += 1
            continue

        try:
            feats = build_features_for_today(hist, open_today=float(open_today), prev_close=float(prev_close))
        except Exception:
            skipped += 1
            continue

        feature_rows.append(feats)
        row_meta.append({"symbol": sym, "features": feats})

    if not feature_rows:
        return {"status": "error", "message": "No tickers could be scored (missing open/prev_close/history).", "skipped": skipped}

    probs_model_5 = predict_probabilities(bundle, feature_rows, target="5pct")
    probs_model_2 = predict_probabilities(bundle, feature_rows, target="2pct")

    # Intraday 5-min bar metrics (open -> now). If the scan runs at ~5 minutes after open,
    # this captures the early impulse + early volume.
    intraday_metrics: dict[str, dict[str, float]] = {}
    try:
        # Prefer the actual NYSE open time for today's session.
        # If unavailable, fall back to a small window ending at now.
        open_utc = timing.get("open_utc")
        if open_utc:
            start = str(open_utc)
        else:
            start = (now_utc - timedelta(minutes=max(1, settings.scan_at_minutes_after_open))).isoformat()
        end = now_utc.isoformat()
        bars = provider.get_intraday_bars(tickers, start_utc=start, end_utc=end, timeframe="1Min")
        intraday_metrics = _compute_5m_metrics(bars, minutes=settings.scan_at_minutes_after_open)
    except Exception:
        intraday_metrics = {}

    sentiment = compute_market_sentiment(
        provider=provider,
        history=history,
        intraday_5m=intraday_metrics,
        news_limit=settings.news_max_articles,
    )

    results = []
    for meta, p_model5, p_model2 in zip(row_meta, probs_model_5, probs_model_2):
        sym = meta["symbol"]
        feats = meta["features"]
        intr = intraday_metrics.get(sym)

        # Expected 5m volume = avg daily vol / 78 (390 minutes / 5)
        expected_5m = None
        try:
            hist = history.get(sym)
            if hist is not None and not hist.empty:
                avg_vol = float(hist["volume"].astype(float).tail(20).mean())
                expected_5m = avg_vol / 78.0
        except Exception:
            expected_5m = None

        p_final_5, adjs = _apply_probability_adjustments(
            p_model=float(p_model5),
            market_score=float(sentiment.get("score") or 0.0),
            intraday=intr,
            expected_5m_vol=expected_5m,
        )
        p_final_2, _ = _apply_probability_adjustments(
            p_model=float(p_model2),
            market_score=float(sentiment.get("score") or 0.0),
            intraday=intr,
            expected_5m_vol=expected_5m,
        )

        score_5 = int(round(p_final_5 * 100))
        score_2 = int(round(p_final_2 * 100))

        reasons = explain_features(feats)
        reasons["market_sentiment"] = sentiment.get("interpretation")
        if intr:
            reasons["intraday"] = intr
        reasons["adjustments"] = adjs

        results.append({
            "symbol": sym,
            # +5% target (primary)
            "prob_5pct": float(p_final_5),
            "prob_model": float(p_model5),
            "score": score_5,
            # +2% target
            "prob_2pct": float(p_final_2),
            "prob_model_2pct": float(p_model2),
            "score_2pct": score_2,
            "features": feats,
            "reasons": reasons,
        })

    results.sort(key=lambda r: r["prob_5pct"], reverse=True)

    scan_id = insert_scan(engine, provider=provider.name, model_version=model_version, market_sentiment=sentiment)
    insert_scores(engine, scan_id, results)

    return {
        "status": "ok",
        "scan_id": scan_id,
        "run_at_utc": now_utc.isoformat(),
        "provider": provider.name,
        "model_version": model_version,
        "timing": timing,
        "tickers_scored": len(results),
        "tickers_skipped": skipped,
        "top10": results[:10],
        "market_sentiment": sentiment,
    }
