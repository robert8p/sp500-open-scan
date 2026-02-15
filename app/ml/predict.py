from __future__ import annotations

from typing import Any

import numpy as np

from app.ml.model_io import load_bundle
from app.features.build import FEATURE_COLUMNS


def load_model_bundle(model_path: str) -> dict[str, Any]:
    return load_bundle(model_path)


def _normalize_target_key(target: str) -> str:
    t = (target or '').strip().lower()
    if t in {'5', '5pct', 'p5', 'five'}:
        return '5pct'
    if t in {'2', '2pct', 'p2', 'two'}:
        return '2pct'
    return t or '5pct'


def _get_target(bundle: dict[str, Any], target: str) -> tuple[list[str], Any]:
    """Return (feature_columns, model) for the requested target."""
    t = _normalize_target_key(target)

    # New multi-target bundle
    targets = bundle.get('targets')
    if isinstance(targets, dict) and targets:
        if t not in targets:
            # fall back to 5pct, then first key
            t = '5pct' if '5pct' in targets else next(iter(targets.keys()))
        td = targets[t]
        cols = list(td.get('feature_columns') or FEATURE_COLUMNS)
        model = td.get('model')
        if model is None:
            raise RuntimeError(f"Model bundle missing model for target '{t}'.")
        return cols, model

    # Legacy single-target bundle
    cols = list(bundle.get('feature_columns') or FEATURE_COLUMNS)
    model = bundle.get('model')
    if model is None:
        raise RuntimeError('Model bundle missing model.')
    return cols, model


def predict_probabilities(bundle: dict[str, Any], feature_rows: list[dict[str, float]], target: str = '5pct') -> list[float]:
    cols, model = _get_target(bundle, target)
    X = np.array([[float(r.get(c, 0.0) or 0.0) for c in cols] for r in feature_rows], dtype=float)
    p = model.predict_proba(X)[:, 1]
    return [float(x) for x in p]


def explain_features(f: dict[str, float]) -> dict[str, Any]:
    signals = []
    gap = f.get("gap", 0.0)
    rsi = f.get("rsi14_prev", 50.0)
    macd_h = f.get("macd_hist_prev", 0.0)
    sma20 = f.get("sma20_dist_prev", 0.0)
    volz = f.get("vol_z20_prev", 0.0)

    if gap >= 0.02:
        signals.append(f"Gap up {gap*100:.1f}% (bullish open)")
    elif gap <= -0.02:
        signals.append(f"Gap down {gap*100:.1f}% (bearish open)")

    if rsi <= 30:
        signals.append(f"RSI {rsi:.0f} (oversold bounce potential)")
    elif rsi >= 70:
        signals.append(f"RSI {rsi:.0f} (overbought / mean-reversion risk)")

    if macd_h > 0:
        signals.append("MACD histogram positive (momentum)")
    elif macd_h < 0:
        signals.append("MACD histogram negative (weak momentum)")

    if sma20 > 0:
        signals.append("Above 20D SMA (trend supportive)")
    elif sma20 < 0:
        signals.append("Below 20D SMA (trend headwind)")

    if volz >= 2:
        signals.append("Unusually high volume yesterday")

    return {"signals": signals[:6]}
