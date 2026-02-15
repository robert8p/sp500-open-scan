from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV

from app.config import get_settings
from app.data_providers.factory import get_provider
from app.features.build import build_training_matrix, FEATURE_COLUMNS
from app.ml.model_io import save_bundle


def _build_dataset(history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for sym, df in tqdm(history.items(), desc="Building features"):
        try:
            m = build_training_matrix(df)
            if m.empty:
                continue
            m = m.copy()
            m["symbol"] = sym
            m["date"] = m.index.astype(str)
            rows.append(m)
        except Exception:
            continue
    if not rows:
        raise RuntimeError("No training data could be built.")
    return pd.concat(rows, axis=0, ignore_index=True)


def _time_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = dataset.copy()
    dataset["date_dt"] = pd.to_datetime(dataset["date"], errors="coerce")
    dataset = dataset.dropna(subset=["date_dt"]).sort_values("date_dt")
    cutoff = dataset["date_dt"].max() - pd.Timedelta(days=90)

    train_df = dataset[dataset["date_dt"] < cutoff]
    val_df = dataset[dataset["date_dt"] >= cutoff]

    if len(train_df) < 1000 or len(val_df) < 200:
        split_idx = int(len(dataset) * 0.8)
        train_df = dataset.iloc[:split_idx]
        val_df = dataset.iloc[split_idx:]
    return train_df, val_df


def _candidate_models() -> list[tuple[str, Pipeline]]:
    return [
        (
            "logreg_l1",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            penalty="l1",
                            solver="liblinear",
                            C=0.2,
                            class_weight="balanced",
                            max_iter=2000,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logreg_l2",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            penalty="l2",
                            solver="lbfgs",
                            C=0.5,
                            class_weight="balanced",
                            max_iter=2000,
                        ),
                    ),
                ]
            ),
        ),
        (
            "hgb",
            Pipeline(
                [
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_depth=3,
                            learning_rate=0.05,
                            max_iter=300,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "rf",
            Pipeline(
                [
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=400,
                            max_depth=10,
                            min_samples_leaf=10,
                            n_jobs=-1,
                            class_weight="balanced_subsample",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]


def _train_single_target(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str,
) -> tuple[dict, CalibratedClassifierCV]:
    """Train, select, and calibrate a probability model for a single label column."""

    X_train_full = train_df[FEATURE_COLUMNS].astype(float)
    y_train = train_df[label_col].astype(int).values
    X_val_full = val_df[FEATURE_COLUMNS].astype(float)
    y_val = val_df[label_col].astype(int).values

    # Sparse feature selection (L1 logreg) – yields a smaller, “optimal-ish” subset of indicators.
    selector = SelectFromModel(
        LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.12,
            class_weight="balanced",
            max_iter=2000,
        )
    )
    selector.fit(X_train_full.values, y_train)
    mask = selector.get_support()
    selected_cols = [c for c, m in zip(FEATURE_COLUMNS, mask) if m]
    if len(selected_cols) < 10:
        selected_cols = list(FEATURE_COLUMNS)

    datasets: list[tuple[str, list[str], np.ndarray, np.ndarray]] = [
        ("full", list(FEATURE_COLUMNS), X_train_full.values, X_val_full.values),
        (
            "selected",
            selected_cols,
            X_train_full[selected_cols].values,
            X_val_full[selected_cols].values,
        ),
    ]

    results = []
    best = None
    best_score = None

    for ds_name, cols, X_tr, X_va in datasets:
        for name, model in _candidate_models():
            label = f"{name}__{ds_name}"
            model.fit(X_tr, y_train)
            p = model.predict_proba(X_va)[:, 1]
            auc = float(roc_auc_score(y_val, p)) if len(np.unique(y_val)) > 1 else float("nan")
            brier = float(brier_score_loss(y_val, p))
            ll = float(log_loss(y_val, np.clip(p, 1e-6, 1 - 1e-6)))
            score = brier  # primary: lower is better
            results.append(
                {
                    "model": label,
                    "auc": auc,
                    "brier": brier,
                    "logloss": ll,
                    "val_rows": int(len(y_val)),
                    "n_features": int(len(cols)),
                }
            )
            if best is None or score < float(best_score):
                best = (label, model, cols)
                best_score = score

    assert best is not None
    best_name, best_model, best_cols = best

    # Calibrate probabilities (sigmoid) to improve “% chance” meaning.
    X_train_best = X_train_full[best_cols].values
    X_val_best = X_val_full[best_cols].values

    calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv=3)
    calibrated.fit(X_train_best, y_train)

    p_cal = calibrated.predict_proba(X_val_best)[:, 1]
    auc_cal = float(roc_auc_score(y_val, p_cal)) if len(np.unique(y_val)) > 1 else float("nan")
    brier_cal = float(brier_score_loss(y_val, p_cal))
    ll_cal = float(log_loss(y_val, np.clip(p_cal, 1e-6, 1 - 1e-6)))

    meta = {
        "model_name": best_name,
        "feature_columns": best_cols,
        "metrics": {
            "candidates": results,
            "selected": best_name,
            "calibrated_val": {"auc": auc_cal, "brier": brier_cal, "logloss": ll_cal},
        },
    }
    return meta, calibrated


def train_and_save() -> dict:
    """Train and save TWO probability models: +5% and +2% intraday targets."""

    settings = get_settings()
    provider = get_provider(settings)

    max_tickers = int(os.getenv("TRAIN_MAX_TICKERS", str(settings.scan_max_tickers)))
    lookback = settings.model_lookback_days

    tickers = provider.get_sp500_tickers(max_tickers=max_tickers)
    history = provider.get_daily_history(tickers, lookback_days=max(lookback, 260))

    dataset = _build_dataset(history)
    train_df, val_df = _time_split(dataset)

    targets_spec = [
        ("5pct", "label", 0.05),
        ("2pct", "label_2pct", 0.02),
    ]

    targets: dict[str, dict] = {}
    for key, label_col, thr in targets_spec:
        meta, model = _train_single_target(train_df, val_df, label_col=label_col)
        targets[key] = {
            "threshold": float(thr),
            "label_column": label_col,
            **meta,
            "model": model,
        }

    version = uuid4().hex[:10]
    trained_at = datetime.now(timezone.utc).isoformat()

    bundle = {
        "version": version,
        "trained_at_utc": trained_at,
        "provider": provider.name,
        "targets": targets,
    }

    # Backward-compatible top-level fields (defaults to 5pct target)
    bundle["model_name"] = targets["5pct"]["model_name"]
    bundle["feature_columns"] = targets["5pct"]["feature_columns"]
    bundle["metrics"] = targets["5pct"]["metrics"]
    bundle["model"] = targets["5pct"]["model"]

    save_bundle(settings.model_path, bundle)

    return {
        "version": version,
        "trained_at_utc": trained_at,
        "provider": provider.name,
        "targets": {
            k: {
                "threshold": v["threshold"],
                "label_column": v["label_column"],
                "model_name": v["model_name"],
                "n_features": len(v["feature_columns"]),
                "calibrated_val": (v.get("metrics") or {}).get("calibrated_val"),
            }
            for k, v in targets.items()
        },
    }


if __name__ == "__main__":
    metrics = train_and_save()
    print("Training complete:")
    print(metrics)
