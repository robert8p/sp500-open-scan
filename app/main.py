from __future__ import annotations

from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pathlib import Path

from app.config import get_settings, Settings
from app.auth import require_admin
from app.db import ensure_db, get_latest_scan, get_engine
from app.jobs.scan import run_scan
from app.ml.train import train_and_save
from app.ml.model_io import bundle_exists, load_bundle


templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app = FastAPI(title="S&P 500 Open Scanner")


@app.on_event("startup")
def _startup():
    settings = get_settings()
    if not settings.admin_token:
        # Render logs will show this warning so users can fix their environment.
        print("WARNING: ADMIN_TOKEN is not set. Admin endpoints are UNPROTECTED. Set ADMIN_TOKEN in Render Environment variables.")
    engine = get_engine(settings)
    ensure_db(engine)
    Path(settings.model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.ticker_cache_path).parent.mkdir(parents=True, exist_ok=True)


def admin_guard(request: Request):
    settings = get_settings()
    require_admin(settings, request.headers.get("authorization"))


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    settings = get_settings()
    engine = get_engine(settings)
    latest_scan, latest_scores = get_latest_scan(engine)
    latest = {"scan": latest_scan, "top": latest_scores}
    model_info = None
    if bundle_exists(settings.model_path):
        b = load_bundle(settings.model_path)
        targets = b.get("targets")
        if isinstance(targets, dict) and targets:
            model_info = {
                "version": b.get("version"),
                "trained_at_utc": b.get("trained_at_utc"),
                "targets": {
                    k: {
                        "model_name": v.get("model_name"),
                        "metrics": (v.get("metrics") or {}).get("calibrated_val"),
                        "threshold": v.get("threshold"),
                    }
                    for k, v in targets.items()
                },
            }
        else:
            # Legacy single-target bundle
            model_info = {
                "version": b.get("version"),
                "trained_at_utc": b.get("trained_at_utc"),
                "targets": {
                    "5pct": {
                        "model_name": b.get("model_name"),
                        "metrics": (b.get("metrics") or {}).get("calibrated_val"),
                        "threshold": 0.05,
                    }
                },
            }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "latest": latest,
            "model": model_info,
        },
    )


@app.get("/api/latest")
def api_latest():
    settings = get_settings()
    engine = get_engine(settings)
    latest_scan, latest_scores = get_latest_scan(engine)
    return JSONResponse({"scan": latest_scan, "top": latest_scores})


@app.post("/api/run-scan")
def api_run_scan(_=Depends(admin_guard)):
    settings = get_settings()
    res = run_scan(settings, force=True)
    return JSONResponse(res)


@app.post("/api/train")
def api_train(_=Depends(admin_guard)):
    metrics = train_and_save()
    return JSONResponse({"status": "ok", "metrics": metrics})