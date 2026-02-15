# S&P 500 Open Scanner (Render Deploy)

A small web app + scheduled job that:

- **Runs just after the US market opens**
- **Scores every S&P 500 ticker** with an estimated probability that it will **reach +5% (or more) or +2% (or more) at any point before the close**
- Produces a **single score (0–100)** per ticker plus a lightweight **market sentiment** snapshot
- Stores the latest results in **Render Postgres** (or SQLite locally) and shows them on a simple dashboard

> ⚠️ This is a demo / research tool, **not** financial advice. The model is a baseline classifier built on daily OHLCV features.
> For production-grade intraday scanning, you will want a paid, reliable market data source.

---

## What “+5% before close” means here

For training labels we use:

**label = 1 if (day_high / day_open - 1) >= 0.05 else 0**

Daily OHLCV is enough to know whether the stock *ever* reached +5% during that day.

## What “+2% before close” means here

For the secondary score we use:

**label_2pct = 1 if (day_high / day_open - 1) >= 0.02 else 0**

The dashboard and API return both +5% (primary) and +2% (secondary) probabilities.

---

## Data provider

This repo uses **Alpaca Market Data**.

Set:

- `DATA_PROVIDER=alpaca`
- `ALPACA_KEY_ID=...`
- `ALPACA_SECRET_KEY=...`

Recommended (paid):

- `ALPACA_FEED=sip` (more complete / fresher consolidated feed)

The scanner also pulls **1-minute bars** from the open to the scan time (default: first 5 minutes) and applies a small **momentum/volume adjustment** on top of the model probability.

---

## Local run (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (Optional but recommended)
export DATA_PROVIDER=alpaca
export ALPACA_KEY_ID="..."
export ALPACA_SECRET_KEY="..."

# Secure admin endpoints
export ADMIN_TOKEN="change-me"

# Local DB (optional). On Render, DATABASE_URL is set automatically.
export DATABASE_URL="sqlite:///./local.db"

# Train model (first time)
python -m app.ml.train

# Run the web app
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

---

## Scheduled scans

There are two ways:

### A) Run a one-off scan
```bash
python -m app.jobs.run_scan
```

### B) Use Render Cron Job (recommended on Render)
On Render, use the included `render.yaml` Blueprint.

---

## Environment variables

Recommended:
- `ADMIN_TOKEN` – protects `/api/run-scan` and `/api/train`.
  If you don't set it, the app will still start, but those endpoints will be **unprotected**.

Database:
- `DATABASE_URL` (Render injects this when you add a Postgres database)

Alpaca:
- `DATA_PROVIDER=alpaca`
- `ALPACA_KEY_ID`
- `ALPACA_SECRET_KEY`
Optional:
- `ALPACA_FEED=iex` (default) or `sip` (paid)

News sentiment (optional):
- `ALPACA_NEWS_LIMIT=50`
- `ALPACA_NEWS_SYMBOLS=SPY,QQQ`

Optional:
- `SCAN_MAX_TICKERS=500` (for testing, set e.g. 50)
- `SCAN_AT_MINUTES_AFTER_OPEN=5` (default: 5)
- `MODEL_LOOKBACK_DAYS=756` (default: ~3 years trading days)

---

## API endpoints

- `GET /` – dashboard
- `GET /api/latest` – latest scan + top scores JSON
- `POST /api/run-scan` – trigger scan (requires `Authorization: Bearer <ADMIN_TOKEN>`)
- `POST /api/train` – retrain model (requires auth)
- `GET /healthz` – health check

---

## Repo layout

- `app/main.py` – FastAPI app + dashboard
- `app/jobs/run_scan.py` – cron entry point
- `app/jobs/scan.py` – scan logic
- `app/ml/train.py` – model training and selection
- `app/ml/predict.py` – prediction helpers
- `app/data_providers/*` – market data providers
- `app/features/*` – feature engineering and indicators
- `app/sentiment/market.py` – market sentiment snapshot
- `app/db.py` – Postgres/SQLite persistence

---

## Why the “optimal combination” claim is cautious

The training pipeline does **model selection** (several sklearn models) and chooses the best on validation metrics, but:

- predicting intraday moves is noisy
- regimes change
- transaction costs and slippage are ignored
- you’ll get better results with richer features (news, options, order book, etc.)

Still, this repo gives you a working end-to-end framework you can evolve.
