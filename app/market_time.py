from __future__ import annotations

from datetime import datetime, timezone, timedelta, date
import pandas_market_calendars as mcal


def get_nyse_open_close_utc(day: date) -> tuple[datetime, datetime] | None:
    cal = mcal.get_calendar("NYSE")
    sched = cal.schedule(start_date=day, end_date=day)
    if sched.empty:
        return None
    open_ts = sched.iloc[0]["market_open"].to_pydatetime()
    close_ts = sched.iloc[0]["market_close"].to_pydatetime()
    # Ensure tz-aware UTC
    if open_ts.tzinfo is None:
        open_ts = open_ts.replace(tzinfo=timezone.utc)
    else:
        open_ts = open_ts.astimezone(timezone.utc)
    if close_ts.tzinfo is None:
        close_ts = close_ts.replace(tzinfo=timezone.utc)
    else:
        close_ts = close_ts.astimezone(timezone.utc)
    return open_ts, close_ts


def should_run_scan(now_utc: datetime, minutes_after_open: int, window_minutes: int) -> tuple[bool, dict]:
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    oc = get_nyse_open_close_utc(now_utc.date())
    if oc is None:
        return False, {"reason": "NYSE closed (holiday/weekend?)"}

    open_utc, close_utc = oc
    target = open_utc + timedelta(minutes=minutes_after_open)
    window_end = target + timedelta(minutes=window_minutes)

    if now_utc < target:
        return False, {"reason": "too early", "open_utc": open_utc.isoformat(), "target_utc": target.isoformat()}
    if now_utc > window_end:
        return False, {"reason": "too late", "open_utc": open_utc.isoformat(), "window_end_utc": window_end.isoformat()}
    if now_utc >= close_utc:
        return False, {"reason": "after close", "close_utc": close_utc.isoformat()}

    return True, {"open_utc": open_utc.isoformat(), "target_utc": target.isoformat(), "window_end_utc": window_end.isoformat()}
