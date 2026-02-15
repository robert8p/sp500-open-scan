from __future__ import annotations

from fastapi import Header, HTTPException
from app.config import Settings


def require_admin(settings: Settings, authorization: str | None = Header(default=None)) -> None:
    # If ADMIN_TOKEN isn't configured, allow access (useful for first boot / local dev).
    # Strongly recommended to set ADMIN_TOKEN in production.
    if not settings.admin_token:
        return

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid token")
