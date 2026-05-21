from __future__ import annotations

from fastapi import Header, HTTPException

from backend.app.core.config import settings


def require_api_key(api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not settings.api_auth_token:
        return None
    if api_key != settings.api_auth_token:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return None
