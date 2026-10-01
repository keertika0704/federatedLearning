"""Shared API-key and same-origin frontend session authentication."""

import logging
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Cookie, HTTPException, Security
from fastapi.security import APIKeyHeader

log = logging.getLogger("uvicorn.error")
PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "models" / "objectiveE").is_dir()
)

for directory in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    env_file = directory / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)
        log.info("Loaded environment from %s", env_file)
        break

API_KEY = (os.getenv("OBJECTIVE_E_API_KEY") or "").strip()
if len(API_KEY) < 16:
    raise RuntimeError(
        "OBJECTIVE_E_API_KEY is missing or shorter than 16 characters. "
        "Set it in the project .env file before starting the backend."
    )

UI_SESSION_TOKEN = secrets.token_urlsafe(32)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    key: str | None = Security(api_key_header),
    ui_session: str | None = Cookie(default=None, alias="carepath_session"),
):
    valid_api_key = key and secrets.compare_digest(key, API_KEY)
    valid_ui_session = ui_session and secrets.compare_digest(ui_session, UI_SESSION_TOKEN)
    if not (valid_api_key or valid_ui_session):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
