from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str
    session_secret: str
    model_path: Path
    cookie_secure: bool
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        secret = os.getenv("SESSION_SECRET", "")
        if len(secret) < 16:
            raise RuntimeError("SESSION_SECRET must contain at least 16 characters")
        return cls(
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql+psycopg://hotel:hotel@localhost:5432/hotel_predictions",
            ),
            session_secret=secret,
            model_path=Path(os.getenv("MODEL_PATH", str(BASE_DIR / "model_xgb.ubj"))),
            cookie_secure=_as_bool(os.getenv("COOKIE_SECURE")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
