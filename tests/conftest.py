from __future__ import annotations

import os

import pytest

test_url = os.getenv("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:")
if test_url.startswith("postgresql") and not test_url.rsplit("/", 1)[-1].endswith("_test"):
    raise RuntimeError(
        "Destructive integration fixtures require an explicitly named *_test database"
    )
os.environ["DATABASE_URL"] = test_url
os.environ["SESSION_SECRET"] = "test-secret-key-with-at-least-32-characters"
os.environ["COOKIE_SECURE"] = "false"

from backend.db import engine  # noqa: E402
from backend.models import Base  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_database():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)
