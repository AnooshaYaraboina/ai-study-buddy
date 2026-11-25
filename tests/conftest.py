import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Isolated, key-free test environment (set before any app import).
_TMP = Path(tempfile.mkdtemp(prefix="studybuddy-tests-"))
os.environ["LLM_OFFLINE"] = "1"
os.environ["DB_PATH"] = str(_TMP / "test.db")
os.environ["LOG_DIR"] = str(_TMP / "logs")
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["INGEST_ON_START"] = "0"
os.environ.setdefault("API_TOKENS", "alice:alice-token,bob:bob-token")
os.environ["RATE_LIMIT_PER_MIN"] = "1000"

# Tests run against a throwaway copy of the ingested index, never the live one.
_LIVE_INDEX = Path(__file__).resolve().parents[1] / "data" / "chroma"
if _LIVE_INDEX.exists() and not os.getenv("CHROMA_HOST"):
    shutil.copytree(_LIVE_INDEX, _TMP / "chroma")
os.environ["CHROMA_DIR"] = str(_TMP / "chroma")

FIXTURES = Path(__file__).parent / "fixtures"


def _has_index() -> bool:
    try:
        from app.vector_store import get_store

        return get_store().count() > 0
    except Exception:
        return False


needs_index = pytest.mark.skipif(not _has_index(), reason="run `python -m app.ingest` first")


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES
