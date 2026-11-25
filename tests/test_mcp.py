"""The notes MCP server works from a real MCP client over stdio."""

import asyncio
import os
import sys
from pathlib import Path

from tests.conftest import needs_index

pytestmark = needs_index
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_mcp_tools_from_client():
    from mcp_client_demo import main

    os.environ["MCP_STUDENT_ID"] = "pytest-mcp"
    out = asyncio.run(main("what is memory poisoning"))
    assert set(out["tools"]) == {"search_notes", "save_flashcard"}
    assert out["hits"] and out["hits"][0]["citation"].endswith("p.17]")
    assert out["card"]["id"] > 0
