"""Minimal MCP client: launches the notes server over stdio, lists tools, calls both.

    python scripts/mcp_client_demo.py "what is memory poisoning"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


def _payload(res):
    """Prefer structured content; fall back to parsing text content blocks."""
    sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if sc is not None:
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    items = [json.loads(c.text) for c in res.content if getattr(c, "text", None)]
    return items[0] if len(items) == 1 and isinstance(items[0], (list, dict)) and not \
        (isinstance(items[0], dict) and "citation" in items[0]) else items


async def main(query: str) -> dict:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "app.mcp_server"], cwd=str(ROOT),
        env={**os.environ, "MCP_STUDENT_ID": os.getenv("MCP_STUDENT_ID", "mcp-demo")},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print("tools:", names)
            res = await session.call_tool("search_notes", {"query": query, "k": 3})
            hits = _payload(res)
            for h in hits:
                print(f"  {h['score']:.3f} {h['citation']}")
            top = hits[0]
            card = await session.call_tool("save_flashcard", {
                "front": query, "back": top["text"].split("\n")[1][:200], "source": top["citation"]})
            card_data = _payload(card)
            print("saved:", card_data)
            return {"tools": names, "hits": hits, "card": card_data}


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "what is memory poisoning"))
