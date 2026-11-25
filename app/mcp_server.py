"""notes MCP server: exposes search_notes and save_flashcard.

Run (stdio, for Claude Desktop / MCP Inspector):
    python -m app.mcp_server
Run (streamable HTTP):
    python -m app.mcp_server --http --port 8765

The MCP server is a local, single-student tool: the student is fixed by the
MCP_STUDENT_ID environment variable, never by a tool argument.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from mcp.server.mcpserver import MCPServer

from app.agents.tools import SaveFlashcardInput, SearchNotesInput, ToolContext
from app.agents.tools import save_flashcard as _save_flashcard
from app.agents.tools import search_notes as _search_notes

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)  # stdout is the MCP channel

server = MCPServer(
    "study-buddy-notes",
    instructions="Search the student's course notes (returns [file p.N] citations) "
                 "and save flashcards to their study deck.",
)


def _ctx() -> ToolContext:
    return ToolContext(student_id=os.getenv("MCP_STUDENT_ID", "mcp-student"), session_id="mcp")


@server.tool()
def search_notes(query: str, day: int | None = None, topic: str | None = None,
                 k: int = 5) -> list[dict]:
    """Search the course notes with hybrid search + rerank.

    Args:
        query: what to look up, e.g. "difference between jailbreak and prompt injection"
        day: optional course day filter (e.g. 22)
        topic: optional topic filter (e.g. "injection")
        k: number of passages (1-10)
    """
    inp = SearchNotesInput(query=query, day=day, topic=topic, k=k)
    return _search_notes(_ctx(), inp)


@server.tool()
def save_flashcard(front: str, back: str, source: str = "", topic: str = "") -> dict:
    """Save a flashcard to the student's deck (SQLite).

    Args:
        front: question or term
        back: answer
        source: citation like "[Day_22_23_AI_Security_Visual_Notes.pdf p.10]"
        topic: optional topic label
    """
    inp = SaveFlashcardInput(front=front, back=back, source=source, topic=topic)
    return _save_flashcard(_ctx(), inp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.http:
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
