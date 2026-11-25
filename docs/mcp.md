# notes MCP server

`app/mcp_server.py` serves two tools over MCP (stdio by default, streamable HTTP with `--http`):

| Tool | Inputs | Returns |
|---|---|---|
| `search_notes` | `query`, `day?`, `topic?`, `k=5` | passages with `[file p.N]` citations + rerank score |
| `save_flashcard` | `front`, `back`, `source?`, `topic?` | saved card (SQLite `flashcards`) |

The student is fixed by `MCP_STUDENT_ID` (a local, single-user tool) — it is never a tool argument.

## Test with MCP Inspector

```bash
# UI
npx @modelcontextprotocol/inspector .venv/bin/python -m app.mcp_server
# CLI (what CI-style checks use)
npx @modelcontextprotocol/inspector --cli .venv/bin/python -m app.mcp_server --method tools/list
npx @modelcontextprotocol/inspector --cli .venv/bin/python -m app.mcp_server \
    --method tools/call --tool-name search_notes --tool-arg query="kill switch" --tool-arg k=2
```

## Python client

```bash
python scripts/mcp_client_demo.py "what is memory poisoning"
```

## Desktop client (Claude Desktop / any MCP host)

Add to `claude_desktop_config.json` (Linux: `~/.config/Claude/`, macOS: `~/Library/Application Support/Claude/`):

```json
{
  "mcpServers": {
    "study-buddy-notes": {
      "command": "/ABSOLUTE/PATH/ai-study-buddy/.venv/bin/python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "/ABSOLUTE/PATH/ai-study-buddy",
      "env": { "MCP_STUDENT_ID": "alice" }
    }
  }
}
```

For Claude Code: `claude mcp add study-buddy-notes -e MCP_STUDENT_ID=alice -- /ABSOLUTE/PATH/.venv/bin/python -m app.mcp_server`
(run from the repo root).
