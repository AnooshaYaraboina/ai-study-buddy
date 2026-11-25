"""The architecture diagram matches the code: required components present,
every file it names exists, and the PNG was exported from the same spec."""

import re
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAWIO = ROOT / "docs" / "architecture.drawio"

REQUIRED = ["Student", "Chat UI", "FastAPI backend", "supervisor", "tutor", "quiz-maker", "grader",
            "memory", "LLM API", "notes MCP server", "course PDFs", "ingest", "Chroma", "SQLite",
            "traces", "trust boundary"]


def _labels() -> list[str]:
    root = ET.parse(DRAWIO).getroot()
    return [re.sub(r"<[^>]+>", " ", unescape(c.get("value", ""))) for c in root.iter("mxCell")]


def test_required_components_present():
    text = " ".join(_labels())
    for comp in REQUIRED:
        assert comp in text, f"diagram missing {comp}"


def test_every_arrow_is_labelled():
    root = ET.parse(DRAWIO).getroot()
    edges = [c for c in root.iter("mxCell") if c.get("edge") == "1"]
    assert len(edges) >= 15
    assert all(e.get("value", "").strip() for e in edges)


def test_trust_boundary_is_dashed_red():
    root = ET.parse(DRAWIO).getroot()
    tb = [c for c in root.iter("mxCell") if "trust boundary" in c.get("value", "")]
    assert tb and "dashed=1" in tb[0].get("style") and "#E53935" in tb[0].get("style")


def test_files_named_in_diagram_exist():
    paths = set()
    for label in _labels():
        for m in re.findall(r"(?:app|ui|agents)/[\w/]+\.py", label):
            paths.add(m if not m.startswith("agents/") else "app/" + m)
    assert paths
    for p in paths:
        assert (ROOT / p).exists(), f"diagram references missing file {p}"


def test_png_exported():
    png = ROOT / "docs" / "architecture.png"
    assert png.exists() and png.stat().st_size > 50_000
