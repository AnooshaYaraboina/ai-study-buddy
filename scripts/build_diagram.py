"""Build docs/architecture.drawio and docs/architecture.png from ONE spec.

The .drawio file opens (and can be edited) in diagrams.net / draw.io desktop.
The PNG is rendered from the same node/edge spec with matplotlib, so the two
never drift apart. If you edit the diagram in draw.io, re-export the PNG with
File > Export as > PNG (or `drawio -x -f png -o docs/architecture.png docs/architecture.drawio`).

    python scripts/build_diagram.py
"""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"
W, H = 1580, 1100

# ---------------------------------------------------------------- palette
C = {
    "user": ("#E8EAF6", "#3949AB"),
    "front": ("#E3F2FD", "#1E88E5"),
    "back": ("#E8F5E9", "#43A047"),
    "agent": ("#FFF3E0", "#FB8C00"),
    "side": ("#F3E5F5", "#8E24AA"),
    "llm": ("#FCE4EC", "#D81B60"),
    "tool": ("#E0F7FA", "#00ACC1"),
    "ingest": ("#FFFDE7", "#F9A825"),
    "store": ("#ECEFF1", "#546E7A"),
    "obs": ("#EFEBE9", "#6D4C41"),
}

# id: (x, y, w, h, layer, title, subtitle, shape)
NODES: dict[str, tuple] = {
    "student": (30, 430, 130, 80, "user", "Student", "User", "actor"),
    "ui": (200, 420, 170, 100, "front", "Chat UI", "Streamlit · ui/app.py", "box"),
    "api": (440, 410, 200, 120, "back", "FastAPI backend",
            "auth · guards · rate limit\napp/api/main.py", "box"),
    "supervisor": (720, 420, 170, 100, "agent", "supervisor",
                   "routes each turn\napp/agents/graph.py", "box"),
    "tutor": (960, 250, 170, 90, "agent", "tutor", "explains, cites [file p.N]", "box"),
    "quiz": (960, 425, 170, 90, "agent", "quiz-maker", "5 MCQs (strict JSON)", "box"),
    "grader": (960, 600, 170, 90, "agent", "grader", "score + tips", "box"),
    "memory": (720, 95, 260, 95, "side", "memory",
               "weak topics · last 6 turns + summary\napp/memory.py", "box"),
    "mcp": (720, 790, 300, 95, "tool", "notes MCP server",
            "search_notes · save_flashcard\napp/mcp_server.py", "box"),
    "llm": (1300, 150, 220, 100, "llm", "LLM API (Claude)",
            "streaming · timeout · retries\ntoken + cost log · app/llm.py", "box"),
    "chroma": (1300, 420, 220, 100, "store", "Chroma",
               "collection: course_notes\nhybrid (BM25 + vector) + rerank", "cyl"),
    "sqlite": (1300, 790, 220, 95, "store", "SQLite",
               "progress + cards\nweak_topics · flashcards", "cyl"),
    "pdfs": (1280, 600, 100, 90, "ingest", "course PDFs", "study guides", "doc"),
    "ingest": (1425, 600, 120, 90, "ingest", "ingest", "parse · chunk\n· embed", "box"),
    "traces": (200, 960, 200, 80, "obs", "traces",
               "Langfuse / OpenTelemetry", "box"),
}

# container frames: (x, y, w, h, label, stroke, dashed, fill)
FRAMES = [
    (420, 40, 840, 890, "trust boundary (backend · agents · tools)", "#E53935", True, None),
    (700, 225, 470, 490, "agents (LangGraph)", "#FB8C00", False, "#FFFAF2"),
]

# (src, dst, label, style)   style: solid | dashed | both (bidirectional)
EDGES = [
    ("student", "ui", "asks / reads", "solid"),
    ("ui", "api", "HTTPS + bearer\n(SSE stream)", "solid"),
    ("api", "supervisor", "guarded turn", "solid"),
    ("supervisor", "tutor", "question", "solid"),
    ("supervisor", "quiz", "\"quiz me\"", "solid"),
    ("supervisor", "grader", "answers", "solid"),
    ("supervisor", "memory", "read / write\nturns + weak topics", "both"),
    ("tutor", "llm", "prompt (stream)", "solid"),
    ("quiz", "llm", "quiz JSON", "solid"),
    ("grader", "llm", "feedback", "solid"),
    ("tutor", "chroma", "retrieve top-5", "solid"),
    ("quiz", "mcp", "search_notes", "both"),
    ("grader", "mcp", "save_flashcard", "both"),
    ("mcp", "sqlite", "flashcards", "solid"),
    ("memory", "sqlite", "persist weak topics", "solid"),
    ("mcp", "chroma", "query", "solid"),
    ("pdfs", "ingest", "parse", "solid"),
    ("ingest", "chroma", "upsert by chunk ID", "solid"),
    ("api", "ingest", "upload (sanitized)", "solid"),
    ("api", "traces", "spans · tokens · cost", "dashed"),
]

# manual waypoints for edges that would otherwise cross boxes
WAYPOINTS: dict[tuple[str, str], list[tuple[float, float]]] = {
    ("memory", "sqlite"): [(1230, 142), (1230, 760), (1410, 760)],
    ("api", "ingest"): [(600, 1075), (1552, 1075), (1552, 645)],
    ("quiz", "mcp"): [(935, 560)],
    ("mcp", "chroma"): [(1200, 837), (1200, 470)],
    ("tutor", "chroma"): [(1180, 295), (1180, 445)],
}


def center(n: str) -> tuple[float, float]:
    x, y, w, h, *_ = NODES[n]
    return x + w / 2, y + h / 2


def anchor(n: str, toward: tuple[float, float]) -> tuple[float, float]:
    """Point on node n's border on the line from its centre toward a point."""
    x, y, w, h, *_ = NODES[n]
    cx, cy = x + w / 2, y + h / 2
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (w / 2) / abs(dx) if dx else float("inf")
    sy = (h / 2) / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def edge_points(src: str, dst: str) -> list[tuple[float, float]]:
    wps = WAYPOINTS.get((src, dst), [])
    first = wps[0] if wps else center(dst)
    last = wps[-1] if wps else center(src)
    return [anchor(src, first), *wps, anchor(dst, last)]


# ---------------------------------------------------------------- draw.io
def build_drawio() -> str:
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for i, (x, y, w, h, label, stroke, dashed, fill) in enumerate(FRAMES):
        style = (
            f"rounded=1;arcSize=2;whiteSpace=wrap;html=1;verticalAlign=top;align=left;"
            f"spacingLeft=10;fontStyle=1;strokeColor={stroke};strokeWidth=2;"
            f"fillColor={fill or 'none'};fontColor={stroke};"
            + ("dashed=1;dashPattern=8 6;" if dashed else "")
        )
        cells.append(
            f'<mxCell id="frame{i}" value="{html.escape(label)}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
            f'height="{h}" as="geometry"/></mxCell>'
        )
    for nid, (x, y, w, h, layer, title, sub, shape) in NODES.items():
        fill, stroke = C[layer]
        value = f"<b>{html.escape(title)}</b><br>" + html.escape(sub).replace("\n", "<br>")
        base = (
            f"whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            "strokeWidth=2;fontSize=12;"
        )
        shape_style = {
            "box": "rounded=1;",
            "cyl": "shape=cylinder3;boundedLbl=1;size=12;",
            "doc": "shape=document;boundedLbl=1;",
            "actor": "shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;",
        }[shape]
        geo_w, geo_h = (40, 60) if shape == "actor" else (w, h)
        gx = x + (w - geo_w) / 2 if shape == "actor" else x
        cells.append(
            f'<mxCell id="{nid}" value="{html.escape(value)}" style="{shape_style}{base}" '
            f'vertex="1" parent="1"><mxGeometry x="{gx}" y="{y}" width="{geo_w}" '
            f'height="{geo_h}" as="geometry"/></mxCell>'
        )
    for i, (s, d, label, style) in enumerate(EDGES):
        st = "edgeStyle=none;html=1;rounded=1;strokeWidth=1.5;fontSize=11;labelBackgroundColor=#ffffff;"
        if style == "dashed":
            st += "dashed=1;strokeColor=#6D4C41;"
        if style == "both":
            st += "startArrow=classic;startFill=1;"
        pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in WAYPOINTS.get((s, d), []))
        pts_xml = f'<Array as="points">{pts}</Array>' if pts else ""
        cells.append(
            f'<mxCell id="e{i}" value="{html.escape(label).replace(chr(10), "&lt;br&gt;")}" '
            f'style="{st}" edge="1" parent="1" source="{s}" target="{d}">'
            f'<mxGeometry relative="1" as="geometry">{pts_xml}</mxGeometry></mxCell>'
        )
    body = "\n        ".join(cells)
    return f"""<mxfile host="build_diagram.py" version="24.0.0">
  <diagram id="arch" name="AI Study Buddy – architecture">
    <mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="10" guides="1" page="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">
      <root>
        {body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""


# ---------------------------------------------------------------- PNG
def build_png(path: Path) -> None:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=130)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(20, 22, "AI Study Buddy — architecture", fontsize=15, fontweight="bold",
            va="center")

    for x, y, w, h, label, stroke, dashed, fill in FRAMES:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
            fc=fill or "none", ec=stroke, lw=2.2,
            ls=(0, (6, 4)) if dashed else "-", zorder=1))
        ax.text(x + 10, y + 16, label, color=stroke, fontsize=10.5, fontweight="bold",
                va="center", zorder=2)

    for s, d, label, style in EDGES:
        pts = edge_points(s, d)
        color = "#6D4C41" if style == "dashed" else "#37474F"
        ls = (0, (5, 4)) if style == "dashed" else "-"
        for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
            ax.plot([x1, x2], [y1, y2], color=color, lw=1.4, ls=ls, zorder=3)
        arrow = "<|-|>" if style == "both" else "-|>"
        if len(pts) > 2 and style == "both":
            # start arrowhead on first segment, end arrowhead on last
            ax.annotate("", xy=pts[0], xytext=pts[1], zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
            arrow = "-|>"
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=3,
                    arrowprops=dict(arrowstyle=arrow, color=color, lw=1.4,
                                    linestyle=ls, shrinkA=0, shrinkB=0))
        # label on the middle of the longest segment
        segs = list(zip(pts[:-1], pts[1:]))
        (a, b) = max(segs, key=lambda p: (p[0][0] - p[1][0]) ** 2 + (p[0][1] - p[1][1]) ** 2)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx, my, label, fontsize=8.2, ha="center", va="center", color="#263238",
                zorder=6, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

    for nid, (x, y, w, h, layer, title, sub, shape) in NODES.items():
        fill, stroke = C[layer]
        if shape == "actor":
            cx = x + w / 2
            ax.add_patch(mpatches.Circle((cx, y + 12), 11, fc=fill, ec=stroke, lw=2, zorder=5))
            ax.plot([cx, cx], [y + 23, y + 50], color=stroke, lw=2, zorder=5)
            ax.plot([cx - 16, cx + 16], [y + 32, y + 32], color=stroke, lw=2, zorder=5)
            ax.plot([cx, cx - 14], [y + 50, y + 68], color=stroke, lw=2, zorder=5)
            ax.plot([cx, cx + 14], [y + 50, y + 68], color=stroke, lw=2, zorder=5)
            ax.text(cx, y + h + 4, title, ha="center", va="center", fontsize=10.5,
                    fontweight="bold", zorder=6)
            continue
        if shape == "cyl":
            ax.add_patch(mpatches.FancyBboxPatch((x, y + 8), w, h - 16,
                         boxstyle="square,pad=0", fc=fill, ec=stroke, lw=2, zorder=4))
            for yy in (y + 8, y + h - 8):
                ax.add_patch(mpatches.Ellipse((x + w / 2, yy), w, 16, fc=fill, ec=stroke,
                             lw=2, zorder=4 if yy > y + 8 else 5))
        else:
            style = "round,pad=0,rounding_size=10" if shape == "box" else "square,pad=0"
            ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle=style,
                         fc=fill, ec=stroke, lw=2, zorder=4))
        ax.text(x + w / 2, y + h * 0.33, title, ha="center", va="center", fontsize=11,
                fontweight="bold", color="#212121", zorder=6)
        ax.text(x + w / 2, y + h * 0.66, sub, ha="center", va="center", fontsize=8.3,
                color="#37474F", zorder=6, linespacing=1.25)

    # legend
    lx, ly = 30, 600
    ax.text(lx, ly, "Layers", fontsize=10, fontweight="bold")
    for i, (name, key) in enumerate([
        ("User", "user"), ("Frontend", "front"), ("Backend", "back"),
        ("Agent layer", "agent"), ("Memory", "side"), ("LLM API", "llm"),
        ("Tool server", "tool"), ("Ingestion", "ingest"), ("Data stores", "store"),
        ("Observability", "obs"),
    ]):
        fill, stroke = C[key]
        ax.add_patch(mpatches.Rectangle((lx, ly + 14 + i * 22), 16, 14, fc=fill, ec=stroke, lw=1.5))
        ax.text(lx + 24, ly + 21 + i * 22, name, fontsize=9, va="center")
    ax.plot([lx, lx + 30], [ly + 248, ly + 248], color="#E53935", lw=2, ls=(0, (6, 4)))
    ax.text(lx + 36, ly + 248, "trust boundary", fontsize=9, va="center")
    ax.plot([lx, lx + 30], [ly + 270, ly + 270], color="#6D4C41", lw=1.4, ls=(0, (5, 4)))
    ax.text(lx + 36, ly + 270, "async telemetry", fontsize=9, va="center")

    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    (DOCS / "architecture.drawio").write_text(build_drawio(), encoding="utf-8")
    build_png(DOCS / "architecture.png")
    print("wrote", DOCS / "architecture.drawio", "and", DOCS / "architecture.png")


if __name__ == "__main__":
    main()
