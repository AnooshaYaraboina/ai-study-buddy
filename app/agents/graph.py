"""LangGraph multi-agent flow.

    load_memory -> supervisor -> {tutor | quiz_maker | grader | tools} -> save_memory -> END

The supervisor routes each turn. The tutor streams tokens through LangGraph's
custom stream channel so the API can forward them as SSE.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterator, Literal, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app import db, memory
from app.agents.grader import format_grade, grade, looks_like_answers, parse_answers
from app.agents.tools import (
    TOOLS,
    GetProgressInput,
    MakeQuizInput,
    SaveFlashcardInput,
    ToolContext,
    get_progress,
    make_quiz,
    run_tool,
    save_flashcard,
)
from app.llm import get_llm
from app.observability import tracer
from app.rag import Answer, Tutor
from app.retrieval.rewrite import rewrite_followup

log = logging.getLogger(__name__)
Route = Literal["tutor", "quiz_maker", "grader", "tools"]
MAX_TOOL_STEPS = 4  # loop cap for the tool-calling agent


class TurnState(TypedDict, total=False):
    student_id: str
    session_id: str
    message: str
    standalone: str
    route: Route
    topic: str
    history: list[dict]
    summary: str
    weak_topics: list[str]
    pending_quiz: list[dict] | None
    reply: str
    citations: list[str]
    sources: list[dict]
    quiz: list[dict] | None
    grade: dict | None
    refused: bool


QUIZ_RE = re.compile(r"\b(quiz|test)\s+me\b|\bquiz\b|\bmcqs?\b|\bpractice questions?\b", re.I)
QUIZ_TOPIC_RE = re.compile(
    r"(?:quiz|test|mcqs?|questions?)(?:\s+me)?(?:\s+\d+)?(?:\s+questions?)?\s+(?:on|about|for|in)\s+(.+)",
    re.I)
PROGRESS_RE = re.compile(r"\b(progress|weak (topics?|spots?|areas?)|how am i doing|my (score|stats)|"
                         r"my flashcards|list (my )?flashcards|show (my )?(cards|flashcards))\b", re.I)
FLASHCARD_RE = re.compile(r"\b(save|make|create|add)\b.{0,20}\bflash ?cards?\b", re.I)


# ------------------------------------------------------------------ nodes
def load_memory(state: TurnState) -> TurnState:
    mem = memory.load(state["student_id"], state["session_id"])
    pending = db.get_pending_quiz(state["student_id"], state["session_id"])
    return {**mem, "pending_quiz": pending}


def supervisor(state: TurnState) -> TurnState:
    msg = state["message"].strip()
    pending = state.get("pending_quiz")
    if pending and looks_like_answers(msg, len(pending)):
        route: Route = "grader"
    elif FLASHCARD_RE.search(msg) or PROGRESS_RE.search(msg):
        route = "tools"
    elif QUIZ_RE.search(msg):
        route = "quiz_maker"
    else:
        route = "tutor"
    with tracer().start_as_current_span("agent.supervisor") as span:
        span.set_attribute("route", route)
    log.info("supervisor -> %s", route)
    return {"route": route}


def tutor(state: TurnState) -> TurnState:
    writer = get_stream_writer()
    standalone = rewrite_followup(state["message"], state.get("history", []))
    final: Answer | None = None
    for item in Tutor().answer_stream(standalone, owner=state["student_id"],
                                      weak=state.get("weak_topics", []),
                                      summary=state.get("summary", "")):
        if isinstance(item, Answer):
            final = item
        elif isinstance(item, dict):
            writer({"type": "replace", "text": item["replace"]})
        else:
            writer({"type": "token", "text": item})
    assert final is not None
    topic = final.hits[0]["metadata"].get("topic", "") if final.hits and not final.refused else ""
    return {"reply": final.text, "citations": final.citations, "standalone": standalone,
            "refused": final.refused, "topic": topic, "sources": final.sources}


def _quiz_topic(state: TurnState) -> str:
    m = QUIZ_TOPIC_RE.search(state["message"])
    if m:
        return m.group(1).strip(" ?.!\"'")
    weak = state.get("weak_topics") or []
    if weak:
        return weak[0]
    for t in reversed(state.get("history", [])):
        if t["role"] == "user" and not QUIZ_RE.search(t["content"]):
            return t.get("standalone") or t["content"]
    return "key concepts"


def quiz_maker(state: TurnState) -> TurnState:
    ctx = ToolContext(state["student_id"], state["session_id"])
    topic = _quiz_topic(state)
    quiz = make_quiz(ctx, MakeQuizInput(topic=topic[:120], n=5))
    for q in quiz:  # one label per quiz keeps weak-topic tracking consistent
        q["topic"] = topic.strip().lower()[:80]
    db.set_pending_quiz(ctx.student_id, ctx.session_id, quiz)
    why = " (one of your weak topics)" if topic in (state.get("weak_topics") or []) else ""
    lines = [f"Here's a 5-question quiz on **{topic}**{why}. Reply with your answers, "
             f"e.g. `A C B D A`.", ""]
    for i, q in enumerate(quiz, 1):
        lines.append(f"**Q{i}. {q['question']}**")
        lines += [f"  {'ABCD'[j]}) {o}" for j, o in enumerate(q["options"])]
        lines.append("")
    return {"reply": "\n".join(lines), "quiz": quiz, "topic": topic,
            "citations": sorted({q["source"] for q in quiz if q.get("source")})}


def grader(state: TurnState) -> TurnState:
    ctx = ToolContext(state["student_id"], state["session_id"])
    quiz = state["pending_quiz"] or []
    g = grade(quiz, parse_answers(state["message"], len(quiz)), ctx)
    db.set_pending_quiz(ctx.student_id, ctx.session_id, None)
    for r in g["results"]:  # weak topics are tracked per quiz topic
        r["topic"] = r.get("topic") or "general"
    memory.update_weak_topics(ctx.student_id, g["results"])
    return {"reply": format_grade(g), "grade": g, "pending_quiz": None,
            "citations": sorted({r["source"] for r in g["results"] if r["source"]})}


def _offline_tools(state: TurnState, ctx: ToolContext) -> str:
    msg = state["message"]
    if FLASHCARD_RE.search(msg):
        m = re.search(r"flash ?cards?\s*(?:for|on|about|:)?\s*(.+)", msg, re.I)
        term = (m.group(1) if m else msg).strip(" ?.!\"'") or "this topic"
        hits = run_tool("search_notes", ctx, {"query": term, "k": 1})
        if not hits or hits[0]["score"] < 0.25:
            return f"I couldn't find “{term}” in your notes, so I didn't save a card."
        body = " ".join(ln for ln in hits[0]["text"].split("\n")[1:4])[:400]
        card = save_flashcard(ctx, SaveFlashcardInput(front=term[:300], back=body or term,
                                                      source=hits[0]["citation"],
                                                      topic=hits[0]["topic"] or ""))
        return f"Saved flashcard #{card['id']}: **{card['front']}** → {card['back'][:160]}… {card['source']}"
    p = get_progress(ctx, GetProgressInput())
    if re.search(r"cards", msg, re.I):
        cards = db.list_flashcards(ctx.student_id, 10)
        return "Your latest flashcards:\n" + ("\n".join(
            f"- **{c['front'][:80]}** → {c['back'][:100]} {c['source']}" for c in cards) or "none yet")
    weak = ", ".join(w["topic"] for w in p["weak_topics"]) or "none yet"
    acc = f"{p['accuracy']:.0%}" if p["accuracy"] is not None else "n/a"
    return (f"You've answered **{p['questions_answered']}** questions (accuracy {acc}).\n"
            f"Weak topics: {weak}\nFlashcards saved: {p['flashcards']}")


def tools(state: TurnState) -> TurnState:
    """Tool-calling agent: Claude picks tools (capped loop); offline = rule-based."""
    ctx = ToolContext(state["student_id"], state["session_id"])
    llm = get_llm()
    if llm.offline:
        return {"reply": _offline_tools(state, ctx)}
    defs = [TOOLS[n].anthropic_schema() for n in ("search_notes", "save_flashcard", "get_progress")]
    messages: list[dict[str, Any]] = [{"role": "user", "content": state["message"]}]
    system = ("You manage the student's study tools. Use tools to act; only save flashcards "
              "whose content comes from search_notes results and include the [file p.N] source. "
              "Reply briefly in plain text.")
    for _ in range(MAX_TOOL_STEPS):
        params = llm._params(system, messages, 2000, None, None)
        params["tools"] = defs
        msg = llm._client.beta.messages.create(**params)
        messages.append({"role": "assistant", "content": [b.model_dump() for b in msg.content]})
        if msg.stop_reason != "tool_use":
            return {"reply": "".join(b.text for b in msg.content if b.type == "text")}
        results = []
        for b in msg.content:
            if b.type != "tool_use":
                continue
            try:
                out = run_tool(b.name, ctx, b.input)
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out, default=str)[:6000]})
            except Exception as e:  # report tool errors back to the model
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": f"error: {e}", "is_error": True})
        messages.append({"role": "user", "content": results})
    return {"reply": "I stopped after too many tool steps. Please try a simpler request."}


def save_memory(state: TurnState) -> TurnState:
    memory.save_turn(state["student_id"], state["session_id"], state["message"],
                     state.get("standalone") or state["message"], state.get("reply", ""))
    return {}


def _route(state: TurnState) -> str:
    return state["route"]


def build_graph():
    g = StateGraph(TurnState)
    g.add_node("load_memory", load_memory)
    g.add_node("supervisor", supervisor)
    g.add_node("tutor", tutor)
    g.add_node("quiz_maker", quiz_maker)
    g.add_node("grader", grader)
    g.add_node("tools", tools)
    g.add_node("save_memory", save_memory)
    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "supervisor")
    g.add_conditional_edges("supervisor", _route,
                            {"tutor": "tutor", "quiz_maker": "quiz_maker",
                             "grader": "grader", "tools": "tools"})
    for n in ("tutor", "quiz_maker", "grader", "tools"):
        g.add_edge(n, "save_memory")
    g.add_edge("save_memory", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_turn(student_id: str, session_id: str, message: str) -> TurnState:
    return get_graph().invoke({"student_id": student_id, "session_id": session_id,
                               "message": message})


def stream_turn(student_id: str, session_id: str, message: str) -> Iterator[dict]:
    """Yield {'type': 'token'|'replace', ...} events, then {'type': 'final', 'state': ...}."""
    final: dict = {}
    for mode, chunk in get_graph().stream(
        {"student_id": student_id, "session_id": session_id, "message": message},
        stream_mode=["custom", "values"],
    ):
        if mode == "custom":
            yield chunk
        else:
            final = chunk
    yield {"type": "final", "state": final}
