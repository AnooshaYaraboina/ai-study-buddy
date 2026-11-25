"""Typed tools used by the agents (and exposed over MCP).

search_notes · make_quiz · save_flashcard · get_progress

``student_id`` is never a tool argument: it comes from the authenticated
``ToolContext``, so the model cannot ask for another student's data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field

from app import db
from app.observability import tracer
from app.rag import cite


@dataclass
class ToolContext:
    student_id: str
    session_id: str = "default"


class SearchNotesInput(BaseModel):
    query: str = Field(min_length=2, max_length=500, description="What to look up in the notes")
    day: int | None = Field(default=None, ge=0, le=60, description="Only notes from this course day")
    topic: str | None = Field(default=None, max_length=80, description="Only notes with this topic")
    k: int = Field(default=5, ge=1, le=10, description="How many passages to return")


class MakeQuizInput(BaseModel):
    topic: str = Field(min_length=2, max_length=120, description="Topic to quiz on")
    n: int = Field(default=5, ge=1, le=10, description="Number of questions")


class SaveFlashcardInput(BaseModel):
    front: str = Field(min_length=2, max_length=300, description="Question / term side")
    back: str = Field(min_length=2, max_length=1000, description="Answer side")
    source: str = Field(default="", max_length=200, description="Citation like [file p.N]")
    topic: str = Field(default="", max_length=80)


class GetProgressInput(BaseModel):
    pass


# ------------------------------------------------------------------ implementations
def search_notes(ctx: ToolContext, inp: SearchNotesInput) -> list[dict]:
    from app.retrieval import get_retriever

    hits = get_retriever().search(inp.query, k=inp.k, owner=ctx.student_id,
                                  day=inp.day, topic=inp.topic)
    return [{"citation": cite(h["metadata"]), "topic": h["metadata"].get("topic"),
             "day": h["metadata"].get("day"), "score": round(h["score"], 3),
             "text": h["text"][:1200]} for h in hits]


def make_quiz(ctx: ToolContext, inp: MakeQuizInput) -> list[dict]:
    from app.agents.quiz import make_quiz as _make
    from app.retrieval import get_retriever

    r = get_retriever()
    hits = r.search(inp.topic, k=8, owner=ctx.student_id)
    pool = r.search("definitions key concepts course overview", k=20, owner=ctx.student_id) + \
        r.search("how it works things to remember", k=20, owner=ctx.student_id)
    qs = _make(inp.topic, hits, pool, n=inp.n)
    return [q.model_dump() for q in qs]


def save_flashcard(ctx: ToolContext, inp: SaveFlashcardInput) -> dict:
    return db.save_flashcard(ctx.student_id, inp.front, inp.back, inp.source, inp.topic)


def get_progress(ctx: ToolContext, inp: GetProgressInput) -> dict:
    return db.progress(ctx.student_id)


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[ToolContext, Any], Any]

    def run(self, ctx: ToolContext, args: dict | BaseModel) -> Any:
        inp = args if isinstance(args, BaseModel) else self.input_model.model_validate(args or {})
        with tracer().start_as_current_span(f"tool.{self.name}") as span:
            span.set_attribute("tool.student", ctx.student_id)
            return self.fn(ctx, inp)

    def anthropic_schema(self) -> dict:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        for p in schema.get("properties", {}).values():
            p.pop("title", None)
        schema["additionalProperties"] = False
        return {"name": self.name, "description": self.description, "input_schema": schema}


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("search_notes", "Search the student's course notes (hybrid search + rerank). "
         "Returns passages with [file p.N] citations.", SearchNotesInput, search_notes),
    Tool("make_quiz", "Create a multiple-choice quiz (4 options each) on a topic from the notes.",
         MakeQuizInput, make_quiz),
    Tool("save_flashcard", "Save a flashcard (front/back) for the current student.",
         SaveFlashcardInput, save_flashcard),
    Tool("get_progress", "Get the current student's quiz accuracy, weak topics and card count.",
         GetProgressInput, get_progress),
]}


def run_tool(name: str, ctx: ToolContext, args: dict) -> Any:
    if name not in TOOLS:
        raise KeyError(f"unknown tool {name}")
    return TOOLS[name].run(ctx, args)
