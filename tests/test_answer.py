"""Answers must be readable in the chat (not just page references) and keep [file p.N]."""

import re
from pathlib import Path

import pytest

from app.answer import compose_answer, split_units
from app.ingest.pdf_loader import load_pdf
from app.rag import CITE_RE

ROOT = Path(__file__).resolve().parents[1]
PDFS = ROOT / "data" / "pdfs"
HANDBOOK = PDFS / "Days_01_21_Master_Revision_Handbook.pdf"
SECURITY = PDFS / "Day_22_23_AI_Security_Visual_Notes.pdf"


class FakeReranker:
    """Deterministic stand-in for the cross-encoder: word overlap with the question."""

    def score(self, query, texts):
        q = set(re.findall(r"[a-z]{3,}", query.lower()))
        return [min(1.0, len(q & set(re.findall(r"[a-z]{3,}", t.lower()))) / 3) for t in texts]


def _hit(text, page=7, file="notes.pdf", score=0.9):
    return {"text": text, "score": score, "metadata": {"file": file, "page": page, "topic": "t"}}


# ------------------------------------------------------------------ layout -> readable lines
@pytest.mark.skipif(not HANDBOOK.exists(), reason="course PDFs not present")
def test_wrapped_definitions_are_rebuilt_as_full_sentences():
    text = load_pdf(HANDBOOK)[6].text  # p.7 "LLM Fundamentals"
    assert ("Large language model (LLM): Neural network trained to estimate likely token "
            "sequences; autoregressive models generate one token at a time.") in text
    assert "Context window: Maximum input plus generated token context available for one " \
           "request, subject to model limits." in text


@pytest.mark.skipif(not SECURITY.exists(), reason="course PDFs not present")
def test_visual_labels_attach_to_their_text():
    p24 = load_pdf(SECURITY)[23].text
    assert "ENTRY: hostile content arrives: message, page, PDF, ticket, tool result" in p24
    assert ("The same attack can be blocked at many layers. Updated OWASP LLM and agentic "
            "risk guidance supports this threat-path view.") in p24
    p10 = load_pdf(SECURITY)[9].text
    assert "PROMPT INJECTION: makes the APPLICATION follow instructions hidden in untrusted " \
           "content → goal: redirect workflow or tools" in p10


# ------------------------------------------------------------------ composer
def test_units_pair_questions_with_answers_and_skip_orphans():
    text = "Title\n▸ Interview flashcards\nQ: What is leakage?\nA: Test data influences training.\n" \
           "A: an orphan answer line that should not be shown alone\nQ: orphan question?"
    units = split_units(0, text)
    assert [(u.kind, u.label) for u in units] == [("qa", "What is leakage?")]


def test_definition_question_leads_with_the_definition():
    hit = _hit("LLM Fundamentals\n▸ Nine definitions worth memorizing\n"
               "Large language model (LLM): Neural network trained to estimate likely token "
               "sequences; autoregressive models generate one token at a time.\n"
               "Token / tokenizer: A token is a model-sized text unit; a tokenizer maps text to "
               "numeric token IDs and back.\nQ: Can LLM output be deterministic?")
    out = compose_answer("what is llm", [hit], reranker=FakeReranker())
    first = out.split("\n")[0]
    assert first.startswith("**Large language model (LLM)** — Neural network trained")
    assert first.endswith("[notes.pdf p.7]")
    assert "Can LLM output be deterministic?" not in out  # never a question without its answer


def test_process_question_keeps_steps_in_order():
    hit = _hit("Threat Path\n▸ How an attack becomes an incident\n"
               "Follow the path from lower-trust content to a privileged effect: every step is a "
               "chance to stop it.\n▸ Five stages, five chances to block\n"
               "ENTRY: hostile content arrives: message, page, PDF, ticket\n"
               "PROMOTION: lower-trust text mistaken for instructions or permission\n"
               "PROPOSAL: model answers or proposes a sensitive tool call\n"
               "ENFORCEMENT: ACLs, tool gateway, schemas, approvals block it", page=24)
    out = compose_answer("how an attack becomes an incident", [hit], reranker=FakeReranker())
    steps = [ln for ln in out.split("\n") if ln.startswith("- ")]
    names = [re.search(r"\*\*(.+?)\*\*", s).group(1) for s in steps]
    assert names == ["Entry", "Promotion", "Proposal", "Enforcement"]
    assert out.startswith("**How an attack becomes an incident** — Follow the path")


def test_every_line_of_an_answer_is_cited_and_readable():
    hit = _hit("Golden Dataset\n▸ Golden datasets and test cases\n"
               "A versioned, curated benchmark of inputs, expected behaviour and metadata.\n"
               "Holdout: never let eval answers leak into prompt examples\n"
               "Versioning: add new failures; never silently rewrite old cases\n"
               "mix · of · tags · only", page=4)
    out = compose_answer("What is a golden dataset?", [hit], reranker=FakeReranker())
    lines = [ln for ln in out.split("\n") if ln and not ln.endswith(":**")]
    for ln in lines:
        assert CITE_RE.search(ln), f"uncited line: {ln}"
        text = CITE_RE.sub("", ln)
        assert len(re.findall(r"[A-Za-z]{3,}", text)) >= 4, f"fragment, not an answer: {ln}"
    assert "A versioned, curated benchmark" in out
    assert "mix · of · tags" not in out  # diagram tag lists are not answers
