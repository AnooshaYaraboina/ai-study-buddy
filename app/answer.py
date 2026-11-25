"""Readable, cited answers composed from the retrieved notes (no-LLM mode).

Without an API key the tutor still has to *answer*, not just point at pages:

1. split the top passages into units: sentences, "Label: text" items, Q/A pairs
2. score every unit against the question with the cross-encoder reranker, plus a
   bonus when its label or its section title matches the question
3. pick a lead answer and supporting key points; for "how / steps / stages"
   questions keep the steps in the order the notes give them
4. render markdown where every line ends with its ``[file p.N]`` citation
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9][a-z0-9+\-/]*")
_STOP = set("""a an the and or of to in on for is are was were be been what which who whom how why
when where does do did can could should would i me my you your it its this that these those with as
at by from about explain tell describe define definition meaning mean means give example examples
vs versus between difference differences please again more work works working use used using role
purpose main some any during should must need needs happen happens have has kind kinds type types""".split())
_LABEL = re.compile(r"^(?P<label>[^:]{2,60}?):\s+(?P<body>\S.*)$")
_NUM_Q = re.compile(r"^\d{1,2}\.\s+(?P<q>.+)$")
_ITEM_NO = re.compile(r"^\d{1,2}[.)]?\s+")
_PROCESS = re.compile(r"\b(steps?|stages?|process|flow|pipeline|lifecycle|becomes?|workflow|"
                      r"sequence|phases?|path|order)\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")
_ACRONYMS = {"ai", "ml", "llm", "llms", "rag", "mcp", "a2a", "acp", "api", "pii", "ci", "cd", "sse",
             "pdf", "acl", "acls", "owasp", "db", "id", "ui", "ux", "slo", "slos", "cia", "gpu",
             "json", "url", "http", "sql", "otel", "rrf", "bm25", "hnsw", "dl", "nlp"}
_SMALL = {"the", "a", "an", "of", "to", "and", "or", "for", "in", "on", "vs", "by", "with", "at"}


def cite(meta: dict) -> str:
    return f"[{meta['file']} p.{meta['page']}]"


def _nice(label: str) -> str:
    """'PROMPT INJECTION' -> 'Prompt injection' (acronyms kept); strip '01 ' numbering."""
    label = _ITEM_NO.sub("", label).strip()
    letters = [c for c in label if c.isalpha()]
    if not letters or not all(c.isupper() for c in letters):
        return label
    words = []
    for i, w in enumerate(label.split()):
        lw = w.lower()
        if lw.strip("(),") in _ACRONYMS or (len(w.strip("(),")) <= 4 and len(label.split()) == 1):
            words.append(w)
        elif i == 0:
            words.append(lw.capitalize())
        else:
            words.append(lw)
    return " ".join(words)


@dataclass
class Unit:
    kind: str  # sent | labeled | qa
    label: str
    body: str
    hit: int
    sec: int
    section: str
    pos: int
    score: float = 0.0  # relevance + bonuses (used to rank)
    ce: float = 0.0  # relevance (cross-encoder + embedding) after quality penalties

    @property
    def text(self) -> str:
        if self.kind == "labeled":
            return f"{self.label}: {self.body}"
        if self.kind == "qa":
            return f"{self.label} {self.body}"
        return self.body

    @property
    def scoring_text(self) -> str:
        """A sentence is scored together with its section title ("Golden datasets and
        test cases. A versioned, curated benchmark ..."), so it keeps its context."""
        if self.kind == "sent" and self.section:
            return f"{self.section}. {self.body}"
        return self.text

    @property
    def fragmentary(self) -> bool:
        """Tag lists from diagrams ("routine · ambiguous · hostile") are weak answers."""
        return self.body.count(" · ") >= 2

    def render(self) -> str:
        body = self.body.replace(" -> ", " → ")
        if self.kind == "labeled":
            return f"**{_nice(self.label)}** — {body}"
        if self.kind == "qa":
            return f"**{self.label}** {body}"
        return body


def _content_words(text: str) -> int:
    return sum(1 for w in re.findall(r"[A-Za-z]{3,}", text))


def _sentences(text: str, max_words: int = 45) -> list[str]:
    if len(text.split()) <= max_words:
        return [text]
    out, cur = [], ""
    for s in _SENT_SPLIT.split(text):
        cand = f"{cur} {s}".strip()
        if cur and len(cand.split()) > max_words:
            out.append(cur)
            cur = s
        else:
            cur = cand
    return out + ([cur] if cur else [])


def split_units(hit: int, text: str) -> list[Unit]:
    """Chunk text (line 0 = chunk title) -> answer units."""
    lines = [ln.strip() for ln in text.split("\n")[1:]]
    units: list[Unit] = []
    sec, section, i = 0, "", 0

    def add(kind: str, label: str, body: str) -> None:
        units.append(Unit(kind, label.strip(), body.strip(), hit, sec, section, len(units)))

    while i < len(lines):
        ln, nxt = lines[i], (lines[i + 1] if i + 1 < len(lines) else "")
        i += 1
        if not ln or ln.isdigit() or (" · " in ln and len(ln.split()) <= 12):
            continue  # page numbers, running headers
        if ln.startswith("▸"):
            sec, section = sec + 1, ln.lstrip("▸ ").strip()
            continue
        if ln.startswith("→") and units:
            units[-1].body += " " + ln
            continue
        if ln.startswith("Q:"):
            if nxt.startswith("A:"):
                add("qa", ln[2:], nxt[2:])
                i += 1
            continue  # never show a question without its answer
        if ln.startswith("A:"):
            continue  # ...or an answer without its question
        m = _NUM_Q.match(ln)
        if (m and nxt and not _NUM_Q.match(nxt) and not nxt.startswith("▸")
                and len(nxt.split()) >= 4 and (ln.endswith("?") or len(ln.split()) <= 9)):
            add("qa", m.group("q"), nxt)
            i += 1
            continue
        m = _LABEL.match(ln)
        if m and len(m.group("label").split()) <= 6:
            if _content_words(m.group("body")) >= 2:
                add("labeled", _ITEM_NO.sub("", m.group("label")), m.group("body"))
            continue
        if len(ln.split()) >= 6:
            for part in _sentences(ln):
                add("sent", "", part)
    return units


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower())}


def _focus(question: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(question.lower()) if w not in _STOP and len(w) > 1}


def _overlap(a: str, b: str) -> float:
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(wa & wb) / max(1, min(len(wa), len(wb)))


def _similarities(question: str, texts: list[str]) -> list[float]:
    from app.ingest.embedder import Embedder

    vecs = Embedder().embed([question, *texts])
    q = vecs[0]
    return [sum(a * b for a, b in zip(q, v)) for v in vecs[1:]]


def compose_answer(question: str, hits: list[dict], reranker=None, max_points: int = 5,
                   max_words: int = 230) -> str:
    """Return a markdown answer built only from ``hits``, every line cited."""
    if not hits:
        return ""
    top = hits[0].get("score", 1.0)
    use = [h for h in hits[:3] if h.get("score", 0.0) >= 0.25 * top] or hits[:1]
    units = [u for i, h in enumerate(use) for u in split_units(i, h["text"])]
    if not units:  # nothing parseable: show the passage's first lines
        first = " ".join(ln for ln in use[0]["text"].split("\n")[1:4] if ln.strip())
        return f"{first} {cite(use[0]['metadata'])}"

    from app.retrieval.hybrid import Reranker

    sections = sorted({(u.hit, u.sec, u.section) for u in units if u.section})
    ce = (reranker or Reranker()).score(question, [u.scoring_text for u in units] +
                                        [s[2] for s in sections])
    sec_score = {(h, s): sc for (h, s, _), sc in zip(sections, ce[len(units):])}
    title_rel: dict[int, float] = {}  # how much each passage's titles match the question
    for (h, _, _), sc in zip(sections, ce[len(units):]):
        title_rel[h] = max(title_rel.get(h, 0.0), sc)
    cos = _similarities(question, [u.text for u in units])
    focus = _focus(question)
    for u, c, sim in zip(units, ce, cos):
        # cross-encoder is precise but peaky at sentence level; embeddings are smoother
        rel = 0.6 * c + 0.4 * min(1.0, max(0.0, (sim - 0.1) / 0.5))
        if u.fragmentary:
            rel *= 0.4
        if _content_words(u.body) < 4:  # diagram captions ("response lost") make poor answers
            rel *= 0.5
        u.ce = rel
        bonus = 0.1 * sec_score.get((u.hit, u.sec), 0.0)
        if focus and u.kind in ("labeled", "qa"):
            lab = _words(u.label)
            inter = len(focus & lab)
            if inter:  # the label names what was asked ("Context window", "RRF")
                cover = inter / len(focus)
                if len(focus) <= 2:
                    cover = max(cover, inter / max(1, len(lab)))
                bonus += 0.25 * cover
        bonus += 0.08 * (len(use) - 1 - u.hit) / max(1, len(use) - 1)  # prefer the top passage
        u.score = rel + bonus

    def good(u: Unit) -> bool:  # worth showing as a key point
        return not u.fragmentary and _content_words(u.body) >= 3

    ranked = sorted(units, key=lambda u: -u.score)
    lead = ranked[0]
    same = [u for u in units if u.hit == lead.hit]
    process = bool(_PROCESS.search(question))
    if process and lead.kind == "labeled":
        # lead is one step of a sequence: show the run of steps, introduced by the page's lead line
        run = [u for u in same if u.sec == lead.sec and good(u)] or [lead]
        intro = next((u for u in same if u.kind == "sent" and u.pos < run[0].pos), None)
        lead, rest = (intro, run) if intro else (run[0], run[1:])
        rest = rest[: max_points + 1]
    elif process:
        rest = [u for u in same if u.pos > lead.pos and good(u)][: max_points + 1]
    else:
        others_on_page = sorted((u for u in same if u is not lead and good(u)),
                                key=lambda u: -u.score)
        titles = " ".join(t for (h, _, t) in sections if h == lead.hit)
        title_words = _words(titles)
        named = len(focus & title_words) / max(1, len(focus)) if focus else 0.0
        topic_page = title_rel.get(lead.hit, 0.0) >= 0.3 or lead.pos == 0 or named >= 0.5
        if topic_page:
            rest = others_on_page[:3]  # the page is about the question: its main items belong
        else:  # e.g. a glossary page: only the entries that match the question
            rest = [u for u in others_on_page if u.ce >= 0.28][:3]
        if lead.hit != 0:  # always let the best-ranked passage contribute
            rest += sorted((u for u in units if u.hit == 0 and good(u)), key=lambda u: -u.score)[:2]
    for u in ranked[1:]:  # strongest points from the other passages
        if len(rest) >= max_points:
            break
        if u.hit != lead.hit and u not in rest and good(u) and u.ce >= 0.28:
            rest.append(u)

    picked, words = [], len(lead.text.split())
    for u in rest:
        if any(_overlap(u.text, p.text) >= 0.7 for p in [lead, *picked]):
            continue
        if words + len(u.text.split()) > max_words:
            break
        picked.append(u)
        words += len(u.text.split())
    if not process:
        picked.sort(key=lambda u: (u.hit, u.pos))

    lead_txt = lead.render()
    if (lead.kind == "sent" and lead.section and len(lead.section.split()) <= 8
            and sec_score.get((lead.hit, lead.sec), 0.0) >= 0.3):
        lead_txt = f"**{lead.section}** — {lead_txt}"
    out = [f"{lead_txt} {cite(use[lead.hit]['metadata'])}"]
    if picked:
        out += ["", "**Key points from your notes:**"]
        out += [f"- {u.render()} {cite(use[u.hit]['metadata'])}" for u in picked]
    return "\n".join(out)
