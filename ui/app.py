"""Streamlit chat UI for AI Study Buddy.

    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
import re
import uuid

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8080")
DEFAULT_TOKEN = os.getenv("UI_DEFAULT_TOKEN", "alice-token")

st.set_page_config(page_title="AI Study Buddy", page_icon="📚", layout="wide")


def headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def api_get(path: str):
    r = httpx.get(f"{API}{path}", headers=headers(), timeout=30)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ state
ss = st.session_state
ss.setdefault("token", DEFAULT_TOKEN)
ss.setdefault("session_id", uuid.uuid4().hex[:12])
ss.setdefault("messages", [])
ss.setdefault("quiz", None)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📚 Study Buddy")
    ss.token = st.text_input("Access token", value=ss.token, type="password",
                             help="Demo tokens: alice-token, bob-token")
    st.caption(f"Session `{ss.session_id}`")
    if st.button("New session", use_container_width=True):
        ss.session_id = uuid.uuid4().hex[:12]
        ss.messages, ss.quiz = [], None
        st.rerun()

    st.subheader("Add notes")
    up = st.file_uploader("Upload a PDF (private to you)", type=["pdf"])
    if up and st.button("Ingest PDF", use_container_width=True):
        with st.spinner("Parsing · chunking · embedding…"):
            r = httpx.post(f"{API}/upload", headers=headers(),
                           files={"file": (up.name, up.getvalue(), "application/pdf")}, timeout=600)
        if r.status_code == 200:
            d = r.json()
            st.success(f"{d['file']}: {d['pages']} pages → {d['chunks']} chunks searchable"
                       + (f" ({d['dropped_hidden_or_injected']} suspicious lines dropped)"
                          if d["dropped_hidden_or_injected"] else ""))
        else:
            st.error(r.json().get("detail", r.text))

    st.subheader("My progress")
    try:
        p = api_get("/progress")
        c1, c2 = st.columns(2)
        c1.metric("Answered", p["questions_answered"])
        c2.metric("Accuracy", f"{p['accuracy']:.0%}" if p["accuracy"] is not None else "–")
        st.metric("Flashcards", p["flashcards"])
        if p["weak_topics"]:
            st.write("**Weak topics** (quizzed next):")
            for w in p["weak_topics"]:
                st.write(f"- {w['topic']} ({w['misses']} missed)")
    except httpx.HTTPError as e:
        st.warning(f"API not reachable at {API}: {e}")

    st.divider()
    st.caption("Try: *What is memory poisoning?* · *explain that again* · *quiz me on "
               "prompt injection* · *quiz me* · *how am I doing?*")

# ------------------------------------------------------------------ rendering
CITE_RE = re.compile(r"\[([^\[\]]+?\.pdf) p\.(\d+)\]")
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#+\-!|~])")


def show_reply(text: str) -> str:
    """Answer text with each [file p.N] reference shown as a soft grey page tag."""
    return CITE_RE.sub(lambda m: f":gray[(📄 {m.group(1)} · p.{m.group(2)})]", text)


def passage_md(text: str) -> str:
    """A retrieved note passage as readable markdown (headings bold, line breaks kept)."""
    out = []
    for ln in text.split("\n")[1:]:  # line 0 is the chunk title
        ln = ln.strip()
        if not ln or (" · " in ln and len(ln.split()) <= 12):
            continue
        if ln.startswith("▸"):
            out.append(f"**{_MD_SPECIAL.sub(r'\\\1', ln.lstrip('▸ '))}**")
        else:
            out.append(_MD_SPECIAL.sub(r"\\\1", ln).replace(" \\-\\> ", " → "))
    return "  \n".join(out)


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📄 Read the source pages ({len(sources)})"):
        for s in sources:
            topic = f" · {s['topic']}" if s.get("topic") else ""
            st.markdown(f"**{s['file']} — page {s['page']}**{topic}")
            st.markdown(passage_md(s["text"]))
            st.divider()


def show_message(m: dict) -> None:
    if m["role"] == "assistant":
        st.markdown(show_reply(m["content"]))
        show_sources(m.get("sources") or [])
    else:
        st.markdown(m["content"])


# ------------------------------------------------------------------ chat
st.header("Ask your notes")
for m in ss.messages:
    with st.chat_message(m["role"]):
        show_message(m)


def send(text: str) -> None:
    ss.messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"):
        box = st.empty()
        buf, final = "", None
        try:
            with httpx.stream("POST", f"{API}/chat/stream", headers=headers(), timeout=300,
                              json={"message": text, "session_id": ss.session_id}) as r:
                if r.status_code != 200:
                    r.read()
                    box.error(r.json().get("detail", r.text))
                    return
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    ev = json.loads(line[6:])
                    if ev["type"] == "token":
                        buf += ev["text"]
                        box.markdown(buf + "▌")
                    elif ev["type"] == "replace":
                        buf = ev["text"]
                    elif ev["type"] == "final":
                        final = ev
                    elif ev["type"] == "error":
                        box.error(ev["error"])
                        return
        except httpx.HTTPError as e:
            box.error(f"API error: {e}")
            return
        reply = (final or {}).get("reply") or buf
        if final and final.get("quiz"):
            ss.quiz = final["quiz"]
            reply = reply.split("\n\n**Q1.")[0]
        elif final and final.get("grade"):
            ss.quiz = None
        sources = (final or {}).get("sources") or []
        box.markdown(show_reply(reply))
        show_sources(sources)
        ss.messages.append({"role": "assistant", "content": reply, "sources": sources})


if ss.quiz:
    with st.form("quiz_form"):
        st.subheader("Quiz")
        picks = []
        for i, q in enumerate(ss.quiz):
            picks.append(st.radio(f"**Q{i+1}. {q['question']}**",
                                  options=list(range(4)),
                                  format_func=lambda j, q=q: f"{'ABCD'[j]}) {q['options'][j]}",
                                  index=None, key=f"q{ss.session_id}{i}"))
        if st.form_submit_button("Submit answers"):
            if any(p is None for p in picks):
                st.warning("Answer every question first.")
            else:
                send(" ".join("ABCD"[p] for p in picks))
                st.rerun()

if prompt := st.chat_input("Ask a question, or type 'quiz me on …'"):
    send(prompt)
    if ss.quiz:
        st.rerun()
