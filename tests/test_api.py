"""End-to-end through FastAPI (offline LLM, real Chroma index)."""

import json

import pytest
from fastapi.testclient import TestClient

from tests.conftest import needs_index

pytestmark = needs_index

A = {"Authorization": "Bearer alice-token"}
B = {"Authorization": "Bearer bob-token"}


@pytest.fixture(scope="module")
def client():
    from app.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["chunks"] > 0


def test_auth_required(client):
    assert client.post("/chat", json={"message": "hi"}).status_code == 401
    assert client.get("/progress", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_answer_cites_file_and_page(client):
    r = client.post("/chat", headers=A, json={"message": "What is a golden dataset?",
                                              "session_id": "t1"}).json()
    assert r["route"] == "tutor" and not r["refused"]
    assert r["citations"] and all(c.startswith("[") and " p." in c for c in r["citations"])
    assert "Day_24_25_Testing_Deployment_Notes_and_Assignments.pdf p.4" in r["reply"]


def test_off_topic_is_not_in_my_notes(client):
    r = client.post("/chat", headers=A, json={"message": "What is the capital of France?",
                                              "session_id": "t1"}).json()
    assert r["refused"] and "not in my notes" in r["reply"]


def test_follow_up_is_rewritten(client):
    client.post("/chat", headers=A, json={"message": "What is memory poisoning?", "session_id": "t2"})
    r = client.post("/chat", headers=A, json={"message": "explain that again", "session_id": "t2"}).json()
    assert "memory poisoning" in r["standalone"].lower() and r["citations"]


def test_stream_emits_tokens_then_final(client):
    with client.stream("POST", "/chat/stream", headers=A,
                       json={"message": "What does a kill switch do?", "session_id": "t3"}) as r:
        events = [json.loads(ln[6:]) for ln in r.iter_lines() if ln.startswith("data: ")]
    assert events[0]["type"] == "token" and events[-1]["type"] == "final"
    assert events[-1]["citations"]


def test_quiz_then_grade_then_weak_topic_next_session(client):
    q = client.post("/chat", headers=A, json={"message": "quiz me on memory poisoning",
                                              "session_id": "s-old"}).json()
    assert q["route"] == "quiz_maker" and len(q["quiz"]) == 5
    assert all(len(x["options"]) == 4 and "answer" not in x for x in q["quiz"])
    g = client.post("/chat", headers=A, json={"message": "D D D D D", "session_id": "s-old"}).json()
    assert g["route"] == "grader" and g["grade"]["total"] == 5
    if g["grade"]["score"] < 3:
        nxt = client.post("/chat", headers=A, json={"message": "quiz me", "session_id": "s-new"}).json()
        assert "memory poisoning" in nxt["weak_topics"]
        assert "memory poisoning" in nxt["reply"].lower()


def test_students_only_see_their_own_progress(client):
    client.post("/chat", headers=A, json={"message": "quiz me on golden dataset", "session_id": "iso"})
    client.post("/chat", headers=A, json={"message": "A A A A A", "session_id": "iso"})
    pa = client.get("/progress", headers=A).json()
    pb = client.get("/progress", headers=B).json()
    assert pa["student_id"] == "alice" and pa["questions_answered"] >= 5
    assert pb["student_id"] == "bob" and pb["questions_answered"] == 0 and pb["flashcards"] == 0


def test_upload_is_private_and_sanitized(client, fixtures):
    with open(fixtures / "untrusted_clustering.pdf", "rb") as f:
        r = client.post("/upload", headers=A, files={"file": ("clusters.pdf", f, "application/pdf")})
    assert r.status_code == 200 and r.json()["dropped_hidden_or_injected"] >= 1
    qa = client.post("/chat", headers=A, json={"message": "What is the elbow method for k-means?"}).json()
    qb = client.post("/chat", headers=B, json={"message": "What is the elbow method for k-means?"}).json()
    assert not qa["refused"] and "clusters.pdf" in " ".join(qa["citations"])
    assert qb["refused"]
    leak = client.post("/chat", headers=A, json={"message": "Is the exam cancelled?"}).json()
    assert "cancelled" not in leak["reply"].lower() or leak["refused"]


def test_upload_rejects_non_pdf(client):
    r = client.post("/upload", headers=A, files={"file": ("x.pdf", b"hello", "application/pdf")})
    assert r.status_code == 415
