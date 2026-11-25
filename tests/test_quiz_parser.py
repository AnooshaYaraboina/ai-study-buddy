import json

import pytest

from app.agents.quiz import QuizParseError, offline_quiz, parse_quiz_json

Q = {"question": "What does RRF fuse?", "options": ["ranked lists", "embeddings", "tokens", "logs"],
     "answer": 0, "explanation": "RRF fuses ranks [h.pdf p.29]", "source": "[h.pdf p.29]",
     "topic": "retrieval"}


def test_strict_json_object():
    qs = parse_quiz_json(json.dumps({"questions": [Q] * 5}))
    assert len(qs) == 5 and qs[0].answer == 0 and qs[0].options[0] == "ranked lists"


def test_code_fences_and_prose_around_json():
    raw = "Sure! Here is your quiz:\n```json\n" + json.dumps({"questions": [Q]}) + "\n```\nGood luck!"
    assert parse_quiz_json(raw)[0].question == Q["question"]


def test_bare_list_and_single_object():
    assert len(parse_quiz_json(json.dumps([Q, Q]))) == 2
    assert len(parse_quiz_json(json.dumps(Q))) == 1


@pytest.mark.parametrize("ans,expected", [(2, 2), ("C", 2), ("c)", 2), ("(B)", 1), ("3", 3),
                                          ("tokens", 2), (4, 3)])
def test_answer_formats_are_normalised(ans, expected):
    q = {**Q, "answer": ans}
    assert parse_quiz_json(json.dumps({"questions": [q]}))[0].answer == expected


def test_lettered_options_and_dict_options():
    q = {**Q, "options": ["A) ranked lists", "B) embeddings", "C) tokens", "D) logs"]}
    assert parse_quiz_json(json.dumps([q]))[0].options[0] == "ranked lists"
    q2 = {**Q, "options": {"A": "ranked lists", "B": "embeddings", "C": "tokens", "D": "logs"}}
    assert parse_quiz_json(json.dumps([q2]))[0].options[3] == "logs"


def test_invalid_items_are_dropped_not_fatal():
    bad = [{**Q, "options": ["a", "b", "c"]}, {**Q, "options": ["a", "a", "b", "c"]},
           {**Q, "answer": 9}, {"question": "x"}, "nonsense", Q]
    qs = parse_quiz_json(json.dumps({"questions": bad}))
    assert len(qs) == 1


@pytest.mark.parametrize("raw", ["", "no json here", "{\"questions\": []}", "[1, 2, 3]",
                                 "{\"questions\": [{\"question\": \"q?\"}]}"])
def test_unusable_output_raises_parse_error(raw):
    with pytest.raises(QuizParseError):
        parse_quiz_json(raw)


def test_truncated_json_with_valid_prefix_object():
    raw = json.dumps(Q) + ', {"question": "cut off'
    assert parse_quiz_json(raw)[0].question == Q["question"]


def test_offline_generator_always_produces_valid_round_trippable_json():
    hit = {"text": "Retrieval\nReciprocal rank fusion\nCombines several ranked lists using reciprocal ranks of items.\n"
                   "Hybrid search\nCombines keyword matching with dense vector similarity for recall.\n"
                   "Reranking reorders candidate passages with a stronger cross-encoder model.",
           "metadata": {"file": "h.pdf", "page": 29, "topic": "retrieval"}}
    pool = [{"text": f"X\nTerm {i}\nDefinition number {i} describes something else entirely here.\n"
                     f"Statement {i} is another fact that is written in the course notes.",
             "metadata": {"file": "o.pdf", "page": i, "topic": "other"}} for i in range(6)]
    qs = offline_quiz("retrieval", [hit], pool, n=3, seed=1)
    assert qs, "generator produced questions"
    again = parse_quiz_json(json.dumps({"questions": [q.model_dump() for q in qs]}))
    assert [q.answer for q in again] == [q.answer for q in qs]
    assert all(len(set(q.options)) == 4 for q in again)
