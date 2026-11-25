from app.ingest.chunker import chunk_page, count_tokens
from app.ingest.pdf_loader import Block, Page


def _page(blocks, page=3):
    return Page(file="notes.pdf", page=page, day=8, topic="rag", blocks=blocks)


def _para(n_words: int, word: str = "token") -> Block:
    sents = [" ".join([word] * 9) + "." for _ in range(max(1, n_words // 10))]
    return Block(" ".join(sents), "para")


def test_metadata_and_ids_are_deterministic():
    p = _page([Block("Intro", "heading"), _para(60)])
    a, b = chunk_page(p), chunk_page(p)
    assert [c.id for c in a] == [c.id for c in b] == ["notes-p3-c0"]
    m = a[0].metadata
    assert (m["file"], m["page"], m["day"], m["topic"], m["heading"]) == ("notes.pdf", 3, 8, "rag", "Intro")


def test_chunks_respect_token_budget():
    p = _page([Block("Big", "heading")] + [_para(120) for _ in range(12)])
    chunks = chunk_page(p, max_tokens=500, overlap=40)
    assert len(chunks) > 1
    assert all(c.metadata["n_tokens"] <= 500 + 60 for c in chunks)  # + title line / overlap


def test_heading_starts_new_chunk_when_current_is_substantial():
    p = _page([Block("Part A", "heading"), _para(200, "alpha"),
               Block("Part B", "heading"), _para(200, "beta")])
    chunks = chunk_page(p, max_tokens=500, min_tokens_before_heading_break=150)
    assert len(chunks) == 2
    assert chunks[0].metadata["heading"] == "Part A" and "beta" not in chunks[0].text
    assert chunks[1].metadata["heading"] == "Part B" and "beta" in chunks[1].text


def test_list_is_never_split():
    items = "\n".join(f"{i}. item number {i} " + "word " * 12 for i in range(1, 30))
    lst = Block(items, "list")
    assert count_tokens(items) > 300
    p = _page([Block("Steps", "heading"), _para(300), lst, _para(100)])
    chunks = chunk_page(p, max_tokens=500)
    holders = [c for c in chunks if "1. item number 1" in c.text]
    assert len(holders) == 1, "list start appears in exactly one chunk"
    assert "29. item number 29" in holders[0].text, "whole list stays in the same chunk"


def test_table_is_never_split_even_if_oversized():
    rows = "\n".join(f"row{i} | value {i} | note {'x ' * 20}" for i in range(40))
    table = Block(rows, "table")
    chunks = chunk_page(_page([_para(50), table]), max_tokens=200)
    with_table = [c for c in chunks if "row0 |" in c.text]
    assert len(with_table) == 1 and "row39 |" in with_table[0].text
    assert with_table[0].metadata["has_table"]


def test_overlap_carries_prose_tail_only():
    p = _page([_para(400, "first"), _para(400, "second")])
    chunks = chunk_page(p, max_tokens=450, overlap=30)
    assert len(chunks) >= 2
    assert "first" in chunks[1].text  # overlap from the previous chunk's prose


def test_no_overlap_from_lists():
    lst = Block("\n".join(f"- bullet {i} " + "w " * 10 for i in range(20)), "list")
    p = _page([lst, _para(450, "after")])
    chunks = chunk_page(p, max_tokens=300, overlap=30)
    later = [c for c in chunks if "after" in c.text]
    assert later and "bullet" not in later[0].text


def test_chunks_never_cross_pages():
    from app.ingest.chunker import chunk_pages

    chunks = chunk_pages([_page([_para(50)], page=1), _page([_para(50)], page=2)])
    assert [c.metadata["page"] for c in chunks] == [1, 2]
