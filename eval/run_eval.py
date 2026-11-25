"""Golden-set evaluation: retrieval recall@5 + answer faithfulness + refusal checks.

    python -m eval.run_eval              # full run, writes eval/report.md + eval/results.json
    python -m eval.run_eval --gate       # exit 1 if a gate fails (used by CI)
    python -m eval.run_eval --judge      # use Claude as faithfulness judge (needs API key)

Metrics
- recall@5     share of answerable questions with >=1 gold page in the top 5 (hit rate)
- strict recall@5  mean |gold ∩ top5| / min(|gold|, 5)
- MRR@5        mean reciprocal rank of the first gold page
- citation rate     answerable answers that cite at least one [file p.N]
- citation validity citations that point at a retrieved passage
- faithfulness      answer claims supported by the passages they cite
                     (offline: content-word overlap >= 0.6; --judge: Claude verdict)
- refusal accuracy  off-topic questions answered with "not in my notes"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from app.llm import get_llm
from app.prompts import NOT_IN_NOTES
from app.rag import CITE_RE, Tutor, cite
from app.retrieval import Retriever
from app.retrieval.hybrid import HybridRetriever

HERE = Path(__file__).parent
GATES = {"recall@5": 0.80, "refusal_accuracy": 1.0, "citation_rate": 1.0, "faithfulness": 0.80}
_W = re.compile(r"[a-z0-9]+")
_STOP = set("""the a an and or of to in on for is are be by with as at from that this it its
into not no can may than then them they their there which what who how when where why""".split())


def load_golden(path: Path = HERE / "golden_set.jsonl") -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _key(meta_or_rel: dict) -> tuple[str, int]:
    return meta_or_rel["file"], int(meta_or_rel["page"])


def retrieval_metrics(retriever, items: list[dict], k: int = 5) -> dict:
    hits = strict = mrr = 0.0
    rows = []
    for it in items:
        gold = {_key(r) for r in it["relevant"]}
        res = retriever.search(it["question"], k=k)
        got = [_key(h["metadata"]) for h in res]
        found = [g for g in got if g in gold]
        hit = bool(found)
        rank = next((i + 1 for i, g in enumerate(got) if g in gold), None)
        hits += hit
        strict += len(set(found)) / min(len(gold), k)
        mrr += 1 / rank if rank else 0
        rows.append({"id": it["id"], "hit": hit, "rank": rank, "top5": [f"{f} p.{p}" for f, p in got]})
    n = len(items)
    return {"recall@5": round(hits / n, 3), "strict_recall@5": round(strict / n, 3),
            "mrr@5": round(mrr / n, 3), "rows": rows}


def _claims(text: str) -> list[str]:
    body = re.sub(r"^From your notes:\s*", "", text.strip())
    parts = [p.strip(" -•\n") for p in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", body)]
    return [p for p in parts if len(_W.findall(p)) >= 4 and not p.startswith("Source:")]


def _supported_offline(claim: str, passages: list[str]) -> bool:
    words = [w for w in _W.findall(CITE_RE.sub("", claim).lower()) if w not in _STOP]
    if not words:
        return True
    ctx = set(_W.findall(" ".join(passages).lower()))
    return sum(w in ctx for w in words) / len(words) >= 0.6


def _supported_judge(claim: str, passages: list[str]) -> bool:
    res = get_llm().complete(
        "You check whether a claim is fully supported by the passages. Answer only YES or NO.",
        [{"role": "user", "content": "Passages:\n" + "\n---\n".join(p[:1500] for p in passages)
          + f"\n\nClaim: {claim}"}],
        task="judge", max_tokens=200, offline=lambda: "YES" if _supported_offline(claim, passages) else "NO")
    return res.text.strip().upper().startswith("YES")


def answer_metrics(tutor: Tutor, items: list[dict], judge: bool) -> dict:
    ans_rows, n_claims, n_supported = [], 0, 0
    cited = valid = total_cites = 0
    refused_ok = false_refusals = 0
    answerable = [i for i in items if i["answerable"]]
    off = [i for i in items if not i["answerable"]]
    for it in items:
        t0 = time.perf_counter()
        a = tutor.answer(it["question"])
        ms = (time.perf_counter() - t0) * 1000
        row = {"id": it["id"], "refused": a.refused, "citations": a.citations,
               "latency_ms": round(ms), "answer": a.text[:600]}
        if it["answerable"]:
            false_refusals += a.refused
            if a.citations:
                cited += 1
            by_cite = {cite(h["metadata"]): h["text"] for h in a.hits}
            for c in CITE_RE.finditer(a.text):
                total_cites += 1
                valid += f"[{c.group(1)} p.{c.group(2)}]" in by_cite
            claims = _claims(a.text) if not a.refused else []
            sup = 0
            for cl in claims:
                cs = [f"[{m.group(1)} p.{m.group(2)}]" for m in CITE_RE.finditer(cl)]
                passages = [by_cite[c] for c in cs if c in by_cite] or [h["text"] for h in a.hits]
                ok = (_supported_judge if judge else _supported_offline)(cl, passages)
                sup += ok
            n_claims += len(claims)
            n_supported += sup
            row["claims"], row["supported"] = len(claims), sup
        else:
            refused_ok += a.refused and NOT_IN_NOTES in a.text
        ans_rows.append(row)
    return {
        "citation_rate": round(cited / len(answerable), 3),
        "citation_validity": round(valid / total_cites, 3) if total_cites else 0.0,
        "faithfulness": round(n_supported / n_claims, 3) if n_claims else 0.0,
        "false_refusal_rate": round(false_refusals / len(answerable), 3),
        "refusal_accuracy": round(refused_ok / len(off), 3) if off else 1.0,
        "rows": ans_rows,
    }


def write_report(res: dict, path: Path) -> None:
    r, a = res["retrieval"]["hybrid+rerank"], res["answers"]
    lines = [
        "# Golden-set evaluation report", "",
        f"- Date: {res['date']}  ·  LLM: `{res['llm']}`  ·  faithfulness judge: `{res['judge']}`",
        f"- Golden set: {res['n_answerable']} answerable questions + {res['n_offtopic']} off-topic",
        f"- Collection: {res['chunks']} chunks · embed `{res['embed_model']}` · rerank `{res['rerank_model']}`",
        "", "## Gates", "", "| Metric | Value | Gate | Pass |", "|---|---|---|---|",
    ]
    for m, g in GATES.items():
        v = r[m] if m in r else a[m]
        lines.append(f"| {m} | {v:.3f} | ≥ {g:.2f} | {'✅' if v >= g else '❌'} |")
    lines += ["", "## Retrieval ablation (k = 5)", "",
              "| Retriever | recall@5 | strict recall@5 | MRR@5 |", "|---|---|---|---|"]
    for name, m in res["retrieval"].items():
        lines.append(f"| {name} | {m['recall@5']:.3f} | {m['strict_recall@5']:.3f} | {m['mrr@5']:.3f} |")
    lines += ["", "## Answers", "",
              f"- citation rate (answers citing file + page): **{a['citation_rate']:.3f}**",
              f"- citation validity (cited page was retrieved): **{a['citation_validity']:.3f}**",
              f"- faithfulness (claims supported by cited notes): **{a['faithfulness']:.3f}**",
              f"- false refusals on answerable questions: **{a['false_refusal_rate']:.3f}**",
              f"- off-topic → \"not in my notes\": **{a['refusal_accuracy']:.3f}**",
              "", "## Misses (hybrid+rerank)", ""]
    misses = [row for row in r["rows"] if not row["hit"]]
    lines += [f"- `{m['id']}` top-5: {', '.join(m['top5'][:3])}…" for m in misses] or ["- none"]
    lines += ["", "## Notes", "",
              "- Gold pages were labelled from the PDF text (pages containing the answer), "
              "not from retriever output. Page numbers are PDF page indices (what a PDF viewer shows).",
              "- The refusal thresholds (`MIN_SCORE`, `MIN_SCORE_SOFT`, `MIN_VECTOR`) were tuned on "
              "this golden set plus ~25 off-topic probes, so refusal numbers are optimistic; "
              "re-check them when the notes change.",
              "- Offline runs use the extractive tutor (no API key); faithfulness is then a "
              "word-overlap check. Run with an `ANTHROPIC_API_KEY` and `--judge` for Claude answers "
              "judged by Claude.", "",
              "_Generated by `python -m eval.run_eval`._", ""]
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--judge", action="store_true", help="LLM faithfulness judge")
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()

    items = load_golden()
    answerable = [i for i in items if i["answerable"]]
    hybrid = HybridRetriever()
    res: dict = {"date": time.strftime("%Y-%m-%d %H:%M"), "llm": get_llm().model
                 if not get_llm().offline else "offline-stub (extractive)",
                 "judge": "claude" if args.judge and not get_llm().offline else "word-overlap",
                 "n_answerable": len(answerable), "n_offtopic": len(items) - len(answerable),
                 "chunks": hybrid.store.count(), "embed_model": hybrid.store.embedder.model_name,
                 "rerank_model": hybrid.reranker.model_name, "retrieval": {}}
    if not args.no_ablation:
        res["retrieval"]["vector only"] = retrieval_metrics(Retriever(hybrid.store), answerable)
        res["retrieval"]["hybrid (BM25+vector, RRF)"] = retrieval_metrics(
            HybridRetriever(hybrid.store, use_rerank=False), answerable)
    res["retrieval"]["hybrid+rerank"] = retrieval_metrics(hybrid, answerable)
    res["answers"] = answer_metrics(Tutor(hybrid), items, args.judge)

    (HERE / "results.json").write_text(json.dumps(res, indent=2))
    write_report(res, HERE / "report.md")
    r, a = res["retrieval"]["hybrid+rerank"], res["answers"]
    print(f"recall@5={r['recall@5']:.3f} strict={r['strict_recall@5']:.3f} mrr={r['mrr@5']:.3f} "
          f"citation_rate={a['citation_rate']:.3f} faithfulness={a['faithfulness']:.3f} "
          f"refusal_acc={a['refusal_accuracy']:.3f} false_refusals={a['false_refusal_rate']:.3f}")
    failed = [m for m, g in GATES.items() if (r.get(m, a.get(m)) or 0) < g]
    if failed:
        print("GATE FAILED:", failed)
    return 1 if (args.gate and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
