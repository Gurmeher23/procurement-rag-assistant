"""Retrieval evaluation harness.

Measures, over eval/questions.jsonl:
  * source hit@k  — did the expected source appear in the top-k passages?
  * keyword recall — fraction of `must_contain` strings found in retrieved text

Run:
    python -m eval.run_eval            # from the project root
"""
from __future__ import annotations

import json
import os

from rag.config import RAGConfig
from rag.loaders import load_documents
from rag.pipeline import RAGPipeline

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)


def load_cases() -> list[dict]:
    with open(os.path.join(HERE, "questions.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(top_k: int = 4) -> dict:
    pipe = RAGPipeline(RAGConfig(top_k=top_k)).ingest(
        load_documents(os.path.join(ROOT, "data")))
    cases = load_cases()

    hits = 0
    kw_found = kw_total = 0
    for case in cases:
        passages = pipe.retrieve(case["question"])
        sources = {p["source"] for p in passages}
        blob = " ".join(p["text"].lower() for p in passages)
        if case["expected_source"] in sources:
            hits += 1
        for kw in case.get("must_contain", []):
            kw_total += 1
            kw_found += int(kw.lower() in blob)
        status = "OK " if case["expected_source"] in sources else "MISS"
        print(f"[{status}] {case['question']}")

    return {
        "cases": len(cases),
        "source_hit@k": round(hits / len(cases), 3),
        "keyword_recall": round(kw_found / max(kw_total, 1), 3),
    }


if __name__ == "__main__":
    metrics = evaluate()
    print("\nMetrics:", json.dumps(metrics, indent=2))
