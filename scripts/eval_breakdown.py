"""Break the benchmark results down by question type, for the README charts.

Reuses the exact corpus, retriever, workflow and judge from `src.eval.runner`,
so the totals line up with `data/eval_results.json`. Writes
`data/eval_breakdown.json`; does not touch the main eval outputs.

Usage:  python scripts/eval_breakdown.py
"""
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.eval.benchmark_dataset import load_or_create_benchmark  # noqa: E402
from src.eval.judge import GenerationJudge  # noqa: E402
from src.eval.metrics import RetrievalMetricsCalculator as M  # noqa: E402
from src.eval.runner import generate_hardened_evaluation_corpus  # noqa: E402
from src.retrieval.base import RetrievalFilter  # noqa: E402
from src.retrieval.local_hybrid_retriever import LocalHybridRetriever  # noqa: E402
from src.workflow.graph import ResearchWorkflow  # noqa: E402


async def main():
    questions = load_or_create_benchmark(ROOT / "data" / "benchmark_questions.json")
    retriever = LocalHybridRetriever()
    await retriever.index_chunks(generate_hardened_evaluation_corpus(questions))
    workflow = ResearchWorkflow(retriever=retriever)

    by_type = defaultdict(lambda: {"n": 0, "recall5": [], "mrr": [], "grounding": [], "coverage": [],
                                   "abstained_ok": [], "failures": defaultdict(int)})
    for q in questions:
        row = by_type[q.question_type]
        row["n"] += 1
        if q.is_answerable:
            hits = [h.model_dump() for h in await retriever.search(
                q.question, k=10, filters=RetrievalFilter(tickers=q.target_tickers), method="hybrid")]
            row["recall5"].append(M.recall_at_k(hits, q.gold_sections, q.gold_keywords, 5))
            row["mrr"].append(M.reciprocal_rank(hits, q.gold_sections, q.gold_keywords))

        res = await workflow.graph.ainvoke({"query": q.question, "retrieval_method": "hybrid",
                                            "retrieval_retry_count": 0})
        ev = GenerationJudge.evaluate_generation(
            item=q, answer=res.get("final_answer", ""), citations=res.get("citations", []),
            retrieved_chunks=res.get("retrieved_chunks", []), uncited_sentences=res.get("uncited_sentences", []))
        if ev["is_abstention"]:
            row["abstained_ok"].append(1.0 if ev["correct_abstention"] else 0.0)
        else:
            row["grounding"].append(ev["factual_grounding"])
            row["coverage"].append(ev["citation_coverage"])
        row["failures"][ev["failure_category"] or "none"] += 1

    def mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    out = {"by_question_type": {
        t: {"n": r["n"], "Recall@5": mean(r["recall5"]), "MRR": mean(r["mrr"]),
            "Factual Grounding": mean(r["grounding"]), "Citation Coverage": mean(r["coverage"]),
            "Correct Abstention": mean(r["abstained_ok"]), "failures": dict(r["failures"])}
        for t, r in by_type.items()}}

    # Judge vs human on the hand-labeled sample, as a 2x2 table.
    sample = json.loads((ROOT / "data" / "hand_labeled_grounding_sample.json").read_text(encoding="utf-8"))
    cm = {"human_yes_judge_yes": 0, "human_yes_judge_no": 0, "human_no_judge_yes": 0, "human_no_judge_no": 0}
    for item in sample:
        judge = GenerationJudge.evaluate_claim_support(item["claim"], item["passage"])
        key = f"human_{'yes' if item['human_label'] else 'no'}_judge_{'yes' if judge else 'no'}"
        cm[key] += 1
    out["judge_confusion"] = cm

    path = ROOT / "data" / "eval_breakdown.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
