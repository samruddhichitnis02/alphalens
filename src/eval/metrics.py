import math
from typing import List, Dict, Any

class RetrievalMetricsCalculator:
    """Calculates Recall@k, MRR, and nDCG@k for retrieval evaluation."""

    @staticmethod
    def is_hit(chunk: Dict[str, Any], gold_sections: List[str], gold_keywords: List[str]) -> bool:
        """Determines if a retrieved chunk matches ground truth evidence."""
        if not gold_sections and not gold_keywords:
            return False
        
        sec_name = chunk.get("section_name", "")
        # Section match
        section_match = any(gs.lower() in sec_name.lower() or sec_name.lower() in gs.lower() for gs in gold_sections)
        
        # Keyword match
        text = chunk.get("text", "").lower()
        keyword_hits = sum(1 for kw in gold_keywords if kw.lower() in text)
        keyword_match = keyword_hits >= max(1, len(gold_keywords) // 3)

        return section_match or keyword_match

    @classmethod
    def recall_at_k(cls, retrieved: List[Dict[str, Any]], gold_sections: List[str], gold_keywords: List[str], k: int) -> float:
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        for chunk in top_k:
            if cls.is_hit(chunk, gold_sections, gold_keywords):
                return 1.0
        return 0.0

    @classmethod
    def reciprocal_rank(cls, retrieved: List[Dict[str, Any]], gold_sections: List[str], gold_keywords: List[str]) -> float:
        for rank, chunk in enumerate(retrieved):
            if cls.is_hit(chunk, gold_sections, gold_keywords):
                return 1.0 / (rank + 1)
        return 0.0

    @classmethod
    def ndcg_at_k(
        cls,
        retrieved: List[Dict[str, Any]],
        gold_sections: List[str],
        gold_keywords: List[str],
        k: int = 10,
        total_relevant: int = None
    ) -> float:
        """
        Calculates Normalized Discounted Cumulative Gain (nDCG@k).
        Strictly bounded in [0.0, 1.0].
        IDCG is computed from the total number of relevant documents that could be ranked in top-k.
        """
        top_k = retrieved[:k]
        if not top_k:
            return 0.0

        relevances = [
            1.0 if cls.is_hit(chunk, gold_sections, gold_keywords) else 0.0
            for chunk in top_k
        ]

        dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))
        if dcg == 0.0:
            return 0.0

        # Number of relevant items that could ideally occupy the top ranks
        hits_found = int(sum(relevances))
        effective_relevant = total_relevant if (total_relevant is not None and total_relevant > 0) else max(hits_found, 1)
        ideal_hits = min(effective_relevant, k)

        idcg = sum(1.0 / math.log2(j + 2) for j in range(ideal_hits))
        if idcg <= 0.0:
            return 0.0

        score = dcg / idcg
        return round(float(min(max(score, 0.0), 1.0)), 4)

