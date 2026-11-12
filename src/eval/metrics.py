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
    def ndcg_at_k(cls, retrieved: List[Dict[str, Any]], gold_sections: List[str], gold_keywords: List[str], k: int) -> float:
        top_k = retrieved[:k]
        dcg = 0.0
        for i, chunk in enumerate(top_k):
            rel = 1.0 if cls.is_hit(chunk, gold_sections, gold_keywords) else 0.0
            dcg += rel / math.log2(i + 2)
        
        # Ideal DCG for binary relevance with 1 hit at position 0
        idcg = 1.0 / math.log2(2)
        return dcg / idcg if idcg > 0 else 0.0

