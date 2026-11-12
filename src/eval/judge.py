import re
from typing import Dict, Any, List
from src.eval.benchmark_dataset import BenchmarkQuestion
from src.eval.metrics import RetrievalMetricsCalculator
from src.utils.logger import get_logger

logger = get_logger("eval.judge")

class GenerationJudge:
    """Evaluates factual grounding, citation correctness, abstention, and assigns failure taxonomy."""

    @classmethod
    def evaluate_generation(
        cls,
        item: BenchmarkQuestion,
        answer: str,
        citations: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
        uncited_sentences: List[str]
    ) -> Dict[str, Any]:
        
        # 1. Check Abstention on Unanswerable Questions
        if not item.is_answerable:
            lower_ans = answer.lower()
            correctly_abstained = any(phrase in lower_ans for phrase in [
                "insufficient evidence",
                "not disclosed",
                "not available",
                "no information",
                "not found in filed sec disclosures"
            ])
            return {
                "is_abstention": True,
                "correct_abstention": correctly_abstained,
                "grounding_score": 1.0 if correctly_abstained else 0.0,
                "citation_correctness": 1.0 if not citations else 0.0,
                "unsupported_claim_rate": 0.0 if correctly_abstained else 1.0,
                "failure_category": None if correctly_abstained else "generation_error"
            }

        # 2. Factual Grounding & Unsupported Claim Rate
        # Ignore introductory boilerplate lines like "Based on filed SEC disclosures:"
        filtered_sentences = [
            s.strip() for s in re.split(r'(?<=[.!?])\s+', answer)
            if len(s.strip()) > 20 and not s.strip().startswith("#") and not s.strip().lower().startswith("based on filed sec disclosures")
        ]
        total_sentences = len(filtered_sentences) or 1
        
        # Filter uncited sentences ignoring introductory boilerplate
        real_uncited = [
            s for s in uncited_sentences
            if not s.strip().lower().startswith("based on filed sec disclosures")
        ]
        unsupported_count = len(real_uncited)
        grounding_score = max(0.0, 1.0 - (unsupported_count / total_sentences))
        unsupported_claim_rate = unsupported_count / total_sentences

        # 3. Citation Correctness
        valid_cites = sum(1 for c in citations if c.get("is_valid", False))
        total_cites = len(citations)
        citation_correctness = (valid_cites / total_cites) if total_cites > 0 else (1.0 if grounding_score > 0.8 else 0.0)

        # 4. Failure Taxonomy Classification
        failure_category = None
        has_retrieval_hit = any(
            RetrievalMetricsCalculator.is_hit(c, item.gold_sections, item.gold_keywords)
            for c in retrieved_chunks[:10]
        )

        if not has_retrieval_hit:
            failure_category = "retrieval_error"
        elif unsupported_claim_rate > 0.3 or citation_correctness < 0.7:
            failure_category = "generation_error"
        elif any(kw.lower() not in answer.lower() for kw in item.gold_keywords[:2]):
            failure_category = "reasoning_error"

        return {
            "is_abstention": False,
            "correct_abstention": None,
            "grounding_score": round(grounding_score, 3),
            "citation_correctness": round(citation_correctness, 3),
            "unsupported_claim_rate": round(unsupported_claim_rate, 3),
            "failure_category": failure_category
        }
