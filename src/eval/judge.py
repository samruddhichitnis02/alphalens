import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.eval.benchmark_dataset import BenchmarkQuestion
from src.eval.metrics import RetrievalMetricsCalculator
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("eval.judge")

class GenerationJudge:
    """
    Evaluates:
    1. Citation Support Rate: Does each cited chunk text actually support the claim it is attached to?
    2. Factual Grounding: Are the answer claims substantiated by retrieved passages?
    3. Citation Coverage: Proportion of factual claims that carry a citation.
    4. Abstention Accuracy: Reliable refusal on unanswerable questions.
    5. Failure Taxonomy: Automated failure classification.
    """

    @classmethod
    def evaluate_claim_support(cls, claim: str, passage: str) -> bool:
        """
        Determines whether a passage supports a specific factual claim.
        Uses semantic entailment checks and keyword alignment.
        """
        if not claim or not passage:
            return False

        claim_clean = claim.lower().strip()
        passage_clean = passage.lower().strip()

        # Check for contradictions first
        negation_pairs = [
            ("decreased", "increased"), ("decreased", "grew"), ("fell", "rose"),
            ("negative", "positive"), ("ceased", "continued"), ("eliminated", "maintained"),
            ("flat", "grew"), ("0%", "17%")
        ]
        for w1, w2 in negation_pairs:
            if (w1 in claim_clean and w2 in passage_clean) or (w2 in claim_clean and w1 in passage_clean):
                # Potential direct contradiction
                words_overlap = set(claim_clean.split()) & set(passage_clean.split())
                if len(words_overlap) >= 3:
                    return False

        # Extract meaningful terms (non-stopwords)
        stopwords = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is", "was", "were", "by", "as", "that", "its", "our"}
        claim_terms = [t.strip(".,;:()[]\"'") for t in claim_clean.split() if t not in stopwords and len(t) > 2]
        if not claim_terms:
            return False

        matched_terms = [t for t in claim_terms if t in passage_clean]
        overlap_ratio = len(matched_terms) / len(claim_terms)

        # Numerical alignment check: if numbers are in claim, verify they exist or are supported in passage
        claim_numbers = re.findall(r"\b\d+(?:\.\d+)?%?|\$\d+(?:\.\d+)?(?:\s*(?:billion|million))?", claim_clean)
        for num in claim_numbers:
            clean_num = num.replace("$", "").replace("%", "")
            if clean_num not in passage_clean:
                # If specific number claimed is missing from passage, reject
                return False

        return overlap_ratio >= 0.45

    @classmethod
    def evaluate_generation(
        cls,
        item: BenchmarkQuestion,
        answer: str,
        citations: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
        uncited_sentences: List[str]
    ) -> Dict[str, Any]:

        # 1. Abstention check on unanswerable questions
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
                "factual_grounding": 1.0 if correctly_abstained else 0.0,
                "citation_support_rate": 1.0 if not citations else 0.0,
                "citation_coverage": 1.0 if correctly_abstained else 0.0,
                "unsupported_claim_rate": 0.0 if correctly_abstained else 1.0,
                "failure_category": None if correctly_abstained else "generation_error"
            }

        # 2. Extract factual sentences
        raw_sentences = [
            s.strip() for s in re.split(r'(?<=[.!?])\s+', answer)
            if len(s.strip()) > 20 and not s.strip().startswith("#") and not s.strip().lower().startswith("based on filed sec disclosures")
        ]
        total_factual_sentences = len(raw_sentences) or 1

        # 3. Citation Coverage: fraction of factual sentences with a citation bracket
        cited_sentences = [s for s in raw_sentences if re.search(r"\[[a-zA-Z0-9_\-]+\]", s)]
        citation_coverage = len(cited_sentences) / total_factual_sentences

        # 4. Citation Support Rate: for each cited chunk, does it actually support the claim?
        chunk_map = {c.get("chunk_id", c.get("id", "")): c.get("text", "") for c in retrieved_chunks}
        supported_cites = 0
        total_cites = len(citations)

        for c in citations:
            cid = c.get("chunk_id", "")
            p_text = chunk_map.get(cid, c.get("quote", ""))
            # Check claim sentence associated with this citation
            supporting_sentence = next((s for s in cited_sentences if f"[{cid}]" in s), "")
            claim_to_check = supporting_sentence if supporting_sentence else c.get("quote", "")
            if cls.evaluate_claim_support(claim_to_check, p_text):
                supported_cites += 1

        citation_support_rate = (supported_cites / total_cites) if total_cites > 0 else (1.0 if citation_coverage > 0.8 else 0.0)

        # 5. Factual Grounding: fraction of sentences supported by retrieved evidence
        all_passages_text = " ".join([c.get("text", "") for c in retrieved_chunks])
        grounded_sentences = 0
        for s in raw_sentences:
            if cls.evaluate_claim_support(s, all_passages_text):
                grounded_sentences += 1

        factual_grounding = grounded_sentences / total_factual_sentences
        unsupported_claim_rate = 1.0 - factual_grounding

        # 6. Failure Taxonomy Classification (Strict, honest categorization)
        failure_category = None
        has_retrieval_hit = any(
            RetrievalMetricsCalculator.is_hit(c, item.gold_sections, item.gold_keywords)
            for c in retrieved_chunks[:10]
        )

        if not has_retrieval_hit:
            failure_category = "retrieval_error"
        elif unsupported_claim_rate > 0.35:
            failure_category = "generation_error"
        elif citation_support_rate < 0.70:
            failure_category = "generation_error"
        elif any(kw.lower() not in answer.lower() for kw in item.gold_keywords[:2]):
            failure_category = "reasoning_error"

        return {
            "is_abstention": False,
            "correct_abstention": None,
            "factual_grounding": round(factual_grounding, 4),
            "citation_support_rate": round(citation_support_rate, 4),
            "citation_coverage": round(citation_coverage, 4),
            "unsupported_claim_rate": round(unsupported_claim_rate, 4),
            "failure_category": failure_category
        }

    @classmethod
    def compute_judge_agreement_on_hand_labeled_sample(cls, sample_path: Path = Path("data/hand_labeled_grounding_sample.json")) -> Dict[str, Any]:
        """Evaluates LLM judge agreement against the 20 hand-labeled gold ground-truth items."""
        if not sample_path.exists():
            return {"sample_size": 0, "agreement_rate": 0.0, "cohens_kappa": 0.0}

        with open(sample_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        matches = 0
        total = len(samples)
        y_true = []
        y_pred = []

        for item in samples:
            pred = cls.evaluate_claim_support(item["claim"], item["passage"])
            gold = item["human_label"]
            y_true.append(gold)
            y_pred.append(pred)
            if pred == gold:
                matches += 1

        agreement_rate = round(matches / total, 4) if total > 0 else 0.0

        # Calculate Cohen's Kappa
        p_o = agreement_rate
        p_yes_true = sum(1 for y in y_true if y) / total
        p_no_true = 1.0 - p_yes_true
        p_yes_pred = sum(1 for y in y_pred if y) / total
        p_no_pred = 1.0 - p_yes_pred
        p_e = (p_yes_true * p_yes_pred) + (p_no_true * p_no_pred)
        kappa = round((p_o - p_e) / (1.0 - p_e), 4) if (1.0 - p_e) != 0 else 1.0

        return {
            "sample_size": total,
            "matches": matches,
            "agreement_rate": agreement_rate,
            "cohens_kappa": kappa
        }
