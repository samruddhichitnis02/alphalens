import pytest
from src.eval.metrics import RetrievalMetricsCalculator

def test_recall_at_k():
    retrieved = [
        {"section_name": "Item 1 - Business", "text": "General overview of company history"},
        {"section_name": "Item 1A - Risk Factors", "text": "Significant cybersecurity breach risks and threats"},
        {"section_name": "Item 7 - MD&A", "text": "Financial results discussion"}
    ]

    gold_sections = ["Item 1A - Risk Factors"]
    gold_keywords = ["cybersecurity"]

    # Hit is at position 1 (index 1), so Recall@1 is 0, Recall@2 is 1.0, Recall@5 is 1.0
    assert RetrievalMetricsCalculator.recall_at_k(retrieved, gold_sections, gold_keywords, k=1) == 0.0
    assert RetrievalMetricsCalculator.recall_at_k(retrieved, gold_sections, gold_keywords, k=2) == 1.0
    assert RetrievalMetricsCalculator.recall_at_k(retrieved, gold_sections, gold_keywords, k=5) == 1.0

def test_reciprocal_rank():
    # Hit at position 1 (rank 2 -> MRR 1/2 = 0.5)
    retrieved = [
        {"section_name": "Item 1 - Business", "text": "General text"},
        {"section_name": "Item 1A - Risk Factors", "text": "Target hit text"}
    ]
    rr = RetrievalMetricsCalculator.reciprocal_rank(retrieved, ["Item 1A - Risk Factors"], ["target"])
    assert rr == 0.5

    # Hit at position 0 (rank 1 -> MRR 1.0)
    retrieved_top = [
        {"section_name": "Item 1A - Risk Factors", "text": "Target hit text"}
    ]
    assert RetrievalMetricsCalculator.reciprocal_rank(retrieved_top, ["Item 1A - Risk Factors"], ["target"]) == 1.0

def test_ndcg_at_k_hand_computed_example():
    """
    Hand-computed test:
    Top 4 retrieved items with relevance [0, 1, 1, 0].
    Number of relevant items = 2.
    DCG@4  = 0/log2(2) + 1/log2(3) + 1/log2(4) + 0/log2(5)
           = 0 + 0.63093 + 0.50000 + 0 = 1.13093
    IDCG@4 = 1/log2(2) + 1/log2(3) = 1.0 + 0.63093 = 1.63093
    nDCG@4 = 1.13093 / 1.63093 = 0.6934 (strictly bounded in [0, 1])
    """
    retrieved = [
        {"section_name": "Item 1 - Business", "text": "Irrelevant general info"},
        {"section_name": "Item 1A - Risk Factors", "text": "Cybersecurity attack threat"},
        {"section_name": "Item 1A - Risk Factors", "text": "Ransomware supply chain breach"},
        {"section_name": "Item 8 - Financials", "text": "Auditor report note"}
    ]
    gold_sections = ["Item 1A - Risk Factors"]
    gold_keywords = ["cybersecurity", "ransomware"]

    ndcg = RetrievalMetricsCalculator.ndcg_at_k(
        retrieved=retrieved,
        gold_sections=gold_sections,
        gold_keywords=gold_keywords,
        k=4,
        total_relevant=2
    )

    assert 0.0 <= ndcg <= 1.0
    assert abs(ndcg - 0.6934) < 1e-3

def test_ndcg_at_k_perfect_ranking():
    retrieved = [
        {"section_name": "Item 1A - Risk Factors", "text": "Hit at rank 1"},
        {"section_name": "Item 1A - Risk Factors", "text": "Hit at rank 2"},
    ]
    ndcg = RetrievalMetricsCalculator.ndcg_at_k(retrieved, ["Item 1A - Risk Factors"], ["hit"], k=5)
    assert ndcg == 1.0

def test_ndcg_at_k_zero_hits():
    retrieved = [
        {"section_name": "Item 1 - Business", "text": "No hits here"}
    ]
    ndcg = RetrievalMetricsCalculator.ndcg_at_k(retrieved, ["Item 1A - Risk Factors"], ["target"], k=5)
    assert ndcg == 0.0

