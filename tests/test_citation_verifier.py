import pytest
from src.extraction.extractor import ExtractorService
from src.workflow.graph import ResearchWorkflow
from src.retrieval.local_hybrid_retriever import LocalHybridRetriever

def test_hallucination_guard_verbatim_check():
    extractor = ExtractorService()
    source_chunk = "The company experienced significant supply chain constraints in its semiconductor procurement during Q3."

    # Exact verbatim quote -> Accepted
    exact_quote = "supply chain constraints in its semiconductor procurement"
    assert extractor.verify_quote_in_chunk(exact_quote, source_chunk) is True

    # Paraphrased / Hallucinated quote -> Rejected!
    hallucinated_quote = "The firm suffered severe shortages of microchips"
    assert extractor.verify_quote_in_chunk(hallucinated_quote, source_chunk) is False

    # Empty quote -> Rejected
    assert extractor.verify_quote_in_chunk("", source_chunk) is False

@pytest.mark.asyncio
async def test_citation_verification_node():
    retriever = LocalHybridRetriever()
    wf = ResearchWorkflow(retriever=retriever)

    state = {
        "final_answer": "Apple grew Services revenue [chunk-101]. Non-existent claim cited [fake-chunk-999]. Uncited sentence floating here.",
        "retrieved_chunks": [
            {
                "chunk_id": "chunk-101",
                "ticker": "AAPL",
                "form_type": "10-K",
                "section_name": "Item 7 - MD&A",
                "filing_date": "2023-10-27",
                "source_url": "https://sec.gov/apple",
                "text": "Services revenue grew 8% year-over-year."
            }
        ]
    }

    verified = await wf.node_verify_citations(state)
    
    valid_cites = verified["citations"]
    invalid_cites = verified["invalid_citations"]
    uncited = verified["uncited_sentences"]

    # Verify chunk-101 is valid
    assert len(valid_cites) == 1
    assert valid_cites[0]["chunk_id"] == "chunk-101"

    # Verify fake-chunk-999 was flagged as invalid
    assert "fake-chunk-999" in invalid_cites

    # Verify uncited sentence was flagged
    assert len(uncited) > 0

