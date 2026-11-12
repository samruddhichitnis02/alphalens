import pytest
from src.retrieval.local_hybrid_retriever import LocalHybridRetriever
from src.workflow.graph import ResearchWorkflow

@pytest.mark.asyncio
async def test_end_to_end_research_workflow():
    retriever = LocalHybridRetriever()
    test_chunks = [
        {
            "id": "c-001",
            "filing_id": "filing-aapl",
            "section_name": "Item 7 - MD&A",
            "text": "Apple total net sales were $383,285 million in fiscal 2023, with Services reaching an all-time record.",
            "extra_metadata": {
                "ticker": "AAPL",
                "form_type": "10-K",
                "filing_date": "2023-10-27",
                "source_url": "https://sec.gov/apple-10k"
            }
        },
        {
            "id": "c-002",
            "filing_id": "filing-msft",
            "section_name": "Item 7 - MD&A",
            "text": "Microsoft Cloud revenue grew 22% driven by Azure enterprise adoption and AI transformations.",
            "extra_metadata": {
                "ticker": "MSFT",
                "form_type": "10-K",
                "filing_date": "2023-07-28",
                "source_url": "https://sec.gov/msft-10k"
            }
        }
    ]
    await retriever.index_chunks(test_chunks)

    workflow = ResearchWorkflow(retriever=retriever)

    # 1. Factual Lookup Query
    res = await workflow.graph.ainvoke({
        "query": "What were Apple's net sales in fiscal 2023?",
        "retrieval_method": "hybrid"
    })

    assert res["question_type"] == "factual_lookup"
    assert "AAPL" in res["target_tickers"]
    assert len(res["retrieved_chunks"]) > 0
    assert len(res["citations"]) > 0
    assert res["citations"][0]["ticker"] == "AAPL"

    # 2. Unanswerable Query (Verification of correct abstention)
    unans_res = await workflow.graph.ainvoke({
        "query": "What is Apple's confidential internal profit forecast for 2035 on Mars?",
        "retrieval_method": "hybrid"
    })
    # Since Martian queries return no relevant chunks or trigger insufficient evidence
    assert "insufficient evidence" in unans_res["final_answer"].lower() or len(unans_res["retrieved_chunks"]) == 0

