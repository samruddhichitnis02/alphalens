import pytest
from src.edgar.chunker import SectionAwareChunker

def test_chunking_within_section_boundaries():
    chunker = SectionAwareChunker(target_tokens=50, overlap_tokens=10)
    
    sec1_text = "This is section one. " * 30
    sec2_text = "This is section two. " * 30
    
    sections = [
        {"section_name": "Item 1A - Risk Factors", "char_start": 0, "char_end": len(sec1_text), "raw_text": sec1_text},
        {"section_name": "Item 7 - MD&A", "char_start": len(sec1_text), "char_end": len(sec1_text) + len(sec2_text), "raw_text": sec2_text}
    ]

    chunks = chunker.chunk_filing_sections(sections, filing_metadata={"ticker": "AAPL"})
    
    # Assert chunks strictly adhere to their source section
    sec1_chunks = [c for c in chunks if c["section_name"] == "Item 1A - Risk Factors"]
    sec2_chunks = [c for c in chunks if c["section_name"] == "Item 7 - MD&A"]

    assert len(sec1_chunks) > 1
    assert len(sec2_chunks) > 1

    for c in sec1_chunks:
        assert "section one" in c["text"]
        assert "section two" not in c["text"] # Never crossed boundary!

    for c in sec2_chunks:
        assert "section two" in c["text"]
        assert "section one" not in c["text"] # Never crossed boundary!

def test_chunk_token_limits_and_traceability():
    target = 100
    chunker = SectionAwareChunker(target_tokens=target, overlap_tokens=20)
    long_text = "NVIDIA accelerated computing platforms enable generative AI training at scale. " * 40
    
    chunks = chunker.chunk_section(
        section_text=long_text,
        section_name="Item 1 - Business",
        section_global_start=500,
        metadata={"ticker": "NVDA", "form_type": "10-K"}
    )

    assert len(chunks) > 1
    for c in chunks:
        assert c["token_count"] <= target
        assert c["char_start"] >= 500
        assert c["char_end"] > c["char_start"]
        assert c["extra_metadata"]["ticker"] == "NVDA"
        assert c["section_name"] == "Item 1 - Business"

