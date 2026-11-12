from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class IngestResponse(BaseModel):
    ticker: str
    cik: str
    company_name: str
    filings_processed: int
    sections_created: int
    chunks_created: int
    extracted_items_count: int
    status: str

class SectionResponse(BaseModel):
    id: str
    filing_id: str
    section_name: str
    section_title: Optional[str] = None
    char_start: int
    char_end: int
    text_length: int

class FilingResponse(BaseModel):
    id: str
    ticker: str
    form_type: str
    accession_number: str
    filing_date: str
    period_of_report: Optional[str] = None
    source_url: str
    sections_count: int = 0
    chunks_count: int = 0

class CitationOutput(BaseModel):
    chunk_id: str
    ticker: str
    form_type: str
    section_name: str
    filing_date: str
    source_url: str
    quote: str
    is_valid: bool = True

class AskRequest(BaseModel):
    query: str = Field(..., description="Analyst question or research query.")
    retrieval_method: str = Field("hybrid", description="'bm25' | 'semantic' | 'hybrid'")
    top_k: int = Field(5, description="Number of passages to consider.")

class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationOutput]
    uncited_sentences: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    question_type: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float

