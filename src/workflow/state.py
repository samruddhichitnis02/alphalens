from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from src.retrieval.base import RetrievalResult, RetrievalFilter

class Citation(BaseModel):
    chunk_id: str
    ticker: str
    form_type: str
    section_name: str
    filing_date: str
    source_url: str
    quote: str
    is_valid: bool = True

class ResearchState(TypedDict):
    # Inputs
    query: str
    retrieval_method: str
    
    # Query Understanding Node outputs
    question_type: str  # factual_lookup, cross_period_comparison, cross_company_comparison, risk_summary, insight_generation
    target_tickers: List[str]
    time_periods: List[str]
    retrieval_filter: Dict[str, Any]
    sub_queries: List[str]
    
    # Retrieval Node outputs
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_retry_count: int
    evidence_sufficient: bool
    
    # Generation Node outputs
    raw_answer: str
    final_answer: str
    insights: Optional[str]
    
    # Citation Verification Node outputs
    citations: List[Dict[str, Any]]
    uncited_sentences: List[str]
    invalid_citations: List[str]
    
    # Observability
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    estimated_cost_usd: float

