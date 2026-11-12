from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class RetrievalResult(BaseModel):
    chunk_id: str
    filing_id: str
    ticker: str
    form_type: str
    section_name: str
    filing_date: str
    period_of_report: Optional[str] = None
    source_url: str
    text: str
    score: float
    retrieval_method: str = "hybrid"

class RetrievalFilter(BaseModel):
    tickers: Optional[List[str]] = None
    form_types: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    sections: Optional[List[str]] = None

class BaseRetriever(ABC):
    """Unified interface for BM25, Semantic kNN, and Hybrid RRF retrievers."""

    @abstractmethod
    async def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Indexes chunks with metadata and embeddings."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[RetrievalFilter] = None,
        method: str = "hybrid"  # "bm25", "semantic", "hybrid"
    ) -> List[RetrievalResult]:
        """Performs search using the requested retrieval strategy and metadata filters."""
        pass

