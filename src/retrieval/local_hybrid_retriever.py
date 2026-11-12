import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np
from src.retrieval.base import BaseRetriever, RetrievalResult, RetrievalFilter
from src.retrieval.embeddings import EmbeddingModel
from src.utils.logger import get_logger

logger = get_logger("retrieval.local")

class SimpleBM25:
    """In-memory BM25 index for zero-dependency local evaluation and unit testing."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens = []
        self.doc_freqs = {}
        self.doc_tokens = []

    def fit(self, documents: List[str]):
        self.corpus_size = len(documents)
        self.doc_tokens = [self._tokenize(doc) for doc in documents]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(self.corpus_size, 1)

        self.doc_freqs = Counter()
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.doc_freqs[t] += 1

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def get_scores(self, query: str) -> List[float]:
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size

        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
            df = self.doc_freqs[token]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            for idx in range(self.corpus_size):
                tf = self.doc_tokens[idx].count(token)
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[idx] / max(self.avg_doc_len, 1e-6)))
                scores[idx] += idf * (num / den)

        return scores


class LocalHybridRetriever(BaseRetriever):
    """
    In-memory hybrid retriever implementing BM25, Semantic cosine similarity, and RRF.
    Enables instant offline testing and evaluation without needing an external OpenSearch cluster.
    """
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[SimpleBM25] = None
        self.embedding_model = EmbeddingModel.get_instance()

    async def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        # Fit BM25
        self.bm25 = SimpleBM25()
        self.bm25.fit(texts)

        # Generate embeddings
        raw_embs = self.embedding_model.encode(texts)
        self.embeddings = np.array(raw_embs, dtype=np.float32)
        
        logger.info(f"Indexed {len(chunks)} chunks in LocalHybridRetriever.")
        return len(chunks)

    def _matches_filters(self, chunk: Dict[str, Any], filters: Optional[RetrievalFilter]) -> bool:
        if not filters:
            return True
        meta = chunk.get("extra_metadata", {})
        ticker = meta.get("ticker", "").upper()
        form_type = meta.get("form_type", "").upper()
        sec_name = chunk.get("section_name", "")
        fdate = meta.get("filing_date", "")

        if filters.tickers and ticker not in [t.upper() for t in filters.tickers]:
            return False
        if filters.form_types and form_type not in [f.upper() for f in filters.form_types]:
            return False
        if filters.sections and sec_name not in filters.sections:
            return False
        if filters.start_date and fdate < filters.start_date:
            return False
        if filters.end_date and fdate > filters.end_date:
            return False
        return True

    async def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[RetrievalFilter] = None,
        method: str = "hybrid"
    ) -> List[RetrievalResult]:
        if not self.chunks:
            return []

        # Find eligible indices by metadata filter
        eligible_indices = [
            i for i, chunk in enumerate(self.chunks)
            if self._matches_filters(chunk, filters)
        ]
        if not eligible_indices:
            return []

        if method == "bm25":
            return self._search_bm25(query, k, eligible_indices)
        elif method == "semantic":
            return self._search_semantic(query, k, eligible_indices)
        else:
            bm25_res = self._search_bm25(query, k * 2, eligible_indices)
            sem_res = self._search_semantic(query, k * 2, eligible_indices)
            return self._reciprocal_rank_fusion(bm25_res, sem_res, k=k)

    def _search_bm25(self, query: str, k: int, eligible_indices: List[int]) -> List[RetrievalResult]:
        scores = self.bm25.get_scores(query)
        scored_pairs = [(i, scores[i]) for i in eligible_indices]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_pairs[:k]:
            c = self.chunks[idx]
            meta = c.get("extra_metadata", {})
            results.append(RetrievalResult(
                chunk_id=c["id"],
                filing_id=c.get("filing_id", ""),
                ticker=meta.get("ticker", ""),
                form_type=meta.get("form_type", ""),
                section_name=c.get("section_name", ""),
                filing_date=meta.get("filing_date", ""),
                period_of_report=meta.get("period_of_report"),
                source_url=meta.get("source_url", ""),
                text=c["text"],
                score=float(score),
                retrieval_method="bm25"
            ))
        return results

    def _search_semantic(self, query: str, k: int, eligible_indices: List[int]) -> List[RetrievalResult]:
        q_emb = np.array(self.embedding_model.encode_query(query), dtype=np.float32)
        # Cosine similarity (already normalized)
        sub_embs = self.embeddings[eligible_indices]
        similarities = np.dot(sub_embs, q_emb)

        scored_pairs = list(zip(eligible_indices, similarities))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_pairs[:k]:
            c = self.chunks[idx]
            meta = c.get("extra_metadata", {})
            results.append(RetrievalResult(
                chunk_id=c["id"],
                filing_id=c.get("filing_id", ""),
                ticker=meta.get("ticker", ""),
                form_type=meta.get("form_type", ""),
                section_name=c.get("section_name", ""),
                filing_date=meta.get("filing_date", ""),
                period_of_report=meta.get("period_of_report"),
                source_url=meta.get("source_url", ""),
                text=c["text"],
                score=float(score),
                retrieval_method="semantic"
            ))
        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_res: List[RetrievalResult],
        sem_res: List[RetrievalResult],
        k: int = 10,
        c: int = 60
    ) -> List[RetrievalResult]:
        rrf_scores: Dict[str, float] = {}
        items_by_id: Dict[str, RetrievalResult] = {}

        for rank, item in enumerate(bm25_res):
            cid = item.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (c + rank + 1))
            items_by_id[cid] = item

        for rank, item in enumerate(sem_res):
            cid = item.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (c + rank + 1))
            items_by_id[cid] = item

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        
        final_results = []
        for cid in sorted_ids:
            res = items_by_id[cid].model_copy()
            res.score = rrf_scores[cid]
            res.retrieval_method = "hybrid_rrf"
            final_results.append(res)

        return final_results

