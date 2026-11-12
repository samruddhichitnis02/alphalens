from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch, helpers
from src.config.settings import settings
from src.retrieval.base import BaseRetriever, RetrievalResult, RetrievalFilter
from src.retrieval.embeddings import EmbeddingModel
from src.utils.logger import get_logger

logger = get_logger("retrieval.opensearch")

class OpenSearchRetriever(BaseRetriever):
    """
    OpenSearch retriever supporting BM25, kNN dense vector search,
    and Hybrid search with Reciprocal Rank Fusion (RRF).
    """
    def __init__(self):
        auth = None
        if settings.OPENSEARCH_USERNAME and settings.OPENSEARCH_PASSWORD:
            auth = (settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD)
            
        self.client = OpenSearch(
            hosts=[settings.OPENSEARCH_HOST],
            http_auth=auth,
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False
        )
        self.index_name = settings.OPENSEARCH_INDEX
        self.embedding_model = EmbeddingModel.get_instance()

    def create_index_if_not_exists(self):
        """Creates OpenSearch index with kNN vector mappings."""
        if not self.client.indices.exists(index=self.index_name):
            body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                },
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "filing_id": {"type": "keyword"},
                        "ticker": {"type": "keyword"},
                        "form_type": {"type": "keyword"},
                        "section_name": {"type": "keyword"},
                        "filing_date": {"type": "date", "format": "yyyy-MM-dd||epoch_millis"},
                        "period_of_report": {"type": "keyword"},
                        "source_url": {"type": "keyword", "index": False},
                        "text": {"type": "text", "analyzer": "standard"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": settings.EMBEDDING_DIMENSION,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib"
                            }
                        }
                    }
                }
            }
            self.client.indices.create(index=self.index_name, body=body)
            logger.info(f"Created OpenSearch index '{self.index_name}' with kNN mappings.")

    async def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        self.create_index_if_not_exists()
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(texts)

        actions = []
        for i, c in enumerate(chunks):
            meta = c.get("extra_metadata", {})
            action = {
                "_index": self.index_name,
                "_id": c["id"],
                "_source": {
                    "chunk_id": c["id"],
                    "filing_id": c["filing_id"],
                    "ticker": meta.get("ticker", ""),
                    "form_type": meta.get("form_type", ""),
                    "section_name": c["section_name"],
                    "filing_date": meta.get("filing_date", "1970-01-01"),
                    "period_of_report": meta.get("period_of_report", ""),
                    "source_url": meta.get("source_url", ""),
                    "text": c["text"],
                    "embedding": embeddings[i]
                }
            }
            actions.append(action)

        success, _ = helpers.bulk(self.client, actions)
        self.client.indices.refresh(index=self.index_name)
        logger.info(f"Indexed {success} chunks into OpenSearch.")
        return success

    def _build_filter_clauses(self, filters: Optional[RetrievalFilter]) -> List[Dict[str, Any]]:
        if not filters:
            return []
        must_clauses = []
        if filters.tickers:
            must_clauses.append({"terms": {"ticker": [t.upper() for t in filters.tickers]}})
        if filters.form_types:
            must_clauses.append({"terms": {"form_type": [f.upper() for f in filters.form_types]}})
        if filters.sections:
            must_clauses.append({"terms": {"section_name": filters.sections}})
        if filters.start_date or filters.end_date:
            date_range = {}
            if filters.start_date:
                date_range["gte"] = filters.start_date
            if filters.end_date:
                date_range["lte"] = filters.end_date
            must_clauses.append({"range": {"filing_date": date_range}})
        return must_clauses

    async def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[RetrievalFilter] = None,
        method: str = "hybrid"
    ) -> List[RetrievalResult]:
        filter_clauses = self._build_filter_clauses(filters)

        if method == "bm25":
            return await self._search_bm25(query, k, filter_clauses)
        elif method == "semantic":
            return await self._search_semantic(query, k, filter_clauses)
        else: # Hybrid RRF
            bm25_res = await self._search_bm25(query, k * 2, filter_clauses)
            sem_res = await self._search_semantic(query, k * 2, filter_clauses)
            return self._reciprocal_rank_fusion(bm25_res, sem_res, k=k)

    async def _search_bm25(self, query: str, k: int, filter_clauses: List[Dict[str, Any]]) -> List[RetrievalResult]:
        query_body = {
            "size": k,
            "query": {
                "bool": {
                    "must": [{"match": {"text": query}}],
                    "filter": filter_clauses
                }
            }
        }
        res = self.client.search(index=self.index_name, body=query_body)
        return self._format_hits(res["hits"]["hits"], "bm25")

    async def _search_semantic(self, query: str, k: int, filter_clauses: List[Dict[str, Any]]) -> List[RetrievalResult]:
        query_vector = self.embedding_model.encode_query(query)
        knn_query = {
            "vector": query_vector,
            "k": k
        }
        if filter_clauses:
            knn_query["filter"] = {"bool": {"must": filter_clauses}}

        query_body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": knn_query
                }
            }
        }
        res = self.client.search(index=self.index_name, body=query_body)
        return self._format_hits(res["hits"]["hits"], "semantic")

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[RetrievalResult],
        semantic_results: List[RetrievalResult],
        k: int = 10,
        c: int = 60
    ) -> List[RetrievalResult]:
        """Combines BM25 and Semantic rank lists via Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        items_by_id: Dict[str, RetrievalResult] = {}

        for rank, item in enumerate(bm25_results):
            cid = item.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (c + rank + 1))
            items_by_id[cid] = item

        for rank, item in enumerate(semantic_results):
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

    def _format_hits(self, hits: List[Dict[str, Any]], method: str) -> List[RetrievalResult]:
        results = []
        for hit in hits:
            src = hit["_source"]
            results.append(RetrievalResult(
                chunk_id=src["chunk_id"],
                filing_id=src["filing_id"],
                ticker=src.get("ticker", ""),
                form_type=src.get("form_type", ""),
                section_name=src.get("section_name", ""),
                filing_date=src.get("filing_date", ""),
                period_of_report=src.get("period_of_report"),
                source_url=src.get("source_url", ""),
                text=src["text"],
                score=float(hit["_score"]),
                retrieval_method=method
            ))
        return results

