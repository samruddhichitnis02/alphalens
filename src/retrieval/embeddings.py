from typing import List
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("retrieval.embeddings")

class EmbeddingModel:
    """Manages local sentence-transformers embedding generation."""
    _instance = None

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = EmbeddingModel()
        return cls._instance

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        self._load_model()
        emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return emb[0].tolist()

