import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # SEC EDGAR Config
    SEC_USER_AGENT: str = "AlphaLensResearch lab@alphalens.ai"
    SEC_MAX_REQUESTS_PER_SECOND: float = 8.0
    DATA_CACHE_DIR: Path = BASE_DIR / "data" / "raw_filings"

    # Database
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/data/alphalens.db"

    # OpenSearch
    OPENSEARCH_HOST: str = "http://localhost:9200"
    OPENSEARCH_INDEX: str = "alphalens-filing-chunks"
    OPENSEARCH_USERNAME: Optional[str] = None
    OPENSEARCH_PASSWORD: Optional[str] = None

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # LLM Provider
    LLM_PROVIDER: str = "claude"  # "claude", "gemini", "ollama", "mock"
    ANTHROPIC_API_KEY: Optional[str] = None
    CLAUDE_EXTRACTION_MODEL: str = "claude-3-5-haiku-latest"
    CLAUDE_RESEARCH_MODEL: str = "claude-3-5-sonnet-latest"

    # Free / Local LLM alternatives
    GEMINI_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Chunking
    DEFAULT_CHUNK_SIZE: int = 400
    DEFAULT_CHUNK_OVERLAP: int = 50

    # Extraction Concurrency & Resilience
    EXTRACTION_SEMAPHORE_LIMIT: int = 5
    MAX_RETRIES: int = 3
    LOG_LEVEL: str = "INFO"

    @property
    def companies_config_path(self) -> Path:
        return BASE_DIR / "config" / "companies.yaml"

    def load_companies_config(self) -> dict:
        if not self.companies_config_path.exists():
            return {"sectors": [], "coverage": {}}
        with open(self.companies_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

settings = Settings()

