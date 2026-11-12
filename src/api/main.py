from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.db.session import init_db
from src.api.routes import router
from src.utils.logger import get_logger

logger = get_logger("api.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing AlphaLens database tables...")
    await init_db()
    logger.info("AlphaLens system startup complete.")
    yield
    logger.info("AlphaLens system shutdown.")

app = FastAPI(
    title="AlphaLens - Investment Research Intelligence API",
    description="Production-grade SEC EDGAR Intelligence System with Section Parsing, Event/Risk Extraction, Hybrid Retrieval & LangGraph Citation Workflow.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "alphalens"}

