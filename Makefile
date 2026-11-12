.PHONY: up down test ingest eval review run clean

up:
	docker compose up -d

down:
	docker compose down

test:
	pytest tests/ -v

ingest:
	python -c "import asyncio; from src.db.session import init_db, AsyncSessionLocal; from src.api.routes import run_ticker_ingestion; asyncio.run(init_db()); [asyncio.run(run_ticker_ingestion(t, AsyncSessionLocal())) for t in ['AAPL', 'MSFT', 'NVDA']]"

eval:
	python -m src.eval.runner

review:
	python -m src.eval.review_cli

run:
	uvicorn src.api.main:app --reload --port 8000

clean:
	rm -rf data/raw_filings data/alphalens.db .pytest_cache

