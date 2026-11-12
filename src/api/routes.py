import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.db.session import get_db
from src.db.models import Company, Filing, Section, Chunk, ExtractedItem
from src.config.settings import settings
from src.edgar.client import EdgarClient
from src.edgar.parser import EdgarParser
from src.edgar.chunker import SectionAwareChunker
from src.extraction.extractor import ExtractorService
from src.retrieval.local_hybrid_retriever import LocalHybridRetriever
from src.retrieval.base import BaseRetriever
from src.workflow.graph import ResearchWorkflow
from src.api.schemas import IngestResponse, FilingResponse, SectionResponse, AskRequest, AskResponse, CitationOutput
from src.utils.logger import get_logger

logger = get_logger("api.routes")
router = APIRouter()

# Global in-process retriever instance (can be swapped with OpenSearchRetriever if OpenSearch is up)
active_retriever: BaseRetriever = LocalHybridRetriever()
edgar_client = EdgarClient()
edgar_parser = EdgarParser()
chunker = SectionAwareChunker()
extractor = ExtractorService()
workflow = ResearchWorkflow(retriever=active_retriever)

async def run_ticker_ingestion(ticker: str, db: AsyncSession) -> IngestResponse:
    cfg = settings.load_companies_config()
    company_info = None
    for s in cfg.get("sectors", []):
        for c in s.get("companies", []):
            if c["ticker"].upper() == ticker.upper():
                company_info = c
                sector_name = s.get("name")
                break
        if company_info:
            break

    if not company_info:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found in companies.yaml config")

    cik = company_info["cik"]
    name = company_info["name"]

    # 1. Ensure Company record
    stmt = select(Company).where(Company.ticker == ticker.upper())
    res = await db.execute(stmt)
    company = res.scalar_one_or_none()
    if not company:
        company = Company(
            ticker=ticker.upper(),
            name=name,
            cik=cik,
            sector=sector_name
        )
        db.add(company)
        await db.flush()

    # 2. Fetch submissions from SEC
    submissions = await edgar_client.get_company_submissions(cik)
    coverage = cfg.get("coverage", {})
    filing_items = edgar_client.filter_recent_filings(
        submissions,
        years_10k=coverage.get("years_10k", 3),
        quarters_10q=coverage.get("quarters_10q", 6),
        months_8k=coverage.get("months_8k", 12)
    )

    total_sections = 0
    total_chunks = 0
    total_extractions = 0
    chunks_to_index = []

    for f_meta in filing_items:
        # Check if filing already exists (idempotency)
        stmt = select(Filing).where(Filing.accession_number == f_meta["accession_number"])
        res = await db.execute(stmt)
        existing_filing = res.scalar_one_or_none()
        if existing_filing:
            logger.info(f"Filing {f_meta['accession_number']} already ingested. Skipping.")
            continue

        # Download/cache
        cached_file = await edgar_client.download_filing_document(f_meta)
        with open(cached_file, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        # Parse sections
        clean_text = edgar_parser.html_to_text(html_content)
        parsed_sections = edgar_parser.parse_filing_sections(clean_text, f_meta["form_type"])

        # Create Filing model
        filing_id = str(uuid.uuid4())
        filing = Filing(
            id=filing_id,
            company_id=company.id,
            accession_number=f_meta["accession_number"],
            form_type=f_meta["form_type"],
            filing_date=f_meta["filing_date"],
            period_of_report=f_meta.get("period_of_report"),
            source_url=f_meta["source_url"],
            local_cache_path=str(cached_file)
        )
        db.add(filing)
        await db.flush()

        for sec in parsed_sections:
            sec_id = str(uuid.uuid4())
            db_section = Section(
                id=sec_id,
                filing_id=filing_id,
                section_name=sec["section_name"],
                section_title=sec.get("section_title"),
                char_start=sec["char_start"],
                char_end=sec["char_end"],
                raw_text=sec["raw_text"]
            )
            db.add(db_section)
            await db.flush()
            total_sections += 1

            # Chunk within section
            sec_chunks = chunker.chunk_section(
                section_text=sec["raw_text"],
                section_name=sec["section_name"],
                section_global_start=sec["char_start"],
                metadata={
                    "ticker": ticker.upper(),
                    "form_type": f_meta["form_type"],
                    "filing_date": f_meta["filing_date"],
                    "period_of_report": f_meta.get("period_of_report"),
                    "source_url": f_meta["source_url"]
                }
            )

            for c in sec_chunks:
                chunk_id = str(uuid.uuid4())
                db_chunk = Chunk(
                    id=chunk_id,
                    filing_id=filing_id,
                    section_id=sec_id,
                    section_name=sec["section_name"],
                    chunk_index=c["chunk_index"],
                    char_start=c["char_start"],
                    char_end=c["char_end"],
                    token_count=c["token_count"],
                    text=c["text"],
                    extra_metadata=c["extra_metadata"]
                )
                db.add(db_chunk)
                total_chunks += 1
                chunks_to_index.append({
                    "id": chunk_id,
                    "filing_id": filing_id,
                    "section_name": sec["section_name"],
                    "text": c["text"],
                    "extra_metadata": c["extra_metadata"]
                })

                # Extraction of events & risks
                if "8-K" in f_meta["form_type"]:
                    ext = await extractor.extract_events_from_8k(c["text"], chunk_id, filing_id)
                    for ev in ext["items"]:
                        db.add(ExtractedItem(
                            chunk_id=chunk_id,
                            filing_id=filing_id,
                            item_type="event",
                            category=ev.event_type,
                            summary_or_description=ev.summary,
                            event_date=ev.date,
                            exact_quote=ev.exact_quote,
                            verified_in_chunk=True,
                            model_used=ext["model_used"],
                            prompt_tokens=ext["prompt_tokens"],
                            completion_tokens=ext["completion_tokens"],
                            estimated_cost_usd=ext["estimated_cost_usd"]
                        ))
                        total_extractions += 1
                elif any(k in sec["section_name"] for k in ["Risk Factors", "MD&A", "Cybersecurity"]):
                    ext = await extractor.extract_risks_from_10k_10q(c["text"], chunk_id, filing_id)
                    for rk in ext["items"]:
                        db.add(ExtractedItem(
                            chunk_id=chunk_id,
                            filing_id=filing_id,
                            item_type="risk",
                            category=rk.category,
                            summary_or_description=rk.description,
                            exact_quote=rk.exact_quote,
                            verified_in_chunk=True,
                            model_used=ext["model_used"],
                            prompt_tokens=ext["prompt_tokens"],
                            completion_tokens=ext["completion_tokens"],
                            estimated_cost_usd=ext["estimated_cost_usd"]
                        ))
                        total_extractions += 1

    await db.commit()

    # Index chunks in retriever
    if chunks_to_index:
        await active_retriever.index_chunks(chunks_to_index)

    return IngestResponse(
        ticker=ticker.upper(),
        cik=cik,
        company_name=name,
        filings_processed=len(filing_items),
        sections_created=total_sections,
        chunks_created=total_chunks,
        extracted_items_count=total_extractions,
        status="success"
    )

@router.post("/ingest/{ticker}", response_model=IngestResponse)
async def ingest_company(ticker: str, db: AsyncSession = Depends(get_db)):
    return await run_ticker_ingestion(ticker, db)

@router.get("/filings", response_model=List[FilingResponse])
async def list_filings(db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Filing)
        .options(selectinload(Filing.company), selectinload(Filing.sections), selectinload(Filing.chunks))
        .order_by(Filing.filing_date.desc())
    )
    res = await db.execute(stmt)
    filings = res.scalars().all()
    return [
        FilingResponse(
            id=f.id,
            ticker=f.company.ticker if f.company else "UNKNOWN",
            form_type=f.form_type,
            accession_number=f.accession_number,
            filing_date=f.filing_date,
            period_of_report=f.period_of_report,
            source_url=f.source_url,
            sections_count=len(f.sections),
            chunks_count=len(f.chunks)
        )
        for f in filings
    ]

@router.get("/filings/{id}/sections", response_model=List[SectionResponse])
async def get_filing_sections(id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Section).where(Section.filing_id == id).order_by(Section.char_start)
    res = await db.execute(stmt)
    sections = res.scalars().all()
    if not sections:
        raise HTTPException(status_code=404, detail=f"No sections found for filing id {id}")
    return [
        SectionResponse(
            id=s.id,
            filing_id=s.filing_id,
            section_name=s.section_name,
            section_title=s.section_title,
            char_start=s.char_start,
            char_end=s.char_end,
            text_length=len(s.raw_text)
        )
        for s in sections
    ]

@router.post("/ask", response_model=AskResponse)
async def ask_research_question(req: AskRequest):
    """Executes the LangGraph research workflow with citations and verification."""
    initial_state = {
        "query": req.query,
        "retrieval_method": req.retrieval_method,
        "retrieval_retry_count": 0
    }
    result_state = await workflow.graph.ainvoke(initial_state)

    citations = [
        CitationOutput(**c) for c in result_state.get("citations", [])
    ]

    return AskResponse(
        query=req.query,
        answer=result_state.get("final_answer", ""),
        citations=citations,
        uncited_sentences=result_state.get("uncited_sentences", []),
        retrieved_chunks=result_state.get("retrieved_chunks", []),
        question_type=result_state.get("question_type", "factual_lookup"),
        latency_seconds=round(result_state.get("latency_seconds", 0.0), 3),
        prompt_tokens=result_state.get("prompt_tokens", 0),
        completion_tokens=result_state.get("completion_tokens", 0),
        estimated_cost_usd=round(result_state.get("estimated_cost_usd", 0.0), 6)
    )

