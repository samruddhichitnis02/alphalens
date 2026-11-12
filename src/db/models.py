import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    String, Text, Integer, Float, Boolean, DateTime, ForeignKey, Index, JSON
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Company(Base):
    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ticker: Mapped[str] = mapped_column(String(10), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    cik: Mapped[str] = mapped_column(String(10), unique=True, index=True, nullable=False)
    sector: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    filings: Mapped[List["Filing"]] = relationship("Filing", back_populates="company", cascade="all, delete-orphan")


class Filing(Base):
    __tablename__ = "filings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), nullable=False, index=True)
    accession_number: Mapped[str] = mapped_column(String(25), unique=True, index=True, nullable=False)
    form_type: Mapped[str] = mapped_column(String(10), index=True, nullable=False)  # 10-K, 10-Q, 8-K
    filing_date: Mapped[str] = mapped_column(String(10), index=True, nullable=False) # YYYY-MM-DD
    period_of_report: Mapped[Optional[str]] = mapped_column(String(10), nullable=True) # YYYY-MM-DD or fiscal quarter
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    local_cache_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    company: Mapped["Company"] = relationship("Company", back_populates="filings")
    sections: Mapped[List["Section"]] = relationship("Section", back_populates="filing", cascade="all, delete-orphan")
    chunks: Mapped[List["Chunk"]] = relationship("Chunk", back_populates="filing", cascade="all, delete-orphan")
    extracted_items: Mapped[List["ExtractedItem"]] = relationship("ExtractedItem", back_populates="filing", cascade="all, delete-orphan")


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filing_id: Mapped[str] = mapped_column(ForeignKey("filings.id", ondelete="CASCADE"), nullable=False, index=True)
    section_name: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    section_title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)

    filing: Mapped["Filing"] = relationship("Filing", back_populates="sections")
    chunks: Mapped[List["Chunk"]] = relationship("Chunk", back_populates="section", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filing_id: Mapped[str] = mapped_column(ForeignKey("filings.id", ondelete="CASCADE"), nullable=False, index=True)
    section_id: Mapped[str] = mapped_column(ForeignKey("sections.id", ondelete="CASCADE"), nullable=False, index=True)
    section_name: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    filing: Mapped["Filing"] = relationship("Filing", back_populates="chunks")
    section: Mapped["Section"] = relationship("Section", back_populates="chunks")
    extracted_items: Mapped[List["ExtractedItem"]] = relationship("ExtractedItem", back_populates="chunk", cascade="all, delete-orphan")


class ExtractedItem(Base):
    __tablename__ = "extracted_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    filing_id: Mapped[str] = mapped_column(ForeignKey("filings.id", ondelete="CASCADE"), nullable=False, index=True)
    item_type: Mapped[str] = mapped_column(String(50), index=True, nullable=False) # "event" or "risk"
    category: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    summary_or_description: Mapped[str] = mapped_column(Text, nullable=False)
    event_date: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    exact_quote: Mapped[str] = mapped_column(Text, nullable=False)
    verified_in_chunk: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="extracted_items")
    filing: Mapped["Filing"] = relationship("Filing", back_populates="extracted_items")

