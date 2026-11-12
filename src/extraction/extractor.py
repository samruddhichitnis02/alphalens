import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from src.config.settings import settings
from src.utils.logger import get_logger
from src.extraction.schemas import FinancialEventItem, FinancialEventList, BusinessRiskItem, BusinessRiskList

logger = get_logger("extraction.extractor")

class ExtractorService:
    """
    Extracts corporate events and business risks from SEC chunks using Claude (or local/mock provider).
    Applies strict verbatim quote hallucination verification and concurrency control.
    """
    def __init__(self):
        self.semaphore = asyncio.Semaphore(settings.EXTRACTION_SEMAPHORE_LIMIT)
        self.provider = settings.LLM_PROVIDER
        self.client = None
        if self.provider == "claude" and settings.ANTHROPIC_API_KEY:
            try:
                import anthropic
                self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            except Exception as e:
                logger.warning(f"Could not initialize Anthropic client: {e}")

    def verify_quote_in_chunk(self, quote: str, chunk_text: str) -> bool:
        """
        Hallucination Guard: Checks whether the extracted quote exists verbatim in the source chunk text.
        Allows minor whitespace tolerance.
        """
        if not quote or not quote.strip():
            return False
        clean_quote = " ".join(quote.strip().split())
        clean_chunk = " ".join(chunk_text.strip().split())
        return clean_quote.lower() in clean_chunk.lower()

    async def extract_events_from_8k(self, chunk_text: str, chunk_id: str, filing_id: str) -> Dict[str, Any]:
        """Extract material financial events from an 8-K chunk."""
        prompt = (
            "You are an institutional financial analyst. Extract all material corporate events mentioned in this SEC 8-K excerpt.\n"
            "Return valid JSON matching this schema:\n"
            "{\n"
            '  "events": [\n'
            '    {"event_type": "earnings"|"guidance_change"|"mergers_and_acquisitions"|"executive_change"|"material_agreement"|"debt_issuance"|"other",\n'
            '     "date": "YYYY-MM-DD or null",\n'
            '     "summary": "Concise summary",\n'
            '     "exact_quote": "Exact verbatim sentence from the text supporting this event"}\n'
            "  ]\n"
            "}\n"
            "CRITICAL RULE: The exact_quote MUST be copied verbatim from the text. If you fabricate or paraphrase the quote, it will be rejected.\n\n"
            f"Filing Text:\n{chunk_text}"
        )

        raw_response, p_tokens, c_tokens, cost = await self._call_llm_with_retry(prompt)
        
        parsed_events = []
        rejected_count = 0
        try:
            # Extract JSON block
            json_match = re.search(r"\{[\s\S]*\}", raw_response)
            if json_match:
                data = json.loads(json_match.group(0))
                event_list = FinancialEventList(**data)
                for ev in event_list.events:
                    is_valid = self.verify_quote_in_chunk(ev.exact_quote, chunk_text)
                    if is_valid:
                        parsed_events.append(ev)
                    else:
                        rejected_count += 1
                        logger.warning(f"Hallucination Guard: Rejected event quote not found verbatim in chunk {chunk_id}: {ev.exact_quote[:60]}...")
        except Exception as e:
            logger.error(f"Failed to parse event extraction JSON: {e}")

        return {
            "chunk_id": chunk_id,
            "filing_id": filing_id,
            "item_type": "event",
            "items": parsed_events,
            "rejected_count": rejected_count,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "estimated_cost_usd": cost,
            "model_used": settings.CLAUDE_EXTRACTION_MODEL if self.client else "deterministic-fallback"
        }

    async def extract_risks_from_10k_10q(self, chunk_text: str, chunk_id: str, filing_id: str) -> Dict[str, Any]:
        """Extract business and financial risks from a 10-K/10-Q Risk Factors or MD&A chunk."""
        prompt = (
            "You are an institutional risk analyst. Extract business and financial risks described in this SEC filing excerpt.\n"
            "Return valid JSON matching this schema:\n"
            "{\n"
            '  "risks": [\n'
            '    {"category": "regulatory"|"supply_chain"|"competition"|"macro"|"cybersecurity"|"litigation"|"concentration"|"operational"|"other",\n'
            '     "description": "Concise risk description",\n'
            '     "exact_quote": "Exact verbatim sentence from the text supporting this risk"}\n'
            "  ]\n"
            "}\n"
            "CRITICAL RULE: The exact_quote MUST be copied verbatim from the text. If you fabricate or paraphrase the quote, it will be rejected.\n\n"
            f"Filing Text:\n{chunk_text}"
        )

        raw_response, p_tokens, c_tokens, cost = await self._call_llm_with_retry(prompt)
        
        parsed_risks = []
        rejected_count = 0
        try:
            json_match = re.search(r"\{[\s\S]*\}", raw_response)
            if json_match:
                data = json.loads(json_match.group(0))
                risk_list = BusinessRiskList(**data)
                for rk in risk_list.risks:
                    is_valid = self.verify_quote_in_chunk(rk.exact_quote, chunk_text)
                    if is_valid:
                        parsed_risks.append(rk)
                    else:
                        rejected_count += 1
                        logger.warning(f"Hallucination Guard: Rejected risk quote not found verbatim in chunk {chunk_id}: {rk.exact_quote[:60]}...")
        except Exception as e:
            logger.error(f"Failed to parse risk extraction JSON: {e}")

        return {
            "chunk_id": chunk_id,
            "filing_id": filing_id,
            "item_type": "risk",
            "items": parsed_risks,
            "rejected_count": rejected_count,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "estimated_cost_usd": cost,
            "model_used": settings.CLAUDE_EXTRACTION_MODEL if self.client else "deterministic-fallback"
        }

    async def _call_llm_with_retry(self, prompt: str) -> tuple[str, int, int, float]:
        """Calls LLM with semaphore concurrency control and exponential backoff retry."""
        async with self.semaphore:
            if not self.client or not settings.ANTHROPIC_API_KEY:
                # Deterministic free fallback for offline testing or without API key
                return self._deterministic_extraction_fallback(prompt)

            for attempt in range(settings.MAX_RETRIES):
                try:
                    resp = await self.client.messages.create(
                        model=settings.CLAUDE_EXTRACTION_MODEL,
                        max_tokens=1024,
                        temperature=0.0,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    text = resp.content[0].text
                    p_tok = resp.usage.input_tokens
                    c_tok = resp.usage.output_tokens
                    # Approx Claude 3.5 Haiku pricing ($0.80 / MTok input, $4.00 / MTok output)
                    cost = (p_tok * 0.80 + c_tok * 4.00) / 1_000_000.0
                    return text, p_tok, c_tok, cost
                except Exception as e:
                    if attempt == settings.MAX_RETRIES - 1:
                        logger.error(f"LLM extraction error after {settings.MAX_RETRIES} attempts: {e}")
                        return self._deterministic_extraction_fallback(prompt)
                    await asyncio.sleep(2 ** attempt)

        return self._deterministic_extraction_fallback(prompt)

    def _deterministic_extraction_fallback(self, prompt: str) -> tuple[str, int, int, float]:
        """
        High precision deterministic extractor for running tests, CI/CD, or offline zero-cost usage.
        Extracts actual verbatim sentences containing risk/event keywords.
        """
        is_risk = "risks" in prompt.lower()
        # Find text after "Filing Text:\n"
        parts = prompt.split("Filing Text:\n")
        chunk_text = parts[1] if len(parts) > 1 else prompt
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
        
        if is_risk:
            risks = []
            risk_keywords = [
                ("cybersecurity", ["cyber", "breach", "security incident", "ransomware", "unauthorized access"]),
                ("supply_chain", ["supply chain", "supplier", "shortage", "manufacturing", "inventory"]),
                ("competition", ["competitor", "competition", "market share", "pricing pressure"]),
                ("regulatory", ["regulat", "compliance", "government investigation", "sec", "doj", "sanctions"]),
                ("litigation", ["lawsuit", "litigation", "legal proceedings", "claims", "patent"]),
                ("macro", ["inflation", "interest rate", "foreign exchange", "recession", "economic downturn"]),
            ]
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 30:
                    continue
                s_lower = s_clean.lower()
                for cat, kws in risk_keywords:
                    if any(kw in s_lower for kw in kws):
                        risks.append({
                            "category": cat,
                            "description": f"Risk related to {cat}: {s_clean[:120]}...",
                            "exact_quote": s_clean
                        })
                        break
                if len(risks) >= 3:
                    break
            return json.dumps({"risks": risks}), len(prompt)//4, len(risks)*40, 0.0
        else:
            events = []
            event_keywords = [
                ("earnings", ["revenue", "net income", "operating results", "eps", "financial results"]),
                ("executive_change", ["appointed", "resigned", "officer", "director", "chief executive"]),
                ("material_agreement", ["agreement", "contract", "definitive", "entered into"]),
                ("debt_issuance", ["notes", "credit facility", "senior notes", "borrowing", "indenture"]),
                ("mergers_and_acquisitions", ["acquisition", "merger", "acquired", "transaction"]),
            ]
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 30:
                    continue
                s_lower = s_clean.lower()
                for etype, kws in event_keywords:
                    if any(kw in s_lower for kw in kws):
                        events.append({
                            "event_type": etype,
                            "date": None,
                            "summary": f"Corporate event ({etype}): {s_clean[:120]}...",
                            "exact_quote": s_clean
                        })
                        break
                if len(events) >= 3:
                    break
            return json.dumps({"events": events}), len(prompt)//4, len(events)*40, 0.0

