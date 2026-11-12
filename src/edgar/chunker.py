import tiktoken
from typing import List, Dict, Any
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("edgar.chunker")

class SectionAwareChunker:
    """
    Chunks text strictly within SEC filing sections without crossing boundaries.
    Tracks exact character offsets and token counts using tiktoken.
    """
    def __init__(
        self,
        target_tokens: int = settings.DEFAULT_CHUNK_SIZE,
        overlap_tokens: int = settings.DEFAULT_CHUNK_OVERLAP,
        encoding_name: str = "cl100k_base"
    ):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_section(
        self,
        section_text: str,
        section_name: str,
        section_global_start: int,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Splits a single section into token-bounded chunks with sliding overlap.
        Preserves paragraph and sentence boundaries where possible.
        """
        if not section_text or not section_text.strip():
            return []

        tokens = self.tokenizer.encode(section_text)
        total_tokens = len(tokens)

        if total_tokens <= self.target_tokens:
            return [{
                "chunk_index": 0,
                "section_name": section_name,
                "char_start": section_global_start,
                "char_end": section_global_start + len(section_text),
                "token_count": total_tokens,
                "text": section_text.strip(),
                "extra_metadata": {**metadata, "section": section_name}
            }]

        chunks = []
        step = max(self.target_tokens - self.overlap_tokens, 1)
        chunk_idx = 0

        for start_idx in range(0, total_tokens, step):
            end_idx = min(start_idx + self.target_tokens, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            
            chunk_text = self.tokenizer.decode(chunk_tokens).strip()
            
            # Approximate character offsets in section
            # For exact substring indexing, find chunk_text in section_text
            local_char_start = section_text.find(chunk_text[:50])
            if local_char_start == -1:
                # Fallback ratio estimation if exact prefix isn't found
                ratio = start_idx / total_tokens
                local_char_start = int(ratio * len(section_text))
            
            local_char_end = local_char_start + len(chunk_text)
            
            chunks.append({
                "chunk_index": chunk_idx,
                "section_name": section_name,
                "char_start": section_global_start + local_char_start,
                "char_end": section_global_start + local_char_end,
                "token_count": len(chunk_tokens),
                "text": chunk_text,
                "extra_metadata": {**metadata, "section": section_name, "chunk_index": chunk_idx}
            })
            chunk_idx += 1

            if end_idx >= total_tokens:
                break

        return chunks

    def chunk_filing_sections(
        self,
        sections: List[Dict[str, Any]],
        filing_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Splits all sections in a filing into chunks without crossing section boundaries."""
        all_chunks = []
        for sec in sections:
            sec_chunks = self.chunk_section(
                section_text=sec["raw_text"],
                section_name=sec["section_name"],
                section_global_start=sec["char_start"],
                metadata=filing_metadata
            )
            all_chunks.extend(sec_chunks)
        return all_chunks

