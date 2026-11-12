import re
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup
from src.utils.logger import get_logger

logger = get_logger("edgar.parser")

# Regex patterns for SEC 10-K / 10-Q Item headers
ITEM_PATTERNS_10K_10Q = [
    (r"(?i)\bitem\s+1a[\.\:\s\-]+risk\s+factors\b", "Item 1A - Risk Factors"),
    (r"(?i)\bitem\s+1c[\.\:\s\-]+cybersecurity\b", "Item 1C - Cybersecurity"),
    (r"(?i)\bitem\s+1[\.\:\s\-]+business\b", "Item 1 - Business"),
    (r"(?i)\bitem\s+2[\.\:\s\-]+properties\b", "Item 2 - Properties"),
    (r"(?i)\bitem\s+3[\.\:\s\-]+legal\s+proceedings\b", "Item 3 - Legal Proceedings"),
    (r"(?i)\bitem\s+7a[\.\:\s\-]+quantitative\s+and\s+qualitative\s+disclosures\b", "Item 7A - Market Risk Disclosures"),
    (r"(?i)\bitem\s+7[\.\:\s\-]+management['’]?s\s+discussion\s+and\s+analysis\b", "Item 7 - MD&A"),
    (r"(?i)\bitem\s+8[\.\:\s\-]+financial\s+statements\b", "Item 8 - Financial Statements"),
    (r"(?i)\bitem\s+9a[\.\:\s\-]+controls\s+and\s+procedures\b", "Item 9A - Controls and Procedures"),
]

# Regex patterns for 8-K Items
ITEM_PATTERNS_8K = [
    (r"(?i)\bitem\s+1\.01[\.\:\s\-]+entry\s+into\s+a\s+material\s+definitive\s+agreement\b", "Item 1.01 Material Agreements"),
    (r"(?i)\bitem\s+1\.02[\.\:\s\-]+termination\s+of\s+a\s+material\s+definitive\s+agreement\b", "Item 1.02 Termination of Material Agreement"),
    (r"(?i)\bitem\s+2\.01[\.\:\s\-]+completion\s+of\s+acquisition\s+or\s+disposition\b", "Item 2.01 M&A Completion"),
    (r"(?i)\bitem\s+2\.02[\.\:\s\-]+results\s+of\s+operations\s+and\s+financial\s+condition\b", "Item 2.02 Earnings & Operations"),
    (r"(?i)\bitem\s+2\.03[\.\:\s\-]+creation\s+of\s+a\s+direct\s+financial\s+obligation\b", "Item 2.03 Debt & Financial Obligations"),
    (r"(?i)\bitem\s+5\.02[\.\:\s\-]+departure\s+of\s+directors\s+or\s+certain\s+officers\b", "Item 5.02 Executive & Director Changes"),
    (r"(?i)\bitem\s+7\.01[\.\:\s\-]+regulation\s+fd\s+disclosure\b", "Item 7.01 Regulation FD Disclosure"),
    (r"(?i)\bitem\s+8\.01[\.\:\s\-]+other\s+events\b", "Item 8.01 Other Events"),
]

class EdgarParser:
    """Parses raw HTML/XML SEC filings into clean text with item/section boundaries."""

    @staticmethod
    def html_to_text(html_content: str) -> str:
        """Converts raw HTML into cleaned, whitespace-normalized plain text."""
        # Fast HTML parsing
        soup = BeautifulSoup(html_content, "html.parser")

        # Strip script, style, and header/footer elements that introduce noise
        for element in soup(["script", "style", "header", "footer"]):
            element.decompose()

        text = soup.get_text(separator="\n")
        # Normalize non-breaking spaces and irregular unicode characters
        text = text.replace("\xa0", " ").replace("&#160;", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    def parse_filing_sections(self, clean_text: str, form_type: str) -> List[Dict[str, Any]]:
        """
        Detects section/item boundaries in cleaned text and splits into structured sections.
        Returns a list of dicts: {section_name, section_title, char_start, char_end, raw_text}
        """
        form = form_type.upper()
        if "8-K" in form:
            patterns = ITEM_PATTERNS_8K
            generic_fallback = r"(?i)\bitem\s+(\d+\.\d+)\b"
        else:
            patterns = ITEM_PATTERNS_10K_10Q
            generic_fallback = r"(?i)\bitem\s+([0-9]{1,2}[a-z]?)\b"

        # Find all section header match positions
        matches: List[Tuple[int, int, str]] = []

        for pat, label in patterns:
            for m in re.finditer(pat, clean_text):
                # Filter out Table of Contents matches (often followed by dots or small page numbers)
                start, end = m.span()
                preview = clean_text[end:end+120]
                if re.match(r"^[\s\.\-]+[0-9]{1,3}\b", preview):
                    # Table of contents line, skip
                    continue
                matches.append((start, end, label))

        # If specific named patterns didn't match (due to company-specific spelling), try generic items
        if not matches:
            for m in re.finditer(generic_fallback, clean_text):
                start, end = m.span()
                matched_label = f"Item {m.group(1).upper()}"
                preview = clean_text[end:end+120]
                if re.match(r"^[\s\.\-]+[0-9]{1,3}\b", preview):
                    continue
                matches.append((start, end, matched_label))

        # Sort matches by appearance in text
        matches.sort(key=lambda x: x[0])

        # Filter duplicates or overlaps
        deduped: List[Tuple[int, int, str]] = []
        for m in matches:
            if not deduped or m[0] >= deduped[-1][1]:
                deduped.append(m)

        sections: List[Dict[str, Any]] = []

        if not deduped:
            # Fallback if no specific item header was found: preserve whole document as general section
            sections.append({
                "section_name": "General",
                "section_title": f"{form_type} Full Filing",
                "char_start": 0,
                "char_end": len(clean_text),
                "raw_text": clean_text
            })
            return sections

        # Preamble before first detected item
        if deduped[0][0] > 0:
            preamble_text = clean_text[0:deduped[0][0]].strip()
            if len(preamble_text) > 100:
                sections.append({
                    "section_name": "Preamble / General",
                    "section_title": "Filing Header & General Information",
                    "char_start": 0,
                    "char_end": deduped[0][0],
                    "raw_text": preamble_text
                })

        # Process each detected section
        for i, (start, end, label) in enumerate(deduped):
            next_start = deduped[i + 1][0] if i + 1 < len(deduped) else len(clean_text)
            sec_text = clean_text[start:next_start].strip()
            if len(sec_text) > 50: # Ignore empty placeholders
                sections.append({
                    "section_name": label,
                    "section_title": label,
                    "char_start": start,
                    "char_end": next_start,
                    "raw_text": sec_text
                })

        return sections

