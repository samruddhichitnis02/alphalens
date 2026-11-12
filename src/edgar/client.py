import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("edgar.client")

class AsyncRateLimiter:
    """Token bucket / timestamp rate limiter to respect SEC fair access (max 10 req/s)."""
    def __init__(self, max_rate: float = 8.0):
        self.rate = max_rate
        self.interval = 1.0 / max_rate
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self.last_request_time = time.monotonic()

class EdgarClient:
    """Async SEC EDGAR client using official SEC Submissions API with disk caching."""
    def __init__(self):
        self.cache_dir = settings.DATA_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limiter = AsyncRateLimiter(settings.SEC_MAX_REQUESTS_PER_SECOND)
        self.headers = {
            "User-Agent": settings.SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }

    async def get_company_submissions(self, cik: str) -> Dict[str, Any]:
        """Fetch company submission history from data.sec.gov."""
        clean_cik = cik.strip().zfill(10)
        cache_path = self.cache_dir / clean_cik / "submissions.json"
        if cache_path.exists():
            import json
            logger.info(f"Using cached submissions for CIK {clean_cik}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        await self.rate_limiter.wait()
        url = f"https://data.sec.gov/submissions/CIK{clean_cik}.json"
        logger.info(f"Fetching submissions metadata from SEC: {url}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return data

    def filter_recent_filings(
        self,
        submissions_data: Dict[str, Any],
        years_10k: int = 3,
        quarters_10q: int = 6,
        months_8k: int = 12
    ) -> List[Dict[str, Any]]:
        """Filter filings based on the project scope."""
        recent = submissions_data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        from datetime import datetime, timedelta
        now = datetime.utcnow()
        date_12m_ago = (now - timedelta(days=365)).strftime("%Y-%m-%d")

        count_10k = 0
        count_10q = 0
        selected_filings = []

        cik = str(submissions_data.get("cik", "")).zfill(10)
        cik_no_zero = str(int(cik)) if cik.isdigit() else cik

        for i in range(len(forms)):
            form = forms[i].upper()
            acc_num = accession_numbers[i]
            filing_date = filing_dates[i]
            report_date = report_dates[i] if i < len(report_dates) else filing_date
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""

            acc_no_dash = acc_num.replace("-", "")
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{acc_no_dash}/{primary_doc}"

            item = {
                "accession_number": acc_num,
                "form_type": form,
                "filing_date": filing_date,
                "period_of_report": report_date,
                "primary_document": primary_doc,
                "source_url": doc_url,
                "cik": cik
            }

            if form == "10-K" and count_10k < years_10k:
                selected_filings.append(item)
                count_10k += 1
            elif form == "10-Q" and count_10q < quarters_10q:
                selected_filings.append(item)
                count_10q += 1
            elif form == "8-K" and filing_date >= date_12m_ago:
                selected_filings.append(item)

        return selected_filings

    async def download_filing_document(self, filing_meta: Dict[str, Any]) -> Path:
        """Download and cache filing HTML/document from SEC Archives."""
        cik = filing_meta["cik"]
        acc_num = filing_meta["accession_number"]
        url = filing_meta["source_url"]

        file_ext = Path(filing_meta["primary_document"]).suffix or ".htm"
        cache_file = self.cache_dir / cik / f"{acc_num}{file_ext}"

        if cache_file.exists() and cache_file.stat().st_size > 0:
            logger.info(f"Filing already cached at {cache_file}")
            return cache_file

        await self.rate_limiter.wait()
        logger.info(f"Downloading filing from {url}")
        
        headers = dict(self.headers)
        headers["Host"] = "www.sec.gov"

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            text_content = resp.text

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8", errors="ignore") as f:
            f.write(text_content)

        return cache_file

