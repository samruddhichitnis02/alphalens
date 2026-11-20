<div align="center">

# AlphaLens

**Ask questions about company reports. Get answers with proof.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-workflow-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Runs locally](https://img.shields.io/badge/runs-100%25%20locally-success.svg)](#run-it-yourself)

<img src="assets/explainer.gif" alt="50-second explainer: what AlphaLens does, in five steps" width="760" />

<sub>50-second explainer · <a href="assets/explainer.mp4">download the MP4</a></sub>

</div>

## What it does

Public companies publish long reports with the SEC: yearly (10-K), quarterly (10-Q) and big-news (8-K) filings, often hundreds of pages each. AlphaLens searches them, answers your question in plain English, and **tags every sentence with the paragraph it came from**. If the reports don't contain the answer, it says so.

## How it works

```mermaid
flowchart LR
    A["1. Download<br/>SEC filings"] --> B["2. Split<br/>into sections and pieces"]
    B --> C["3. Search<br/>keywords + meaning"]
    C --> D["4. Answer<br/>with a source per sentence"]
    D --> E["5. Check<br/>every source is real"]
    C -. "nothing relevant" .-> F["Say 'not enough info'"]
```

<details>
<summary>Detailed architecture</summary>

```mermaid
flowchart TD
    subgraph Ingest ["Ingestion"]
        SEC["SEC EDGAR API"] -->|"rate limit 8 req/s"| Client["Async EdgarClient"]
        Client --> Cache["Disk cache"]
        Cache --> Parser["Section-aware parser<br/>(10-K/10-Q/8-K Items)"]
        Parser --> Chunker["Chunker (400 tokens,<br/>never crosses sections)"]
    end
    subgraph Store ["Storage and indexing"]
        Chunker --> DB[("PostgreSQL / SQLite")]
        Chunker --> Extractor["LLM extractor<br/>+ verbatim-quote guard"]
        Chunker --> Embed["all-MiniLM-L6-v2 embeddings"]
        Embed --> Index[("OpenSearch or in-memory<br/>BM25 + kNN")]
    end
    subgraph Flow ["LangGraph workflow"]
        Q["Question"] --> N1["Understand query"] --> N2["Retrieve (RRF fusion,<br/>per company)"]
        N2 --> N3{"Evidence?"}
        N3 -->|"weak"| N3b["Reformulate"] --> N2
        N3 -->|"none"| Abstain["Not enough info"]
        N3 -->|"yes"| N4["Generate with [chunk_id]"] --> N5["Verify citations"]
    end
```

</details>

## Results at a glance

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/kpis_dark.png">
  <img src="assets/kpis.png" alt="Headline numbers: 96% right passage in top 5, 97% citations supported, 100% trick questions declined, 31 ms per question, $0 per question">
</picture>

### The benchmark: 59 hand-checked questions

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/question_mix_dark.png">
  <img src="assets/question_mix.png" alt="Question types: 15 fact lookups, 12 compare across years, 12 compare two companies, 10 risk summaries, 10 trick questions" width="640">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmark_coverage_dark.png">
  <img src="assets/benchmark_coverage.png" alt="Questions per company (AAPL and MSFT 10 each, others 6 to 8) and the filing sections answers come from (MD&A 32, Risk Factors 15)">
</picture>

### Search: does it find the right paragraph?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/retrieval_benchmark_dark.png">
  <img src="assets/retrieval_benchmark.png" alt="All three search methods find the right passage in the top 5 for 95.9% of questions; meaning search ranks it highest" width="720">
</picture>

### Answers: can you trust them?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/answer_quality_dark.png">
  <img src="assets/answer_quality.png" alt="100% correct refusals, 97% citations support their sentence, 81% sentences grounded, 60% factual sentences cited" width="640">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/scores_by_type_dark.png">
  <img src="assets/scores_by_type.png" alt="Heatmap of scores by question type: comparing two companies scores lowest on search (83%) and citations (52%); risk summaries lowest on grounding (59%)" width="720">
</picture>

### Errors: where does it go wrong?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/failure_breakdown_dark.png">
  <img src="assets/failure_breakdown.png" alt="Errors by question type: weak comparisons are the most common error, search misses the rarest; all 10 trick questions handled correctly" width="720">
</picture>

### Grader check: does the automatic grader match a human?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/judge_agreement_dark.png">
  <img src="assets/judge_agreement.png" alt="Grader vs human on 20 examples: 15 agree, 5 disagree, kappa 0.50" width="440">
</picture>

<details>
<summary>Limits of these numbers</summary>

- The benchmark uses a small hand-written set of filing-style paragraphs (`generate_hardened_evaluation_corpus` in [`src/eval/runner.py`](src/eval/runner.py)), not the full SEC archive.
- Answers come from the free offline generator, not Claude. Set `LLM_PROVIDER=claude` and re-run `make eval` to measure a real model.
- The prompt-version and model-tier rows in `eval_results.json` are fixed values, not measurements, so they're not shown.
- The ticker-filter ablation lowers precision (0.95 to 0.21) because the hit check matches on section and keywords, not company. This needs a fix.
- The grader never assigns `extraction_error`, so that category is left out.

</details>

## Coverage

| Sector | Companies |
| :--- | :--- |
| Technology | AAPL, MSFT, NVDA, AMZN, META |
| Financials & Healthcare | JPM, PFE |
| Consumer, Energy & Auto | TSLA, WMT, XOM |

Filings: 10-K (last 3 years), 10-Q (last 6 quarters), 8-K (last 12 months). Configured in [`config/companies.yaml`](config/companies.yaml).

## Run it yourself

```bash
git clone https://github.com/samruddhichitnis02/alphalens.git
cd alphalens
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install uv && uv pip install -e .
cp .env.example .env               # set SEC_USER_AGENT (name + email, required by the SEC)
```

| Command | What it does |
| :--- | :--- |
| `make test` | Run the unit tests |
| `make eval` | Run the 59-question benchmark |
| `make run` | Start the API at http://localhost:8000 (docs at `/docs`) |
| `make up` / `make down` | Start / stop PostgreSQL + OpenSearch in Docker |

Without Docker, set `DATABASE_URL` to the SQLite line in `.env.example` and `LLM_PROVIDER=mock`. Everything then runs locally for free.

Refresh the charts and video after `make eval`:

```bash
python scripts/eval_breakdown.py            # per-question-type numbers
python scripts/generate_charts.py           # all charts in assets/
python scripts/generate_explainer_video.py  # explainer GIF + MP4 (MP4 needs imageio-ffmpeg)
```

## API

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/v1/ingest/{ticker}` | Download and index one company's filings |
| `GET` | `/api/v1/filings` | List indexed filings |
| `GET` | `/api/v1/filings/{id}/sections` | Sections of one filing |
| `POST` | `/api/v1/ask` | Ask a question, get a cited answer |

```bash
curl -X POST http://localhost:8000/api/v1/ask -H "Content-Type: application/json" \
  -d '{"query": "Compare Microsoft Azure and Amazon AWS cloud growth", "retrieval_method": "hybrid", "top_k": 5}'
```

## Project layout

```
src/edgar/        download, parse sections, chunk
src/retrieval/    BM25, embeddings, hybrid RRF (local or OpenSearch)
src/workflow/     LangGraph: understand -> retrieve -> check -> answer -> verify
src/extraction/   structured facts with the verbatim-quote guard
src/eval/         benchmark, metrics, grounding judge, review CLI
src/api/          FastAPI routes
scripts/          chart, breakdown and explainer-video generators
```
