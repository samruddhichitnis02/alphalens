"""Generate README charts straight from the evaluation output.

Every number drawn here is read from `data/eval_results.json` (written by
`python -m src.eval.runner`) or `data/benchmark_questions.json`. Nothing is
typed in by hand, so re-running the eval and then this script keeps the
README charts honest.

Each chart is written twice - `<name>.png` (light) and `<name>_dark.png` - so
the README can serve the right one for GitHub's light and dark themes.

Usage:  python scripts/generate_charts.py
"""
import json
import logging
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # quiet font fallback chatter

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RESULTS = json.loads((ROOT / "data" / "eval_results.json").read_text(encoding="utf-8"))
QUESTIONS = json.loads((ROOT / "data" / "benchmark_questions.json").read_text(encoding="utf-8"))

# Validated categorical palette (light / dark steps of the same hues).
THEMES = {
    "light": {
        "surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "suffix": "",
    },
    "dark": {
        "surface": "#0d1117", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "suffix": "_dark",
    },
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


def _style(ax, t, grid_axis="x"):
    fig = ax.figure
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], length=0)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _title(ax, text, t):
    ax.set_title(text, pad=26, color=t["ink"])


def _subtitle(ax, text, t):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=t["ink2"], fontsize=10.5, va="bottom")


def _save(fig, name, t):
    out = ASSETS / f"{name}{t['suffix']}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.25, facecolor=t["surface"])
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)}")


def _hbar(t, name, title, subtitle, labels, values, fmt, xmax, xlabel):
    """Single-series horizontal bar chart, largest at top, one colour."""
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(labels) + 1.4))
    _style(ax, t)
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, height=0.56, color=t["series"][0])
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.012, yi, fmt(v), va="center", ha="left", color=t["ink"], fontsize=11, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=t["ink"], fontsize=11)
    ax.set_xlim(0, xmax * 1.1)
    ax.set_xlabel(xlabel, color=t["ink2"])
    _title(ax, title, t)
    _subtitle(ax, subtitle, t)
    _save(fig, name, t)


def chart_retrieval(t):
    """Dot plot: three search methods across four ranking metrics."""
    rc = RESULTS["retrieval_comparison"]
    methods = [("bm25", "Keyword search (BM25)"), ("semantic", "Meaning search (embeddings)"), ("hybrid", "Hybrid (both, fused with RRF)")]
    metrics = [
        ("Recall@5", "Right passage in top 5"),
        ("Recall@10", "Right passage in top 10"),
        ("MRR", "How high the first right\npassage ranks (MRR)"),
        ("nDCG@10", "Overall ranking quality\n(nDCG@10)"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    _style(ax, t)
    offsets = [0.22, 0, -0.22]
    for mi, (key, _) in enumerate(metrics):
        base = len(metrics) - 1 - mi
        for (mkey, mlabel), off, color in zip(methods, offsets, t["series"]):
            v = rc[mkey][key]
            ax.plot([0.85, v], [base + off] * 2, color=t["grid"], linewidth=1.2, zorder=1)
            ax.scatter([v], [base + off], s=70, color=color, edgecolors=t["surface"], linewidths=2, zorder=3,
                       label=mlabel if mi == 0 else None)
            ax.text(v + 0.003, base + off, f"{v:.3f}", va="center", ha="left", fontsize=9.5, color=t["ink2"])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m[1] for m in metrics][::-1], color=t["ink"], fontsize=10.5)
    ax.set_xlim(0.85, 1.0)
    ax.set_ylim(-0.6, len(metrics) - 0.4)
    ax.set_xlabel("Score (1.0 = perfect)  ·  axis starts at 0.85 to show the small gaps", color=t["ink2"], fontsize=9.5)
    _title(ax, "Finding the right passage: three search methods", t)
    n = sum(1 for q in QUESTIONS if q.get("is_answerable", True))
    _subtitle(ax, f"{n} answerable benchmark questions · higher is better", t)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.42, -0.2), ncol=3, frameon=False, fontsize=9.5,
                    handletextpad=0.3, columnspacing=1.2)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    _save(fig, "retrieval_benchmark", t)


def chart_answer_quality(t):
    g = RESULTS["generation_metrics"]
    rows = [
        ("Says \"not enough info\" when\nthe answer isn't in the filings", g["Correct Abstention Rate"]),
        ("Citations that really back\nup their sentence", g["Citation Support Rate"]),
        ("Sentences supported by\nthe retrieved filings", g["Factual Grounding Rate"]),
        ("Factual sentences that\ncarry a citation", g["Citation Coverage Rate"]),
    ]
    _hbar(t, "answer_quality", "How trustworthy are the answers?",
          f"{RESULTS['dataset_size']} benchmark questions · offline answer generator (no paid API calls)",
          [r[0] for r in rows], [r[1] * 100 for r in rows], lambda v: f"{v:.0f}%", 100, "Percent")


def chart_question_mix(t):
    names = {
        "factual_lookup": "Look up a fact",
        "cross_period_comparison": "Compare across years",
        "cross_company_comparison": "Compare two companies",
        "risk_summary": "Summarise risks",
        "unanswerable": "Trick question (no answer)",
    }
    counts = Counter(q["question_type"] for q in QUESTIONS)
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    _hbar(t, "question_mix", "What the benchmark asks",
          f"{len(QUESTIONS)} hand-checked questions across 5 types",
          [names.get(k, k) for k, _ in items], [v for _, v in items], lambda v: f"{v:.0f}", max(counts.values()),
          "Number of questions")


def chart_failures(t):
    ft = RESULTS["failure_taxonomy"]
    total = RESULTS["dataset_size"]
    names = {
        "reasoning_error": "Reasoning: found the facts,\ncompared them badly",
        "generation_error": "Writing: sentence missing\na citation",
        "retrieval_error": "Search: right passage\nnot found",
    }
    # The judge never assigns "extraction_error" (see src/eval/judge.py), so its
    # count is always 0 by construction - leave it off rather than imply a result.
    items = sorted(((k, v) for k, v in ft.items() if k != "extraction_error"), key=lambda kv: -kv[1])
    clean = total - sum(ft.values())
    _hbar(t, "failure_breakdown", "Where answers went wrong",
          f"Main error per question, {total} questions · {clean} had no error",
          [names.get(k, k) for k, _ in items], [v for _, v in items], lambda v: f"{v:.0f}", max(ft.values()),
          "Number of questions")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    for theme in THEMES.values():
        chart_retrieval(theme)
        chart_answer_quality(theme)
        chart_question_mix(theme)
        chart_failures(theme)
