"""Generate README charts straight from the evaluation output.

Every number drawn here is read from `data/eval_results.json` (written by
`python -m src.eval.runner`), `data/eval_breakdown.json` (written by
`scripts/eval_breakdown.py`) or `data/benchmark_questions.json`. Nothing is
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
_BREAKDOWN_PATH = ROOT / "data" / "eval_breakdown.json"  # written by scripts/eval_breakdown.py
BREAKDOWN = json.loads(_BREAKDOWN_PATH.read_text(encoding="utf-8")) if _BREAKDOWN_PATH.exists() else None

# Validated categorical palette (light / dark steps of the same hues).
THEMES = {
    "light": {
        "surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "neutral": "#c3c2b7", "tile": "#f6f6f4",
        "ramp": ["#e8f1fc", "#86b6ef", "#2a78d6", "#0d366b"],
        "suffix": "",
    },
    "dark": {
        "surface": "#0d1117", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "neutral": "#52514e", "tile": "#161b22",
        "ramp": ["#15202e", "#184f95", "#3987e5", "#b7d3f6"],
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


TYPE_NAMES = {
    "factual_lookup": "Look up a fact",
    "cross_period_comparison": "Compare across years",
    "cross_company_comparison": "Compare two companies",
    "risk_summary": "Summarise risks",
    "unanswerable": "Trick question (no answer)",
}


def _luminance(rgb):
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def chart_kpis(t):
    """Headline numbers as a strip of stat tiles."""
    rc, g = RESULTS["retrieval_comparison"]["hybrid"], RESULTS["generation_metrics"]
    tiles = [
        (f"{rc['Recall@5'] * 100:.0f}%", "right passage\nin the top 5"),
        (f"{g['Citation Support Rate'] * 100:.0f}%", "citations that back\nup their sentence"),
        (f"{g['Correct Abstention Rate'] * 100:.0f}%", "trick questions\ncorrectly declined"),
        (f"{g['p50 Latency (s)'] * 1000:.0f} ms", "typical time\nper question"),
        (f"${g['Avg Cost per Question ($)']:.0f}", "cost per question\n(offline mode)"),
    ]
    fig = plt.figure(figsize=(10, 1.9))
    fig.patch.set_facecolor(t["surface"])
    n = len(tiles)
    for i, (value, label) in enumerate(tiles):
        ax = fig.add_axes([i / n + 0.006, 0.04, 1 / n - 0.012, 0.92])
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=t["tile"], zorder=0))
        ax.text(0.5, 0.64, value, ha="center", va="center", fontsize=26, fontweight="bold", color=t["ink"],
                transform=ax.transAxes)
        ax.text(0.5, 0.25, label, ha="center", va="center", fontsize=10, color=t["ink2"], transform=ax.transAxes,
                linespacing=1.3)
    _save(fig, "kpis", t)


def chart_type_heatmap(t):
    """Question type x metric grid, one-hue sequential colour."""
    from matplotlib.colors import LinearSegmentedColormap

    bt = BREAKDOWN["by_question_type"]
    types = [k for k in TYPE_NAMES if k in bt and bt[k]["Recall@5"] is not None]
    metrics = [("Recall@5", "Right passage\nin top 5"), ("MRR", "Ranked\nfirst (MRR)"),
               ("Factual Grounding", "Sentences\nsupported"), ("Citation Coverage", "Sentences\nwith a citation")]
    data = [[bt[k][m] for m, _ in metrics] for k in types]
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    lo, hi = 0.4, 1.0

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    _style(ax, t, grid_axis="both")
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    for r, row in enumerate(data):
        for c, v in enumerate(row):
            ink = "#0b0b0b" if _luminance(cmap((v - lo) / (hi - lo))) > 0.45 else "#ffffff"
            ax.text(c, r, f"{v * 100:.0f}%", ha="center", va="center", color=ink, fontsize=13, fontweight="bold")
    for x in range(len(metrics) + 1):  # surface-coloured gaps between cells
        ax.axvline(x - 0.5, color=t["surface"], linewidth=4)
    for y in range(len(types) + 1):
        ax.axhline(y - 0.5, color=t["surface"], linewidth=4)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], color=t["ink"], fontsize=10.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([f"{TYPE_NAMES[k]}  (n={bt[k]['n']})" for k in types], color=t["ink"], fontsize=10.5)
    # Explicit placement: with ticks on top, set_title would collide with the subtitle.
    ax.text(0, 1.42, "Scores by question type", transform=ax.transAxes, color=t["ink"], fontsize=14,
            fontweight="bold")
    ax.text(0, 1.32, "Comparing two companies is hardest to search; risk summaries are least grounded",
            transform=ax.transAxes, color=t["ink2"], fontsize=10.5)
    ax.text(1.0, -0.05, "darker = higher" if t["suffix"] == "" else "brighter = higher", transform=ax.transAxes, ha="right", va="top", color=t["muted"],
            fontsize=9)
    _save(fig, "scores_by_type", t)


def chart_failures(t):
    """Stacked bars: what went wrong, per question type."""
    bt = BREAKDOWN["by_question_type"]
    # The judge never assigns "extraction_error" (see src/eval/judge.py), so it is left off.
    cats = [("none", "No error", t["neutral"]), ("retrieval_error", "Search missed", t["series"][0]),
            ("generation_error", "Missing citation", t["series"][1]),
            ("reasoning_error", "Weak comparison", t["series"][2])]
    types = [k for k in TYPE_NAMES if k in bt]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    _style(ax, t)
    y = list(range(len(types)))[::-1]
    left = [0] * len(types)
    for key, label, color in cats:
        vals = [bt[k]["failures"].get(key, 0) for k in types]
        ax.barh(y, vals, left=left, height=0.58, color=color, edgecolor=t["surface"], linewidth=2, label=label)
        for yi, l, v in zip(y, left, vals):
            if v:
                ink = "#0b0b0b" if key == "none" and t["suffix"] == "" else "#ffffff"
                ax.text(l + v / 2, yi, str(v), ha="center", va="center", color=ink, fontsize=10.5, fontweight="bold")
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(y)
    ax.set_yticklabels([TYPE_NAMES[k] for k in types], color=t["ink"], fontsize=10.5)
    ax.set_xlabel("Number of questions", color=t["ink2"])
    ax.set_title("What went wrong, by question type", pad=46, color=t["ink"])
    ft = RESULTS["failure_taxonomy"]
    ax.text(0, 1.13, f"Most misses happen after search: {ft['reasoning_error']} weak comparisons, "
            f"{ft['generation_error']} missing citations, {ft['retrieval_error']} search misses",
            transform=ax.transAxes, color=t["ink2"], fontsize=10.5)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=9.5,
                    handlelength=1.2, columnspacing=1.4)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    _save(fig, "failure_breakdown", t)


def chart_judge(t):
    """2x2 grid: automatic grader vs a human on 20 hand-labeled pairs."""
    cm = BREAKDOWN["judge_confusion"]
    agr = RESULTS["judge_human_agreement"]
    grid = [[cm["human_yes_judge_yes"], cm["human_yes_judge_no"]],
            [cm["human_no_judge_yes"], cm["human_no_judge_no"]]]
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    _style(ax, t, grid_axis="both")
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    for r in range(2):
        for c in range(2):
            color = t["series"][2] if r == c else t["series"][1]
            ax.add_patch(plt.Rectangle((c + 0.03, 1 - r + 0.03), 0.94, 0.94, color=color))
            ax.text(c + 0.5, 1 - r + 0.56, str(grid[r][c]), ha="center", va="center", fontsize=30, fontweight="bold",
                    color="#ffffff")
            ax.text(c + 0.5, 1 - r + 0.26, "agree" if r == c else "disagree", ha="center", va="center", fontsize=10.5,
                    color="#ffffff")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Grader:\nsupported", "Grader:\nnot supported"], color=t["ink"], fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["Human:\nsupported", "Human:\nnot supported"], color=t["ink"], fontsize=10)
    ax.set_title("Does the grader agree with a human?", pad=66, color=t["ink"], fontsize=13)
    ax.text(0, 1.17, f"{agr['matches']} of {agr['sample_size']} hand-labeled examples · "
            f"Cohen's kappa {agr['cohens_kappa']:.2f} (moderate)", transform=ax.transAxes, color=t["ink2"], fontsize=10)
    _save(fig, "judge_agreement", t)


def chart_coverage(t):
    """Which companies and filing sections the benchmark questions target."""
    by_co = Counter(tk for q in QUESTIONS for tk in q["target_tickers"])
    by_sec = Counter(s.replace(" Executive & Director Changes", " Exec Changes (8-K)").replace(" Disclosures", "")
                     for q in QUESTIONS for s in q["gold_sections"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.9})
    for ax, counts, title in ((a1, by_co, "Questions per company"), (a2, by_sec, "Sections the answers come from")):
        _style(ax, t)
        items = sorted(counts.items(), key=lambda kv: -kv[1])
        y = list(range(len(items)))[::-1]
        ax.barh(y, [v for _, v in items], height=0.6, color=t["series"][0])
        for yi, (_, v) in zip(y, items):
            ax.text(v + max(counts.values()) * 0.02, yi, str(v), va="center", color=t["ink"], fontsize=10,
                    fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([k for k, _ in items], color=t["ink"], fontsize=10)
        ax.set_xlim(0, max(counts.values()) * 1.15)
        ax.set_title(title, pad=12, color=t["ink"], fontsize=12.5)
    _save(fig, "benchmark_coverage", t)


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    for theme in THEMES.values():
        chart_kpis(theme)
        chart_retrieval(theme)
        chart_answer_quality(theme)
        chart_question_mix(theme)
        chart_coverage(theme)
        if BREAKDOWN:
            chart_type_heatmap(theme)
            chart_failures(theme)
            chart_judge(theme)
        else:
            print("skipped breakdown charts: run `python scripts/eval_breakdown.py` first")
