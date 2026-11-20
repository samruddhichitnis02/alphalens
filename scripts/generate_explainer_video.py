"""Render the ~45 second plain-English explainer for the README.

Outputs
  assets/explainer.gif  - plays inline on GitHub
  assets/explainer.mp4  - sharper version (needs `pip install imageio imageio-ffmpeg`)

The result numbers shown at the end are read from data/eval_results.json, so
the video stays in sync with the latest `make eval` run.

Usage:  python scripts/generate_explainer_video.py
"""
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RESULTS = json.loads((ROOT / "data" / "eval_results.json").read_text(encoding="utf-8"))

W, H = 960, 540
FPS = 10

BG = (13, 17, 23)
CARD = (22, 27, 34)
LINE = (48, 54, 61)
INK = (240, 239, 236)
INK2 = (170, 176, 184)
BLUE = (57, 135, 229)
ORANGE = (235, 104, 52)
AQUA = (27, 175, 122)
GREEN = (63, 185, 80)


# --------------------------------------------------------------------- fonts
def _font_path(bold=False, mono=False):
    win = Path("C:/Windows/Fonts")
    name = "consola.ttf" if mono else ("segoeuib.ttf" if bold else "segoeui.ttf")
    if (win / name).exists():
        return str(win / name)
    from matplotlib import font_manager  # fallback: DejaVu ships with matplotlib

    family = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    return font_manager.findfont(font_manager.FontProperties(family=family, weight="bold" if bold else "normal"))


_cache = {}


def font(size, bold=False, mono=False):
    key = (size, bold, mono)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(_font_path(bold, mono), size)
    return _cache[key]


# ------------------------------------------------------------------- helpers
def ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def fade(t, start, dur=0.5):
    return ease((t - start) / dur)


def mix(c, a, bg=BG):
    return tuple(int(bg[i] + (c[i] - bg[i]) * a) for i in range(3))


def text(d, xy, s, f, color, a=1.0, anchor="la", bg=BG):
    if a <= 0:
        return
    d.text(xy, s, font=f, fill=mix(color, a, bg), anchor=anchor)


def wrap(s, f, max_w):
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if f.getlength(trial) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def para(d, xy, s, f, color, max_w, a=1.0, gap=8, anchor_center=False, bg=BG):
    x, y = xy
    for line in wrap(s, f, max_w):
        if anchor_center:
            text(d, (x, y), line, f, color, a, anchor="ma", bg=bg)
        else:
            text(d, (x, y), line, f, color, a, bg=bg)
        y += f.size + gap
    return y


def card(d, box, a=1.0, outline=LINE, fill=CARD, width=2):
    d.rounded_rectangle(box, radius=14, fill=mix(fill, a), outline=mix(outline, a), width=width)


def check(d, cx, cy, size, color, a=1.0):
    pts = [(cx - size * 0.5, cy), (cx - size * 0.15, cy + size * 0.35), (cx + size * 0.5, cy - size * 0.4)]
    d.line(pts, fill=mix(color, a), width=max(3, int(size / 5)), joint="curve")


def progress(d, i, n):
    """Small scene-progress dots at the bottom so viewers know where they are."""
    x0 = W / 2 - (n - 1) * 9
    for k in range(n):
        r = 4
        c = INK if k == i else LINE
        d.ellipse((x0 + k * 18 - r, H - 24 - r, x0 + k * 18 + r, H - 24 + r), fill=c)


def header(d, t, kicker, title):
    text(d, (60, 44), kicker.upper(), font(15, bold=True), BLUE, fade(t, 0))
    text(d, (60, 68), title, font(34, bold=True), INK, fade(t, 0.1))


# -------------------------------------------------------------------- scenes
def scene_title(d, t):
    a = fade(t, 0.1, 0.8)
    text(d, (W / 2, 170), "AlphaLens", font(78, bold=True), INK, a, anchor="ma")
    text(d, (W / 2, 275), "Ask questions about company reports.", font(30), INK2, fade(t, 0.8), anchor="ma")
    text(d, (W / 2, 318), "Get answers with proof.", font(30, bold=True), AQUA, fade(t, 1.4), anchor="ma")
    w = 200 * ease((t - 1.8) / 0.8)
    if w > 0:
        d.rounded_rectangle((W / 2 - w / 2, 385, W / 2 + w / 2, 389), radius=2, fill=BLUE)


def scene_problem(d, t):
    header(d, t, "The problem", "Company reports are huge")
    docs = [("10-K", "Yearly report", BLUE), ("10-Q", "Every quarter", ORANGE), ("8-K", "Big news", AQUA)]
    for i, (code, label, color) in enumerate(docs):
        a = fade(t, 0.5 + i * 0.35)
        if a <= 0:
            continue
        x = 140 + i * 240
        rise = (1 - a) * 25
        for k in range(3, 0, -1):  # a little stack of pages
            off = k * 7
            card(d, (x + off, 150 + off + rise, x + 150 + off, 330 + off + rise), a * 0.6)
        card(d, (x, 150 + rise, x + 150, 330 + rise), a, outline=color)
        text(d, (x + 75, 175 + rise), code, font(30, bold=True), color, a, anchor="ma", bg=CARD)
        for ln in range(5):
            lw = 100 if ln % 2 == 0 else 80
            d.rounded_rectangle((x + 25, 228 + rise + ln * 18, x + 25 + lw, 234 + rise + ln * 18), radius=3,
                                fill=mix(LINE, a, CARD))
        text(d, (x + 75, 360), label, font(20), INK2, a, anchor="ma")
    para(d, (W / 2, 412), "Every public company in the US files these with the SEC. Each one can be "
         "hundreds of pages. Reading them by hand takes days.", font(22), INK, 760, fade(t, 2.0), anchor_center=True)


STEPS = [
    ("Download", BLUE, "Downloads the filings of 10 big companies straight from the SEC's free EDGAR website."),
    ("Split", ORANGE, "Cuts each filing into its sections (like Risk Factors) and then into small, readable pieces."),
    ("Search", AQUA, "When you ask a question, it finds the pieces that match - by exact words AND by meaning."),
    ("Answer", BLUE, "It writes an answer where every sentence points to the exact piece it came from."),
    ("Check", GREEN, "It double-checks every pointer. If the reports don't say it, it answers \"not enough info\"."),
]
STEP_SECS = 3.6


def scene_pipeline(d, t):
    header(d, t, "How it works", "Five simple steps")
    active = min(int(max(t - 0.6, 0) / STEP_SECS), len(STEPS) - 1)
    bw, gap, y0 = 150, 32, 175
    x0 = (W - (len(STEPS) * bw + (len(STEPS) - 1) * gap)) / 2
    for i, (name, color, _) in enumerate(STEPS):
        a = fade(t, 0.2 + i * 0.12)
        x = x0 + i * (bw + gap)
        on = i == active
        done = i < active
        card(d, (x, y0, x + bw, y0 + 96), a, outline=color if (on or done) else LINE, width=3 if on else 2,
             fill=mix(color, 0.18, CARD) if on else CARD)
        text(d, (x + bw / 2, y0 + 16), str(i + 1), font(22, bold=True), color if (on or done) else INK2, a, anchor="ma",
             bg=CARD)
        text(d, (x + bw / 2, y0 + 52), name, font(22, bold=True), INK if (on or done) else INK2, a, anchor="ma",
             bg=CARD)
        if i < len(STEPS) - 1:
            ax = x + bw + 6
            d.line((ax, y0 + 48, ax + gap - 12, y0 + 48), fill=mix(INK2, a), width=2)
            d.polygon([(ax + gap - 12, y0 + 42), (ax + gap - 4, y0 + 48), (ax + gap - 12, y0 + 54)], fill=mix(INK2, a))
    # caption for the active step
    local = (t - 0.6) - active * STEP_SECS
    ca = fade(local, 0, 0.4)
    name, color, caption = STEPS[active]
    card(d, (100, 320, W - 100, 450), ca)
    text(d, (130, 340), f"Step {active + 1}: {name}", font(22, bold=True), color, ca, bg=CARD)
    para(d, (130, 378), caption, font(22), INK, W - 260, ca, bg=CARD)


def scene_example(d, t):
    header(d, t, "Example", "You ask, it answers - with sources")
    q = "How did Apple's Services revenue change in 2023?"
    typed = q[: int(max(t - 0.6, 0) * 28)]
    card(d, (80, 140, W - 80, 200), fade(t, 0.3))
    text(d, (104, 156), "You:", font(20, bold=True), BLUE, fade(t, 0.3), bg=CARD)
    text(d, (164, 156), typed + ("|" if len(typed) < len(q) and int(t * 4) % 2 == 0 else ""), font(20), INK,
         fade(t, 0.3), bg=CARD)

    a = fade(t, 2.6)
    if a > 0:
        card(d, (80, 220, W - 80, 340), a)
        text(d, (104, 236), "AlphaLens:", font(20, bold=True), AQUA, a, bg=CARD)
        text(d, (104, 270), "Services sales hit a record $85.2 billion, up 9%,", font(21), INK, a, bg=CARD)
        text(d, (104, 300), "driven by ads, cloud and the App Store", font(21), INK, a, bg=CARD)
        tag = "[AAPL 10-K · Item 7]"
        tx = 104 + font(21).getlength("driven by ads, cloud and the App Store ")
        text(d, (tx, 303), tag, font(19, mono=True), ORANGE, a, bg=CARD)
    b = fade(t, 3.8)
    if b > 0:
        check(d, 110, 372, 22, GREEN, b)
        text(d, (134, 360), "Source checked: that sentence really is in Apple's annual report.", font(19), INK2, b)

    c = fade(t, 5.4)
    if c > 0:
        card(d, (80, 405, W - 80, 470), c)
        text(d, (104, 416), "You:", font(18, bold=True), BLUE, c, bg=CARD)
        text(d, (160, 416), "What will Apple's stock price be next year?", font(18), INK, c, bg=CARD)
        e = fade(t, 6.4)
        text(d, (104, 442), "AlphaLens:", font(18, bold=True), AQUA, e, bg=CARD)
        text(d, (204, 442), "The filings don't contain enough information to answer that.", font(18), INK, e, bg=CARD)


def scene_results(d, t):
    header(d, t, "Does it work?", "Tested on 59 hand-checked questions")
    rc = RESULTS["retrieval_comparison"]["hybrid"]
    gm = RESULTS["generation_metrics"]
    stats = [
        (rc["Recall@5"] * 100, "of answerable questions:", "right passage in the top 5", BLUE),
        (gm["Citation Support Rate"] * 100, "of citations really back", "up their sentence", AQUA),
        (gm["Correct Abstention Rate"] * 100, "of trick questions were", "correctly declined", GREEN),
    ]
    cw, gap = 250, 30
    x0 = (W - (3 * cw + 2 * gap)) / 2
    for i, (val, l1, l2, color) in enumerate(stats):
        a = fade(t, 0.4 + i * 0.4)
        x = x0 + i * (cw + gap)
        card(d, (x, 160, x + cw, 380), a)
        shown = val * ease((t - 0.4 - i * 0.4) / 1.4)
        text(d, (x + cw / 2, 185), f"{shown:.0f}%", font(64, bold=True), color, a, anchor="ma", bg=CARD)
        text(d, (x + cw / 2, 290), l1, font(19), INK, a, anchor="ma", bg=CARD)
        text(d, (x + cw / 2, 318), l2, font(19), INK, a, anchor="ma", bg=CARD)
    para(d, (W / 2, 418), "Measured on a small practice set of filing text, with a free offline "
         "answer writer. Full numbers are in the README.", font(18), INK2, 760, fade(t, 2.2), anchor_center=True)


def scene_outro(d, t):
    text(d, (W / 2, 150), "AlphaLens", font(56, bold=True), INK, fade(t, 0.1), anchor="ma")
    text(d, (W / 2, 232), "Read less. Ask more. Always see the source.", font(26), AQUA, fade(t, 0.6), anchor="ma")
    a = fade(t, 1.2)
    card(d, (W / 2 - 210, 300, W / 2 + 210, 410), a)
    for i, cmd in enumerate(["make test", "make eval", "make run"]):
        text(d, (W / 2 - 180, 314 + i * 30), "$", font(20, mono=True), INK2, a, bg=CARD)
        text(d, (W / 2 - 154, 314 + i * 30), cmd, font(20, mono=True), INK, a, bg=CARD)
    text(d, (W / 2, 432), "Runs on your laptop - no paid API key needed", font(18), INK2, fade(t, 1.6), anchor="ma")


SCENES = [
    (scene_title, 4.0),
    (scene_problem, 6.5),
    (scene_pipeline, 0.6 + STEP_SECS * len(STEPS) + 0.4),
    (scene_example, 9.5),
    (scene_results, 7.0),
    (scene_outro, 4.5),
]


def render_frames():
    frames = []
    for si, (fn, dur) in enumerate(SCENES):
        n = int(dur * FPS)
        for f in range(n):
            t = f / FPS
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            fn(d, t)
            progress(d, si, len(SCENES))
            # quick fade-to-black between scenes
            out = min(1.0, (dur - t) / 0.3)
            if out < 1:
                img = Image.blend(Image.new("RGB", (W, H), BG), img, max(out, 0))
            frames.append(img)
    return frames


def main():
    ASSETS.mkdir(exist_ok=True)
    frames = render_frames()
    print(f"rendered {len(frames)} frames ({len(frames) / FPS:.0f}s)")

    gif_frames = [f.resize((800, 450), Image.LANCZOS).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
                  for f in frames]
    gif_path = ASSETS / "explainer.gif"
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=int(1000 / FPS), loop=0,
                       optimize=True)
    print(f"saved {gif_path.relative_to(ROOT)} ({gif_path.stat().st_size / 1e6:.1f} MB)")

    try:
        import imageio.v2 as imageio
        import numpy as np

        mp4_path = ASSETS / "explainer.mp4"
        with imageio.get_writer(mp4_path, fps=FPS * 3, codec="libx264", quality=8, pixelformat="yuv420p",
                                macro_block_size=4) as w:
            for f in frames:
                arr = np.asarray(f)
                for _ in range(3):  # 30 fps container, smoother in players
                    w.append_data(arr)
        print(f"saved {mp4_path.relative_to(ROOT)} ({mp4_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:  # imageio-ffmpeg not installed
        print(f"skipped mp4 ({exc}); install with: pip install imageio imageio-ffmpeg")


if __name__ == "__main__":
    main()
