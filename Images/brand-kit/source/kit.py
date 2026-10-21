"""Gradient Labs brand kit: loss-landscape mark (navy, orange descent path) + outlined Space Grotesk wordmark.

usage: python kit.py <out dir>   (fonts/sg400|500|700.ttf: Space Grotesk static TTFs, OFL)
"""
import sys
from pathlib import Path

import pymupdf
import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image

from mark import mark

HERE = Path(__file__).parent
OUT = Path(sys.argv[1])
NAME = "Gradient Labs"
TAGLINE = "Every step closer to the answer."


# ---------------- wordmark outlines ----------------
def text_path(text, weight, tracking_em=0.0):
    """SVG path d + bounds for text at 1 unit per em; baseline y=0, kerned by HarfBuzz."""
    path = HERE / "fonts" / f"sg{weight}.ttf"
    font = TTFont(path)
    upm = font["head"].unitsPerEm
    face = hb.Face(path.read_bytes())
    hbf = hb.Font(face)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hbf, buf, {"kern": True, "liga": False})
    gs, order = font.getGlyphSet(), font.getGlyphOrder()
    pen = SVGPathPen(gs, ntos=lambda v: f"{v:.4f}".rstrip("0").rstrip("."))
    bp = BoundsPen(gs)
    x = 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        m = (1 / upm, 0, 0, -1 / upm, x + pos.x_offset / upm, 0)
        g = gs[order[info.codepoint]]
        g.draw(TransformPen(pen, m))
        g.draw(TransformPen(bp, m))
        x += pos.x_advance / upm + tracking_em
    return pen.getCommands(), bp.bounds


MAIN, MAIN_B = text_path("GRADIENT", 700, 0.02)
# LABS: smaller, letter-spaced, left-aligned under GRADIENT
TR, TS = 0.55, 0.40
SUB, SUB_B = text_path("LABS", 500, TR)
TAG, TAG_B = text_path(TAGLINE, 400)

# ---------------- the mark ----------------
NAVY = ("#050f3d", "#1336b8", "#4f7bff")      # on light grounds
BRIGHT = ("#0f2a8a", "#2f62ff", "#a9c2ff")    # on dark grounds
WHITE = ("#ffffff", "#ffffff", "#ffffff")
ORANGE, ORANGE_DARK = "#ff7a2e", "#ff9a4d"

MARK = {"navy": mark(NAVY, ORANGE), "bright": mark(BRIGHT, ORANGE_DARK), "white": mark(WHITE, "#ffffff")}
SMALL = dict(rings=11, sectors=56, shrink=0.74, dots=6, path_w=1.9)
ICON = {"navy": mark(NAVY, ORANGE, **SMALL), "bright": mark(BRIGHT, ORANGE_DARK, **SMALL),
        "white": mark(WHITE, "#ffffff", **SMALL)}

INK = {"light": dict(mark="navy", text="#0b1530", sub="#1336b8", tag="#4a5578"),
       "dark": dict(mark="bright", text="#eef2ff", sub="#a9c2ff", tag="#9aa8c7")}
BG_LIGHT, BG_DARK = "#ffffff", "#070b18"


def place_mark(m, x, y, w):
    body, mw, mh = m
    k = w / mw
    return f'<g transform="translate({x:.2f} {y:.2f}) scale({k:.5f})">{body}</g>', mh * k


def word(x, top, cap, ink, tagline=False):
    """GRADIENT / LABS (/ tagline) block; GRADIENT cap height = cap; returns svg, width, height."""
    k = cap / -MAIN_B[1]
    w = (MAIN_B[2] - MAIN_B[0]) * k
    s = (f'<path transform="translate({x - MAIN_B[0] * k:.2f} {top + cap:.2f}) scale({k:.5f})" '
         f'fill="{ink["text"]}" d="{MAIN}"/>')
    k2 = k * TS
    scap = -SUB_B[1] * k2
    sy = top + cap + cap * 0.42 + scap
    s += (f'<path transform="translate({x - SUB_B[0] * k2:.2f} {sy:.2f}) scale({k2:.5f})" '
          f'fill="{ink["sub"]}" d="{SUB}"/>')
    h = sy - top
    if tagline:
        k3 = k * 0.26
        gy = sy + cap * 0.70 + (-TAG_B[1] * k3)
        s += (f'<path transform="translate({x - TAG_B[0] * k3:.2f} {gy:.2f}) scale({k3:.5f})" '
              f'fill="{ink["tag"]}" d="{TAG}"/>')
        h = gy - top + (TAG_B[3] * k3)
        w = max(w, (TAG_B[2] - TAG_B[0]) * k3)
    return s, w, h


def svg(w, h, body, bg=None, rx=0):
    rect = f'<rect width="{w}" height="{h}" rx="{rx}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{NAME}"><title>{NAME}</title>{rect}{body}</svg>\n')


def lockup(W, H, theme, bg, tagline=False, cap=None, mark_w=None, cx=None):
    """Mark left, words right, the pair centred on (cx or W/2, H/2)."""
    ink = INK[theme]
    cap = cap or H * 0.2
    mark_w = mark_w or cap * 4.4
    m = MARK[ink["mark"]]
    mh = m[2] * mark_w / m[1]
    gap = cap * 0.6
    _, ww, wh = word(0, 0, cap, ink, tagline)
    total = mark_w + gap + ww
    x0 = (cx if cx is not None else W / 2) - total / 2
    mk, _ = place_mark(m, x0, H / 2 - mh / 2, mark_w)
    wd, _, _ = word(x0 + mark_w + gap, H / 2 - wh / 2, cap, ink, tagline)
    return svg(W, H, mk + wd, bg)


def stacked(W, theme, bg):
    ink = INK[theme]
    cap = W * 0.085
    m = MARK[ink["mark"]]
    mw = W * 0.66
    mh = m[2] * mw / m[1]
    _, ww, wh = word(0, 0, cap, ink)
    gap = cap * 0.9
    y0 = (W - (mh + gap + wh)) / 2
    mk, _ = place_mark(m, (W - mw) / 2, y0, mw)
    wd, _, _ = word((W - ww) / 2, y0 + mh + gap, cap, ink)
    return svg(W, W, mk + wd, bg)


def icon(S, bg, which, frac=0.84, rx=0, fine=False):
    m = (MARK if fine else ICON)[which]
    mw = S * frac
    mh = m[2] * mw / m[1]
    mk, _ = place_mark(m, (S - mw) / 2, (S - mh) / 2, mw)
    return svg(S, S, mk, bg, rx)


def mark_only(m):
    body, w, h = m
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-30 -30 {w + 60:.0f} {h + 60:.0f}" width="{w + 60:.0f}" '
            f'height="{h + 60:.0f}" role="img" aria-label="{NAME}"><title>{NAME}</title>{body}</svg>\n')


SVGS = {
    "logo-mark-for-light.svg": mark_only(MARK["navy"]),
    "logo-mark-for-dark.svg": mark_only(MARK["bright"]),
    "logo-mark-mono-white.svg": mark_only(MARK["white"]),
    "logo-mark-small-for-dark.svg": mark_only(ICON["bright"]),
    "logo-mark-small-for-light.svg": mark_only(ICON["navy"]),
    "logo-horizontal-light.svg": lockup(1400, 480, "light", BG_LIGHT),
    "logo-horizontal-dark.svg": lockup(1400, 480, "dark", BG_DARK),
    "logo-horizontal-transparent-for-light.svg": lockup(1400, 480, "light", None),
    "logo-horizontal-transparent-for-dark.svg": lockup(1400, 480, "dark", None),
    "logo-horizontal-tagline-dark.svg": lockup(1400, 540, "dark", BG_DARK, tagline=True, cap=480 * 0.2),
    "logo-horizontal-tagline-light.svg": lockup(1400, 540, "light", BG_LIGHT, tagline=True, cap=480 * 0.2),
    "logo-stacked-dark.svg": stacked(1200, "dark", BG_DARK),
    "logo-stacked-light.svg": stacked(1200, "light", BG_LIGHT),
    "logo-icon-dark-square.svg": icon(1024, BG_DARK, "bright"),
    "logo-icon-light-square.svg": icon(1024, BG_LIGHT, "navy"),
    "logo-icon-transparent.svg": icon(1024, None, "navy"),
    "logo-icon-mono-white.svg": icon(1024, None, "white"),
}


def png(src, width, dest, alpha=True):
    d = pymupdf.open(stream=src.encode(), filetype="svg")
    k = width / d[0].rect.width
    dest.parent.mkdir(parents=True, exist_ok=True)
    d[0].get_pixmap(matrix=pymupdf.Matrix(k, k), alpha=alpha).save(dest)


(OUT / "svg").mkdir(parents=True, exist_ok=True)
for name, src in SVGS.items():
    (OUT / "svg" / name).write_text(src, encoding="utf-8")
    w = int(src.split('width="')[1].split('"')[0])
    png(src, w * 2, OUT / "logo" / name.replace(".svg", ".png"))

# web
fav = icon(64, "#0b1a5c", "bright", frac=0.9, rx=14)
(OUT / "web").mkdir(parents=True, exist_ok=True)
(OUT / "web" / "favicon.svg").write_text(fav, encoding="utf-8")
for s in (16, 32, 48):
    png(fav, s, OUT / "web" / f"favicon-{s}.png")
Image.open(OUT / "web" / "favicon-48.png").save(OUT / "web" / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
app = icon(512, "#0b1a5c", "bright", frac=0.82)
for s in (180, 192, 512):
    png(app, s, OUT / "web" / f"app-icon-{s}.png", alpha=False)


def banner(W, H, cap, tagline, cx=None):
    return lockup(W, H, "dark", BG_DARK, tagline=tagline, cap=cap, cx=cx)


SOCIAL = {
    "web/website-og-image-1200x630": (banner(1200, 630, 88, True), 1200),
    "linkedin/linkedin-company-cover-1128x191": (banner(1128, 191, 42, False, cx=1128 * 0.6), 1128),
    "linkedin/linkedin-personal-banner-1584x396": (banner(1584, 396, 66, True, cx=1584 * 0.62), 1584),
    "linkedin/linkedin-post-1200x627": (banner(1200, 627, 88, True), 1200),
    "linkedin/linkedin-post-square-1080x1080": (stacked(1080, "dark", BG_DARK), 1080),
    "linkedin/linkedin-logo-400": (icon(400, BG_LIGHT, "navy", frac=0.92, fine=True), 400),
    "linkedin/linkedin-logo-400-dark": (icon(400, BG_DARK, "bright", frac=0.88, fine=True), 400),
}
for name, (src, w) in SOCIAL.items():
    png(src, w, OUT / f"{name}.png", alpha=False)
    png(src, w * 2, OUT / f"{name}@2x.png", alpha=False)

# preview sheet
tiles = ["logo/logo-horizontal-light.png", "logo/logo-horizontal-dark.png", "logo/logo-horizontal-tagline-dark.png",
         "logo/logo-stacked-light.png", "logo/logo-icon-light-square.png", "logo/logo-icon-dark-square.png",
         "web/app-icon-512.png", "linkedin/linkedin-personal-banner-1584x396.png"]
sheet = Image.new("RGB", (1600, 1500), "#d9dee8")
pos = [(10, 10, 790, 271), (800, 10, 790, 271), (10, 291, 790, 305), (800, 291, 380, 380), (1190, 291, 200, 200),
       (1190, 501, 200, 200), (1400, 291, 190, 190), (10, 720, 1580, 395)]
for t, (x, y, w, h) in zip(tiles, pos):
    im = Image.open(OUT / t).convert("RGBA")
    im.thumbnail((w, h))
    sheet.paste(im, (x, y), im)
fv = Image.open(OUT / "web" / "favicon-32.png").convert("RGBA")
sheet.paste(fv, (10, 1130), fv)
fv = Image.open(OUT / "web" / "favicon-16.png").convert("RGBA")
sheet.paste(fv, (52, 1138), fv)
sheet.save(OUT / "preview.png")
print("done")
