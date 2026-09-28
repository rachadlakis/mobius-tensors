import os, json, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from gen3 import render, tp, FB, FL, BG, INK, icon_ribbon
import gen2
from inf import mix as _mix, hexc as _hexc
_A, _B = (0x25, 0x5c, 0xe8), (0x14, 0x24, 0x7a)
gen2.col = lambda t, l: _hexc([v * (0.35 + 0.90 * l) for v in _mix(_A, _B, 1 - abs(1 - 2 * t))])
D = "brand"; os.makedirs(D, exist_ok=True)
CY, VI, TXT = "#7ae1ff", "#a78bfa", "#eaf4ff"

def mesh(s, cx, cy): return icon_ribbon(s, cx, cy)

def word(x, y, size, tc, sc):
    # MOBIUS at `size`, TENSORS below; returns svg + width
    d1 = tp(FB, "MOBIUS", x, y, size, 0.04)
    d2 = tp(FL, "TENSORS", x + size * 0.04, y + size * 0.65, size * 0.46, 0.42)
    return '<path d="%s" fill="%s"/><path d="%s" fill="%s"/>' % (d1, tc, d2, sc)

def adv(font, txt, size, track):
    gs=font.getGlyphSet(); cm=font.getBestCmap(); sc=size/font['head'].unitsPerEm
    return sum(gs[cm[ord(c)]].width*sc for c in txt)+track*size*(len(txt)-1)

def svg(w, h, body, bg=None):
    b = '<rect width="%d" height="%d" fill="%s"/>' % (w, h, bg) if bg else ""
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">%s%s</svg>' % (w, h, w, h, b, body)

def backdrop(w, h, ribbon_scale=None, ribbon_cx=None, ribbon_cy=None):
    # navy base + glow + faint dot grid (tensor field)
    g = ('<defs><radialGradient id="gl" cx="0.75" cy="0.45" r="0.6"><stop offset="0" stop-color="#1b2a55" stop-opacity="0.9"/>'
         '<stop offset="1" stop-color="%s" stop-opacity="0"/></radialGradient>'
         '<radialGradient id="gl2" cx="0.15" cy="0.9" r="0.5"><stop offset="0" stop-color="#2a1f5c" stop-opacity="0.6"/><stop offset="1" stop-color="%s" stop-opacity="0"/></radialGradient>'
         '<pattern id="dots" width="%d" height="%d" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="%.1f" fill="#7ae1ff" fill-opacity="0.10"/></pattern></defs>'
         % (BG, BG, max(18, h // 14), max(18, h // 14), max(1.1, h / 400)))
    return g + '<rect width="%d" height="%d" fill="%s"/><rect width="100%%" height="100%%" fill="url(#gl)"/><rect width="100%%" height="100%%" fill="url(#gl2)"/><rect width="100%%" height="100%%" fill="url(#dots)"/>' % (w, h, BG)

A = {}  # name -> (svg, transparent?)

# --- master icons / lockups (vector) ---
A["logo-icon-transparent"] = (svg(1024, 1024, mesh(470, 512, 512)), True)
A["logo-icon-dark-square"] = (svg(1024, 1024, mesh(400, 512, 512), BG), False)
A["logo-icon-light-square"] = (svg(1024, 1024, mesh(400, 512, 512), "#ffffff"), False)
A["logo-horizontal-dark"] = (svg(1400, 480, mesh(215, 260, 240) + word(530, 262, 150, TXT, CY), BG), False)
A["logo-horizontal-light"] = (svg(1400, 480, mesh(215, 260, 240) + word(530, 262, 150, INK, "#2554d8"), "#ffffff"), False)
A["logo-horizontal-transparent-for-dark"] = (svg(1400, 480, mesh(215, 260, 240) + word(530, 262, 150, TXT, CY)), True)
A["logo-horizontal-transparent-for-light"] = (svg(1400, 480, mesh(215, 260, 240) + word(530, 262, 150, INK, "#2554d8")), True)
A["logo-horizontal-tagline-dark"] = (svg(1400, 540, mesh(215, 260, 270) + word(530, 262, 150, TXT, CY) + '<path d="%s" fill="#9fb3c8"/>' % tp(FL, "Endless vision. Infinite progress.", 536, 440, 44, 0.04), BG), False)
A["logo-stacked-dark"] = (svg(1200, 1200, mesh(330, 600, 440) + word(355, 830, 160, TXT, CY), BG), False)

# --- LinkedIn ---
# profile/company logo: square, icon kept inside the circle crop
A["linkedin-logo-400"] = (svg(400, 400, backdrop(400, 400) + mesh(150, 200, 200)), False)
# company page cover 1128x191 (logo overlaps bottom-left on mobile -> keep content centre/right)
def cover(w, h, tagline=True):
    body = backdrop(w, h)
    tall = w / h < 5
    body += mesh(h * (0.46 if tall else 0.62), w * (0.83 if tall else 0.80), h * 0.47)
    ws = h * (0.20 if tall else 0.30)
    body += word(w * (0.40 if tall else 0.34), h * (0.48 if tall else 0.50), ws, TXT, CY)
    if tagline:
        body += '<path d="%s" fill="#9fb3c8"/>' % tp(FL, "Endless vision. Infinite progress.", w * (0.40 if tall else 0.34) + ws * 0.04, h * (0.48 if tall else 0.50) + ws * 1.35, ws * 0.30, 0.03)
    return svg(w, h, body)
A["linkedin-company-cover-1128x191"] = (cover(1128, 191), False)
A["linkedin-personal-banner-1584x396"] = (cover(1584, 396), False)
# post / share images
def post(w, h):
    body = backdrop(w, h) + mesh(min(w, h) * 0.30, w / 2, h * 0.40)
    ws = min(w, h) * 0.11
    wm_w = adv(FB, 'MOBIUS', ws, 0.04)
    body += word(w / 2 - wm_w / 2, h * 0.70, ws, TXT, CY)
    tw = adv(FL, "Endless vision. Infinite progress.", ws * 0.46, 0.04)
    body += '<path d="%s" fill="#9fb3c8"/>' % tp(FL, "Endless vision. Infinite progress.", w / 2 - tw / 2, h * 0.70 + ws * 1.55, ws * 0.46, 0.04)
    return svg(w, h, body)
A["linkedin-post-1200x627"] = (post(1200, 627), False)
A["linkedin-post-square-1080x1080"] = (post(1080, 1080), False)
A["website-og-image-1200x630"] = (post(1200, 630), False)

# --- app / web icons ---
for sz in (180, 192, 512):
    A[f"app-icon-{sz}"] = (svg(sz, sz, '<rect width="%d" height="%d" fill="%s"/>' % (sz, sz, BG) + mesh(sz * 0.40, sz / 2, sz / 2)), False)
A["favicon-source"] = (svg(256, 256, '<rect width="256" height="256" rx="56" fill="%s"/>' % BG + icon_ribbon(118, 128, 128)), True)

json.dump({k: v[1] for k, v in A.items()}, open(D + "/_manifest.json", "w"))
for k, (s, t) in A.items():
    open(f"{D}/{k}.svg", "w").write(s)
print(len(A))
