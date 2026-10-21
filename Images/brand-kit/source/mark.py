"""Gradient Labs mark: a loss landscape of fine tensor cells, with a gradient-descent path to its lowest point."""
import math

import numpy as np


def g(x, y, cx, cy, sx, sy):
    return np.exp(-((x - cx) ** 2 / (2 * sx * sx) + (y - cy) ** 2 / (2 * sy * sy)))


# hills and one deep valley; the descent starts high on the big hill
BUMPS = [(-0.52, 0.48, 0.25, 0.25, 0.95), (0.60, 0.45, 0.21, 0.21, 0.60), (0.20, -0.70, 0.22, 0.19, 0.45),
         (-0.62, -0.40, 0.18, 0.18, 0.30), (0.0, 0.0, 0.42, 0.30, -0.70)]
ZS = 0.70


def height(x, y):
    return ZS * sum(a * g(x, y, cx, cy, sx, sy) for cx, cy, sx, sy, a in BUMPS)


def grad(x, y, e=1e-4):
    return np.array([(height(x + e, y) - height(x - e, y)) / (2 * e),
                     (height(x, y + e) - height(x, y - e)) / (2 * e)])


def descent(start=(-0.47, 0.36), lr=0.035, steps=160):
    p = np.array(start, float)
    out = [p.copy()]
    for _ in range(steps):
        p = p - lr * grad(*p)
        out.append(p.copy())
    return np.array(out)


def view(P, az, el):
    """World (x, y, z) -> screen (sx, sy down) and depth (bigger = farther)."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    c, s = math.cos(az), math.sin(az)
    x, y = c * x - s * y, s * x + c * y
    up = z * math.cos(el) + y * math.sin(el)
    depth = y * math.cos(el) - z * math.sin(el)
    return np.stack([x, -up], -1), depth


def mix(a, b, t):
    a = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    t = max(0.0, min(1.0, t))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def ramp(pal, t):
    return mix(pal[0], pal[1], t * 2) if t < 0.5 else mix(pal[1], pal[2], (t - 0.5) * 2)


LIGHT = np.array([-0.45, -0.55, 0.75])
LIGHT /= np.linalg.norm(LIGHT)


def mark(pal, path_col, rings=22, sectors=110, shrink=0.80, az=0.55, el=0.56, dots=9, path_w=1.0):
    """SVG body of the mark normalised to 1000 wide; returns (body, w, h)."""
    rs = np.linspace(0.0, 1.0, rings + 1)
    ts = np.linspace(0, 2 * np.pi, sectors + 1)
    R, T = np.meshgrid(rs, ts, indexing="ij")
    X, Y = R * np.cos(T), R * np.sin(T)
    P = np.stack([X, Y, height(X, Y)], -1)
    zmin, zmax = P[..., 2].min(), P[..., 2].max()
    polys = []
    for i in range(rings):
        for j in range(sectors):
            q = np.array([P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]])
            nrm = np.cross(q[1] - q[0], q[3] - q[0])
            if np.linalg.norm(nrm) < 1e-9:
                nrm = np.cross(q[2] - q[0], q[3] - q[1])
            nrm /= np.linalg.norm(nrm) + 1e-9
            if nrm[2] < 0:
                nrm = -nrm
            lit = max(0.0, nrm @ LIGHT) ** 1.6
            hgt = ((q[:, 2].mean() - zmin) / (zmax - zmin)) ** 0.8
            t = 0.55 * lit + 0.45 * hgt
            c = q.mean(0)
            # bilinear inset: radial gaps (between rings) thinner than angular ones, so
            # rings seen edge-on in the valley don't turn into thick stripes
            a, b = (1 - shrink * 1.1) / 2, (1 - shrink) / 2
            uv = [(a, b), (1 - a, b), (1 - a, 1 - b), (a, 1 - b)]
            q2 = np.array([(1 - u) * (1 - v) * q[0] + u * (1 - v) * q[1] + u * v * q[2] + (1 - u) * v * q[3]
                           for u, v in uv])
            scr, dep = view(q2, az, el)
            polys.append((scr, dep.mean(), t))
    # descent path, lifted a hair above the surface
    D = descent()
    Dz = np.stack([D[:, 0], D[:, 1], height(D[:, 0], D[:, 1]) + 0.02], -1)
    dscr, _ = view(Dz, az, el)
    pts = np.concatenate([p for p, *_ in polys])
    lo, hi = pts.min(0), pts.max(0)
    k = 1000 / (hi[0] - lo[0])
    polys.sort(key=lambda p: -p[1])
    out = []
    for q, _, t in polys:
        col = ramp(pal, t)
        xy = (q - lo) * k
        out.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in xy)}" fill="{col}" '
                   f'stroke="{col}" stroke-width="1" stroke-linejoin="round"/>')
    # dots: pick steps so they bunch up as the descent slows near the minimum
    idx = np.unique(np.round(np.geomspace(1, len(D), dots) - 1).astype(int))
    d = (dscr - lo) * k
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in d[: idx[-1] + 1])
    out.append(f'<polyline points="{line}" fill="none" stroke="{path_col}" stroke-width="{9 * path_w:.1f}" '
               f'stroke-linecap="round" stroke-linejoin="round" opacity="0.8"/>')
    for m, i in enumerate(idx):
        r = (13 + 8 * m / max(1, len(idx) - 1)) * path_w
        out.append(f'<circle cx="{d[i, 0]:.1f}" cy="{d[i, 1]:.1f}" r="{r:.1f}" fill="{path_col}"/>')
    x, y = d[idx[-1]]
    out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{44 * path_w:.1f}" fill="none" stroke="{path_col}" '
               f'stroke-width="{6 * path_w:.1f}" opacity="0.6"/>')
    return "".join(out), 1000.0, float((hi[1] - lo[1]) * k)
