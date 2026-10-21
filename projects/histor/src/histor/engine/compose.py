"""Composed real-case clips for the scenario bench (scenarios.py): drawn layers over a real
video, rendered at run time, so the bench needs no extra footage in git.

A recipe (a spec's `compose`) is a base clip and layers, each with `from`/`to` in seconds of the
composed clip and its own `seed` (default: its place in the list), so every run gives the same
frames:

    {"base": "data/vtest.avi", "start": 0, "seconds": 40,
     "layers": [{"kind": "item", "at": [400, 420], "size": [70, 50], "from": 20}]}

Points and boxes may be pixels or 0..1 of the frame (like zones). Layer kinds:

    flame        a flame cut from a labelled fire picture (soft alpha), flickering: a smooth
                 random warp of 3-6 px, brightness 0.85-1.15 and size 0.9-1.1 every frame
    poster       the same flame, still (a picture of a fire)
    smoke        a grey plume (alpha 0.3-0.6) rising 5-15 px/s and growing
    item         a cardboard box, a parcel or a ladder that appears (or is carried in) and stays
    slide        a van sliding from box `a` to box `b` over `seconds`, then parked
    cover        a hand over the lens
    shift        the camera turned by `share` of the width (1/16)
    fast         a white cart crossing the picture in `seconds` (0.8)
    warm_mover   red/orange cloth moving along `path` (a fire look-alike)
    lamp         a bright warm disc with bloom, still
    turn_signal  an orange light blinking at `hz` (1.5)
    person       a person (cut once from data/basketball2.png) walking along `path`, falling
                 at `fall_at` over `fall_seconds` and lying there
    light        the whole picture dimming to `gain` from `from` to `to` (dusk)
    shrink       the whole picture `factor` times smaller in a black frame (a far camera)

Night variants are frame transforms applied after composing (and to plain clips): colour,
grey, low, ir, dark (VARIANTS). Flicker is shared with training/fire_smoke/make_test_video.py.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
SPRITES = PROJECT / "data" / "sprites"
FIRE_PICTURES = PROJECT / "training" / "fire_smoke" / "data" / "v2_test" / "all"
PERSON_SOURCE = PROJECT / "data" / "basketball2.png"
VARIANTS = ("colour", "grey", "low", "ir", "dark")
WARM = ("red", "orange", "yellow", "white")  # pose.colour_names: what flames look like
FLAME = "indoor_4b0809db8d8d"  # flames on a dark grate: the warmest of the large fire boxes


# ------------------------------------------------------------------ shared pieces

class Flicker:
    """What flames (or smoke) do to their patch of the picture, frame after frame: a smooth
    random warp of `amp` pixels and a brightness change of `gain`. `smooth` 0 draws a new warp
    every frame (flames); towards 1 it drifts slowly (smoke). A 4-channel patch (BGR + alpha)
    gets the warp on its alpha too, the brightness on BGR only."""

    def __init__(self, rng: np.random.Generator, amp: tuple[float, float] = (3.0, 6.0),
                 gain: tuple[float, float] = (0.85, 1.15), smooth: float = 0.0) -> None:
        self.rng, self.amp, self.gain, self.smooth = rng, amp, gain, smooth
        self._field: np.ndarray | None = None  # (2, 4, 4) coarse warp, -1..1

    def __call__(self, patch: np.ndarray) -> np.ndarray:
        h, w = patch.shape[:2]
        new = self.rng.uniform(-1, 1, (2, 4, 4))
        self._field = new if self._field is None else (
            self.smooth * self._field + (1 - self.smooth) * new)
        amp = self.rng.uniform(*self.amp)
        dx, dy = (cv2.resize(f.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * amp
                  for f in self._field)
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        out = cv2.remap(patch, xs + dx, ys + dy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        gain = self.rng.uniform(*self.gain)
        out[..., :3] = np.clip(out[..., :3].astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return out


def _px(values: list[float], size: tuple[int, int]) -> list[int]:
    """Pixels, or 0..1 of the frame (all values <= 1) scaled to `size` (w, h)."""
    w, h = size
    if max(abs(v) for v in values) <= 1:
        return [int(round(v * (w if i % 2 == 0 else h))) for i, v in enumerate(values)]
    return [int(round(v)) for v in values]


def paste(frame: np.ndarray, sprite: np.ndarray, alpha: np.ndarray, x: int, y: int
          ) -> np.ndarray:
    """Blend a BGR `sprite` with `alpha` (0..1) into `frame` at top-left (x, y), clipped to the
    frame. Returns the alpha as placed in the frame (frame-sized)."""
    h, w = frame.shape[:2]
    placed = np.zeros((h, w), np.float32)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sprite.shape[1], w), min(y + sprite.shape[0], h)
    if x1 <= x0 or y1 <= y0:
        return placed
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x, None].astype(np.float32)
    s = sprite[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    region = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(region * (1 - a) + s * a, 0, 255).astype(np.uint8)
    placed[y0:y1, x0:x1] = a[..., 0]
    return placed


def yolo_boxes(label: Path, size: tuple[int, int]) -> list[tuple[int, list[int]]]:
    """A YOLO label file -> (class, [x1, y1, x2, y2] pixels) for an image of `size` (w, h)."""
    w, h = size
    v = [float(x) for x in label.read_text().split()]
    boxes = []
    for i in range(0, len(v) - 4, 5):
        c, cx, cy, bw, bh = v[i:i + 5]
        boxes.append((int(c), [int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                               int((cx + bw / 2) * w), int((cy + bh / 2) * h)]))
    return boxes


@functools.lru_cache(maxsize=8)
def flame_sprite(stem: str = FLAME, pictures: Path = FIRE_PICTURES
                 ) -> tuple[np.ndarray, np.ndarray]:
    """A flame cut from a held-out fire picture: BGR and a soft alpha (the warm and bright
    pixels inside its largest labelled fire box, feathered 5 px), so its edge is the flame's,
    not a rectangle."""
    from histor.engine.pose import COLOURS, colour_names

    labels = pictures / "labels" / "test"
    image = cv2.imread(str(pictures / "images" / "test" / f"{stem}.jpg"))
    if image is None:
        raise FileNotFoundError(pictures / "images" / "test" / f"{stem}.jpg")
    h, w = image.shape[:2]
    _, (x1, y1, x2, y2) = max(((c, b) for c, b in yolo_boxes(labels / f"{stem}.txt", (w, h))
                               if c == 0), key=lambda cb: (cb[1][2] - cb[1][0]) * (cb[1][3] - cb[1][1]))
    crop = image[max(y1, 0):y2, max(x1, 0):x2]
    names = colour_names(crop.reshape(-1, 3)).reshape(crop.shape[:2])
    value = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 2]
    warm = np.isin(names, [COLOURS.index(n) for n in WARM]) | (value >= 220)
    mask = cv2.morphologyEx(warm.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    pad = 8  # room for the feather and the warp
    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0)  # feathered ~5 px
    return crop, alpha


def person_sprite() -> tuple[np.ndarray, np.ndarray]:
    """The left player of data/basketball2.png cut out with GrabCut (BGR + alpha), cached in
    data/sprites/person.png."""
    cached = SPRITES / "person.png"
    rgba = cv2.imread(str(cached), cv2.IMREAD_UNCHANGED) if cached.exists() else None
    if rgba is not None:
        return rgba[..., :3].copy(), rgba[..., 3].astype(np.float32) / 255
    image = cv2.imread(str(PERSON_SOURCE))
    if image is None:
        raise FileNotFoundError(PERSON_SOURCE)
    # a grey picture: GrabCut needs hints (head, torso, legs sure; the door edge behind not)
    mask = np.full(image.shape[:2], cv2.GC_BGD, np.uint8)
    mask[72:478, 28:175] = cv2.GC_PR_BGD
    mask[150:290, 50:125] = mask[290:465, 45:135] = cv2.GC_PR_FGD
    for x1, y1, x2, y2 in ((65, 160, 110, 270), (70, 300, 95, 440), (105, 300, 125, 440)):
        mask[y1:y2, x1:x2] = cv2.GC_FGD
    cv2.circle(mask, (82, 108), 16, cv2.GC_FGD, -1)
    mask[85:145, 103:122] = cv2.GC_BGD
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, (0, 0, 0, 0), bg, fg, 8, cv2.GC_INIT_WITH_MASK)  # rect unused
    person: np.ndarray = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    person = cv2.morphologyEx(person, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(person)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    alpha = cv2.GaussianBlur(person[box].astype(np.float32), (5, 5), 0)
    SPRITES.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cached), np.dstack([image[box], (alpha * 255).astype(np.uint8)]))
    return image[box].copy(), alpha


def _scaled(sprite: np.ndarray, alpha: np.ndarray, height: float
            ) -> tuple[np.ndarray, np.ndarray]:
    k = max(height, 2) / sprite.shape[0]
    size = (max(int(sprite.shape[1] * k), 1), max(int(height), 2))
    return (cv2.resize(sprite, size, interpolation=cv2.INTER_AREA),
            cv2.resize(alpha, size, interpolation=cv2.INTER_AREA))


def _rotated(sprite: np.ndarray, alpha: np.ndarray, degrees: float
             ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Turned about its feet (bottom centre), on a canvas that holds it all; and where the
    feet are on that canvas."""
    h, w = sprite.shape[:2]
    side = 2 * int(np.hypot(h, w)) + 2
    m = cv2.getRotationMatrix2D((w / 2, h), degrees, 1.0)
    m[:, 2] += (side / 2 - w / 2, side / 2 - h)
    rgba = np.dstack([sprite, (alpha * 255).astype(np.uint8)])
    out = cv2.warpAffine(rgba, m, (side, side), flags=cv2.INTER_LINEAR)
    a = out[..., 3].astype(np.float32) / 255
    ys, xs = np.nonzero(a > 0.02)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return out[box][..., :3], a[box], (side / 2 - xs.min(), side / 2 - ys.min())


# ------------------------------------------------------------------ layers

class Layer:
    def __init__(self, spec: dict[str, Any], rng: np.random.Generator) -> None:
        self.spec, self.rng = spec, rng
        self.start = float(spec.get("from", 0.0))
        end = spec.get("to")
        self.end = float("inf") if end is None else float(end)

    def active(self, t: float) -> bool:
        return self.start <= t < self.end

    def apply(self, frame: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray | None]:
        """The frame with this layer drawn, and what it covers (alpha 0..1, None: all)."""
        raise NotImplementedError


class Flame(Layer):
    still = False

    def __init__(self, spec: dict[str, Any], rng: np.random.Generator) -> None:
        super().__init__(spec, rng)
        sprite, alpha = flame_sprite(spec.get("source", FLAME))
        self.sprite, self.alpha = _scaled(sprite, alpha, spec.get("height", 120))
        self.flicker = None if self.still else Flicker(rng)

    def apply(self, frame, t):
        sprite, alpha = self.sprite, self.alpha
        if self.flicker is not None:
            rgba = self.flicker(np.dstack([sprite, (alpha * 255).astype(np.uint8)]))
            k = self.rng.uniform(0.9, 1.1)
            rgba = cv2.resize(rgba, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
            sprite, alpha = rgba[..., :3], rgba[..., 3].astype(np.float32) / 255
        x, y = _px(self.spec["at"], frame.shape[1::-1])  # the flame's base (bottom centre)
        return frame, paste(frame, sprite, alpha, x - sprite.shape[1] // 2, y - sprite.shape[0])


class Poster(Flame):
    still = True


class Smoke(Layer):
    def __init__(self, spec, rng):
        super().__init__(spec, rng)
        self.rise = rng.uniform(5, 15)
        self.opacity = rng.uniform(0.3, 0.6)
        self.width = spec.get("width", 60)
        self.jitter = rng.uniform(-0.3, 0.3, 200)

    def apply(self, frame, t):
        h, w = frame.shape[:2]
        x, y = _px(self.spec["at"], (w, h))  # where it comes from
        mask = np.zeros((h, w), np.float32)
        age = t - self.start
        for k in range(min(int(age / 0.5) + 1, len(self.jitter))):  # a puff every 0.5 s
            puff_age = age - k * 0.5
            r = self.width / 2 * (1 + 0.15 * puff_age)
            cx = x + self.jitter[k] * self.width + 2 * puff_age * self.jitter[-k - 1]
            cy = y - self.rise * puff_age
            cv2.circle(mask, (int(cx), int(cy)), int(r), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), self.width / 6) * self.opacity
        grey = np.full_like(frame, 200)
        out = (frame * (1 - mask[..., None]) + grey * mask[..., None]).astype(np.uint8)
        return out, mask


def _item_sprite(kind: str, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """A cardboard box, a parcel or a ladder, drawn with shading."""
    image = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.float32)
    if kind == "ladder":
        rail = max(w // 8, 2)
        for x in (0, w - rail):
            image[:, x:x + rail] = (175, 180, 185)
            image[:, x + rail - 1:x + rail] = (120, 125, 130)  # shade
            alpha[:, x:x + rail] = 1
        for y in range(h // 10, h, max(h // 6, 4)):
            cv2.rectangle(image, (0, y), (w - 1, y + max(h // 40, 2)), (165, 170, 175), -1)
            alpha[y:y + max(h // 40, 2) + 1, :] = 1
        return image, alpha
    top = h // 4
    image[top:] = (55, 105, 160)                      # the front: brown cardboard
    image[top:, w * 3 // 4:] = (45, 85, 135)          # the side in shade
    image[:top] = (85, 140, 190)                      # the top, lit
    tape = (205, 205, 195)
    image[:, w // 2 - max(w // 16, 1):w // 2 + max(w // 16, 1)] = tape
    if kind == "parcel":
        cv2.rectangle(image, (w // 8, top + h // 6), (w // 8 + w // 4, top + h // 3),
                      (240, 240, 240), -1)        # the address label
    cv2.rectangle(image, (0, 0), (w - 1, h - 1), (30, 60, 95), 1)
    cv2.line(image, (0, top), (w - 1, top), (40, 75, 115), 1)
    alpha[:] = 1
    return image, alpha


class Item(Layer):
    def __init__(self, spec, rng):
        super().__init__(spec, rng)
        self.carry = 2.0 if spec.get("carried") else 0.0
        self.start -= self.carry  # carried in: on screen 2 s before it is put down

    def apply(self, frame, t):
        size = frame.shape[1::-1]
        x, y = _px(self.spec["at"], size)            # top-left where it is left
        w, h = _px(self.spec.get("size", [70, 50]), size)
        sprite, alpha = _item_sprite(self.spec.get("item", "box"), w, h)
        put = self.start + self.carry
        if t < put:  # being carried in from the left by a dark figure
            x -= int(150 * (put - t) / self.carry)
        mask = paste(frame, sprite, alpha, x, y)
        if self.carry and t < put + 2.0:  # the carrier, walking on and out after putting it down
            walker = x + w + int(max(0.0, t - put) * 120)
            body = np.full((int(h * 2.6), max(w // 2, 12), 3), 35, np.uint8)
            mask = np.maximum(mask, paste(frame, body, np.ones(body.shape[:2], np.float32),
                                          walker, y + h - body.shape[0]))
        return frame, mask


class Slide(Layer):
    def apply(self, frame, t):
        size = frame.shape[1::-1]
        a = np.array(_px(self.spec["a"], size), float)
        b = np.array(_px(self.spec["b"], size), float)
        k = min(1.0, (t - self.start) / max(self.spec.get("seconds", 3.0), 1e-6))
        x1, y1, x2, y2 = (a + (b - a) * k).astype(int)
        w, h = max(x2 - x1, 2), max(y2 - y1, 2)
        van = np.empty((h, w, 3), np.uint8)
        van[:] = (60, 70, 150)
        cv2.rectangle(van, (10, h // 3), (w - 10, h // 3 + max(h // 12, 4)), (235, 235, 235), -1)
        for wx in (20, w // 2):
            cv2.rectangle(van, (wx, h // 10), (wx + w // 6, h // 4), (20, 20, 20), -1)
        noise = self.rng.normal(0, 3, van.shape)  # the sensor's noise is over the van too
        van = np.clip(van + noise, 0, 255).astype(np.uint8)
        return frame, paste(frame, van, np.ones((h, w), np.float32), x1, y1)


class Cover(Layer):
    def apply(self, frame, t):
        hand = np.clip(25 + self.rng.normal(0, 2, frame.shape), 0, 255).astype(np.uint8)
        return hand, None


class Shift(Layer):
    def apply(self, frame, t):
        h, w = frame.shape[:2]
        dx, dy = int(w * self.spec.get("share", 1 / 16)), self.spec.get("dy", 4)
        out = cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                             borderMode=cv2.BORDER_REPLICATE)
        return out, None


class Fast(Layer):
    def apply(self, frame, t):
        h, w = frame.shape[:2]
        seconds = self.spec.get("seconds", 0.8)
        if t - self.start > seconds:
            return frame, np.zeros((h, w), np.float32)
        cw, ch = self.spec.get("size", [60, 40])
        y = _px([0, self.spec.get("y", 0.6)], (w, h))[1]
        x = int(-cw + (w + cw) * (t - self.start) / seconds)
        cart = np.full((ch, cw, 3), 245, np.uint8)
        for wx in (cw // 5, cw * 4 // 5):
            cv2.circle(cart, (wx, ch - 4), 4, (30, 30, 30), -1)
        return frame, paste(frame, cart, np.ones((ch, cw), np.float32), x, y)


def _along(path: list[list[float]], k: float, size: tuple[int, int]) -> tuple[int, int]:
    (ax, ay), (bx, by) = (_px(p, size) for p in path)
    k = min(max(k, 0.0), 1.0)
    return int(ax + (bx - ax) * k), int(ay + (by - ay) * k)


class WarmMover(Layer):
    def apply(self, frame, t):
        size = frame.shape[1::-1]
        span = (self.end - self.start) if np.isfinite(self.end) else 10.0
        x, y = _along(self.spec["path"], (t - self.start) / span, size)
        w, h = self.spec.get("size", [40, 70])
        cloth = np.empty((h, w, 3), np.uint8)
        cloth[:] = (40, 50, 200)                                   # red coat
        sway = int(3 * np.sin(t * 6))
        for i in range(1, 4):                                      # folds, catching the light
            fx = i * w // 4 + sway
            cv2.line(cloth, (fx, 0), (fx - sway, h), (30, 110, 235), max(w // 10, 2))
        alpha = np.zeros((h, w), np.float32)
        cv2.fillConvexPoly(alpha, np.array([[w // 4, 0], [w * 3 // 4, 0], [w - 1, h - 1],
                                            [0, h - 1]], np.int32), 1.0)
        return frame, paste(frame, cloth, cv2.GaussianBlur(alpha, (5, 5), 0), x - w // 2, y - h)


def _glow(frame: np.ndarray, centre: tuple[int, int], radius: int,
          core: tuple[int, int, int], halo: tuple[int, int, int]) -> np.ndarray:
    """A light: a `core` disc in a `halo` of bloom (drawn around it only)."""
    side = 8 * radius + 1
    disc = np.zeros((side, side), np.float32)
    cv2.circle(disc, (side // 2, side // 2), radius, 1.0, -1)
    bloom = cv2.GaussianBlur(disc, (0, 0), radius) * 0.6
    x, y = centre[0] - side // 2, centre[1] - side // 2
    placed = paste(frame, np.full((side, side, 3), halo, np.uint8), bloom, x, y)
    return np.maximum(placed, paste(frame, np.full((side, side, 3), core, np.uint8),
                                    cv2.GaussianBlur(disc, (3, 3), 0), x, y))


class Lamp(Layer):
    def apply(self, frame, t):
        centre = tuple(_px(self.spec["at"], frame.shape[1::-1]))
        return frame, _glow(frame, centre, self.spec.get("radius", 14), (200, 235, 255),
                            (80, 160, 255))


class TurnSignal(Layer):
    def apply(self, frame, t):
        if ((t - self.start) * self.spec.get("hz", 1.5) * 2) % 2 >= 1:  # off half the time
            return frame, np.zeros(frame.shape[:2], np.float32)
        centre = tuple(_px(self.spec["at"], frame.shape[1::-1]))
        return frame, _glow(frame, centre, self.spec.get("radius", 8), (60, 180, 255),
                            (0, 120, 255))


class Person(Layer):
    def __init__(self, spec, rng):
        super().__init__(spec, rng)
        self.sprite, self.alpha = _scaled(*person_sprite(), spec.get("height", 200))
        self.fall_at = float(spec.get("fall_at", self.start + 10))
        self.fall_seconds = float(spec.get("fall_seconds", 1.0))

    def apply(self, frame, t):
        size = frame.shape[1::-1]
        walk = max(self.fall_at - self.start, 1e-6)
        x, y = _along(self.spec["path"], (t - self.start) / walk, size)  # where the feet are
        angle = 90.0 * min(max((t - self.fall_at) / self.fall_seconds, 0.0), 1.0)
        if angle == 0:
            bob = int(2 * np.sin((t - self.start) * 2 * np.pi * 2)) if t < self.fall_at else 0
            s, a = self.sprite, self.alpha
            return frame, paste(frame, s, a, x - s.shape[1] // 2, y - s.shape[0] + bob)
        s, a, (fx, fy) = _rotated(self.sprite, self.alpha, -angle)  # falls forward, to the right
        return frame, paste(frame, s, a, int(x - fx), int(y - fy))


class Light(Layer):
    def active(self, t):
        return t >= self.start  # stays dim after `to`

    def apply(self, frame, t):
        span = (self.end - self.start) if np.isfinite(self.end) else 1.0
        k = min((t - self.start) / max(span, 1e-6), 1.0)
        gain = 1 + (self.spec.get("gain", 0.3) - 1) * k
        return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8), None


class Shrink(Layer):
    def apply(self, frame, t):
        h, w = frame.shape[:2]
        f = self.spec.get("factor", 4)
        small = cv2.resize(frame, (w // f, h // f), interpolation=cv2.INTER_AREA)
        out = np.zeros_like(frame)
        x, y = _px(self.spec.get("at", [0, 0]), (w, h))
        small = small[:max(h - y, 0), :max(w - x, 0)]
        out[y:y + small.shape[0], x:x + small.shape[1]] = small
        return out, None


LAYERS: dict[str, type[Layer]] = {
    "flame": Flame, "poster": Poster, "smoke": Smoke, "item": Item, "slide": Slide,
    "cover": Cover, "shift": Shift, "fast": Fast, "warm_mover": WarmMover, "lamp": Lamp,
    "turn_signal": TurnSignal, "person": Person, "light": Light, "shrink": Shrink}


# ------------------------------------------------------------------ the composer

class Composer:
    """A recipe -> (frame, t) frames. Base: a video (from `start`, for `seconds`) or a still
    image (held for `seconds` at `fps`)."""

    def __init__(self, recipe: dict[str, Any], root: Path = PROJECT,
                 keep_masks: bool = False) -> None:
        unknown = [layer.get("kind") for layer in recipe.get("layers", [])
                   if layer.get("kind") not in LAYERS]
        if unknown:
            raise ValueError(f"compose: unknown layer kinds {unknown}; use {sorted(LAYERS)}")
        self.base = root / recipe["base"]
        self.start = float(recipe.get("start", 0.0))
        self.seconds = recipe.get("seconds")
        self.fps = float(recipe.get("fps", 10.0))
        self.layers = [LAYERS[spec["kind"]](spec, np.random.default_rng(spec.get("seed", i)))
                       for i, spec in enumerate(recipe.get("layers", []))]
        self.keep_masks = keep_masks
        self.masks: list[np.ndarray | None] = []  # per layer, from the last frame (keep_masks)

    def _base(self) -> Iterator[np.ndarray]:
        image = cv2.imread(str(self.base)) if self.base.suffix.lower() in (
            ".png", ".jpg", ".jpeg", ".bmp") else None
        if image is not None:
            frames = int(round((self.seconds or 10.0) * self.fps))
            for _ in range(frames):
                yield image.copy()
            return
        cap = cv2.VideoCapture(str(self.base))
        if not cap.isOpened():
            raise FileNotFoundError(self.base)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        try:
            for _ in range(int(round(self.start * self.fps))):  # read, not seek: exact frames
                if not cap.read()[0]:
                    return
            limit = None if self.seconds is None else int(round(self.seconds * self.fps))
            n = 0
            while limit is None or n < limit:
                ok, image = cap.read()
                if not ok:
                    return
                yield image
                n += 1
        finally:
            cap.release()

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        for i, frame in enumerate(self._base()):
            t = i / self.fps
            masks: list[np.ndarray | None] = []
            for layer in self.layers:
                mask = None
                if layer.active(t):
                    frame, mask = layer.apply(frame, t)
                masks.append(mask)
            if self.keep_masks:
                self.masks = masks
            yield frame, t


# ------------------------------------------------------------------ night variants

def variant(name: str, seed: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    """A frame transform for a night look of the same scene:
    colour  as filmed
    grey    black and white (a camera that dropped colour at dusk)
    low     low light: x0.25 brightness (with a 1.3 gamma, the mean kept at x0.25), noise 3
    ir      infrared: grey x0.8+30, contrast stretched, bloom on bright spots, blur, noise 6
    dark    x0.04: an unlit camera (tamper `dark`, below dark_mean 12)"""
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; use {list(VARIANTS)}")
    rng = np.random.default_rng(seed)

    def grey(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    def low(frame: np.ndarray) -> np.ndarray:
        f = frame.astype(np.float32)
        curved = 255 * (f / 255) ** 1.3
        dim = curved * (0.25 * f.mean() / max(float(curved.mean()), 1e-6))
        return np.clip(dim + rng.normal(0, 3, f.shape), 0, 255).astype(np.uint8)

    def ir(frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) * 0.8 + 30
        lo, hi = np.percentile(g, (1, 99))
        g = (g - lo) * 255 / max(hi - lo, 1.0)
        g += 60 * cv2.GaussianBlur((g > 230).astype(np.float32), (0, 0), 6)
        g = cv2.GaussianBlur(g, (3, 3), 0) + rng.normal(0, 6, g.shape)
        return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def dark(frame: np.ndarray) -> np.ndarray:
        return (frame.astype(np.float32) * 0.04).astype(np.uint8)

    return {"colour": lambda frame: frame, "grey": grey, "low": low, "ir": ir,
            "dark": dark}[name]
