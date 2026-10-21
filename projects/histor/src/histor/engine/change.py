"""What changed in the picture: one analysis per camera and frame, shared by everything that
needs it (moving things for search in motion.py; fire and new still items in later steps), so
the frame is shrunk and compared with its background once, not once per feature. Its options
are the `change` section of the config (configs/default.yaml says what each one means).

On a small copy of each frame (`width` px wide, the height in proportion):

    shake       the grey copy is first lined up with the learned background (phase
                correlation), so a camera swaying in the wind by a few pixels is not change; a
                bigger jump (the camera was turned) starts the background again (`restarted`)
    background  OpenCV MOG2 learns, per pixel, what is normal there, including what always
                moves (leaves, water, a flag): `fg` is what differs from it, without shadows
                (their share is `shadow_share`: smoke can look like a shadow)
    light       when most of the picture changes at once (clouds, lamps switched on), it is
                light, not an object (`light`); `fg` is still given, for who wants it
    flicker     `diff`: what changed since the previous frame, which a background cannot
                learn away (flames)
    colours     `warm` (red, orange, yellow: colours.py), `bright` and `colourless` (a grey or
                infrared picture); only when a consumer asks (`fire_masks`), to save the time

Boxes go between frame pixels and the small copy with to_small / to_frame; regions(mask) and
share_in(mask, box) take and give frame pixels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from histor.engine import boxes, colours

log = logging.getLogger(__name__)
# options of the analysis that used to sit in search.motion (still read from there)
ANALYSER_KEYS = ("width", "light_change", "max_shift", "warmup_frames", "history",
                 "var_threshold")
_OPEN, _CLOSE = np.ones((3, 3), np.uint8), np.ones((7, 7), np.uint8)
_warned = False


@dataclass
class Change:
    """One frame's analysis. Masks are uint8 0/1 at the analysis size (`shape`)."""

    scale: float              # analysis pixels per frame pixel
    shape: tuple[int, int]    # analysis height, width
    small: np.ndarray = field(repr=False)    # BGR (grey for a grey frame) small copy
    grey: np.ndarray = field(repr=False)     # its grey, unblurred (tamper.prepare's pixels)
    blurred: np.ndarray = field(repr=False)  # grey, 5x5 Gaussian, lined up with the background
    fg: np.ndarray = field(repr=False)       # differs from the background (no shadows)
    fg_share: float = 0.0
    shadow_share: float = 0.0
    light: bool = False       # most of the picture changed at once
    shift: tuple[float, float] = (0.0, 0.0)  # shake cancelled, analysis pixels
    restarted: bool = False   # the view jumped: the background is learned again
    ready: bool = False       # the background has seen `warmup_frames` frames
    frozen: bool = False      # grey identical to the previous frame's
    width: int = 320          # the analyser's width
    diff: np.ndarray | None = field(default=None, repr=False)    # flicker since the last frame
    warm: np.ndarray | None = field(default=None, repr=False)    # with fire_masks
    bright: np.ndarray | None = field(default=None, repr=False)  # with fire_masks
    colourless: bool | None = None                               # with fire_masks

    def to_small(self, box: Any) -> np.ndarray:
        return boxes.scale_box(box, self.scale)

    def to_frame(self, box: Any) -> np.ndarray:
        return boxes.scale_box(box, 1 / self.scale)

    def regions(self, mask: np.ndarray, min_area: float = 0.0,
                max_area: float = 1.0) -> list[np.ndarray]:
        """Connected parts of `mask` covering min_area..max_area of the picture: boxes in frame
        pixels."""
        sh, sw = self.shape
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        return [self.to_frame([x, y, x + bw, y + bh])
                for x, y, bw, bh, area in stats[1:count]
                if min_area * sw * sh <= area <= max_area * sw * sh]

    def share_in(self, mask: np.ndarray, box: Any) -> float:
        """Share of the box (frame pixels) that is set in `mask`; 0 for a box outside."""
        sh, sw = self.shape
        x1, y1, x2, y2 = boxes.clip_box(self.to_small(box), sw, sh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float(mask[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))].mean())


class ChangeAnalyser:
    def __init__(self, width: int = 320, light_change: float = 0.4, max_shift: float = 0.08,
                 warmup_frames: int = 15, history: int = 300, var_threshold: float = 25.0,
                 diff_threshold: int = 12, warm_min_value: int = 120, bright_value: int = 220,
                 colourless_saturation: float = 12.0, fire_masks: bool = False) -> None:
        self.width = width
        self.light_change = light_change
        self.max_shift = max_shift
        self.warmup = warmup_frames
        self.history, self.var_threshold = history, var_threshold
        self.diff_threshold = diff_threshold
        self.warm_min_value = warm_min_value
        self.bright_value = bright_value
        self.colourless_saturation = colourless_saturation
        self.fire_masks = fire_masks
        self._window: np.ndarray | None = None
        self._previous: np.ndarray | None = None  # last blurred (flicker)
        self._last_grey: np.ndarray | None = None  # last grey (frozen)
        self.frames = 0          # frames since the background (re)started
        self.light_changes = 0   # ready frames taken as a change of light
        self.restarts = 0        # camera moved: background learned again
        self._restart()
        self.restarts = 0

    @classmethod
    def from_config(cls, config: dict[str, Any], **flags: Any) -> ChangeAnalyser:
        """config: the whole config; the `change` section, over the analysis keys an older
        config has in search.motion."""
        global _warned
        old = {k: v for k, v in ((config.get("search") or {}).get("motion") or {}).items()
               if k in ANALYSER_KEYS}
        if old and not _warned:
            _warned = True
            log.info("search.motion %s: moved to change: (still read)", sorted(old))
        options = {**old, **(config.get("change") or {})}
        options.pop("enabled", None)
        return cls(**options, **flags)

    def _restart(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(self.history, self.var_threshold,
                                                      detectShadows=True)
        self.frames = 0
        self._previous = None
        self.restarts += 1

    def update(self, frame: np.ndarray, t: float) -> Change:
        """t: the frame's time (seconds; only differences matter)."""
        h, w = frame.shape[:2]
        scale = self.width / w
        small = cv2.resize(frame, (self.width, max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        shift, restarted = (0.0, 0.0), False
        if self.frames:
            blurred, shift, restarted = self._align(blurred)
        self.frames += 1
        mask = self._bg.apply(blurred)
        fg: np.ndarray = (mask > 200).astype(np.uint8)  # 255 = foreground, 127 = shadow
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _OPEN)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _CLOSE)  # a blurred streak
        fg_share = float(fg.mean())
        ready = self.frames > self.warmup
        light = fg_share > self.light_change
        if ready and light:
            self.light_changes += 1
        frozen = self._last_grey is not None and self._last_grey.shape == grey.shape and \
            not cv2.absdiff(self._last_grey, grey).any()
        self._last_grey = grey
        change = Change(scale, grey.shape[:2], small, grey, blurred, fg, fg_share,
                        float((mask == 127).mean()), light, shift, restarted, ready, frozen,
                        self.width)
        if self.fire_masks:
            self._fire_masks(change)
        self._previous = blurred
        return change

    def _fire_masks(self, change: Change) -> None:
        previous = self._previous
        change.diff = (cv2.absdiff(change.blurred, previous) > self.diff_threshold).astype(
            np.uint8) if previous is not None and previous.shape == change.blurred.shape \
            else np.zeros_like(change.fg)
        if change.small.ndim == 2:  # a grey picture: no colour at all
            change.warm = np.zeros_like(change.fg)
            change.bright = (change.grey >= self.bright_value).astype(np.uint8)
            change.colourless = True
            return
        hsv = cv2.cvtColor(change.small, cv2.COLOR_BGR2HSV)
        change.warm = colours.warm_mask(hsv, self.warm_min_value)
        change.bright = (hsv[..., 2] >= self.bright_value).astype(np.uint8)
        change.colourless = float(hsv[..., 1].mean()) < self.colourless_saturation

    def _align(self, small: np.ndarray) -> tuple[np.ndarray, tuple[float, float], bool]:
        """small lined up with the background: (image, shift cancelled, restarted)."""
        ref = self._bg.getBackgroundImage()
        if ref is None or ref.shape[:2] != small.shape[:2]:
            return small, (0.0, 0.0), False
        if ref.ndim == 3:
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        if self._window is None or self._window.shape != small.shape:
            self._window = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(ref.astype(np.float32),
                                                small.astype(np.float32), self._window)
        if (abs(dx) < 0.5 and abs(dy) < 0.5) or response < 0.1:  # still, or no clear shift
            return small, (0.0, 0.0), False
        sh, sw = small.shape
        if max(abs(dx) / sw, abs(dy) / sh) > self.max_shift:  # turned: a new view
            self._restart()
            return small, (0.0, 0.0), True
        shift = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        return (cv2.warpAffine(small, shift, (sw, sh), borderMode=cv2.BORDER_REPLICATE),
                (float(dx), float(dy)), False)
