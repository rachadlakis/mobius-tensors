"""The one colour table: names of OpenCV hues (0-179) and when a pixel has colour at all.
Clothing colours (pose.colour_names) and the warm mask of the change analysis (change.py) both
read it."""

from __future__ import annotations

import numpy as np

# (upper hue, exclusive, name), from hue 0; red wraps round at both ends
HUE_BINS = ((8, "red"), (20, "orange"), (33, "yellow"), (85, "green"), (128, "blue"),
            (150, "purple"), (170, "pink"), (180, "red"))
CHROMA_S, CHROMA_V = 60, 50  # saturation and value a pixel needs to have a hue at all
WARM = ("red", "orange", "yellow")  # what flames look like

_WARM_HUE = np.zeros(256, bool)
_lower = 0
for _upper, _name in HUE_BINS:
    _WARM_HUE[_lower:_upper] = _name in WARM
    _lower = _upper


def chroma(s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pixels with a colour (not grey, white or black)."""
    return (s >= CHROMA_S) & (v >= CHROMA_V)


def warm_mask(hsv: np.ndarray, min_value: int = CHROMA_V) -> np.ndarray:
    """uint8 0/1: red, orange or yellow pixels of an HSV image at least `min_value` bright."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (chroma(s, v) & (v >= min_value) & _WARM_HUE[h]).astype(np.uint8)
