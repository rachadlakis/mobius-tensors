"""The per-camera change analysis (engine/change.py) that motion, tamper and fire share."""

from __future__ import annotations

import statistics
import time

import cv2
import numpy as np
import pytest

from histor.engine.change import ChangeAnalyser
from histor.engine.tamper import TamperCheck, prepare
from kit import noisy, scene_noise

W, H = 640, 360


def learned(analyser: ChangeAnalyser, base: np.ndarray, frames: int = 40,
            seed: int | None = None) -> float:
    """The background learned on `frames` of `base` (noisy when `seed` is given): the time."""
    t = 0.0
    for i in range(frames):
        analyser.update(base if seed is None else noisy(base, seed + i), t)
        t += 0.08
    return t


def with_square(base: np.ndarray, x: int, colour: tuple[int, int, int] = (255, 255, 255)
                ) -> np.ndarray:
    frame = base.copy()
    cv2.rectangle(frame, (x, 150), (x + 40, 190), colour, -1)
    return frame


def test_ready_after_warmup_and_fg_of_moving_square():
    analyser, base = ChangeAnalyser(warmup_frames=15), scene_noise()
    changes = [analyser.update(base, i * 0.08) for i in range(16)]
    assert [c.ready for c in changes] == [False] * 15 + [True]
    learned(analyser, base)
    change = analyser.update(with_square(base, 300), 5.0)
    assert change.ready and not change.light and change.shape == (180, 320)
    (box,) = change.regions(change.fg, 0.002)
    assert np.allclose(box, [300, 150, 340, 190], atol=6)  # frame pixels
    assert change.share_in(change.fg, [300, 150, 340, 190]) > 0.8


def test_shake_aligned_no_fg():
    analyser, base = ChangeAnalyser(), scene_noise()
    t = learned(analyser, base)
    for i in range(20):
        dx, dy = (3, -3) if i % 2 else (-3, 3)  # swaying in the wind
        shaken = cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H),
                                borderMode=cv2.BORDER_REPLICATE)
        change = analyser.update(shaken, t)
        t += 0.08
        assert change.shift != (0.0, 0.0) and not change.regions(change.fg, 0.002)
    assert analyser.restarts == 0


def test_turn_restarts():
    analyser, base = ChangeAnalyser(), scene_noise()
    t = learned(analyser, base)
    change = analyser.update(np.roll(base, 100, axis=1), t)  # 16% of the width
    assert change.restarted and not change.ready and analyser.restarts == 1
    assert not analyser.update(np.roll(base, 100, axis=1), t + 0.08).restarted


def test_light_flag_but_mask_kept():
    analyser, base = ChangeAnalyser(), scene_noise()
    t = learned(analyser, base)
    change = analyser.update(cv2.add(base, np.full_like(base, 90)), t)  # lamps switched on
    assert change.light and change.fg_share > 0.4 and change.fg.any()
    assert analyser.light_changes == 1


def test_shadow_share_on_darker_patch():
    analyser, base = ChangeAnalyser(), scene_noise()
    t = learned(analyser, base)
    frame = base.copy()
    frame[100:260, 200:440] = (frame[100:260, 200:440] * 0.5).astype(np.uint8)  # a cloud
    change = analyser.update(frame, t)
    assert change.shadow_share > 0.05
    assert change.share_in(change.fg, [200, 100, 440, 260]) < 0.05  # a shadow is not a thing


def test_diff_marks_flicker_not_static_noise():
    analyser, base = ChangeAnalyser(fire_masks=True), scene_noise()
    t = learned(analyser, base, seed=0)
    for i in range(10):
        assert analyser.update(noisy(base, 100 + i), t).diff.mean() < 0.005  # camera noise
        t += 0.08
    flicker = []
    for i in range(10):
        frame = noisy(base, 200 + i)
        frame[150:190, 300:340] = 60 if i % 2 else 230  # a flame's flicker
        flicker.append(analyser.update(frame, t))
        t += 0.08
    assert all(c.share_in(c.diff, [300, 150, 340, 190]) > 0.5 for c in flicker[1:])


def test_warm_and_bright_masks_and_colourless():
    analyser = ChangeAnalyser(fire_masks=True)
    frame = np.full((H, W, 3), 90, np.uint8)
    frame[:, :320] = (40, 120, 60)  # green grass
    frame[100:200, 100:200] = (0, 140, 255)  # orange, BGR
    change = analyser.update(frame, 0.0)
    assert change.share_in(change.warm, [100, 100, 200, 200]) > 0.9
    assert change.warm.sum() == pytest.approx(change.warm[50:100, 50:100].size, rel=0.1)
    assert not change.colourless
    grey = np.full((H, W, 3), 60, np.uint8)
    grey[200:260, 400:460] = 250  # an infrared hot spot
    change = analyser.update(grey, 0.1)
    assert change.colourless and not change.warm.any()
    assert change.share_in(change.bright, [400, 200, 460, 260]) > 0.9
    plain = ChangeAnalyser().update(frame, 0.0)
    assert plain.warm is None and plain.diff is None  # only when asked (fire)


def test_grey_identical_to_tamper_measure():
    frame = scene_noise(3, (576, 768))
    change = ChangeAnalyser().update(frame, 0.0)
    assert change.width == 320 and np.array_equal(change.grey, prepare(frame))
    wide = ChangeAnalyser(width=480).update(frame, 0.0)
    check = TamperCheck(interval_seconds=0)
    check.update(frame, 0.0, change=wide)  # another width: tamper keeps its own copy
    assert np.array_equal(check.last_measure.grey, prepare(frame))


def test_regions_scaled_to_frame_pixels():
    analyser = ChangeAnalyser()
    change = analyser.update(np.zeros((720, 1280, 3), np.uint8), 0.0)
    mask = np.zeros(change.shape, np.uint8)
    mask[10:20, 30:50] = 1   # 200 px of 320 x 180
    mask[100:102, 100:102] = 1  # 4 px
    assert change.scale == 0.25
    (box,) = change.regions(mask, min_area=0.001)
    assert box.tolist() == [120, 40, 200, 80]
    assert len(change.regions(mask)) == 2 and not change.regions(mask, max_area=0.00005)
    assert change.to_small([120, 40, 200, 80]).tolist() == [30, 10, 50, 20]


def test_share_in_box():
    change = ChangeAnalyser().update(np.zeros((720, 1280, 3), np.uint8), 0.0)
    mask = np.zeros(change.shape, np.uint8)
    mask[10:20, 30:40] = 1
    assert change.share_in(mask, [120, 40, 200, 80]) == pytest.approx(0.5)
    assert change.share_in(mask, [120, 40, 160, 80]) == 1.0
    assert change.share_in(mask, [2000, 2000, 2100, 2100]) == 0.0  # outside the picture
    assert change.share_in(mask, [130, 50, 130, 60]) == 0.0         # empty box


def test_old_search_motion_keys_still_configure_the_analyser():
    old = ChangeAnalyser.from_config({"search": {"motion": {"enabled": True, "history": 100,
                                                            "min_area": 0.004}}})
    assert old.history == 100 and old.width == 320
    new = ChangeAnalyser.from_config({"search": {"motion": {"history": 100}},
                                      "change": {"history": 50, "width": 240}})
    assert new.history == 50 and new.width == 240  # the `change` section wins
    assert ChangeAnalyser.from_config({}, fire_masks=True).fire_masks


def test_cpu_budget():
    """One 1280x720 frame's whole analysis, fire masks included (README budget: 5 ms)."""
    analyser = ChangeAnalyser(fire_masks=True)
    base = scene_noise(1, (720, 1280))
    looks = [base] + [noisy(base, k) for k in range(6)]
    spent = []
    for i in range(330):
        frame = looks[i % 7]
        frame = with_square(frame, (i * 23) % 1200, (0, 140, 255))  # an orange mover
        start = time.perf_counter()
        analyser.update(frame, i * 0.04)
        if i >= 30:
            spent.append((time.perf_counter() - start) * 1000)
    median, p95 = statistics.median(spent), float(np.percentile(spent, 95))
    print(f"change analysis 1280x720: median {median:.2f} ms, p95 {p95:.2f} ms")
    assert median < 10 and p95 < 20
