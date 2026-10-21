"""Composed clips for the scenario bench (engine/compose.py): deterministic layers, flames that
flicker where posters stay still, night looks."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
import pytest

from histor.engine.compose import (FIRE_PICTURES, PERSON_SOURCE, SPRITES, Composer, Flicker,
                                   variant)
from kit import scene_shapes

FLAMES = pytest.mark.skipif(not FIRE_PICTURES.exists(),
                            reason="held-out fire pictures (training/fire_smoke: hf_upload.py --get)")
PERSON = pytest.mark.skipif(not (PERSON_SOURCE.exists() or (SPRITES / "person.png").exists()),
                            reason="data/basketball2.png")


@pytest.fixture
def base(tmp_path):
    cv2.imwrite(str(tmp_path / "base.png"), scene_shapes())  # 640 x 360, still
    return tmp_path


def frames(root, layers, seconds=2.0, keep_masks=False):
    composer = Composer({"base": "base.png", "seconds": seconds, "fps": 10, "layers": layers},
                        root, keep_masks=keep_masks)
    out = []
    for frame, t in composer:
        out.append((frame.copy(), t, list(composer.masks)))
    return out


def digest(clip) -> str:
    h = hashlib.md5()
    for frame, _, _ in clip:
        h.update(frame.tobytes())
    return h.hexdigest()


ALL = [{"kind": "smoke", "at": [0.5, 0.9], "from": 0.2},
       {"kind": "item", "at": [300, 250], "carried": True, "from": 1},
       {"kind": "slide", "a": [-0.3, 0.3, 0, 0.9], "b": [0, 0.3, 0.3, 0.9], "from": 0},
       {"kind": "fast", "from": 0.5},
       {"kind": "warm_mover", "path": [[0.1, 0.8], [0.9, 0.8]], "from": 0, "to": 2},
       {"kind": "lamp", "at": [0.8, 0.2]},
       {"kind": "turn_signal", "at": [0.2, 0.2]},
       {"kind": "light", "from": 1, "to": 2, "gain": 0.5}]


def test_layers_deterministic_by_seed(base):
    assert digest(frames(base, ALL)) == digest(frames(base, ALL))
    reseeded = [{**layer, "seed": 9} for layer in ALL]
    assert digest(frames(base, reseeded)) != digest(frames(base, ALL))


@FLAMES
@PERSON
def test_flame_and_person_deterministic(base):
    layers = [{"kind": "flame", "at": [0.3, 0.9], "height": 100},
              {"kind": "person", "path": [[0.6, 0.95], [0.8, 0.95]], "height": 200,
               "fall_at": 1, "fall_seconds": 0.5}]
    assert digest(frames(base, layers)) == digest(frames(base, layers))



def test_flicker_warps_and_changes_brightness():
    """What fire_loop.mp4 and the flame layer get every frame: never the same patch twice."""
    patch = scene_shapes()[100:200, 100:220]
    flicker = Flicker(np.random.default_rng(0))
    a, b = flicker(patch), flicker(patch)
    for out in (a, b):
        assert np.abs(out.astype(np.int16) - patch).mean() > 4
    assert np.abs(a.astype(np.int16) - b).mean() > 4
    warp_only = Flicker(np.random.default_rng(0), gain=(1.0, 1.0))(patch)
    assert np.abs(warp_only.astype(np.int16) - patch).mean() > 4
    mean_kept = Flicker(np.random.default_rng(0), amp=(0.0, 0.0), gain=(1.0, 1.0))(patch)
    assert (mean_kept == patch).all()

@FLAMES
def test_flame_flickers_poster_does_not(base):
    for kind, inside in (("flame", lambda d: d > 4), ("poster", lambda d: d < 0.5)):
        clip = frames(base, [{"kind": kind, "at": [0.5, 0.9], "height": 150}], keep_masks=True)
        diffs, outside = [], []
        for (a, _, ma), (b, _, mb) in zip(clip, clip[1:]):
            d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
            covered = (ma[0] > 0.01) | (mb[0] > 0.01)
            diffs.append(d[covered].mean())
            outside.append(d[~covered].mean())
        assert inside(np.mean(diffs)), (kind, np.mean(diffs))
        assert np.mean(outside) < 0.05, (kind, np.mean(outside))  # a soft cut, no rectangle


def test_variants():
    rng = np.random.default_rng(0)
    frame = cv2.resize(rng.integers(0, 255, (36, 64, 3), dtype=np.uint8), (640, 360))
    grey = variant("grey")(frame)
    assert (grey[..., 0] == grey[..., 1]).all() and (grey[..., 1] == grey[..., 2]).all()
    assert abs(variant("low")(frame).mean() - frame.mean() * 0.25) <= 2
    assert variant("dark")(frame).mean() < 12
    ir = variant("ir")(frame)
    assert cv2.cvtColor(ir, cv2.COLOR_BGR2HSV)[..., 1].max() == 0
    assert variant("colour")(frame) is frame
    with pytest.raises(ValueError, match="unknown variant"):
        variant("sepia")


def test_item_appears_and_stays_at_box(base):
    clip = frames(base, [{"kind": "item", "at": [300, 250], "size": [70, 50], "from": 1}],
                  seconds=3, keep_masks=True)
    for frame, t, (mask,) in clip:
        if t < 1:
            assert mask is None
            continue
        ys, xs = np.nonzero(mask > 0.5)
        assert (xs.min(), ys.min(), xs.max(), ys.max()) == (300, 250, 369, 299)


def test_slide_stops_at_target(base):
    clip = frames(base, [{"kind": "slide", "a": [0, 100, 100, 200], "b": [300, 100, 400, 200],
                          "seconds": 1, "from": 0.5}], seconds=3, keep_masks=True)
    lefts = []
    for _, t, (mask,) in clip:
        if mask is not None:
            lefts.append((t, int(np.nonzero(mask.max(axis=0) > 0.5)[0].min())))
    assert lefts[0] == (0.5, 0)
    moving = [x for t, x in lefts if t <= 1.5]
    assert moving == sorted(moving) and moving[-1] == 300
    assert {x for t, x in lefts if t >= 1.5} == {300}  # parked
