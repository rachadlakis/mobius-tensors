import cv2
import numpy as np
from histor.engine.motion import MotionWatch

W, H = 640, 360


def scene(seed: int = 0) -> np.ndarray:
    """A textured still view (edges everywhere, like a real yard)."""
    noise = np.random.default_rng(seed).integers(0, 255, (H // 8, W // 8, 3), dtype=np.uint8)
    return cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC)


def learned(watch: MotionWatch, base: np.ndarray, frames: int = 40) -> float:
    t = 0.0
    for _ in range(frames):
        assert watch.update(base, t) == []
        t += 0.08
    return t


def test_fast_object_crossing_in_half_a_second_is_reported():
    watch, base = MotionWatch(), scene()
    t = learned(watch, base)
    reported = []
    for x in range(0, W - 40, 100):  # 40 px square, 100 px a frame: 6 frames at 12.5 fps
        frame = base.copy()
        cv2.rectangle(frame, (x, 150), (x + 40, 190), (255, 255, 255), -1)
        reported += watch.update(frame, t)
        t += 0.08
    assert len({i for i, _ in reported}) == 1          # one object, followed across the jumps
    x1, y1, x2, y2 = reported[0][1]
    assert 130 <= y1 <= 160 and 180 <= y2 <= 210        # where the square was


def test_tracked_objects_are_left_to_the_indexer():
    watch, base = MotionWatch(), scene()
    t = learned(watch, base)
    for x in range(0, 400, 50):
        frame = base.copy()
        cv2.rectangle(frame, (x, 150), (x + 40, 190), (255, 255, 255), -1)
        assert watch.update(frame, t, covered=[np.array([x - 5, 145, x + 45, 195])]) == []
        t += 0.08


def test_camera_shake_is_not_motion():
    watch, base = MotionWatch(), scene()
    t = learned(watch, base)
    for i in range(20):
        dx, dy = (3, -2) if i % 2 else (-2, 3)  # swaying in the wind
        shaken = cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H),
                                borderMode=cv2.BORDER_REPLICATE)
        assert watch.update(shaken, t) == []
        t += 0.08


def test_light_change_is_not_motion():
    watch, base = MotionWatch(), scene()
    t = learned(watch, base)
    darker = (base * 0.5).astype(np.uint8)  # a cloud: taken as shadow
    for _ in range(5):
        assert watch.update(darker, t) == []
        t += 0.08
    brighter = cv2.add(base, np.full_like(base, 90))  # lamps switched on
    for _ in range(5):
        assert watch.update(brighter, t) == []
        t += 0.08
    assert watch.light_changes > 0


def test_busy_indexer_gets_the_blob_again():
    watch, base = MotionWatch(), scene()
    t = learned(watch, base)
    first = []
    x = 0
    while not first:
        frame = base.copy()
        cv2.rectangle(frame, (x, 150), (x + 40, 190), (255, 255, 255), -1)
        first = watch.update(frame, t)
        x, t = x + 30, t + 0.08
    watch.retry([first[0][0]])
    frame = base.copy()
    cv2.rectangle(frame, (x, 150), (x + 40, 190), (255, 255, 255), -1)
    assert [i for i, _ in watch.update(frame, t)] == [first[0][0]]
