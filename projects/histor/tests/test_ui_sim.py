import threading

import cv2
import numpy as np
from histor.ui import (IMAGE_FPS, IMAGE_SECONDS, MODULES, file_frames, modules_on, thumb_caption,
                       with_modules)


def test_photos_and_videos_play_as_one_camera(tmp_path):
    photo = tmp_path / "a.png"
    cv2.imwrite(str(photo), np.zeros((40, 60, 3), np.uint8))
    video = tmp_path / "b.avi"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (60, 40))
    for _ in range(5):
        out.write(np.full((40, 60, 3), 128, np.uint8))
    out.release()
    frames = list(file_frames([str(photo), str(video)], real_speed=False, loop=False,
                              stop=threading.Event()))
    names = [n for _, _, n in frames]
    assert names.count("a.png") == IMAGE_FPS * IMAGE_SECONDS and names.count("b.avi") == 5
    times = [t for _, t, _ in frames]
    assert times == sorted(times)                         # the video goes on after the photo
    assert times[IMAGE_FPS * IMAGE_SECONDS] == IMAGE_SECONDS


def test_loop_stops_when_asked(tmp_path):
    photo = tmp_path / "a.png"
    cv2.imwrite(str(photo), np.zeros((8, 8, 3), np.uint8))
    stop = threading.Event()
    seen = 0
    for _ in file_frames([str(photo)], real_speed=False, loop=True, stop=stop):
        seen += 1
        if seen == 3 * IMAGE_FPS * IMAGE_SECONDS:        # three passes, then Stop
            stop.set()
    assert seen == 3 * IMAGE_FPS * IMAGE_SECONDS


def test_modules_ticked_override_the_config():
    config = {"pose": {"enabled": True, "cameras": ["laptop"]}, "plates": {"enabled": False}}
    assert modules_on(config) == ["Pose: person down, hands up"]
    chosen = with_modules(config, ["Number plates"])
    assert chosen["plates"] == {"enabled": True, "cameras": []}   # on for the simulated camera
    assert chosen["pose"]["enabled"] is False and config["pose"]["enabled"] is True  # a copy
    assert all(chosen[s]["enabled"] is (s == "plates") for s in MODULES.values())
    assert with_modules(config, None) is config


def test_thumb_caption():
    assert thumb_caption("thumbs/d/laptop_1791968124389_1.jpg").startswith("laptop ")
    assert thumb_caption("snapshots/y1.jpg") == "y1.jpg"
