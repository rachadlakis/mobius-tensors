import numpy as np

from detector import YoloxOnnxDetector, decode_yolox, letterbox, nms


def make_postprocessor(**kwargs) -> YoloxOnnxDetector:
    """Detector without an ONNX session, for testing pre/post-processing only."""
    det = YoloxOnnxDetector.__new__(YoloxOnnxDetector)
    det.input_size = (64, 64)
    det.class_names = ("person", "car")
    det.conf_threshold = 0.3
    det.nms_threshold = 0.45
    det.class_agnostic_nms = False
    det.allowed_ids = None
    det.__dict__.update(kwargs)
    return det


def test_letterbox_keeps_aspect_and_pads():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    blob, ratio = letterbox(img, (64, 64))
    assert blob.shape == (3, 64, 64)
    assert ratio == 64 / 200
    assert blob[0, -1, 0] == 114  # bottom padding


def test_nms_suppresses_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.5) == [0, 2]


def test_postprocess_decodes_box_and_class():
    # 64x64 input -> grids 8x8 + 4x4 + 2x2 = 84 anchors; 5 + 2 classes.
    raw = np.zeros((84, 7), dtype=np.float32)
    # Anchor at stride-8 grid cell (x=3, y=2): index 2 * 8 + 3 = 19.
    raw[19] = [0.5, 0.5, np.log(2), np.log(1), 0.9, 0.1, 0.8]
    dets = make_postprocessor().postprocess(raw, ratio=0.5, image_shape=(128, 128))
    assert len(dets) == 1
    d = dets[0]
    assert d.label == "car"
    # center (28, 20), size (16, 8) in input pixels -> scaled by 1/0.5.
    assert np.allclose([d.x1, d.y1, d.x2, d.y2], [40, 32, 72, 48])
    assert np.isclose(d.score, 0.72)


def test_class_filter():
    raw = np.zeros((84, 7), dtype=np.float32)
    raw[19] = [0.5, 0.5, 0, 0, 0.9, 0.1, 0.8]
    det = make_postprocessor(allowed_ids={0})
    assert det.postprocess(raw, 1.0, (64, 64)) == []


def test_decode_shape():
    assert decode_yolox(np.zeros((84, 7), np.float32), (64, 64)).shape == (84, 7)
