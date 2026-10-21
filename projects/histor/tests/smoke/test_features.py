"""Smoke tests of single features with the real models, each with its own skip reason (not the
all-models requirement of test_smoke.py). Run with the other smoke tests: uv run pytest -m smoke
"""

from __future__ import annotations

import json

import pytest
from smoke_kit import DATA, MODELS, run

pytestmark = pytest.mark.smoke

BENCH_NEEDS = [MODELS / "yolox_nano.onnx", MODELS / "clip_image.onnx",
               MODELS / "fire_smoke_yolov9s_640.onnx", DATA / "vtest.avi"]
BENCH = pytest.mark.skipif(not all(p.exists() for p in BENCH_NEEDS),
                           reason=f"the street suite needs {[p.name for p in BENCH_NEEDS]}")


def bench(*args: str) -> tuple[int, str]:
    out = run("-m", "histor.scenarios", "--suite", "street", "--only",
              "street_cart", "--quick", *args, timeout=600)
    return out.returncode, out.stdout + out.stderr


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    path = tmp_path_factory.mktemp("bench") / "r.json"
    code, text = bench("--out", str(path))
    return code, text, path


@BENCH
def test_scenarios_cli_quick(quick_run):
    code, text, path = quick_run
    assert code == 0, text
    result = json.loads(path.read_text(encoding="utf-8"))
    assert "recall" in result["summary"]["colour"]
    assert result["runs"][0]["timings"]["detect"]["mean"] > 0


@BENCH
def test_scenarios_compare_detects_regression(quick_run, tmp_path):
    _, _, path = quick_run
    doctored = json.loads(path.read_text(encoding="utf-8"))
    doctored["runs"][0]["score"]["hits"].append(  # a hit this run cannot have
        {"expect": 7, "t": 1.0, "delay": 0.0, "lag": 0.0, "fire": False})
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(doctored), encoding="utf-8")
    code, text = bench("--out", str(tmp_path / "again.json"), "--compare", str(baseline))
    assert code == 1 and "FAIL street_cart [colour]: expect #7 was hit" in text, text


FIRE_ONLY_NEEDS = [MODELS / "yolox_nano.onnx", MODELS / "fire_smoke_yolov9s_640.onnx",
                   DATA / "vtest.avi"]


@pytest.mark.skipif(not all(p.exists() for p in FIRE_ONLY_NEEDS),
                    reason=f"needs {[p.name for p in FIRE_ONLY_NEEDS]}")
def test_fire_only_pipeline_has_change_analysis(tmp_path):
    """Fire alone (no search, no tracker) still gets the camera's change analysis, with the
    fire masks, ready after its warm-up."""
    from histor.engine.pipeline import Pipeline
    from histor.engine.stream import iter_frames

    config = {"model": {"path": str(MODELS / "yolox_nano.onnx")},
              "store": {"path": str(tmp_path / "db.sqlite")},
              "alerts": {"console": False},
              "fire": {"enabled": True,
                       "model": {"path": str(MODELS / "fire_smoke_yolov9s_640.onnx")}}}
    pipe = Pipeline.from_config(config, source="smoke", live=False, start_time=0.0)
    assert pipe.indexer is None and pipe.tracker is None and pipe.change is not None
    try:
        for i, f in enumerate(iter_frames(str(DATA / "vtest.avi"))):
            result = pipe.process(f.image, f.timestamp)
            assert result.change is not None and result.change.ready == (i >= 15)
            if i == 16:
                break
        assert result.change.warm is not None and result.change.diff is not None
    finally:
        pipe.close()
        pipe.store.close()
