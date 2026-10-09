from pathlib import Path

import yaml

import shared  # noqa: F401
from ask import format_answer
from detector import COCO_CLASSES

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_embeds_every_coco_class():
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    assert config["model"]["classes"] is None
    assert config["search"]["classes"] == list(COCO_CLASSES)


def test_format_answer():
    answer = {
        "query": {"text": "red car", "cameras": "all", "when": "yesterday"},
        "results": [{
            "camera": "gate", "label": "car", "kind": "object", "score": 0.31234,
            "first_seen": "2026-10-02T18:05:00", "last_seen": "2026-10-02T18:05:40",
            "thumb": "thumbs/1.jpg",
        }],
    }
    text = format_answer(answer, Path("media"))
    assert "'red car' on all, yesterday" in text
    assert "gate" in text and "2026-10-02 18:05:00 -> 18:05:40" in text and "car" in text
    assert "nothing found" in format_answer({**answer, "results": []}, Path("media"))
