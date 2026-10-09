"""Record a camera: detect and track every object, embed objects and scenes, save to the store.

Same recorder as engine/main.py, run with Argus's config (configs/default.yaml),
which embeds every COCO class instead of only people and vehicles.

    uv run -m argus.record --source gate
    uv run -m argus.record --source data/vtest.avi --camera street \
        --start-time 2026-09-25T14:00
"""

from __future__ import annotations
from argus.engine.main import main

if __name__ == "__main__":
    main()
