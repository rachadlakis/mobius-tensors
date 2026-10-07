"""Record a camera: detect and track every object, embed objects and scenes, save to the store.

Same recorder as object-detection's src/main.py, run with Argus's config (configs/default.yaml),
which embeds every COCO class instead of only people and vehicles.

    uv run src/record.py --source gate
    uv run src/record.py --source ../object-detection/data/vtest.avi --camera street \
        --start-time 2026-09-25T14:00
"""

from __future__ import annotations

import shared  # noqa: F401
from main import main

if __name__ == "__main__":
    main()
