"""Make ../object-detection/src importable.

Argus reuses its detector, tracker, CLIP encoder, event store and search instead of copying
them. Import this module before any of those: `import shared  # noqa: F401`.
"""

from __future__ import annotations

import sys
from pathlib import Path

OBJECT_DETECTION_SRC = Path(__file__).resolve().parents[2] / "object-detection" / "src"

if not OBJECT_DETECTION_SRC.is_dir():
    raise ImportError(f"Argus needs the object-detection project at {OBJECT_DETECTION_SRC}")
if str(OBJECT_DETECTION_SRC) not in sys.path:
    sys.path.append(str(OBJECT_DETECTION_SRC))
