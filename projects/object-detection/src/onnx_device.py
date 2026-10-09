"""ONNX Runtime sessions on a GPU when there is one, else on the CPU.

`device` in a config section (model, search.clip, pose):
    auto        the first GPU this ONNX Runtime build can use (CUDA, then DirectML), else CPU
    cpu         always the CPU
    cuda        NVIDIA GPU (install the `gpu` dependency group); cuda:1 = the second GPU
    directml    any GPU on Windows (onnxruntime-directml); directml:1 = the second GPU
    coreml      Apple Neural Engine / GPU (in the standard macOS wheel)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}
AUTO_ORDER = ("cuda", "directml")  # CoreML only on request: it runs some models only in part

_dlls_loaded = False


def providers_for(device: str = "auto", available: list[str] | None = None) -> list[Any]:
    """ONNX Runtime providers for a device name, CPU last as the fallback for unsupported ops."""
    if available is None:
        import onnxruntime as ort

        available = ort.get_available_providers()
    name, _, index = (device or "auto").lower().partition(":")
    if name == "auto":
        name = next((d for d in AUTO_ORDER if PROVIDERS[d] in available), "cpu")
    if name not in PROVIDERS:
        raise ValueError(f"Unknown device {device!r}: use auto, {', '.join(PROVIDERS)}")
    if name == "cpu":
        return [PROVIDERS["cpu"]]
    if PROVIDERS[name] not in available:
        hint = {"cuda": "uv sync --no-group cpu --group gpu",
                "directml": "uv pip install onnxruntime-directml"}.get(name, "")
        raise RuntimeError(
            f"device {device!r}: this ONNX Runtime has no {PROVIDERS[name]} "
            f"(it has {', '.join(available)}). {hint}".strip())
    provider: Any = PROVIDERS[name]
    if index:
        provider = (provider, {"device_id": int(index)})
    return [provider, PROVIDERS["cpu"]]


def create_session(model: str | Path, device: str = "auto", num_threads: int = 0) -> Any:
    """An InferenceSession on `device`. Logs the device actually used: ONNX Runtime falls back
    to the CPU (with its own error message) when the GPU libraries or driver are missing."""
    global _dlls_loaded
    import onnxruntime as ort

    providers = providers_for(device, ort.get_available_providers())
    names = [p if isinstance(p, str) else p[0] for p in providers]
    if names[0] == PROVIDERS["cuda"] and not _dlls_loaded and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()  # CUDA / cuDNN from the nvidia-* wheels of the `gpu` group
        _dlls_loaded = True
    options = ort.SessionOptions()
    if num_threads > 0:
        options.intra_op_num_threads = num_threads
    session = ort.InferenceSession(str(model), options, providers=providers)
    used = session.get_providers()[0]
    if used != names[0]:
        log.warning("%s: wanted %s, running on %s", Path(model).name, names[0], used)
    else:
        log.info("%s: running on %s", Path(model).name, used)
    return session
