"""A local vision-language model that reads evidence images for the chat (`look` tool).

CLIP ranks images but can't count, read or describe what someone is doing. When a question needs
that, the chat model passes thumbnails it already found to `look`, and this model answers in a
sentence. It runs on the site's own server (Ollama, llama.cpp, any OpenAI-compatible server with
vision), so the images never leave it, even when the chat model is Claude: only the text answer
goes back to the chat model.

    chat:
      vision:
        api: ollama                    # or openai: llama.cpp, LM Studio, vLLM...
        model: qwen3.5:4b              # `ollama pull qwen3.5:4b` (Apache-2.0, reads images)
        base_url: http://127.0.0.1:11434
        max_images: 4                  # per call
        max_side: 640                  # images are shrunk to this many pixels, longest side
        think: false                   # Ollama: no reasoning before the answer (see below)

Seconds per image on a CPU, growing with image size: keep max_side small. Thinking models
(qwen3.5) answer in about a second with `think: false`; with thinking on, a yes/no question can
run to thousands of tokens of reasoning (minutes on a CPU, past the alert check's timeout).
"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from histor.llm import OLLAMA_CONTEXT, THINK, ollama_root, post_json

VISION_APIS = ("ollama", "openai")
PROMPT = (
    "You look at security camera images for someone who asked a question about them. Answer "
    "only from what you can see. Be short and concrete: counts, colours, actions, text. If the "
    "image is too small, dark or blurred to tell, say so instead of guessing. When there are "
    "several images, refer to them as image 1, image 2, ... in the order given.")


def encode_jpeg(image: np.ndarray, max_side: int, quality: int = 85) -> str:
    """BGR image -> base64 JPEG, shrunk so its longest side is at most max_side."""
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode the image")
    return base64.b64encode(buf.tobytes()).decode()


class VisionModel:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434/v1",
                 api_key: str | None = None, max_images: int = 4, max_side: int = 640,
                 temperature: float = 0.0, timeout: float = 300, enabled: bool = True,
                 api: str = "openai", num_ctx: int = OLLAMA_CONTEXT, think: bool = False) -> None:
        """api: ollama (its own API: the context size is set here, images count in it) or
        openai (any OpenAI-compatible server). think: Ollama only; models without thinking
        ignore false."""
        if api not in VISION_APIS:
            raise ValueError("chat.vision needs Ollama or an OpenAI-compatible server "
                             "(api: ollama or openai)")
        self.model = model
        self.api = api
        self.num_ctx = int(num_ctx)
        self.think = bool(think)
        self.url = (ollama_root(base_url) + "/api/chat" if api == "ollama"
                    else base_url.rstrip("/") + "/chat/completions")
        self.api_key = api_key
        self.max_images = max(1, int(max_images))
        self.max_side = max(64, int(max_side))
        self.temperature = temperature
        self.timeout = timeout

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> VisionModel | None:
        """`chat.vision` section; None when missing or `enabled: false`."""
        if not config or not config.get("model") or config.get("enabled") is False:
            return None
        return cls(**{k: v for k, v in config.items() if k != "note"})

    def ask(self, images: list[np.ndarray], question: str, system: str = PROMPT) -> str:
        """Images (BGR) + a question -> the model's answer, without its thinking.
        system: the instructions (alert verification has its own, see engine/verify.py)."""
        if not images:
            raise ValueError("No image to look at")
        if len(images) > self.max_images:
            raise ValueError(f"At most {self.max_images} images per look")
        jpegs = [encode_jpeg(image, self.max_side) for image in images]
        if self.api == "ollama":
            body: dict[str, Any] = {
                "model": self.model, "stream": False, "think": self.think,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": question, "images": jpegs}]}
        else:
            content: list[dict[str, Any]] = [{"type": "text", "text": question}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpeg}}
                for jpeg in jpegs]
            body = {"model": self.model, "temperature": self.temperature, "stream": False,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": content}]}
        reply = post_json(self.url, body, self.api_key, self.timeout, self.model)
        message = reply["message"] if self.api == "ollama" else reply["choices"][0]["message"]
        return THINK.sub("", message.get("content") or "").strip() or "(no answer)"
