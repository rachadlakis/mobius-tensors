"""CLIP image/text embeddings with ONNX Runtime (no PyTorch): the engine behind text search.

Images and sentences are mapped into the same 512-number space, so "a person in a red shirt"
lands close to crops of people in red shirts. Models come from `export_clip.py`.
"""

from __future__ import annotations

import gzip
import html
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
CONTEXT_LENGTH = 77
# Exports made before the model name was written into the ONNX metadata, and the vectors they
# stored (model column empty), are this model.
LEGACY_MODEL = "ViT-B-32-quickgelu/laion400m_e32"


@lru_cache
def bytes_to_unicode() -> dict[int, str]:
    """Reversible byte -> printable unicode map used by CLIP's byte-level BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Tokenizer:
    """Port of OpenCLIP's SimpleTokenizer (lower-case cleaning, 77-token context)."""

    def __init__(self, vocab_path: str | Path) -> None:
        import regex  # \p{L} classes; optional `search` dependency

        lines = gzip.open(vocab_path).read().decode("utf-8").split("\n")
        merges: list[tuple[str, str]] = [
            (a, b) for a, b in (line.split() for line in lines[1:49152 - 256 - 2 + 1])]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab] + ["".join(m) for m in merges]
        vocab += ["<start_of_text>", "<end_of_text>"]
        self.encoder = {v: i for i, v in enumerate(vocab)}
        self.bpe_ranks = {m: i for i, m in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.sot, self.eot = self.encoder["<start_of_text>"], self.encoder["<end_of_text>"]
        self.pat = regex.compile(
            r"""<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|"""
            r"""[^\s\p{L}\p{N}]+""", regex.IGNORECASE)
        self._regex = regex
        self.cache: dict[str, str] = {}

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, 1 << 30))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = tuple(new)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        text = " ".join(html.unescape(html.unescape(text)).split()).lower()
        ids: list[int] = []
        for token in self._regex.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self.bpe(token).split(" "))
        return ids

    def __call__(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), CONTEXT_LENGTH), dtype=np.int64)
        for row, text in enumerate(texts):
            ids = [self.sot, *self.encode(text)][: CONTEXT_LENGTH - 1] + [self.eot]
            out[row, : len(ids)] = ids
        return out


def preprocess(image: np.ndarray, size: int = 224) -> np.ndarray:
    """BGR crop -> normalized CHW float32. Pads to a square instead of center-cropping, so a
    tall person crop keeps head and feet (a center crop would cut both off)."""
    h, w = image.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), (MEAN[::-1] * 255).astype(np.uint8), dtype=np.uint8)
    y, x = (side - h) // 2, (side - w) // 2
    canvas[y:y + h, x:x + w] = image
    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_CUBIC)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def session_model(session: Any) -> str:
    """The CLIP model an ONNX session was exported from (metadata `clip_model`)."""
    meta = session.get_modelmeta().custom_metadata_map or {}
    return meta.get("clip_model") or LEGACY_MODEL


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


class ClipEncoder:
    def __init__(
        self,
        image: str | Path | None = "models/clip_image.onnx",
        text: str | Path | None = "models/clip_text.onnx",
        vocab: str | Path = "models/clip_bpe_vocab.txt.gz",
        num_threads: int = 0,
        batch_size: int = 16,
        device: str = "auto",
    ) -> None:
        """Load only what a process needs: the recorder embeds images, the API embeds text."""
        from argus.engine.onnx_device import create_session

        for path in (image, text):
            if path and not Path(path).exists():
                raise FileNotFoundError(
                    f"{path} not found. Create the CLIP models with: "
                    "uv run --group export -m argus.engine.export_clip")
        self._image = create_session(image, device, num_threads) if image else None
        self._text = create_session(text, device, num_threads) if text else None
        self.tokenizer = Tokenizer(vocab) if text else None
        self.batch_size = batch_size
        models = {session_model(s) for s in (self._image, self._text) if s is not None}
        if len(models) > 1:
            raise ValueError(f"CLIP image and text models differ: {sorted(models)}. "
                             "Export both again with export_clip.py.")
        self.model = models.pop() if models else LEGACY_MODEL  # tag for stored vectors

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        """BGR images of any size -> (N, 512) unit vectors."""
        assert self._image is not None, "image encoder not loaded"
        if not images:
            return np.zeros((0, 512), dtype=np.float32)
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = np.stack([preprocess(im) for im in images[i:i + self.batch_size]])
            out.append(self._image.run(None, {"pixels": batch})[0])
        return _normalize(np.concatenate(out))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        assert self._text is not None and self.tokenizer is not None, "text encoder not loaded"
        return _normalize(self._text.run(None, {"tokens": self.tokenizer(texts)})[0])

    def embed_query(self, description: str) -> np.ndarray:
        """Average of a few phrasings: more robust than the raw words alone."""
        prompts = [description, f"a photo of {description}", f"a cctv image of {description}"]
        return _normalize(self.embed_texts(prompts).mean(0, keepdims=True))[0]
