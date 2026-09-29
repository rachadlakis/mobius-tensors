"""Export an OpenCLIP model to two ONNX files (image + text encoders) for torch-free search.

Only needed once, on any machine with internet; the server then runs CLIP with ONNX Runtime.

    uv run --group export src/export_clip.py      # -> models/clip_image.onnx, clip_text.onnx, vocab

Default weights: ViT-B-32 (quickgelu) trained on LAION-400M, downloaded from the OpenCLIP
GitHub release (MIT licence). Kept in float32: int8 quantization was tested and changed
search rankings too much (top-1 agreement fell to 50-70%), so it is opt-in via --quantize.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

WEIGHTS_URL = (
    "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
    "vit_b_32-quickgelu-laion400m_e32-46683a32.pt"
)
ARCH = "ViT-B-32-quickgelu"


def main() -> None:
    import numpy as np
    import onnxruntime as ort
    import open_clip
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="Local .pt file (default: download)")
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--quantize", action="store_true", help="int8 (4x smaller, less accurate)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    weights = args.weights or args.out_dir / "clip_vit_b_32_laion400m.pt"
    if not weights.exists():
        print(f"Downloading {WEIGHTS_URL}")
        urllib.request.urlretrieve(WEIGHTS_URL, weights)

    model, _, _ = open_clip.create_model_and_transforms(ARCH, pretrained=str(weights))
    model.eval()

    class Image(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m = model

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            return self.m.encode_image(pixels)

    class Text(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m = model

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return self.m.encode_text(tokens)

    tokenizer = open_clip.get_tokenizer(ARCH)
    exports = {
        "clip_image": (Image(), torch.randn(2, 3, 224, 224), "pixels"),
        "clip_text": (Text(), tokenizer(["a photo of a person", "a red car"]), "tokens"),
    }
    for name, (module, example, input_name) in exports.items():
        fp32 = args.out_dir / f"{name}.fp32.onnx"
        torch.onnx.export(
            module, (example,), str(fp32), input_names=[input_name], output_names=["embedding"],
            dynamic_axes={input_name: {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
        final = args.out_dir / f"{name}.onnx"
        if not args.quantize:
            fp32.replace(final)
        else:
            quantize_dynamic(str(fp32), str(final), weight_type=QuantType.QInt8, per_channel=True)
            # Sanity check: the int8 model must agree with the fp32 one.
            a = ort.InferenceSession(str(fp32)).run(None, {input_name: example.numpy()})[0]
            b = ort.InferenceSession(str(final)).run(None, {input_name: example.numpy()})[0]
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            print(f"{name}: int8 vs fp32 cosine = {cos.min():.4f}")
            fp32.unlink()
        print(f"Wrote {final} ({final.stat().st_size / 1e6:.0f} MB)")

    # The BPE vocabulary the tokenizer needs, next to the models.
    vocab_src = Path(open_clip.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"
    shutil.copy(vocab_src, args.out_dir / "clip_bpe_vocab.txt.gz")
    with gzip.open(vocab_src) as f:
        assert f.readline()  # readable
    print(f"Wrote {args.out_dir / 'clip_bpe_vocab.txt.gz'}")


if __name__ == "__main__":
    main()
