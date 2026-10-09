"""Export the pose model (body keypoints) to ONNX, for torch-free pose estimation.

Only needed once, on any machine with internet; the recorder then runs it with ONNX Runtime.

    uv run --group export src/export_pose.py      # -> models/pose_vitpose_b.onnx (~340 MB)

Default: ViTPose-B "simple" (usyd-community/vitpose-base-simple on Hugging Face): Apache-2.0,
fine-tuned on COCO keypoints only (backbone pre-trained on ImageNet-1k with MAE), 75.8 AP on
COCO val. It takes 256x192 person crops and returns 17 heatmaps of 64x48 (see src/pose.py).

The faster RTMPose-t (`uv run src/download_model.py pose_rtmpose_t`) was trained on seven
datasets, some of them research-only: use it for trials, not on client sites.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODEL_ID = "usyd-community/vitpose-base-simple"


def main() -> None:
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import VitPoseForPoseEstimation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--out", type=Path, default=Path("models/pose_vitpose_b.onnx"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model = VitPoseForPoseEstimation.from_pretrained(args.model)
    model.eval()

    class Heatmaps(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m = model

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            return self.m(pixel_values=pixels).heatmaps

    example = torch.randn(2, 3, 256, 192)
    torch.onnx.export(
        Heatmaps(), (example,), str(args.out), input_names=["input"], output_names=["heatmaps"],
        dynamic_axes={"input": {0: "batch"}, "heatmaps": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    # Sanity check: ONNX Runtime must give the same heatmaps as PyTorch.
    with torch.no_grad():
        expected = Heatmaps()(example).numpy()
    got = ort.InferenceSession(str(args.out)).run(None, {"input": example.numpy()})[0]
    print(f"max difference vs PyTorch: {np.abs(expected - got).max():.2e}")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
