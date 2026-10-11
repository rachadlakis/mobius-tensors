<img src="../../Images/brand-kit/logo/logo-horizontal-light.png" alt="Mobius Tensors" width="200" align="right" />

# Distributed Training

*A Mobius Tensors project.*

Training and fine-tuning models across several GPUs or machines.

| | |
| --- | --- |
| **Status** | Not started: [projects/distributed-training](../../projects/distributed-training/) is an empty folder |
| **Code** | — |
| **Docs** | — |

## Draft scope (to confirm)

*From the services list in the website repo's `ideas.txt` ("distributed training", "train LLMs,
object detection, VLMs", "fine-tuning as a service"); nothing is decided yet.*

- Training and fine-tuning LLMs on a customer's data, as a service.
- Training and fine-tuning vision-language models (VLMs).
- Training object detection models for a customer's own objects.
- Training detectors at scale, on rented GPUs.

First internal use: the detectors Histor needs and doesn't have yet (our own YOLOX plate finder,
fire and smoke, per-customer objects). These train in hours on one rented GPU, so they don't need
multi-GPU training yet ([Histor DOCUMENTATION.md, section 45.3](../../projects/histor/DOCUMENTATION.md#453-models-and-training)).
