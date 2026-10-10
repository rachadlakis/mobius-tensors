# Distributed Training

Training and fine-tuning models across several GPUs or machines.

| | |
| --- | --- |
| **Status** | Not started: [projects/distributed-training](../../projects/distributed-training/) is an empty folder |
| **Code** | — |
| **Docs** | — |

## Draft scope (to confirm)

*From the services list in the website repo's `ideas.txt` ("distributed training", "train LLMs,
object detection, VLMs", "fine-tuning as a service"); nothing is decided yet.*

- Fine-tuning language and vision-language models on a customer's data, as a service.
- Training detectors at scale, on rented GPUs.

First internal use: the detectors Argus needs and doesn't have yet (our own YOLOX plate finder,
fire and smoke, per-customer objects). These train in hours on one rented GPU, so they don't need
multi-GPU training yet ([Argus DOCUMENTATION.md, section 45.3](../../projects/argus/DOCUMENTATION.md#453-models-and-training)).
