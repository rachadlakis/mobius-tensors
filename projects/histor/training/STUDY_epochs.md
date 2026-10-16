# Study: do more epochs catch more fire and smoke?

Started 2026-10-15. Question from the user: would training longer than 100 epochs make the
fire/smoke detector miss fewer fires? Money is not the constraint. A missed fire is the outcome
that matters.

## What we know before the study

**Every run so far changed more than one thing at once:**
- Run 1 (Tiny 60) and run 2 (Small 60) used data v1.
- Runs 3 and 4 (100 epochs) used data v2, and run 3 was stopped and resumed.

So no pair of runs isolates the effect of epochs.

**The differences seen so far are within noise.** Paired bootstrap over the 2,332 old-source test
images (v2 split, 1,000 resamples): fires and smoke caught at a fixed per-image false-alarm rate,
difference against Small 60:

| Comparison | Fire, 2 % false | Fire, 5 % false | Smoke, 2 % false | Smoke, 5 % false |
| --- | --- | --- | --- | --- |
| Small 100 − Small 60 | +4.4 [−1.4, +11.7] | −1.8 [−3.2, +0.8] | +1.1 [−8.3, +7.4] | +2.1 [−2.8, +7.5] |
| Tiny 100 − Small 60 | +1.5 [−4.8, +8.9] | −2.0 [−4.9, +0.2] | +3.2 [−8.1, +9.2] | +0.0 [−3.8, +4.5] |

Points, with the 95 % interval in brackets. Every interval includes zero.

**What the curves show:**
- Validation rises at the end of every run, at 60 and at 100 epochs alike. That rise is the
  schedule ending (learning rate decay, then no mosaic for the last 15 epochs), not proof that more
  epochs are needed.
- Small at 100 epochs: validation flat over the last 3 epochs while the training loss still falls
  fast (2.75 → 2.58), the first sign of overfitting.
- Tiny at 100 epochs: still creeping up (0.3689 → 0.3691).

## Design

One change only: the number of epochs. Everything else is fixed:
- data v2 (the HF backup `Rachad/histor-fire-smoke`);
- 640 px, batch 32, 12 workers, LibreYOLO 1.6.0, the same settings as runs 3–4;
- `--patience 0` (no early stopping);
- seed 0;
- the same GPU type.

A repeat with a different seed measures how much a result moves by chance alone.

| Arm | Model | Epochs | Seed | Status | GPU hours (est.) |
| --- | --- | --- | --- | --- | --- |
| T60 | Tiny | 60 | 0 | new | 2.8 |
| T100 | Tiny | 100 | 0 | **done** (run 4) | – |
| T100-s1 | Tiny | 100 | 1 | new: training noise | 4.7 |
| T150 | Tiny | 150 | 0 | new | 7.1 |
| T200 | Tiny | 200 | 0 | new | 9.4 |
| S60 | Small | 60 | 0 | new (run 2 used v1 data) | 3.6 |
| S100 | Small | 100 | 0 | **done** (run 3, resumed at 63; the resume restores optimizer, EMA and schedule) | – |
| S150 | Small | 150 | 0 | new | 8.9 |
| S200 | Small | 200 | 0 | new | 11.8 |

That is 7 new runs, about 49 GPU hours plus about 3 hours of setup and scoring. At $0.72/h
(RTX PRO 4500 Blackwell), that's **about $38**. Run in parallel on 7 pods, it takes about 12 hours
of wall time (the longest arm).

## Measurements

All on the v2 test split. It's the same for every arm, and no arm trains on it or picks a
checkpoint with it.

**Primary, what matters for safety:** fires caught and smoke caught per image at a fixed false-alarm
rate (2 % and 5 % of pictures without that class). Reported:
- on old sources and on indoor;
- as paired bootstrap 95 % intervals against the 100-epoch arm of the same size.

**Secondary:**
- mAP50 and mAP50-95;
- distant smoke (Pyro);
- the validation curve and train/validation gap per epoch (overfitting);
- CPU speed is the same within a size, so it isn't measured.

**Later, once we have them:** per-event detection on real fire videos (time to the first alarm).
No permissively licensed fire-video test set has been picked yet.

## Decision rule (fixed before the results)

A longer schedule becomes the default only if all of these hold:
1. It improves fires caught or smoke caught at 5 % false alarms by **at least 2 points**, with the
   95 % interval above zero.
2. It loses nothing elsewhere (indoor, Pyro) beyond the seed-to-seed spread of the T100 pair.
3. The validation curve shows no overfitting at the chosen checkpoint.

Otherwise we keep 100 epochs and spend effort on data instead: indoor CCTV fire and smoke,
pictures where fire fills the frame, and hard negatives.

## Results

*To fill in.*
