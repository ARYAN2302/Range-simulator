# Range-Simulator

Synthetic RF data generation + a trained model, for estimating drone distance from a single
passive radio receiver.

## Status: not a working capability yet

Two honest problems, stated plainly:

**1. It needs to already know the starting distance.** Everything in this repo tracks *changes*
in distance over time — it does not estimate distance from a cold start (no prior information).
That's not a bug we ran out of time to fix. A single receiver picking up signal strength cannot
tell "close + weak transmitter" apart from "far + strong transmitter" — the information isn't
there. Fixing this needs either a precisely known transmit power, or a real reference distance
at the start of tracking. Neither exists in this repo.

**2. The one bucket-accuracy result we have is from a single test, not proven to repeat.** We
only tested on one held-out flight. We have not checked whether the same number holds up on a
different held-out flight. Given this exact type of result has swung from very good to
completely broken across different sessions earlier in this project, one result should not be
trusted as a working number yet.

**What would actually fix problem #1:** angle. If you know the *direction* the signal is coming
from, and you have more than one antenna, geometry alone can give you position over time — no
knowledge of transmit power needed. This repo doesn't do that yet. A separate two-antenna
direction-finding model exists and would need to be combined with this one.

## The bucket-prediction numbers

Tested on one held-out real flight (`979_S4`), distance buckets: 0-100m, 100-300m, 300-700m,
700-1500m, 1500m+.

| method | % of readings in the correct bucket |
|---|---|
| just guess the most common bucket, every time | 39.5% |
| assume the drone never moved from its starting bucket | 22.7% |
| **raw signal strength, tracked over time** | **55.5%** |
| trained model, tracked over time | 47.9% |
| trained model, but with random (untrained) weights | 44.0% |

Reading this correctly: tracking beats both baselines, which is a real effect on this one flight.
But raw signal strength alone beats the trained model — the model isn't adding value yet. And
this is one flight, not a validated number (see problem #2 above).

## Why this got built this way

The real recorded data can't train a model directly: within almost every recording session,
distance and elapsed time move together (as the drone flew further, time also passed), so any
model trained on it just learns to read the clock. This was proven several independent ways.

The fix: generate synthetic training data where distance is randomized independently of
everything else, train on that, then carefully bring real data back in for calibration.

## What's in this repo

- `synth_pipeline/simulate.py` — generates synthetic training data: real captured burst shapes,
  fake (but physically correct) distance-vs-signal-strength relationship on top, plus realistic
  hardware noise (via `torchsig`).
- `raptor_model/` — the neural network (tokenizer + encoder + decoder), based on the published
  [Radio-FM paper](https://arxiv.org/abs/2608.05793). Fixed two real bugs in it this session (see
  below).
- `synth_pipeline/l1_rssi.py` — the raw signal-strength measurement, as its own standalone
  function (no model needed to compute it).
- `synth_pipeline/train_raptorfm_joint.py` — trains the model two ways at once: predicting
  distance (on fake data) and reconstructing real radio signals (on real data, no labels needed,
  so no risk of the clock-reading problem above).
- `synth_pipeline/l1l2l3_pipeline.py`, `bucket_pipeline_979.py` — the full pipeline: signal
  strength + model prediction → tracked over time → converted to a bucket prediction, scored
  against real held-out flights.
- `synth_pipeline/diagnose_transfer.py`, `ablate_physfeat.py` — diagnostic scripts used to find
  out *why* the model wasn't beating a random baseline (see next section).

## What we found, in order

1. **The model wasn't beating an untrained random network.** Ran a fair test (same setup, one
   with the trained model, one with random weights) — they were roughly tied.
2. **Found why:** the network's main output is deliberately normalized in a way that erases
   absolute signal strength — the one thing that actually correlates with real distance. A single
   free number the code already computed (`log_rms`, the raw signal strength) predicted real
   distance far better than the trained model did.
3. **Fixed it** by feeding the raw signal strength to the model's output layer directly. Numbers
   jumped a lot — but a follow-up test showed almost all of that jump was just from the raw
   signal strength itself, not anything the model learned.
4. **Re-tested properly** (fixed a data-generation bug, fixed an unseeded random-comparison bug,
   used a stricter validation method): the trained model still didn't clearly beat random, though
   the gap got smaller.
5. **Tested on a drone the model had never seen at all** (excluded from every training stage, not
   just the labels) — it generalized about as well to the new drone as to a held-out session of a
   drone it already knew. That's a real, useful finding on its own.
6. **Realized the metric was wrong.** Everything above used rank correlation, which checks
   ordering, not whether a prediction lands in the right distance bucket. Switched to actual
   bucket accuracy — see the numbers above.

## Two real bugs fixed in the model architecture

- The network's cross-channel attention block was adding its own output twice (a residual-connection
  bug) and skipping a normalization step the original paper's design has.
- One of the realistic hardware-noise effects (`DigitalAGC`) was accidentally erasing the exact
  signal-strength-vs-distance relationship the whole pipeline depends on. Removed it.

## Hardware note

The real deployed system has 4 synchronized receive channels (2× AD9361), not the single channel
used here. That's what would enable the angle-based approach described above. Not built yet —
recorded as a metadata reference only (`simulate.hardware_spec()`).

## Usage

Paths in these scripts are hardcoded to the original development machine — adjust `DATASET_DIR`
and the Raptor-repo path near the top of each file before running elsewhere. Needs `torch`,
`numpy`, `pandas`, `scipy`, `scikit-learn`, `torchsig`.

```bash
# Generate a synthetic dataset
python synth_pipeline/generate_dataset.py --name my_dataset --n_train 20000 --n_val 3000

# Train the model
python synth_pipeline/train_raptorfm_joint.py --dataset_name my_dataset --epochs 15

# Adapt + evaluate on a held-out real session
python synth_pipeline/train_bitfit_raptorfm.py --real_holdout_session <session> \
  --pretrained_ckpt <checkpoint>.pt

# Full pipeline -> bucket accuracy
python synth_pipeline/bucket_pipeline_979.py --l2_ckpt <checkpoint>.pt --real_scale <value> \
  --eval_session <session>
```
