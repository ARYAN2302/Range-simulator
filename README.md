# Range-Simulator

Confound-free synthetic data generation + a Radio-FM-style foundation model, built for passive
single-channel RF drone range estimation. This repo captures the synthetic-pretraining research
track of a larger project; the production/EKF-tracker side lives elsewhere.

## Status: not a demonstrated working capability yet

Read this before anything else in this README. The metric used through most of this
investigation (Spearman rank correlation) answers "is the ranking of readings preserved,"
not "does a given reading land in the right distance bucket" — which was the actual original
goal. That mismatch went uncorrected for most of a working session and inflated how positive
earlier results looked.

Once corrected to score actual bucket classification accuracy (see `bucket_pipeline_979.py`),
two things became clear:

1. **Every tracked result in this repo — including the bucket-accuracy numbers — assumes the
   tracker is given the true distance at the start of tracking (an "anchor").** No cold-start
   capability (estimate distance/bucket with zero prior information) was ever built or
   demonstrated. This is not a fixable bug; it follows from a formal observability proof: a
   single passive receiver cannot distinguish "close + weak transmitter" from "far + strong
   transmitter" from signal strength alone — that ambiguity is only resolved by knowing the
   transmit power precisely, or by an external reference distance at first detection.
2. **The one bucket-accuracy result produced (55.5% tracked vs. 39.5% majority-class baseline,
   on `979_S4`) is a single held-out session — n=1.** It has not been validated across multiple
   held-out sessions (a leave-one-session-out sweep across all four `979` sessions), and this
   project has repeatedly seen results swing from strong to catastrophic across sessions for the
   same method. Treat this number as unvalidated, not as a demonstrated capability.

**A path forward identified but not yet built**: bearings-only tracking (Target Motion
Analysis) using angle-only measurements from a coherent antenna array. Angle is a pure geometry
measurement, independent of the transmitter's power, so it doesn't need either a calibrated
transmit power or an external reference distance. RSS-based ranging (everything in this repo)
would demote to a secondary, supporting signal fused into that tracker, not the primary range
source.

## The problem this exists to solve

The team's real captured RF dataset (single-channel IQ, DJI OcuSync-family drones) cannot support
a session/drone-transferable range model trained directly on it: within almost every recording
session, **distance is aliased with elapsed recording time** (Spearman up to 1.00), a classical
unrandomized-experimental-design confound. Every black-box approach tried on the real data alone —
gradient-boosted trees, CNNs, self-supervised transformers, dual-stream cross-attention transformers
— hit this same wall. It's a data identifiability problem, not a capacity or architecture problem.

The fix: stop training on the confounded real data directly. Generate **synthetic** training data
where distance and every nuisance parameter are drawn independently per sample (no "session"
concept exists at all), pretrain the actual predictor on that, and bring real data back in only
through careful, validated adaptation.

## What's in this repo

- **`synth_pipeline/simulate.py`** — the channel simulator. Takes real captured burst waveforms
  (for realistic protocol/signal *structure* only — their original amplitude is discarded) and
  imposes a fully controlled, confound-free distance-vs-power relationship on top: an explicit
  link-budget model (EIRP + gain + power-law path loss, exponent ~2.2–2.9, validated against real
  sessions with genuine path-loss signal) plus torchsig's TX/RX hardware-impairment chains
  (clock drift/jitter, IQ imbalance, phase noise, quantization — deliberately excluding
  `DigitalAGC`/`CoarseGainChange`, which are designed to *remove* exactly the amplitude-vs-range
  mapping this pipeline exists to teach). Every generated dataset self-checks
  `Spearman(generation_order, distance) < 0.15`.
- **`synth_pipeline/generate_dataset.py`** — production dataset-generation CLI: generates once,
  caches train/val splits to disk with a full manifest (generation params, confound check,
  deployed-hardware reference spec), instead of regenerating data inline in every training run.
  Supports `--exclude_drones` for genuine unseen-drone holdout (excludes a drone's waveforms from
  synthetic *sourcing*, not just its labels).
- **`raptor_model/`** — the backbone: `RAPTORFMTokenizer` + `RAPTORFMEncoder` (dual-channel,
  intra-channel attention + inter-channel interaction, per
  [Radio-FM (arXiv:2608.05793)](https://arxiv.org/abs/2608.05793)) + `RAPTORFM`, the fully
  assembled model (tokenizer + encoder + masked-reconstruction decoder + fusion + range head).
  Fixed here: the inter-channel block was double-counting its residual and missing the RMSNorm
  the paper's own architecture diagram shows before it.
- **`synth_pipeline/train_raptorfm_joint.py`** — joint masked-reconstruction (self-supervised,
  60% Channel-Independent masking — the paper's own ablation-identified optimum) + supervised
  range regression. Reconstruction needs no distance label, so real IQ from **any session** can
  be used for it with zero confound risk — that risk only exists in the distance-regression
  signal, not in reconstructing the waveform itself.
- **`synth_pipeline/train_bitfit_*.py`** — downstream adaptation via BitFit (bias/norm-only
  fine-tuning), with a seeded random-backbone control and session- or drone-level held-out
  validation (never the test session/drone itself).
- **`synth_pipeline/l1l2l3_pipeline.py`** — the full inference pipeline: L1 (a raw physical
  RSSI-like feature, `log_rms`) → L2 (the RAPTORFM range prediction) → L3 (a VB-adaptive Kalman
  filter, Särkkä & Nummenmaa 2009, fusing either/both over time with per-session LOSO-fit
  calibration slopes). This is the temporal-fusion test that single-window comparisons can't
  answer: L2's raw per-window prediction can look weak while still producing excellent *tracked*
  output once fused over time.
- **`synth_pipeline/diagnose_transfer.py`**, **`ablate_physfeat.py`**, **`residual_correction.py`**
  — the diagnostic battery used to find *why* early transfer attempts failed: domain-gap tests
  (CORAL/MMD/linear separability), per-layer probing, and CKA between pretrained and random
  backbones. This is what surfaced the RMSNorm bug (see below) rather than a domain-gap or
  insufficient-pretraining explanation.

## What we found, in order

1. **First pretrain-vs-random comparison** (frozen linear probe on `h_cls`, 1000 epochs, 3 real
   held-out sessions): random-init backbone matched or beat the synthetic-pretrained one. Adding
   torchsig realism didn't fix it.
2. **Diagnosis**: `h_cls` is RMSNorm'd at every transformer block, which structurally discards
   absolute amplitude. A single free scalar the tokenizer already computes — `log_rms`, the
   log-RMS of the raw window, zero trainable parameters — correlated with real distance far more
   strongly (Spearman up to −0.89) than anything the learned embedding produced. The architecture
   has a dedicated scale-preserving branch for this (`h_phys`) that no training script had used.
3. **Fix**: feed the head `[h_cls, h_phys, log_rms]` instead of `h_cls` alone. Real-session
   Spearman jumped to 0.85+. An ablation then showed this gain was *almost entirely* `log_rms`
   itself (identical for pretrained and random backbones, since it has no learned parameters) —
   but isolating `[h_cls, h_phys]` alone did show a small, real, consistent pretraining edge on
   the two sessions with genuine physical signal.
4. **A properly controlled re-test** (AGC-impairment bug fixed in the generator, random control
   properly seeded, session-level not window-level validation, BitFit instead of a naive linear
   probe): pretraining still didn't durably beat random on single-window comparisons, but the gap
   narrowed substantially, and further narrowed again once genuine self-supervised (not just
   supervised-regression) pretraining was used on real IQ data.
5. **The L1→L2→L3 pipeline test, with genuine unseen-drone holdout**: L2 was pretrained (SSL +
   supervised) and adapted (BitFit) using only one drone's data (`Mini_2`/`Mini_Pro_4`, 4
   sessions) — the target drone (`mini_5_pro`) was excluded from synthetic waveform sourcing,
   the SSL reconstruction pool, and adaptation entirely, not just its distance labels. L1 was
   also decoupled into its own standalone, model-free function (`l1_rssi.py`) rather than being
   read out of the L2 model's internals.

   Tracked (VB-AKF) Spearman on the three never-seen `mini_5_pro` sessions:

   | session | log_rms | L2 pred (pretrained) | L2 pred (random) | fused (pretrained) | fused (random) |
   |---|---|---|---|---|---|
   | mini5_S1 | 0.601 | 0.495 | 0.539 | 0.661 | 0.617 |
   | mini5_S2 | **−0.956** | −0.934 | −0.735 | −0.952 | −0.939 |
   | mini5_S3 | 0.888 | 0.892 | 0.878 | 0.907 | 0.899 |

   Compared against a same-drone validation session (`979_S1`, held out from gradient updates
   throughout, never from the target drone's synthetic/SSL/adaptation exclusion since it's the
   *training* drone) evaluated the same way: log_rms 0.720, L2 pred 0.708 (pretrained) / 0.715
   (random), fused 0.710 / 0.716.

   `mini5_S2` (n=270, the smallest session) is catastrophically negative for *every* method
   including raw `log_rms` — this is the known data-quality/anchor pathology flagged earlier in
   the investigation, not a drone-generalization failure. Excluding it, unseen-drone performance
   (`mini5_S1`: 0.5–0.66, `mini5_S3`: 0.88–0.91) is essentially on par with the same-drone
   validation session (0.71–0.72) — the pipeline generalizes to a genuinely unseen drone about as
   well as it generalizes to a held-out session of a drone it trained on. Pretrained vs. random
   stays mixed and close throughout (neither consistently wins), consistent with the small-but-
   real pretraining edge found earlier, not a dominant one. Single-window (non-tracked) numbers
   on the same unseen-drone sessions were far weaker and barely distinguishable between methods
   (mini5_S1: log_rms 0.460 / pretrained 0.448 / random 0.450; mini5_S3: 0.640 / 0.644 / 0.657) —
   the L3 temporal fusion is doing real, substantial work here, not just window-by-window scoring.
6. **Bucket-accuracy re-evaluation** (`bucket_pipeline_979.py`, same-drone only — 979 dataset,
   `979_S1/S2/S3` train, `979_S4` held out validation, coarse5 bucket scheme
   0-100/100-300/300-700/700-1500/1500+m): anchor-conditioned tracked bucket accuracy 55.5%
   (`log_rms`) / 47.9% (pretrained model) / 44.0% (random control), vs. a 39.5% majority-class
   baseline and 22.7% freeze-at-anchor baseline. The confusion matrix shows real, systematic
   bias (under-predicting the 100-300m bucket, over-predicting 300-700m) — this is a single
   held-out session, not cross-validated, see Status section above.

## Deployed-hardware reference (metadata only, not yet used)

The actual production system is 2× AD9361, hardware-synchronized (4 RX channels, 2 TX auxiliary),
61.44 Msps, 56 MHz instantaneous bandwidth per chip, ~2M samples/channel per raw capture. This
pipeline currently trains single-channel on short per-burst windows — the hardware spec is
recorded (`simulate.hardware_spec()`, stamped into every generated dataset's manifest) as a
forward-reference for a future multi-channel extension, not the current data shape. A companion
two-antenna DOA (bearing) model exists separately; the natural extension for full position is
L1 (per-antenna RSSI) → L2a (this repo's range model) + L2b (the DOA model) → L3 (a genuine
nonlinear EKF/UKF fusing range and bearing).

## Other open questions

- One of three unseen-drone test sessions (`mini5_S2`) has a known ground-truth/anchor pathology
  that makes every method fail on it, including the raw physical feature -- it should be treated
  as a data-quality issue to fix/exclude, not folded into a single "unseen drone" headline number
  without the caveat above.
- Pretrained-vs-random-backbone remains genuinely close on both the unseen-drone and same-drone
  validation sessions -- the SSL pretraining recipe used here has a small, real edge in some
  configurations, not a dominant one. Whether a different objective, more waveform diversity, or
  more real IQ diversity would widen that edge is untested.
- Full 2D/3D position estimation is blocked on actually building/using the multi-channel capture
  path — not attempted here.

## Usage

Scripts carry absolute paths from the original development machine
(`/home/naveen/Desktop/Learned representation/...`) at the top of each file — adjust the path
constants (`DATASET_DIR` in `simulate.py`/`real_data.py`, the `sys.path.insert(...)` calls
pointing at the Raptor repo, and each script's `OUT`/checkpoint paths) for your own environment
before running. Requires `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, and `torchsig`.

```bash
# Generate a confound-free synthetic dataset
python synth_pipeline/generate_dataset.py --name my_dataset --n_train 20000 --n_val 3000

# Joint SSL + supervised pretraining
python synth_pipeline/train_raptorfm_joint.py --dataset_name my_dataset --epochs 15

# BitFit adaptation + evaluation on a held-out real session
python synth_pipeline/train_bitfit_raptorfm.py --real_holdout_session <session> \
  --pretrained_ckpt <checkpoint>.pt

# Full L1->L2->L3 pipeline evaluation
python synth_pipeline/l1l2l3_pipeline.py --l2_ckpt <checkpoint>.pt --real_scale <value> \
  --eval_sessions <session1>,<session2>
```
