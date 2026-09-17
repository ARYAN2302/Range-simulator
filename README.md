# Range-Simulator

Confound-free synthetic data generation + a Radio-FM-style foundation model, built for passive
single-channel RF drone range estimation. This repo captures the synthetic-pretraining research
track of a larger project; the production/EKF-tracker side lives elsewhere.

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
5. **The L1→L2→L3 pipeline test**: once L2's prediction is fused over time through the VB-AKF
   tracker (matching how this would actually be deployed, rather than judged window-by-window),
   both `log_rms` and the model's own prediction reach Spearman 0.8–0.97 on sessions with real
   signal — dramatically higher than any single-window number produced all session, and on at
   least one session the model's own prediction *beat* the raw physical feature once tracked.

## Deployed-hardware reference (metadata only, not yet used)

The actual production system is 2× AD9361, hardware-synchronized (4 RX channels, 2 TX auxiliary),
61.44 Msps, 56 MHz instantaneous bandwidth per chip, ~2M samples/channel per raw capture. This
pipeline currently trains single-channel on short per-burst windows — the hardware spec is
recorded (`simulate.hardware_spec()`, stamped into every generated dataset's manifest) as a
forward-reference for a future multi-channel extension, not the current data shape. A companion
two-antenna DOA (bearing) model exists separately; the natural extension for full position is
L1 (per-antenna RSSI) → L2a (this repo's range model) + L2b (the DOA model) → L3 (a genuine
nonlinear EKF/UKF fusing range and bearing).

## Status / honest open questions

- Whether SSL pretraining's real transferable edge survives a *fully* clean protocol (this run's
  SSL pool excluded the target session's/drone's data; a stricter version would also verify no
  indirect leakage through session-recognition shortcuts).
- The L1→L2→L3 pipeline numbers above are genuinely encouraging but were run with differing
  degrees of drone/session holdout across iterations — check each script's own docstring/manifest
  for exactly what was held out before citing a specific number.
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
