#!/usr/bin/env python3
"""Synthetic drone-RF channel simulator.

Design principle (the whole point of this pipeline): take REAL captured burst waveforms
(so signal/protocol structure is realistic) and impose a FULLY CONTROLLED, SYNTHETIC
distance-vs-power relationship on top -- with distance and every nuisance parameter drawn
INDEPENDENTLY per sample (not as a function of generation order or of each other). This is
the randomized-block design principle from the DOE research, implemented in software: it
structurally prevents the exact confound (distance aliased with session/time) that sank
every real-data approach earlier this session, because there is no "session" here at all --
every synthetic sample is an independent draw.

Physics grounding (from this session's real, validated findings, not generic textbook values):
  - amplitude-domain power-law exponent ~1.1-1.45 (implies power exponent ~2.2-2.9),
    matching published A2G LOS measurements once we corrected for amplitude-vs-power units.
  - link budget: received amplitude level set by (drone EIRP + RX gain), both known/dtawable
    quantities, exactly as investigated for the anchor problem.
  - realistic hardware/channel nuisance layered on top via torchsig's Impairments (TX-side:
    clock drift/jitter, IQ imbalance, nonlinear PA, phase noise, quantization; RX-side: the
    same plus AGC and coarse gain change) -- these operate purely on the waveform itself, with
    no knowledge of distance_m, so they add realism without reopening the confound.

Current status: TX/RX hardware impairment CHAINS are lifted directly from torchsig's
`Impairments(level=2)` preset (see `_get_torchsig_chains` below), applied around our own
explicit power-law path-loss + gain + thermal-noise model. torchsig's own statistical Fading
model and its RandAugment ml_transforms (ChannelSwap/TimeReversal/AddSlope/RandomDropSamples)
are deliberately excluded: Fading would impose a second, uncontrolled amplitude perturbation on
top of the path-loss law we already control, and the ml_transforms are classification-style
augmentations (AddSlope in particular injects its own amplitude trend) that would fight the
very distance-vs-amplitude relationship this pipeline exists to teach.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path

from torchsig.signals.signal_types import Signal
from torchsig.transforms.impairments import Impairments

# ---------------------------------------------------------------------------
# Deployed hardware reference (armory.in production system). METADATA ONLY for now: training
# stays single-channel (project scope decision), and T_CHUNK below is a short per-burst
# training window, not a full raw capture. Recorded here -- and stamped into every generated
# dataset's manifest via `hardware_spec()` -- so a future multi-channel/full-capture-length
# extension has the real numbers on hand instead of re-deriving them.
#   - 2x AD9361, hardware-synchronized -> 4 RX channels total, 2 TX auxiliary channels
#   - sample rate: 61.44 Msps per channel (== FS below, already matched)
#   - raw capture buffer: 2,000,000 samples/channel (~32.55 ms) per capture event
#   - instantaneous bandwidth: 56 MHz per AD9361 channel
# ---------------------------------------------------------------------------
HW_N_AD9361 = 2
HW_N_RX_CHANNELS = 4
HW_N_TX_AUX_CHANNELS = 2
HW_SAMPLES_PER_CHANNEL_PER_CAPTURE = 2_000_000
HW_BANDWIDTH_HZ = 56e6
HW_SAMPLE_RATE_HZ = 61.44e6  # == FS below

def hardware_spec():
    """Deployed-hardware reference numbers, for stamping into dataset manifests."""
    return dict(
        n_ad9361=HW_N_AD9361, n_rx_channels=HW_N_RX_CHANNELS, n_tx_aux_channels=HW_N_TX_AUX_CHANNELS,
        samples_per_channel_per_capture=HW_SAMPLES_PER_CHANNEL_PER_CAPTURE,
        bandwidth_hz=HW_BANDWIDTH_HZ, sample_rate_hz=HW_SAMPLE_RATE_HZ,
        capture_duration_s=HW_SAMPLES_PER_CHANNEL_PER_CAPTURE / HW_SAMPLE_RATE_HZ,
        note=("Training currently uses a single channel and short per-burst windows "
              "(T_CHUNK samples), not full multi-channel captures -- this block is a metadata "
              "reference for a future multi-channel extension, not the current data shape."),
    )

_TORCHSIG_CACHE = {}
_AGC_LIKE_TRANSFORMS = {'DigitalAGC', 'CoarseGainChange'}

def _get_torchsig_chains(level=2, seed=0, exclude_agc=True):
    """Build (once, cached) torchsig's TX and RX hardware-impairment transform chains.

    Reuses ONE `Impairments` object per (level, seed) so its internal RNGs are constructed
    once; each call to a transform naturally advances that RNG, so repeated per-sample use
    is already i.i.d. without needing to reseed on every draw.

    exclude_agc (default True): drop DigitalAGC and CoarseGainChange from the RX chain.
    These are specifically designed to renormalize received amplitude toward a reference
    level -- i.e. to REMOVE the amplitude-vs-range mapping this whole pipeline exists to
    teach. Including them (the original default) was a bug: it meant every synthetic sample
    downstream of the RX chain had its distance-carrying amplitude information partially
    scrambled by design, before the backbone ever saw it.
    """
    key = (level, seed, exclude_agc)
    if key not in _TORCHSIG_CACHE:
        imp = Impairments(level=level, seed=seed)
        tx_chain = imp.get_signal_transforms()
        if level >= 2:
            tx_chain = tx_chain[:-1]   # drop channel_models=[Fading]; we impose our own path loss
        rx_chain = imp.get_dataset_transforms()[1:]   # drop the RandAugment ml_transforms
        if exclude_agc:
            rx_chain = [t for t in rx_chain
                        if type(getattr(t, 'transform', t)).__name__ not in _AGC_LIKE_TRANSFORMS]
        _TORCHSIG_CACHE[key] = (tx_chain, rx_chain)
    return _TORCHSIG_CACHE[key]


def _run_chain(chain, w):
    sig = Signal(data=np.ascontiguousarray(w, dtype=np.complex64))
    for t in chain:
        sig = t(sig)
    return sig.data

DATASET_DIR = Path("/home/naveen/Desktop/Learned representation/experiments/rf_range_packet_dataset_v1")
T_CHUNK = 4096
FS = 61.44e6

EIRP_DBM = {'Mini_2': 26.0, 'Mini_Pro_4': 33.0, 'mini_5_pro': 33.0}
BANDS_HZ = {'2440': 2.44e9, '5770': 5.77e9}

class WaveformPool:
    """Loads a pool of real burst waveforms into memory once, as (drone, band) -> list of arrays.
    These supply realistic protocol/signal STRUCTURE only -- their original amplitude/distance
    association is discarded entirely; we impose our own controlled physics on top.

    exclude_drones: drone name(s) to leave out of the pool entirely -- for a genuine unseen-drone
    generalization test, this must exclude the target drone's waveforms here too, not just its
    distance labels, since burst SHAPE (not just imposed amplitude) comes from these templates.
    """
    def __init__(self, max_per_group=400, seed=0, exclude_drones=None):
        exclude_drones = set(exclude_drones or [])
        packets = pd.read_parquet(DATASET_DIR / "packets.parquet")
        captures = pd.read_parquet(DATASET_DIR / "captures.parquet").set_index('capture_id')
        packets['drone'] = packets['capture_id'].map(captures['drone'])
        packets['freq_band'] = packets['capture_id'].map(captures['frequency'])
        if exclude_drones:
            packets = packets[~packets['drone'].isin(exclude_drones)]
        rng = np.random.default_rng(seed)
        self.pool = {}
        for (drone, band), g in packets.groupby(['drone', 'freq_band']):
            paths = g['iq_file'].tolist()
            rng.shuffle(paths)
            arrs = []
            for p in paths[:max_per_group]:
                try:
                    a = np.load(p, mmap_mode='r')
                    if len(a) < T_CHUNK:
                        continue
                    arrs.append(p)  # store path, load+crop lazily per draw (memory-friendly)
                except Exception:
                    continue
            if arrs:
                self.pool[(drone, str(band))] = arrs
        print(f"WaveformPool: {sum(len(v) for v in self.pool.values())} usable templates across {len(self.pool)} (drone,band) groups")

    def draw_window(self, rng):
        """Return (raw_window[T_CHUNK] complex64, drone, band)."""
        key = list(self.pool.keys())[rng.integers(0, len(self.pool))]
        drone, band = key
        path = self.pool[key][rng.integers(0, len(self.pool[key]))]
        arr = np.load(path, mmap_mode='r')
        L = len(arr)
        s = rng.integers(0, max(1, L - T_CHUNK))
        w = np.asarray(arr[s:s+T_CHUNK], dtype=np.complex64)
        if len(w) < T_CHUNK:
            w = np.pad(w, (0, T_CHUNK - len(w)))
        return w, drone, band


def apply_channel(window, distance_m, drone, band, rng,
                   exponent_mean=2.55, exponent_std=0.35,   # power-law exponent, centered on this
                                                              # session's validated range 2.2-2.9
                   gain_db_range=(25, 45),                   # plausible RX gain sweep (robustness)
                   nuisance_db_std=6.0,                      # per-sample independent nuisance
                                                              # (antenna orientation etc.) -- this
                                                              # is what real data could never cleanly
                                                              # separate from distance; here it's
                                                              # drawn independently BY CONSTRUCTION.
                   noise_floor_scale=1.0,
                   use_torchsig=True, torchsig_level=2, torchsig_seed=0, torchsig_exclude_agc=True):
    """Impose a controlled, confound-free distance-vs-power relationship on a real waveform,
    with torchsig TX/RX hardware-impairment chains layered around it (see module docstring)."""
    w = window.copy()
    amp = np.abs(w)
    rms = np.sqrt(np.mean(amp**2)) + 1e-12
    w = w / rms  # strip original (uncontrolled) amplitude entirely

    if use_torchsig:
        tx_chain, rx_chain = _get_torchsig_chains(torchsig_level, torchsig_seed, torchsig_exclude_agc)
        w = _run_chain(tx_chain, w)
        # TX impairments (e.g. NonlinearAmplifier compression, Quantize) can shift the RMS
        # level; re-normalize so the amplitude scale below is fully determined by OUR gain/
        # path-loss model, not by whatever intensity torchsig happened to draw this sample.
        rms2 = np.sqrt(np.mean(np.abs(w) ** 2)) + 1e-12
        w = w / rms2

    eirp_dbm = EIRP_DBM.get(drone, 30.0)
    gain_db = rng.uniform(*gain_db_range)
    nuisance_db = rng.normal(0, nuisance_db_std)   # i.i.d. per sample, independent of distance
    exponent = max(0.5, rng.normal(exponent_mean, exponent_std))  # power-law exponent, this draw

    # received power (dB, arbitrary reference) = link budget - path loss (log-distance power law)
    d_ref = 100.0  # reference distance, arbitrary; only relative levels matter here
    path_loss_db = 10 * exponent * np.log10(max(distance_m, 1.0) / d_ref)
    level_db = eirp_dbm + gain_db + nuisance_db - path_loss_db
    amplitude_scale = 10 ** (level_db / 20.0)  # dB -> linear amplitude factor

    w = w * amplitude_scale

    # constant-ish receiver (thermal) noise floor, added pre-ADC -- independent of distance, as
    # in reality. Added before the RX chain so RX-side digital effects (AGC, quantization) see
    # the same signal+noise mix a real receiver's digital backend would.
    noise_level = noise_floor_scale * rng.uniform(0.5, 1.5)
    noise = (rng.normal(0, 1, w.shape) + 1j*rng.normal(0, 1, w.shape)).astype(np.complex64) * noise_level
    w = w + noise

    if use_torchsig:
        w = _run_chain(rx_chain, w)

    return w.astype(np.complex64)


def make_synthetic_batch(pool: WaveformPool, n, rng, dist_range=(10, 8000), channel_kwargs=None):
    """Generate n synthetic (window, distance_m, drone, band) samples, fully i.i.d."""
    channel_kwargs = channel_kwargs or {}
    out_w = np.empty((n, T_CHUNK, 2), dtype=np.float32)
    out_d = np.empty(n, dtype=np.float64)
    out_drone = []
    out_band = []
    for i in range(n):
        raw, drone, band = pool.draw_window(rng)
        d = float(np.exp(rng.uniform(np.log(dist_range[0]), np.log(dist_range[1]))))  # log-uniform
        w = apply_channel(raw, d, drone, band, rng, **channel_kwargs)
        out_w[i, :, 0] = w.real; out_w[i, :, 1] = w.imag
        out_d[i] = d; out_drone.append(drone); out_band.append(band)
    return out_w, out_d, out_drone, out_band


DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

def load_cached_dataset(name, split="train"):
    """Load a dataset previously written by generate_dataset.py: returns (W, D, drones, bands,
    manifest). Lets training scripts reuse one generated dataset deterministically across runs
    instead of regenerating an in-process synthetic pool every time."""
    d = DATASETS_DIR / name
    npz = np.load(d / f"{split}.npz", allow_pickle=True)
    manifest = json.load(open(d / "manifest.json"))
    return npz["W"], npz["D"], list(npz["drones"]), list(npz["bands"]), manifest


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pool = WaveformPool(max_per_group=300)
    W, D, drones, bands = make_synthetic_batch(pool, 200, rng, channel_kwargs=dict(use_torchsig=True, torchsig_seed=0))
    print("batch shape", W.shape, "distance range", D.min(), D.max())
    print("hardware reference:", json.dumps(hardware_spec(), indent=2))

    # ---- confound sanity check (the whole point of this pipeline): distance vs generation
    # order must be ~uncorrelated, unlike every real session this whole investigation found ----
    order = np.arange(len(D))
    from scipy.stats import spearmanr
    rho = spearmanr(order, D).statistic
    print(f"Spearman(generation_order, distance) = {rho:.3f}  (real sessions were up to 1.000 -- "
          f"this MUST be near zero by construction)")
    assert abs(rho) < 0.15, "confound leaked into the synthetic generator -- fix before training!"
    print("CONFOUND CHECK PASSED: synthetic distance is not aliased with generation order.")
