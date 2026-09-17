#!/usr/bin/env python3
"""Production synthetic-dataset generator.

Decouples data generation from training: every earlier script in this pipeline (train_synth.py,
train_mixed.py, train_frozen_adapter.py) built its own synthetic pool in-process, so re-running
or comparing experiments meant regenerating data from scratch each time with no durable record
of exactly what was generated. This script generates ONCE, writes train/val splits to disk, and
stamps a manifest with every generation parameter plus the deployed-hardware reference spec
(see simulate.hardware_spec) -- so a dataset is a reproducible, inspectable artifact rather than
a side effect of a training run.

Usage:
    python generate_dataset.py --name v1 --n_train 20000 --n_val 3000
    python generate_dataset.py --name v1_no_torchsig --no-use_torchsig   # ablation / comparison

Output layout: datasets/<name>/{train.npz, val.npz, manifest.json}
Load with simulate.load_cached_dataset(name, split).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from simulate import WaveformPool, make_synthetic_batch, T_CHUNK, FS, hardware_spec, DATASETS_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', type=str, required=True, help="dataset name -> datasets/<name>/")
    p.add_argument('--n_train', type=int, default=20000)
    p.add_argument('--n_val', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dist_lo', type=float, default=10.0)
    p.add_argument('--dist_hi', type=float, default=8000.0)
    p.add_argument('--exponent_mean', type=float, default=2.55)
    p.add_argument('--exponent_std', type=float, default=0.35)
    p.add_argument('--gain_db_lo', type=float, default=25.0)
    p.add_argument('--gain_db_hi', type=float, default=45.0)
    p.add_argument('--nuisance_db_std', type=float, default=6.0)
    p.add_argument('--noise_floor_scale', type=float, default=1.0)
    p.add_argument('--use_torchsig', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--torchsig_level', type=int, default=2, choices=[0, 1, 2])
    p.add_argument('--max_per_group', type=int, default=400,
                    help="max real burst waveforms loaded per (drone,band) group in WaveformPool")
    p.add_argument('--exclude_drones', type=str, default='', help="comma-separated drone names to "
                    "exclude from WaveformPool entirely -- for a genuine unseen-drone eval, e.g. mini_5_pro")
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    out_dir = DATASETS_DIR / args.name
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already exists -- pass --overwrite to regenerate it")
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_kwargs = dict(
        exponent_mean=args.exponent_mean, exponent_std=args.exponent_std,
        gain_db_range=(args.gain_db_lo, args.gain_db_hi),
        nuisance_db_std=args.nuisance_db_std, noise_floor_scale=args.noise_floor_scale,
        use_torchsig=args.use_torchsig, torchsig_level=args.torchsig_level, torchsig_seed=args.seed,
    )
    print(f"Generating dataset '{args.name}' -> {out_dir}", flush=True)
    print(f"channel_kwargs: {channel_kwargs}", flush=True)

    exclude_drones = [d.strip() for d in args.exclude_drones.split(',') if d.strip()]
    rng = np.random.default_rng(args.seed)
    pool = WaveformPool(max_per_group=args.max_per_group, seed=args.seed, exclude_drones=exclude_drones)

    t0 = time.time()
    print("Generating train split...", flush=True)
    Wtr, Dtr, drones_tr, bands_tr = make_synthetic_batch(
        pool, args.n_train, rng, dist_range=(args.dist_lo, args.dist_hi), channel_kwargs=channel_kwargs)
    print("Generating val split...", flush=True)
    Wva, Dva, drones_va, bands_va = make_synthetic_batch(
        pool, args.n_val, rng, dist_range=(args.dist_lo, args.dist_hi), channel_kwargs=channel_kwargs)
    gen_time_s = time.time() - t0

    # confound check -- every dataset this pipeline produces must satisfy this, by construction
    order = np.arange(len(Dtr))
    rho = float(spearmanr(order, Dtr).statistic)
    if abs(rho) >= 0.15:
        raise RuntimeError(f"confound leaked into generated dataset: Spearman(order,distance)={rho:.3f}")

    np.savez_compressed(out_dir / "train.npz", W=Wtr, D=Dtr,
                         drones=np.array(drones_tr, dtype=object), bands=np.array(bands_tr, dtype=object))
    np.savez_compressed(out_dir / "val.npz", W=Wva, D=Dva,
                         drones=np.array(drones_va, dtype=object), bands=np.array(bands_va, dtype=object))

    manifest = dict(
        name=args.name, generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_train=len(Wtr), n_val=len(Wva), t_chunk=T_CHUNK, fs_hz=FS,
        dist_range_m=[args.dist_lo, args.dist_hi], seed=args.seed,
        exclude_drones=exclude_drones,
        channel_kwargs=channel_kwargs,
        confound_check_spearman_order_vs_distance=rho,
        generation_time_s=round(gen_time_s, 1),
        hardware_reference=hardware_spec(),
    )
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=2)

    print(f"\nDONE in {gen_time_s:.1f}s -> {out_dir}", flush=True)
    print(f"confound check: Spearman(order,distance)={rho:.3f}  (must be < 0.15)", flush=True)


if __name__ == "__main__":
    main()
