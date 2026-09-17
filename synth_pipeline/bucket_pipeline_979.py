#!/usr/bin/env python3
"""L1 -> L2 -> L3 pipeline, scored by actual BUCKET classification accuracy -- not Spearman.

Scope, deliberately narrow per explicit instruction: 979 dataset ONLY (Mini_2, Mini_Pro_4),
979_S1/S2/S3 as train (979_S1 as inner-val for epoch selection), 979_S4 held out as the single
validation session. No cross-drone claim is made here at all.

Bucket scheme: the project's own established "coarse5" convention --
  (0,100), (100,300), (300,700), (700,1500), (1500,inf)

IMPORTANT LIMITATION, stated up front, not buried: the VB-AKF tracker (L3) is initialized from
the TRUE distance at t=0 (an anchor). This script reports how well the tracked estimate lands in
the correct bucket over the REST of the session given that anchor -- it does NOT demonstrate
cold-start bucket prediction with no prior information, which remains unsolved (see project
history: single-sensor RSS ranging is provably unobservable without either a calibrated
transmit-power or an external reference). Baselines included specifically to make this limit
visible: a majority-class baseline (always guess the session's most common true bucket) and a
freeze-at-anchor baseline (always guess the anchor's own bucket, i.e. assume zero motion) --
if the tracker doesn't clearly beat freeze-at-anchor, the anchor is doing all the work.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/naveen/Desktop/Learned representation/Raptor")
from real_data import load_sessions
from l1l2l3_pipeline import (load_session_ordered, build_l2_model, extract_l1_l2,
                              fit_shared_slope, run_vb_akf_k)

BUCKETS = [(0, 100), (100, 300), (300, 700), (700, 1500), (1500, np.inf)]
BUCKET_LABELS = ["0-100", "100-300", "300-700", "700-1500", "1500+"]


def to_bucket(distance_m):
    d = np.asarray(distance_m)
    idx = np.zeros(d.shape, dtype=int)
    for i, (lo, hi) in enumerate(BUCKETS):
        idx = np.where((d >= lo) & (d < hi), i, idx)
    idx = np.where(d >= BUCKETS[-1][0], len(BUCKETS) - 1, idx)
    return idx


def evaluate_session_buckets(all_df, held_out, feats, warm_frac=0.1):
    ho = all_df[all_df['session_id'] == held_out].sort_values('elapsed_s').reset_index(drop=True)
    band = ho['band'].iloc[0]
    same_band_sessions = sorted(all_df[all_df['band'] == band]['session_id'].unique())
    train_sessions = [s for s in same_band_sessions if s != held_out]
    if len(train_sessions) < 2:
        return None

    bs, rs = [], []
    for feat in feats:
        b, r = fit_shared_slope(all_df, feat, train_sessions)
        bs.append(b); rs.append(r)
    R_pop = np.diag([r ** 2 for r in rs])

    times = ho['elapsed_s'].values
    Z = ho[feats].values
    true_u = ho['logd'].values
    u_est, u_std = run_vb_akf_k(times, Z, np.array(bs), u0=true_u[0], R_pop=R_pop)

    warm = max(5, int(warm_frac * len(ho)))
    pred_bucket = to_bucket(10 ** u_est[warm:])
    true_bucket = to_bucket(10 ** true_u[warm:])
    anchor_bucket = to_bucket(10 ** true_u[0])

    tracked_acc = float((pred_bucket == true_bucket).mean())
    majority_label = np.bincount(true_bucket, minlength=len(BUCKETS)).argmax()
    majority_acc = float((true_bucket == majority_label).mean())
    freeze_acc = float((true_bucket == anchor_bucket).mean())
    rho = float(spearmanr(u_est[warm:], true_u[warm:]).statistic)  # kept for continuity, not the headline
    cm = confusion_matrix(true_bucket, pred_bucket, labels=list(range(len(BUCKETS))))

    return dict(session=held_out, band=band, feats=feats, n=len(ho), n_scored=len(pred_bucket),
                tracked_bucket_accuracy=round(tracked_acc, 3),
                majority_class_baseline=round(majority_acc, 3),
                freeze_at_anchor_baseline=round(freeze_acc, 3),
                vb_spearman=round(rho, 3),
                confusion_matrix=cm.tolist(), bucket_labels=BUCKET_LABELS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--l2_ckpt', type=str, required=True)
    p.add_argument('--real_scale', type=float, required=True)
    p.add_argument('--eval_session', type=str, default='979_S4')
    p.add_argument('--out_name', type=str, default='bucket_979')
    args = p.parse_args()

    OUT = Path(__file__).resolve().parent / f"results_{args.out_name}"
    OUT.mkdir(parents=True, exist_ok=True)

    captures = load_sessions()
    sessions_979 = sorted([s for s in captures['session_id'].unique() if s.startswith('979')])
    print(f"979 sessions: {sessions_979}", flush=True)
    l2_model = build_l2_model(args.l2_ckpt)

    dfs = []
    for sid in sessions_979:
        df = load_session_ordered(sid, captures)
        df = extract_l1_l2(df, l2_model, args.real_scale)
        df['session_id'] = sid
        df['logd'] = np.log10(df['distance_m'].clip(lower=1))
        dfs.append(df)
        print(f"  {sid}: {len(df)} packets, distance [{df['distance_m'].min():.0f},{df['distance_m'].max():.0f}]m", flush=True)
    all_df = __import__('pandas').concat(dfs, ignore_index=True)

    print(f"\nBucket scheme: {list(zip(BUCKET_LABELS, BUCKETS))}", flush=True)
    print(f"\n=== Evaluating {args.eval_session} (anchor-conditioned tracking, NOT cold-start) ===", flush=True)
    results = {}
    for feats in [['log_rms'], ['l2_pred'], ['log_rms', 'l2_pred']]:
        r = evaluate_session_buckets(all_df, args.eval_session, feats)
        if r is None:
            continue
        key = '+'.join(feats)
        results[key] = r
        print(f"\n--- features: {key} ---", flush=True)
        print(f"n_scored={r['n_scored']}  tracked_bucket_accuracy={r['tracked_bucket_accuracy']:.3f}  "
              f"majority_class_baseline={r['majority_class_baseline']:.3f}  "
              f"freeze_at_anchor_baseline={r['freeze_at_anchor_baseline']:.3f}  "
              f"(vb_spearman={r['vb_spearman']:.3f}, shown for continuity only)", flush=True)
        cm = np.array(r['confusion_matrix'])
        print(f"confusion matrix (rows=true, cols=predicted), labels={BUCKET_LABELS}:")
        print(cm)

    json.dump(results, open(OUT / "results.json", "w"), indent=2)
    print(f"\nSaved to {OUT / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
