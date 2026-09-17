#!/usr/bin/env python3
"""L1 -> L2 -> L3 pipeline: RSSI feature(s) -> Radio-FM range prediction -> EKF/VB-AKF tracker.

This is the temporal-fusion test that every single-window Spearman comparison run today could
not answer: does L3's Kalman fusion over time recover good tracking even where L2's per-window
prediction alone looked weak? Generalizes f13_vb_akf.py's VB-adaptive KF (Sarkka & Nummenmaa)
from its hardcoded 2-feature [wrms_mean, env_peak_mean] design to an arbitrary set of k
log-scale measurement features, computed fresh from raw windows (not the old feature-cache
parquet) -- so any combination of L1 (log_rms) and L2 (RAPTORFM prediction) can be tracked.

State: [u, v, o_1, ..., o_k] -- u=log10(distance), v=du/dt, o_i = calibration offset per feature.
Measurement model: z_i = o_i + b_i*u + noise_i, with b_i LOSO-fit (same-band, other sessions).
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/naveen/Desktop/Learned representation/Raptor")
from real_data import load_sessions, DATASET_DIR
from src.models.raptorfm import RAPTORFM

T_CHUNK = 4096
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------- ordered (time-preserving) per-session loader ----------------
def load_session_ordered(session_id, captures):
    """Returns a DataFrame with one row per packet, in chronological order, with columns:
    elapsed_s, distance_m, band, drone, window (T_CHUNK complex64 array), iq_file."""
    packets = pd.read_parquet(DATASET_DIR / "packets.parquet")
    sub_caps = captures[captures['session_id'] == session_id].sort_values('timestamp')
    t0 = sub_caps['timestamp'].min()
    cap_row = sub_caps.set_index('capture_id')
    pkts = packets[packets['capture_id'].isin(cap_row.index)].copy()
    pkts['timestamp'] = pkts['capture_id'].map(cap_row['timestamp'])
    pkts = pkts.sort_values(['timestamp', 'packet_index']).reset_index(drop=True)

    rows = []
    for _, pkt in pkts.iterrows():
        try:
            arr = np.load(pkt['iq_file'], mmap_mode='r')
        except Exception:
            continue
        if len(arr) < T_CHUNK:
            continue
        w = np.asarray(arr[:T_CHUNK], dtype=np.complex64)
        row = cap_row.loc[pkt['capture_id']]
        rows.append(dict(
            elapsed_s=(row['timestamp'] - t0).total_seconds(),
            distance_m=float(row['distance_m']),
            band=str(row['frequency']), drone=row['drone'], window=w,
        ))
    return pd.DataFrame(rows)


# ---------------- L1/L2 feature extraction ----------------
def build_l2_model(ckpt_path):
    m = RAPTORFM(d_model=192, n_heads=6, d_ff=768, n_layers=6, d_physical=64,
                 patch=8, stride=8, mask_ratio=0.6, mode="C").to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    m.load_state_dict(sd, strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def extract_l1_l2(df, l2_model, real_scale, bs=64):
    """Adds log_rms (L1) and l2_pred (L2) columns to df, in the SAME row order (preserves
    chronological order)."""
    W = np.stack(df['window'].values)
    Wn = (W / real_scale).astype(np.complex64)
    Wr = np.stack([Wn.real, Wn.imag], axis=-1).astype(np.float32)  # [N,T,2]
    log_rms = np.empty(len(Wr), dtype=np.float32)
    l2_pred = np.empty(len(Wr), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(Wr), bs):
            xb = torch.from_numpy(Wr[s:s+bs]).unsqueeze(2).float().to(DEVICE)
            out = l2_model(xb)
            log_rms[s:s+bs] = out['log_rms'].squeeze(-1).float().cpu().numpy()
            l2_pred[s:s+bs] = out['range_pred'].float().cpu().numpy()
    df = df.copy()
    df['log_rms'] = log_rms
    df['l2_pred'] = l2_pred
    return df


# ---------------- generalized VB-AKF (k measurement features) ----------------
def fit_shared_slope(all_df, feat, sessions):
    rows = all_df[all_df['session_id'].isin(sessions)]
    y = rows[feat].values
    Xs = rows['logd'].values.reshape(-1, 1)
    dummies = pd.get_dummies(rows['session_id'], drop_first=False).values.astype(float)
    X = np.hstack([Xs, dummies])
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    resid = y - reg.predict(X)
    return float(reg.coef_[0]), float(resid.std())


def run_vb_akf_k(times, Z, b, u0, R_pop, q_pos=0.0001, q_offset=1e-5, v_leak=1.0,
                  nu0=15, forget_rho=0.99, n_iter=3):
    """Z: [n, k]. b: [k]. R_pop: [k,k]. State: [u, v, o_1..o_k]."""
    n, k = Z.shape
    d = k
    x = np.zeros(2 + k); x[0] = u0
    for i in range(k):
        x[2 + i] = Z[0, i] - b[i] * u0
    P = np.diag([0.05 ** 2, 0.02 ** 2] + [0.3 ** 2] * k)
    nu = float(nu0); V = (nu0 - d - 1) * R_pop.copy()
    us = [x[0]]; u_std = [np.sqrt(P[0, 0])]
    H = np.zeros((k, 2 + k))
    for i in range(k):
        H[i, 0] = b[i]; H[i, 2 + i] = 1.0
    for t in range(1, n):
        dt = max(times[t] - times[t - 1], 1e-3)
        decay = np.exp(-v_leak * dt)
        F = np.eye(2 + k)
        F[0, 1] = dt * decay; F[1, 1] = decay
        Q = np.zeros((2 + k, 2 + k))
        Q[:2, :2] = q_pos * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        for i in range(k):
            Q[2 + i, 2 + i] = q_offset * dt
        x_pred = F @ x; P_pred = F @ P @ F.T + Q

        nu_pred = forget_rho * (nu - d - 1) + d + 1
        V_pred = forget_rho * V
        z = Z[t]
        R_hat = V_pred / (nu_pred - d - 1)
        x_it, P_it = x_pred.copy(), P_pred.copy()
        for _ in range(n_iter):
            S = H @ P_pred @ H.T + R_hat
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_it = x_pred + K @ (z - H @ x_pred)
            P_it = P_pred - K @ H @ P_pred
            resid = z - H @ x_it
            Lambda = np.outer(resid, resid) + H @ P_it @ H.T
            nu_new = nu_pred + 1
            V_new = V_pred + Lambda
            R_hat = V_new / (nu_new - d - 1)
        x, P = x_it, P_it
        nu, V = nu_new, V_new
        us.append(x[0]); u_std.append(np.sqrt(max(P[0, 0], 0)))
    return np.array(us), np.array(u_std)


def backtest_session(all_df, held_out, feats, warm_frac=0.1):
    """feats: list of column names to fuse as measurements (e.g. ['log_rms'], ['l2_pred'],
    or ['log_rms','l2_pred']). Returns dict with vb_spearman, mae_dex, n."""
    ho = all_df[all_df['session_id'] == held_out].sort_values('elapsed_s').reset_index(drop=True)
    if len(ho) < 20:
        return None
    band = ho['band'].iloc[0]
    same_band_sessions = sorted(all_df[all_df['band'] == band]['session_id'].unique())
    train_sessions = [s for s in same_band_sessions if s != held_out]
    if len(train_sessions) < 2:
        return None

    bs, rs = [], []
    for feat in feats:
        b, r = fit_shared_slope(all_df, feat, train_sessions)
        bs.append(b); rs.append(r)
    k = len(feats)
    R_pop = np.diag([r ** 2 for r in rs])  # cross-feature correlation unknown here -> diagonal

    times = ho['elapsed_s'].values
    Z = ho[feats].values
    true_u = ho['logd'].values
    u_est, u_std = run_vb_akf_k(times, Z, np.array(bs), u0=true_u[0], R_pop=R_pop)
    warm = max(5, int(warm_frac * len(ho)))
    rho = float(spearmanr(u_est[warm:], true_u[warm:]).statistic)
    mae_dex = float(np.mean(np.abs(u_est[warm:] - true_u[warm:])))
    maxerr = float(np.max(np.abs(u_est[warm:] - true_u[warm:])))
    return dict(session=held_out, band=band, n=len(ho), feats=feats,
                vb_spearman=round(rho, 3), mae_dex=round(mae_dex, 3), max_abs_err=round(maxerr, 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--l2_ckpt', type=str, required=True)
    p.add_argument('--real_scale', type=float, required=True,
                   help="scale factor used when the L2 model was trained on real data (from its manifest)")
    p.add_argument('--eval_sessions', type=str, required=True, help="comma-separated session ids to backtest")
    p.add_argument('--out_name', type=str, default='l1l2l3_eval')
    args = p.parse_args()

    OUT = Path(__file__).resolve().parent / f"results_{args.out_name}"
    OUT.mkdir(parents=True, exist_ok=True)

    captures = load_sessions()
    all_sessions = sorted(captures['session_id'].unique())
    print(f"Loading L2 model from {args.l2_ckpt}...", flush=True)
    l2_model = build_l2_model(args.l2_ckpt)

    print("Loading and extracting L1/L2 features for ALL sessions (needed for LOSO slope fits)...", flush=True)
    dfs = []
    for sid in all_sessions:
        df = load_session_ordered(sid, captures)
        if len(df) == 0:
            continue
        df = extract_l1_l2(df, l2_model, args.real_scale)
        df['session_id'] = sid
        df['logd'] = np.log10(df['distance_m'].clip(lower=1))
        dfs.append(df)
        print(f"  {sid}: {len(df)} packets", flush=True)
    all_df = pd.concat(dfs, ignore_index=True)

    eval_sessions = [s.strip() for s in args.eval_sessions.split(',')]
    feat_configs = [['log_rms'], ['l2_pred'], ['log_rms', 'l2_pred']]
    rows = []
    for held_out in eval_sessions:
        for feats in feat_configs:
            r = backtest_session(all_df, held_out, feats)
            if r:
                rows.append(r)
                print(json.dumps(r), flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)
    print("\n=== SUMMARY (mean vb_spearman by feature set) ===")
    print(res.groupby(res['feats'].apply(tuple))['vb_spearman'].agg(['mean', 'count']))


if __name__ == "__main__":
    main()
