#!/usr/bin/env python3
"""Real-data loader for the mixing curriculum. Extracts (window, true_distance, drone, band,
session_id) tuples from the canonical real dataset, using the same 7-session structure and
T_CHUNK=4096 windowing convention established throughout this project's earlier diagnostics."""
import numpy as np, pandas as pd
from pathlib import Path

DATASET_DIR = Path("/home/naveen/Desktop/Learned representation/experiments/rf_range_packet_dataset_v1")
T_CHUNK = 4096

def load_sessions():
    captures = pd.read_parquet(DATASET_DIR / "captures.parquet")
    captures['timestamp'] = pd.to_datetime(captures['timestamp'])
    captures = captures.sort_values(['dataset', 'timestamp']).reset_index(drop=True)
    captures['gap_s'] = captures.groupby('dataset')['timestamp'].diff().dt.total_seconds()
    captures['session_id'] = (captures['dataset'].astype(str) + "_S" +
        (captures['gap_s'].isna() | (captures['gap_s'] > 60)).groupby(captures['dataset']).cumsum().astype(str))
    return captures

def real_windows_for_sessions(session_ids, max_windows_per_packet=4, seed=0):
    """Returns (windows [N,T_CHUNK,2] float32 raw amplitude, distances [N], drones, bands, session_ids)."""
    captures = load_sessions()
    packets = pd.read_parquet(DATASET_DIR / "packets.parquet")
    cap_row = captures.set_index('capture_id')
    sub_caps = captures[captures['session_id'].isin(session_ids)]
    cids = set(sub_caps['capture_id'])
    pkts = packets[packets['capture_id'].isin(cids)]

    rng = np.random.default_rng(seed)
    W = []; D = []; drones = []; bands = []; sids = []
    for _, pkt in pkts.iterrows():
        try:
            arr = np.load(pkt['iq_file'], mmap_mode='r')
        except Exception:
            continue
        L = len(arr)
        if L < T_CHUNK:
            continue
        nw = min(max_windows_per_packet, L // T_CHUNK)
        if nw <= 0:
            continue
        starts = [0] if nw == 1 else [int(round(i*(L-T_CHUNK)/(nw-1))) for i in range(nw)]
        row = cap_row.loc[pkt['capture_id']]
        d = float(row['distance_m'])
        drone = row['drone']; band = str(row['frequency']); sid = row['session_id']
        for s in starts:
            c = np.asarray(arr[s:s+T_CHUNK], dtype=np.complex64)
            if len(c) < T_CHUNK:
                c = np.pad(c, (0, T_CHUNK - len(c)))
            w = np.stack([c.real, c.imag], -1).astype(np.float32)
            W.append(w); D.append(d); drones.append(drone); bands.append(band); sids.append(sid)
    if not W:
        return None, None, [], [], []
    idx = rng.permutation(len(W))
    W = np.stack(W)[idx]; D = np.array(D)[idx]
    drones = [drones[i] for i in idx]; bands = [bands[i] for i in idx]; sids = [sids[i] for i in idx]
    return W, D, drones, bands, sids

if __name__ == "__main__":
    captures = load_sessions()
    sessions = sorted(captures['session_id'].unique())
    print("Sessions:", sessions)
    W, D, drones, bands, sids = real_windows_for_sessions(sessions[:1])
    print(f"session {sessions[0]}: {len(W)} windows, distance range [{D.min():.0f},{D.max():.0f}]m")
