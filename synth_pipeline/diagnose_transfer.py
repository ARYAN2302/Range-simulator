#!/usr/bin/env python3
"""Decisive sim-to-real transfer diagnostics on CACHED forward passes -- no new training.

Directly implements the "diagnostic battery" from deep_research_brief_6_sim2real.md's answer
(D1/D2/D3): distinguishes (H1) small-gap-but-useless-task, (H2) blocking domain gap, and
(H3) insufficient real data/noise-floor as explanations for why the frozen-adapter comparison
found the torchsig-pretrained backbone did not durably beat a random-backbone control.

Computes, per transformer layer (tokenizer output + each of 6 encoder blocks):
  - CORAL distance + MMD^2 (RBF, median heuristic) + linear domain separability
    (synthetic vs. pooled real), on the PRETRAINED backbone -- tests H2 (domain gap).
  - Ridge-regression probe Spearman(pred, true log10(distance)): fit on synthetic, tested on
    synthetic held-out AND fit on 6-session real pool, tested on each of the 3 held-out real
    sessions -- for both the pretrained and random backbones -- tests H1 vs H3 (does *any*
    layer transfer, and does pretraining help at any layer even if final h_cls doesn't).
  - A trivial log-RMS-only baseline probe at the same real sessions, for reference (this is the
    "amplitude shortcut" the supervised synthetic pretraining is hypothesized to have learned).
  - Linear CKA (Kornblith et al. 2019) between pretrained and random backbones, per layer, on
    the SAME real windows, outlier-trimmed -- tests whether high representational similarity
    explains the random-backbone catch-up seen in the frozen-adapter sweep.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/naveen/Desktop/Learned representation/Raptor")
from real_data import load_sessions, real_windows_for_sessions
from simulate import load_cached_dataset
from src.models.raptorfm_tokenizer import RAPTORFMTokenizer, RoPE
from src.models.raptorfm_encoder import RAPTORFMEncoder

D = 192
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 1234  # fixed, unlike train_frozen_adapter.py's unseeded random-control (a noise source
             # flagged after the sweep) -- this script's random backbone is reproducible.
HELD_OUT_SESSIONS = ["979_S4", "mini5_S3", "mini5_S2"]
CKPT = "/home/naveen/Desktop/Learned representation/experiments/synth_pipeline/results_synth_torchsig_v1/best.pt"
N_SUBSAMPLE = 1500

torch.manual_seed(SEED)
np.random.seed(SEED)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = RAPTORFMTokenizer(d_model=D, d_physical=64, patch=8, stride=8, mask_ratio=0.0, n_heads=6)
        self.enc = RAPTORFMEncoder(d_model=D, n_heads=6, d_ff=768, n_layers=6, d_physical=64)

    def layerwise(self, x):
        """Returns (list of 7 [B,D] CLS embeddings -- tokenizer-out + after each of 6 blocks,
        log_rms [B,1])."""
        t = self.tok(x, None)
        t["i_masked"] = t["i_t"]; t["q_masked"] = t["q_t"]
        t["mask_i"] = torch.zeros_like(t["mask_i"]); t["mask_q"] = torch.zeros_like(t["mask_q"])
        h_i, h_q = t["i_masked"], t["q_masked"]
        cos, sin = t["cos"], t["sin"]
        rope = RoPE(self.enc.d_model)
        layers = [((h_i[:, 0] + h_q[:, 0]) / 2.0)]
        for block in self.enc.blocks:
            h_i, h_q = block(h_i, h_q, cos, sin, rope)
            layers.append((h_i[:, 0] + h_q[:, 0]) / 2.0)
        return layers, t["log_rms"]


def load_backbone(pretrained):
    bb = Backbone().to(DEVICE)
    if pretrained:
        sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
        bb_sd = {k: v for k, v in sd.items() if k.startswith('tok.') or k.startswith('enc.')}
        bb.load_state_dict(bb_sd, strict=True)
    bb.eval()
    for p in bb.parameters():
        p.requires_grad_(False)
    return bb


def extract_layerwise(bb, W, bs=48):
    """W: [N,T,2] float32 raw amplitude. Returns (layers: list of 7 [N,D] np arrays, log_rms [N])."""
    n_layers = 7
    outs = [np.empty((len(W), D), dtype=np.float32) for _ in range(n_layers)]
    log_rms = np.empty(len(W), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(W), bs):
            xb = torch.from_numpy(W[s:s+bs]).unsqueeze(2).float().to(DEVICE)
            layers, lr = bb.layerwise(xb)
            for li, layer in enumerate(layers):
                outs[li][s:s+bs] = layer.cpu().numpy()
            log_rms[s:s+bs] = lr.squeeze(-1).cpu().numpy()
    return outs, log_rms


# ---------------- metrics ----------------
def coral_distance(X, Y):
    Cx = np.cov(X, rowvar=False); Cy = np.cov(Y, rowvar=False)
    return float(np.linalg.norm(Cx - Cy, ord='fro'))


def mmd2_rbf(X, Y, n_sub=500, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) > n_sub: X = X[rng.choice(len(X), n_sub, replace=False)]
    if len(Y) > n_sub: Y = Y[rng.choice(len(Y), n_sub, replace=False)]
    XY = np.vstack([X, Y])
    sub = XY[rng.choice(len(XY), min(300, len(XY)), replace=False)]
    d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
    sigma2 = np.median(d2[d2 > 0]) + 1e-12
    def rbf(A, B): return np.exp(-np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1) / (2 * sigma2))
    return float(rbf(X, X).mean() + rbf(Y, Y).mean() - 2 * rbf(X, Y).mean())


def linear_separability(X, Y, seed=0):
    Xall = np.vstack([X, Y]); yall = np.concatenate([np.zeros(len(X)), np.ones(len(Y))])
    Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.3, random_state=seed, stratify=yall)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def probe_spearman(Xtr, ytr, Xte, yte, alpha=10.0):
    reg = Ridge(alpha=alpha).fit(Xtr, ytr)
    pred = reg.predict(Xte)
    return float(spearmanr(pred, yte).statistic)


def linear_cka(X, Y, trim_std=5.0):
    def trim(A):
        norms = np.linalg.norm(A, axis=1)
        keep = norms < (norms.mean() + trim_std * norms.std())
        return keep
    keep = trim(X) & trim(Y)
    X, Y = X[keep], Y[keep]
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, ord='fro') ** 2
    nx = np.linalg.norm(X.T @ X, ord='fro'); ny = np.linalg.norm(Y.T @ Y, ord='fro')
    return float(hsic / (nx * ny + 1e-12))


def main():
    rng = np.random.default_rng(0)
    print("Loading synthetic val set (torchsig_v1)...", flush=True)
    Wsyn, Dsyn, _, _, _ = load_cached_dataset("torchsig_v1", "val")
    idx = rng.choice(len(Wsyn), min(N_SUBSAMPLE, len(Wsyn)), replace=False)
    Wsyn, Dsyn = Wsyn[idx], Dsyn[idx]
    Usyn = np.log10(Dsyn)

    print("Loading pooled real windows (all 7 sessions, for domain-gap tests)...", flush=True)
    captures = load_sessions()
    all_sessions = sorted(captures['session_id'].unique())
    Wreal_pool_raw, Dreal_pool, _, _, sids_pool = real_windows_for_sessions(all_sessions, max_windows_per_packet=3, seed=0)
    idxp = rng.choice(len(Wreal_pool_raw), min(N_SUBSAMPLE, len(Wreal_pool_raw)), replace=False)
    Wreal_pool_raw, Dreal_pool = Wreal_pool_raw[idxp], Dreal_pool[idxp]

    print("Loading per-held-out-session real train/test splits (matches frozen-adapter protocol)...", flush=True)
    held_out_data = {}
    for sess in HELD_OUT_SESSIONS:
        train_sessions = [s for s in all_sessions if s != sess]
        Wtr_raw, Dtr, _, _, _ = real_windows_for_sessions(train_sessions, max_windows_per_packet=3, seed=0)
        Wte_raw, Dte, _, _, _ = real_windows_for_sessions([sess], max_windows_per_packet=8, seed=0)
        i1 = rng.choice(len(Wtr_raw), min(N_SUBSAMPLE, len(Wtr_raw)), replace=False)
        held_out_data[sess] = dict(Wtr=Wtr_raw[i1], Utr=np.log10(Dtr[i1]), Wte=Wte_raw, Ute=np.log10(Dte))

    # global scale: match training convention (scale by real data's own RMS, synth already unit-ish)
    real_scale = float(np.sqrt(np.mean(Wreal_pool_raw[:200] ** 2)) + 1e-12)
    syn_scale = float(np.sqrt(np.mean(Wsyn[:200] ** 2)) + 1e-12)
    Wreal_pool = (Wreal_pool_raw / real_scale).astype(np.float32)
    Wsyn_n = (Wsyn / syn_scale).astype(np.float32)
    for sess in HELD_OUT_SESSIONS:
        held_out_data[sess]['Wtr'] = (held_out_data[sess]['Wtr'] / real_scale).astype(np.float32)
        held_out_data[sess]['Wte'] = (held_out_data[sess]['Wte'] / real_scale).astype(np.float32)

    print("\nLoading backbones...", flush=True)
    bb_pre = load_backbone(pretrained=True)
    bb_rand = load_backbone(pretrained=False)

    print("Extracting layerwise embeddings (this is the only GPU-heavy step)...", flush=True)
    layers_syn_pre, logrms_syn = extract_layerwise(bb_pre, Wsyn_n)
    layers_syn_rand, _ = extract_layerwise(bb_rand, Wsyn_n)
    layers_realpool_pre, logrms_realpool = extract_layerwise(bb_pre, Wreal_pool)
    layers_realpool_rand, _ = extract_layerwise(bb_rand, Wreal_pool)

    held_out_layers = {}
    for sess in HELD_OUT_SESSIONS:
        d = held_out_data[sess]
        L_tr_pre, _ = extract_layerwise(bb_pre, d['Wtr'])
        L_te_pre, lr_te = extract_layerwise(bb_pre, d['Wte'])
        L_tr_rand, _ = extract_layerwise(bb_rand, d['Wtr'])
        L_te_rand, _ = extract_layerwise(bb_rand, d['Wte'])
        held_out_layers[sess] = dict(tr_pre=L_tr_pre, te_pre=L_te_pre, tr_rand=L_tr_rand, te_rand=L_te_rand,
                                      logrms_te=lr_te, Utr=d['Utr'], Ute=d['Ute'])
    print("Done extracting.\n", flush=True)

    report = {}

    # ---- D1: domain gap (pretrained backbone, final layer = index 6) ----
    Xs, Xr = layers_syn_pre[6], layers_realpool_pre[6]
    report['domain_gap_h_cls_pretrained'] = dict(
        coral_distance=coral_distance(Xs, Xr),
        mmd2_rbf=mmd2_rbf(Xs, Xr),
        linear_separability=linear_separability(Xs, Xr),
    )
    print("=== D1: domain gap (synthetic vs. pooled real, h_cls, pretrained backbone) ===")
    print(json.dumps(report['domain_gap_h_cls_pretrained'], indent=2))

    # ---- D2: per-layer probing ----
    print("\n=== D2: per-layer probe Spearman(pred, true log10(distance)) ===")
    report['per_layer_probe'] = {}
    # synthetic in-distribution (train/test split within synth-val)
    n = len(Usyn); split = int(n * 0.7)
    syn_probe = []
    for li in range(7):
        Xtr, Xte = layers_syn_pre[li][:split], layers_syn_pre[li][split:]
        ytr, yte = Usyn[:split], Usyn[split:]
        syn_probe.append(probe_spearman(Xtr, ytr, Xte, yte))
    report['per_layer_probe']['synthetic_indist_pretrained'] = syn_probe
    print(f"synthetic in-dist (pretrained), by layer 0-6: {[round(v,3) for v in syn_probe]}")

    for sess in HELD_OUT_SESSIONS:
        d = held_out_layers[sess]
        pre_probe = [probe_spearman(d['tr_pre'][li], d['Utr'], d['te_pre'][li], d['Ute']) for li in range(7)]
        rand_probe = [probe_spearman(d['tr_rand'][li], d['Utr'], d['te_rand'][li], d['Ute']) for li in range(7)]
        report['per_layer_probe'][sess] = dict(pretrained=pre_probe, random=rand_probe)
        print(f"{sess:10s} pretrained by layer 0-6: {[round(v,3) for v in pre_probe]}")
        print(f"{sess:10s} random     by layer 0-6: {[round(v,3) for v in rand_probe]}")

    # ---- log-RMS-only trivial baseline ----
    print("\n=== log-RMS-only baseline Spearman(log_rms, true log10(distance)) ===")
    report['logrms_baseline'] = dict(synthetic=float(spearmanr(logrms_syn, Usyn).statistic))
    print(f"synthetic: {report['logrms_baseline']['synthetic']:.3f}")
    for sess in HELD_OUT_SESSIONS:
        d = held_out_layers[sess]
        rho = float(spearmanr(d['logrms_te'], d['Ute']).statistic)
        report['logrms_baseline'][sess] = rho
        print(f"{sess}: {rho:.3f}")

    # ---- D3: CKA pretrained vs random, per layer, on pooled real ----
    print("\n=== D3: linear CKA(pretrained, random) per layer, on pooled real windows ===")
    cka_vals = [linear_cka(layers_realpool_pre[li], layers_realpool_rand[li]) for li in range(7)]
    report['cka_pretrained_vs_random_real'] = cka_vals
    print(f"CKA by layer 0-6: {[round(v,3) for v in cka_vals]}")

    out_path = Path(__file__).resolve().parent / "diagnose_transfer_report.json"
    json.dump(report, open(out_path, "w"), indent=2)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
