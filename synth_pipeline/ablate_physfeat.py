#!/usr/bin/env python3
"""Isolates the pretrained backbone's marginal contribution once log_rms is in the feature mix.

log_rms is computed directly from raw IQ (sigma = sqrt(mean(iq**2))) with zero trainable
parameters -- it is IDENTICAL for the pretrained and random-control backbones. So the
frozen-adapter [h_cls,h_phys,log_rms] sweep no longer cleanly isolates "does pretraining help":
both conditions get the same powerful log_rms feature for free. This script fits a fast Ridge
probe (no 1000-epoch head training) on three feature subsets -- log_rms alone, [h_cls,h_phys]
only (excludes the free feature), and the full concat -- for both backbones, on the same
real train/test split used by train_frozen_adapter.py. Comparing [h_cls,h_phys]-only between
backbones is the clean test of whether synthetic pretraining adds anything beyond log_rms.
"""
import sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/naveen/Desktop/Learned representation/Raptor")
from real_data import load_sessions, real_windows_for_sessions
from src.models.raptorfm_tokenizer import RAPTORFMTokenizer
from src.models.raptorfm_encoder import RAPTORFMEncoder

D = 192
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = "/home/naveen/Desktop/Learned representation/experiments/synth_pipeline/results_synth_torchsig_v2_physfeat/best.pt"
HELD_OUT_SESSIONS = ["979_S4", "mini5_S3", "mini5_S2"]
torch.manual_seed(1234)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = RAPTORFMTokenizer(d_model=D, d_physical=64, patch=8, stride=8, mask_ratio=0.0, n_heads=6)
        self.enc = RAPTORFMEncoder(d_model=D, n_heads=6, d_ff=768, n_layers=6, d_physical=64)

    def forward(self, x):
        t = self.tok(x, None)
        t["i_masked"] = t["i_t"]; t["q_masked"] = t["q_t"]
        t["mask_i"] = torch.zeros_like(t["mask_i"]); t["mask_q"] = torch.zeros_like(t["mask_q"])
        o = self.enc(t)
        return o["h_cls"], o["h_phys"], o["log_rms"]


def load_backbone(pretrained):
    bb = Backbone().to(DEVICE)
    if pretrained:
        sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
        bb_sd = {k: v for k, v in sd.items() if k.startswith('tok.') or k.startswith('enc.')}
        bb.load_state_dict(bb_sd, strict=True)
    bb.eval()
    for p in bb.parameters(): p.requires_grad_(False)
    return bb


def extract(bb, W, bs=48):
    hcls = np.empty((len(W), D), dtype=np.float32)
    hphys = np.empty((len(W), D), dtype=np.float32)
    lrms = np.empty((len(W), 1), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(W), bs):
            xb = torch.from_numpy(W[s:s+bs]).unsqueeze(2).float().to(DEVICE)
            a, b, c = bb(xb)
            hcls[s:s+bs] = a.cpu().numpy(); hphys[s:s+bs] = b.cpu().numpy(); lrms[s:s+bs] = c.cpu().numpy()
    return hcls, hphys, lrms


def probe(Xtr, ytr, Xte, yte, alpha=10.0):
    reg = Ridge(alpha=alpha).fit(Xtr, ytr)
    return float(spearmanr(reg.predict(Xte), yte).statistic)


def main():
    captures = load_sessions()
    all_sessions = sorted(captures['session_id'].unique())
    bb_pre = load_backbone(True)
    bb_rand = load_backbone(False)

    print(f"{'session':10s} {'feature_set':16s} {'pretrained':>10s} {'random':>10s}")
    for sess in HELD_OUT_SESSIONS:
        train_sessions = [s for s in all_sessions if s != sess]
        Wtr_raw, Dtr, _, _, _ = real_windows_for_sessions(train_sessions, max_windows_per_packet=3, seed=0)
        Wte_raw, Dte, _, _, _ = real_windows_for_sessions([sess], max_windows_per_packet=8, seed=0)
        scale = float(np.sqrt(np.mean(Wtr_raw[:200] ** 2)) + 1e-12)
        Wtr = (Wtr_raw / scale).astype(np.float32); Wte = (Wte_raw / scale).astype(np.float32)
        Utr = np.log10(Dtr); Ute = np.log10(Dte)

        hcls_tr_p, hphys_tr_p, lrms_tr_p = extract(bb_pre, Wtr)
        hcls_te_p, hphys_te_p, lrms_te_p = extract(bb_pre, Wte)
        hcls_tr_r, hphys_tr_r, lrms_tr_r = extract(bb_rand, Wtr)
        hcls_te_r, hphys_te_r, lrms_te_r = extract(bb_rand, Wte)

        # log_rms is backbone-independent -- sanity check they match
        assert np.allclose(lrms_tr_p, lrms_tr_r, atol=1e-4), "log_rms should be backbone-independent!"

        feat_sets = {
            'log_rms_only': (lrms_tr_p, lrms_te_p, lrms_tr_r, lrms_te_r),
            'hcls_hphys_only': (np.concatenate([hcls_tr_p, hphys_tr_p], -1), np.concatenate([hcls_te_p, hphys_te_p], -1),
                                 np.concatenate([hcls_tr_r, hphys_tr_r], -1), np.concatenate([hcls_te_r, hphys_te_r], -1)),
            'full_concat': (np.concatenate([hcls_tr_p, hphys_tr_p, lrms_tr_p], -1), np.concatenate([hcls_te_p, hphys_te_p, lrms_te_p], -1),
                            np.concatenate([hcls_tr_r, hphys_tr_r, lrms_tr_r], -1), np.concatenate([hcls_te_r, hphys_te_r, lrms_te_r], -1)),
        }
        for name, (Xtr_p, Xte_p, Xtr_r, Xte_r) in feat_sets.items():
            rho_p = probe(Xtr_p, Utr, Xte_p, Ute)
            rho_r = probe(Xtr_r, Utr, Xte_r, Ute)
            print(f"{sess:10s} {name:16s} {rho_p:10.3f} {rho_r:10.3f}")


if __name__ == "__main__":
    main()
