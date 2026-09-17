#!/usr/bin/env python3
"""Physics-baseline + learned-residual-correction model (KalmanNet-style: physics scaffold +
learned residual, per deep_research_brief_5's foundation-model recommendation and brief_6's
D2.4 regularize-toward-physics guidance).

The ablation (ablate_physfeat.py) found that naively concatenating log_rms with [h_cls,h_phys]
into one linear/MLP head makes things WORSE than log_rms alone -- the learned features are
correlated with log_rms in a way a joint fit mishandles. This script instead uses a two-stage
design that cannot make that mistake by construction:

  Stage 1 (physics baseline): fit U_base = a*log_rms + b on the 6-session real training pool
    (log_rms is backbone-independent, so this baseline is identical regardless of backbone).
  Stage 2 (learned residual correction): fit a Ridge regression from [h_cls,h_phys] to the
    STAGE-1 RESIDUAL (Utr - Ubase_tr) on the same training pool, for both the pretrained and
    random-control backbones. At test time: U_pred = U_base(log_rms) + correction([h_cls,h_phys]).

Reports Spearman/R^2 for the log_rms-only baseline (the bar to beat) and for baseline+correction
under both backbones, sweeping the correction's Ridge regularization strength (a stronger prior
toward "no correction" is itself a test of whether the pretrained features have anything
non-noisy to add).
"""
import sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/naveen/Desktop/Learned representation/Raptor")
from real_data import load_sessions, real_windows_for_sessions
from src.models.raptorfm_tokenizer import RAPTORFMTokenizer
from src.models.raptorfm_encoder import RAPTORFMEncoder

D = 192
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = "/home/naveen/Desktop/Learned representation/experiments/synth_pipeline/results_synth_torchsig_v2_physfeat/best.pt"
HELD_OUT_SESSIONS = ["979_S4", "mini5_S3", "mini5_S2"]
CORRECTION_ALPHAS = [1.0, 10.0, 50.0, 200.0, 1000.0]
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


def main():
    captures = load_sessions()
    all_sessions = sorted(captures['session_id'].unique())
    bb_pre = load_backbone(True)
    bb_rand = load_backbone(False)

    print(f"{'session':10s} {'model':28s} {'spearman':>9s} {'r2':>9s}")
    for sess in HELD_OUT_SESSIONS:
        train_sessions = [s for s in all_sessions if s != sess]
        Wtr_raw, Dtr, _, _, _ = real_windows_for_sessions(train_sessions, max_windows_per_packet=3, seed=0)
        Wte_raw, Dte, _, _, _ = real_windows_for_sessions([sess], max_windows_per_packet=8, seed=0)
        scale = float(np.sqrt(np.mean(Wtr_raw[:200] ** 2)) + 1e-12)
        Wtr = (Wtr_raw / scale).astype(np.float32); Wte = (Wte_raw / scale).astype(np.float32)
        Utr = np.log10(Dtr); Ute = np.log10(Dte)

        hcls_tr_p, hphys_tr_p, lrms_tr = extract(bb_pre, Wtr)
        hcls_te_p, hphys_te_p, lrms_te = extract(bb_pre, Wte)
        hcls_tr_r, hphys_tr_r, lrms_tr_r = extract(bb_rand, Wtr)
        hcls_te_r, hphys_te_r, lrms_te_r = extract(bb_rand, Wte)
        assert np.allclose(lrms_tr, lrms_tr_r, atol=1e-4)  # backbone-independent, as established

        # ---- Stage 1: physics baseline from log_rms alone ----
        base_reg = Ridge(alpha=10.0).fit(lrms_tr, Utr)
        Ubase_tr = base_reg.predict(lrms_tr)
        Ubase_te = base_reg.predict(lrms_te)
        resid_tr = Utr - Ubase_tr

        rho_base = float(spearmanr(Ubase_te, Ute).statistic)
        r2_base = float(r2_score(Ute, Ubase_te))
        print(f"{sess:10s} {'log_rms baseline (Stage 1)':28s} {rho_base:9.3f} {r2_base:9.3f}")

        for tag, Xtr_feat, Xte_feat in [
            ('pretrained', np.concatenate([hcls_tr_p, hphys_tr_p], -1), np.concatenate([hcls_te_p, hphys_te_p], -1)),
            ('random', np.concatenate([hcls_tr_r, hphys_tr_r], -1), np.concatenate([hcls_te_r, hphys_te_r], -1)),
        ]:
            # select correction alpha via 3-fold CV on the TRAINING pool only (never touches
            # the held-out test session) -- avoids picking the alpha that happens to look best
            # on the test set, which would silently inflate every number below.
            kf = KFold(n_splits=3, shuffle=True, random_state=0)
            best_alpha, best_cv_rho = None, -np.inf
            for alpha in CORRECTION_ALPHAS:
                cv_rhos = []
                for tr_idx, va_idx in kf.split(Xtr_feat):
                    reg = Ridge(alpha=alpha).fit(Xtr_feat[tr_idx], resid_tr[tr_idx])
                    pred_va = Ubase_tr[va_idx] + reg.predict(Xtr_feat[va_idx])
                    cv_rhos.append(spearmanr(pred_va, Utr[va_idx]).statistic)
                mean_cv_rho = float(np.nanmean(cv_rhos))
                if mean_cv_rho > best_cv_rho:
                    best_cv_rho, best_alpha = mean_cv_rho, alpha

            corr_reg = Ridge(alpha=best_alpha).fit(Xtr_feat, resid_tr)
            Ufinal_te = Ubase_te + corr_reg.predict(Xte_feat)
            rho = float(spearmanr(Ufinal_te, Ute).statistic)
            r2 = float(r2_score(Ute, Ufinal_te))
            delta = rho - rho_base
            print(f"{sess:10s} {'+' + tag + f' correction (a={best_alpha:g}, cv_rho={best_cv_rho:.3f})':28s} {rho:9.3f} {r2:9.3f}   (Δspearman vs baseline: {delta:+.3f})")
        print()


if __name__ == "__main__":
    main()
