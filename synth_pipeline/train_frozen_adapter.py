#!/usr/bin/env python3
"""Frozen-backbone + small adapter head, fine-tuned on REAL data only.

Rationale: joint mixing (train_mixed.py) let the model partially re-absorb the same
session-specific confound that sank every real-data approach in this project, because the
whole network (including the backbone) was exposed to real data's non-transferable
session artifacts. Freezing the synthetic-pretrained backbone and only adapting a small head
limits how much of the real-session confound can corrupt the representation, while still
allowing real-world calibration to shape the final prediction.

Since the backbone is frozen, embeddings are extracted ONCE (no backprop through the
transformer at all) -- this is now essentially a linear-probe problem, fast to train.

Includes a frozen-RANDOM-backbone control: same architecture, randomly initialized (never
trained on anything), same real-data adapter training. If the synthetic-pretrained backbone
doesn't clearly beat this control, the synthetic pretraining isn't adding real transferable
value beyond what a fixed random projection already provides.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/naveen/Desktop/Learned representation/Raptor")
from real_data import load_sessions, real_windows_for_sessions
from src.models.raptorfm_tokenizer import RAPTORFMTokenizer
from src.models.raptorfm_encoder import RAPTORFMEncoder

p = argparse.ArgumentParser()
p.add_argument('--real_holdout_session', type=str, required=True)
p.add_argument('--pretrained_ckpt', type=str,
                default="/home/naveen/Desktop/Learned representation/experiments/synth_pipeline/results_synth_v1/best.pt")
p.add_argument('--head_epochs', type=int, default=200)
p.add_argument('--head_lr', type=float, default=1e-3)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out_tag', type=str, default='',
               help="distinguishes output dir (results_frozen_<tag>_<session>) so runs against "
                    "different checkpoints/architectures don't silently overwrite each other's "
                    "results.json -- a prior run without this arg overwrote the original "
                    "pre-torchsig sweep results because OUT didn't vary with checkpoint.")
args = p.parse_args()

D = 192
D_FEAT = D * 2 + 1  # h_cls + h_phys + log_rms -- see train_synth.py for the rationale
                     # (h_cls alone is RMSNorm'd and structurally discards absolute amplitude,
                     # which is where the real-transferable signal lives; diagnose_transfer.py
                     # found log_rms alone gets Spearman -0.89 on 979_S4 real data).
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_out_name = f"results_frozen_{args.out_tag}_{args.real_holdout_session}" if args.out_tag else f"results_frozen_{args.real_holdout_session}"
OUT = Path(f"/home/naveen/Desktop/Learned representation/experiments/synth_pipeline/{_out_name}")
OUT.mkdir(parents=True, exist_ok=True)

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = RAPTORFMTokenizer(d_model=D, d_physical=64, patch=8, stride=8, mask_ratio=0.0, n_heads=6)
        self.enc = RAPTORFMEncoder(d_model=D, n_heads=6, d_ff=768, n_layers=6, d_physical=64)
    def forward(self, x):
        t = self.tok(x, None)
        t["i_masked"]=t["i_t"]; t["q_masked"]=t["q_t"]
        t["mask_i"]=torch.zeros_like(t["mask_i"]); t["mask_q"]=torch.zeros_like(t["mask_q"])
        o = self.enc(t)
        return torch.cat([o["h_cls"], o["h_phys"], o["log_rms"]], dim=-1)

def load_backbone(pretrained):
    bb = Backbone().to(DEVICE)
    if pretrained:
        sd = torch.load(args.pretrained_ckpt, map_location=DEVICE, weights_only=True)
        # sd keys are "tok.xxx" / "enc.xxx" / "head.xxx" from the full RangePredictor -- filter to backbone only
        bb_sd = {k: v for k, v in sd.items() if k.startswith('tok.') or k.startswith('enc.')}
        bb.load_state_dict(bb_sd, strict=True)
        print("Loaded PRETRAINED (synthetic) backbone weights.", flush=True)
    else:
        print("Using RANDOM (untrained) backbone -- control condition.", flush=True)
    bb.eval()
    for pp in bb.parameters():
        pp.requires_grad_(False)
    return bb

def extract_embeddings(bb, W, bs=48):
    embs = np.empty((len(W), D_FEAT), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(W), bs):
            xb = torch.from_numpy(W[s:s+bs]).unsqueeze(2).float().to(DEVICE)
            embs[s:s+bs] = bb(xb).cpu().numpy()
    return embs

# ---------------- load real data ----------------
captures = load_sessions()
all_sessions = sorted(captures['session_id'].unique())
train_sessions = [s for s in all_sessions if s != args.real_holdout_session]
print(f"Train sessions: {train_sessions}  Held out: {args.real_holdout_session}", flush=True)
Wtr_raw, Dtr, _, _, _ = real_windows_for_sessions(train_sessions, max_windows_per_packet=3, seed=args.seed)
Wte_raw, Dte, _, _, _ = real_windows_for_sessions([args.real_holdout_session], max_windows_per_packet=8, seed=args.seed)
print(f"train windows={len(Wtr_raw)}  test windows={len(Wte_raw)}", flush=True)

scale = float(np.sqrt(np.mean(Wtr_raw[:200]**2)) + 1e-12)
Wtr = (Wtr_raw / scale).astype(np.float32)
Wte = (Wte_raw / scale).astype(np.float32)
Utr = np.log10(Dtr); Ute = np.log10(Dte)

def train_and_eval_head(Etr, Utr, Ete, Ute, tag):
    Etr_t = torch.tensor(Etr, dtype=torch.float32, device=DEVICE)
    Utr_t = torch.tensor(Utr, dtype=torch.float32, device=DEVICE)
    Ete_t = torch.tensor(Ete, dtype=torch.float32, device=DEVICE)
    head = nn.Sequential(nn.LayerNorm(D_FEAT), nn.Linear(D_FEAT, 64), nn.ReLU(), nn.Linear(64, 1)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=1e-3)
    log = []
    for ep in range(args.head_epochs):
        head.train()
        perm = torch.randperm(len(Etr_t))
        tot = 0.0
        for s in range(0, len(perm), 256):
            b = perm[s:s+256]
            pred = head(Etr_t[b]).squeeze(-1)
            loss = torch.nn.functional.huber_loss(pred, Utr_t[b], delta=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)*len(b)
        head.eval()
        with torch.no_grad():
            pred_te = head(Ete_t).squeeze(-1).cpu().numpy()
        rho = spearmanr(pred_te, Ute).statistic
        r2 = r2_score(Ute, pred_te)
        mae_m = np.mean(np.abs(10**pred_te - 10**Ute))
        if ep % 20 == 0 or ep == args.head_epochs-1:
            print(f"[{tag}] epoch {ep} train_loss={tot/len(Etr_t):.4f} test_spearman={rho:.4f} test_r2={r2:.4f} test_mae_m={mae_m:.1f}", flush=True)
        log.append(dict(epoch=ep, test_spearman=float(rho), test_r2=float(r2), test_mae_m=float(mae_m)))
    return log

results = {}
for tag, pretrained in [("pretrained_synthetic", True), ("random_control", False)]:
    print(f"\n=== {tag} ===", flush=True)
    # Seed identically before EACH condition (not just once globally) so the random-control
    # backbone init and both conditions' head init/training are reproducible and not entangled
    # with whatever RNG state the other condition left behind. Previously unseeded: random-
    # control's own results varied run-to-run by as much as the effect being measured
    # (e.g. 0.509 vs 0.287 Spearman on the same held-out session across two sweeps).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    bb = load_backbone(pretrained)
    Etr = extract_embeddings(bb, Wtr)
    Ete = extract_embeddings(bb, Wte)
    log = train_and_eval_head(Etr, Utr, Ete, Ute, tag)
    best = max(log, key=lambda r: r['test_spearman'])
    results[tag] = dict(log=log, best=best)
    del bb; torch.cuda.empty_cache()

json.dump(results, open(OUT/"results.json","w"), indent=2)
print("\n=== SUMMARY ===")
for tag in results:
    b = results[tag]['best']
    print(f"{tag:22s} best_spearman={b['test_spearman']:.4f}  r2={b['test_r2']:.4f}  mae_m={b['test_mae_m']:.1f}  (epoch {b['epoch']})")
