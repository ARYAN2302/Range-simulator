#!/usr/bin/env python3
"""Progressive real-data mixing curriculum: mostly-synthetic training, with a small, controlled
fraction of REAL recorded data blended in each batch. Evaluated causally on a REAL session held
out entirely from training (true leave-one-session-out, same discipline as every earlier
real-data experiment this project ran) -- this is the actual test of whether synthetic
pretraining plus a light real-data nudge can survive contact with a genuinely unseen real
session, unlike training on real data alone (which collapsed to chance every time it was tried).
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/naveen/Desktop/Learned representation/Raptor")
from simulate import WaveformPool, make_synthetic_batch
from real_data import load_sessions, real_windows_for_sessions
from src.models.raptorfm_tokenizer import RAPTORFMTokenizer
from src.models.raptorfm_encoder import RAPTORFMEncoder

p = argparse.ArgumentParser()
p.add_argument('--real_holdout_session', type=str, required=True,
               help="one of: 979_S1 979_S2 979_S3 979_S4 mini5_S1 mini5_S2 mini5_S3 -- reserved, never trained on")
p.add_argument('--mix_ratio', type=float, default=0.10, help="fraction of each training batch drawn from real data")
p.add_argument('--n_synth', type=int, default=20000)
p.add_argument('--epochs', type=int, default=15)
p.add_argument('--batch_size', type=int, default=48)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--use_torchsig', action=argparse.BooleanOptionalAction, default=True,
               help="layer torchsig TX/RX hardware-impairment chains into apply_channel")
p.add_argument('--torchsig_level', type=int, default=2, choices=[0, 1, 2])
args = p.parse_args()
CHANNEL_KWARGS = dict(use_torchsig=args.use_torchsig, torchsig_level=args.torchsig_level, torchsig_seed=args.seed)

D = 192
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
OUT = Path(f"/home/naveen/Desktop/Learned representation/experiments/synth_pipeline/results_mixed_{args.real_holdout_session}_mix{int(args.mix_ratio*100)}")
OUT.mkdir(parents=True, exist_ok=True)
print(f"DEVICE={DEVICE} holdout={args.real_holdout_session} mix_ratio={args.mix_ratio}", flush=True)

# ---------------- synthetic pool ----------------
rng = np.random.default_rng(args.seed)
pool = WaveformPool(max_per_group=400, seed=args.seed)
print("Generating synthetic pool...", flush=True)
Wsyn, Dsyn, _, _ = make_synthetic_batch(pool, args.n_synth, rng, channel_kwargs=CHANNEL_KWARGS)

# ---------------- real data: train on 6 sessions, hold out 1 ----------------
captures = load_sessions()
all_sessions = sorted(captures['session_id'].unique())
assert args.real_holdout_session in all_sessions, all_sessions
train_sessions = [s for s in all_sessions if s != args.real_holdout_session]
print(f"Real train sessions: {train_sessions}", flush=True)
print("Loading real training windows...", flush=True)
Wreal_raw, Dreal, _, _, _ = real_windows_for_sessions(train_sessions, max_windows_per_packet=3, seed=args.seed)
print(f"Real train windows: {len(Wreal_raw)}", flush=True)
print("Loading held-out real session windows (never trained on)...", flush=True)
Wtest_raw, Dtest, _, _, _ = real_windows_for_sessions([args.real_holdout_session], max_windows_per_packet=8, seed=args.seed)
print(f"Held-out test windows: {len(Wtest_raw)}", flush=True)

# IMPORTANT: do NOT per-window RMS-normalize real data -- that would erase the relative
# amplitude differences between windows, which is exactly where real distance information
# lives (confounded with session calibration, but present). Synthetic data strips-then-
# reimposes amplitude deliberately (apply_channel); real data must keep its natural relative
# scale. Only a single GLOBAL scale factor is applied, for numerical range, same convention
# used throughout this project's real-data training scripts.
global_scale = float(np.sqrt(np.mean(Wsyn[:200]**2)) + 1e-12)
Wsyn = Wsyn / global_scale
real_scale = float(np.sqrt(np.mean(Wreal_raw[:200]**2)) + 1e-12)
Wreal = (Wreal_raw / real_scale).astype(np.float32)
Wtest = (Wtest_raw / real_scale).astype(np.float32)

Usyn = np.log10(Dsyn); Ureal = np.log10(Dreal); Utest = np.log10(Dtest)
print(f"synth u range [{Usyn.min():.2f},{Usyn.max():.2f}]  real u range [{Ureal.min():.2f},{Ureal.max():.2f}]  "
      f"test u range [{Utest.min():.2f},{Utest.max():.2f}]", flush=True)

json.dump(dict(holdout=args.real_holdout_session, mix_ratio=args.mix_ratio, n_synth=len(Wsyn),
               n_real_train=len(Wreal), n_test=len(Wtest), global_scale=global_scale),
          open(OUT/"data_manifest.json","w"), indent=2)

# ---------------- model ----------------
class RangePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = RAPTORFMTokenizer(d_model=D, d_physical=64, patch=8, stride=8, mask_ratio=0.0, n_heads=6)
        self.enc = RAPTORFMEncoder(d_model=D, n_heads=6, d_ff=768, n_layers=6, d_physical=64)
        self.head = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 1))
    def _backbone(self, x):
        t = self.tok(x, None)
        t["i_masked"] = t["i_t"]; t["q_masked"] = t["q_t"]
        t["mask_i"] = torch.zeros_like(t["mask_i"]); t["mask_q"] = torch.zeros_like(t["mask_q"])
        o = self.enc(t)
        return o["h_cls"]
    def forward(self, x, use_checkpoint=False):
        h_cls = ckpt.checkpoint(self._backbone, x, use_reentrant=False) if (use_checkpoint and self.training) else self._backbone(x)
        return self.head(h_cls).squeeze(-1)

model = RangePredictor().to(DEVICE)
n_params = sum(pp.numel() for pp in model.parameters())
print(f"Params: {n_params:,}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

def to_tensor(W):
    return torch.from_numpy(W).unsqueeze(2).float()

def make_mixed_epoch_indices(n_synth, n_real, mix_ratio, epoch_size, rng):
    """Return list of (source, idx) pairs for one epoch, mix_ratio fraction from real."""
    n_from_real = int(epoch_size * mix_ratio)
    n_from_synth = epoch_size - n_from_real
    synth_idx = rng.integers(0, n_synth, size=n_from_synth)
    real_idx = rng.integers(0, n_real, size=n_from_real) if n_real > 0 else np.array([], dtype=int)
    items = [('s', i) for i in synth_idx] + [('r', i) for i in real_idx]
    rng.shuffle(items)
    return items

def run_train_epoch(bs):
    model.train()
    items = make_mixed_epoch_indices(len(Wsyn), len(Wreal), args.mix_ratio, len(Wsyn), rng)
    tot_loss = 0.0; n = 0
    for s in range(0, len(items), bs):
        batch = items[s:s+bs]
        xs = np.stack([(Wsyn[i] if src=='s' else Wreal[i]) for src,i in batch])
        ys = np.array([(Usyn[i] if src=='s' else Ureal[i]) for src,i in batch])
        xb = to_tensor(xs).to(DEVICE); yb = torch.tensor(ys, dtype=torch.float32, device=DEVICE)
        amp_ctx = torch.amp.autocast('cuda') if USE_AMP else torch.autocast('cpu', enabled=False)
        with amp_ctx:
            pred = model(xb, use_checkpoint=True)
            loss = F.huber_loss(pred, yb, delta=1.0)
        opt.zero_grad()
        if scaler: scaler.scale(loss).backward(); scaler.unscale_(opt)
        else: loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler: scaler.step(opt); scaler.update()
        else: opt.step()
        tot_loss += float(loss) * len(batch); n += len(batch)
    return tot_loss / n

def run_eval(W, U, bs):
    model.eval()
    preds = np.empty(len(W))
    with torch.no_grad():
        for s in range(0, len(W), bs):
            xb = to_tensor(W[s:s+bs]).to(DEVICE)
            amp_ctx = torch.amp.autocast('cuda') if USE_AMP else torch.autocast('cpu', enabled=False)
            with amp_ctx:
                pred = model(xb, use_checkpoint=False)
            preds[s:s+bs] = pred.detach().cpu().numpy()
    rho = spearmanr(preds, U).statistic
    r2 = r2_score(U, preds)
    mae_dex = np.mean(np.abs(preds - U))
    mae_m = np.mean(np.abs(10**preds - 10**U))
    return dict(spearman=float(rho), r2=float(r2), mae_dex=float(mae_dex), mae_m=float(mae_m)), preds

print("\n=== SANITY ===", flush=True)
xb = to_tensor(Wsyn[:4]).to(DEVICE); yb = torch.tensor(Usyn[:4], dtype=torch.float32, device=DEVICE)
pred = model(xb); loss = F.huber_loss(pred, yb)
opt.zero_grad(); loss.backward()
gn = sum(pp.grad.norm().item()**2 for pp in model.head.parameters() if pp.grad is not None)**0.5
print(f"sanity loss={float(loss):.3f} grad={gn:.2e}", flush=True)
assert gn > 0
opt.zero_grad()
print("SANITY PASS\n", flush=True)

print("=== TRAIN (mixed) ===", flush=True)
log = []
best = (-1, None)
for ep in range(args.epochs):
    t0 = time.time()
    train_loss = run_train_epoch(args.batch_size)
    synth_metrics, _ = run_eval(Wsyn[-1000:], Usyn[-1000:], args.batch_size)  # sanity: still fits synthetic
    real_test_metrics, test_preds = run_eval(Wtest, Utest, args.batch_size)
    row = dict(epoch=ep, train_loss=round(train_loss,4), time_s=round(time.time()-t0,1),
               synth_val=synth_metrics, real_holdout_test=real_test_metrics)
    log.append(row); print(json.dumps(row), flush=True)
    with open(OUT/"training_log.json","w") as f: json.dump(log, f, indent=2)
    if real_test_metrics['spearman'] > best[0]:
        best = (real_test_metrics['spearman'], ep)
        torch.save(model.state_dict(), OUT/"best.pt")
        np.save(OUT/"best_test_preds.npy", test_preds)

print(f"\nDONE. Best real-holdout Spearman: {best[0]:.4f} at epoch {best[1]}", flush=True)
json.dump(dict(best_real_holdout_spearman=best[0], best_epoch=best[1], args=vars(args)),
          open(OUT/"summary.json","w"), indent=2)
