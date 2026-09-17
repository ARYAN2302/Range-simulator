#!/usr/bin/env python3
"""Train the Radio-FM-style backbone (RAPTORFMTokenizer/Encoder, unchanged architecture) as the
actual range PREDICTOR, on the synthetic confound-free pipeline. Pure-synthetic first (this run);
real-data mixing is a later, separate step once this is validated.

Proper batched supervised training (not the per-capture loop style used for the earlier
dual-stream experiments) -- valid here because every synthetic sample is already an
independent, fully-labeled example; there's no packet/capture hierarchy to aggregate over.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

sys.path.insert(0, "/home/naveen/Desktop/Learned representation")
sys.path.insert(0, "/home/naveen/Desktop/Learned representation/Raptor")
from experiments.synth_pipeline.simulate import WaveformPool, make_synthetic_batch, T_CHUNK, load_cached_dataset
from src.models.raptorfm_tokenizer import RAPTORFMTokenizer
from src.models.raptorfm_encoder import RAPTORFMEncoder

p = argparse.ArgumentParser()
p.add_argument('--n_train', type=int, default=20000)
p.add_argument('--n_val', type=int, default=3000)
p.add_argument('--epochs', type=int, default=15)
p.add_argument('--batch_size', type=int, default=48)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out_name', type=str, default='results_synth_v1')
p.add_argument('--exponent_std', type=float, default=0.35)
p.add_argument('--gain_db_lo', type=float, default=25)
p.add_argument('--gain_db_hi', type=float, default=45)
p.add_argument('--nuisance_db_std', type=float, default=6.0)
p.add_argument('--use_torchsig', action=argparse.BooleanOptionalAction, default=True,
               help="layer torchsig TX/RX hardware-impairment chains into apply_channel")
p.add_argument('--torchsig_level', type=int, default=2, choices=[0, 1, 2])
p.add_argument('--dataset_name', type=str, default=None,
               help="if set, load a pre-generated dataset from datasets/<name>/ (see "
                    "generate_dataset.py) instead of generating in-process; the channel/"
                    "torchsig args above are then ignored (the cached dataset's manifest "
                    "records what was actually used)")
args = p.parse_args()
CHANNEL_KWARGS = dict(exponent_std=args.exponent_std,
                       gain_db_range=(args.gain_db_lo, args.gain_db_hi),
                       nuisance_db_std=args.nuisance_db_std,
                       use_torchsig=args.use_torchsig, torchsig_level=args.torchsig_level,
                       torchsig_seed=args.seed)

D = 192
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
OUT = Path(f"/home/naveen/Desktop/Learned representation/experiments/synth_pipeline/{args.out_name}")
OUT.mkdir(parents=True, exist_ok=True)
print(f"DEVICE={DEVICE} AMP={USE_AMP} OUT={OUT}", flush=True)

# ---------------- synthetic data: cached dataset (production path) or in-process ----------------
if args.dataset_name:
    print(f"Loading cached dataset '{args.dataset_name}'...", flush=True)
    Wtr, Dtr, drones_tr, bands_tr, manifest = load_cached_dataset(args.dataset_name, "train")
    Wva, Dva, drones_va, bands_va, _ = load_cached_dataset(args.dataset_name, "val")
    print(f"Loaded manifest: {json.dumps(manifest, indent=2)}", flush=True)
else:
    rng = np.random.default_rng(args.seed)
    pool = WaveformPool(max_per_group=400, seed=args.seed)
    print("Generating synthetic train set...", flush=True)
    Wtr, Dtr, drones_tr, bands_tr = make_synthetic_batch(pool, args.n_train, rng, channel_kwargs=CHANNEL_KWARGS)
    print("Generating synthetic val set...", flush=True)
    Wva, Dva, drones_va, bands_va = make_synthetic_batch(pool, args.n_val, rng, channel_kwargs=CHANNEL_KWARGS)
    print(f"Channel kwargs: {CHANNEL_KWARGS}", flush=True)

# global scale from train only (consistent with the rest of this project's convention)
global_scale = float(np.sqrt(np.mean(Wtr[:200]**2)) + 1e-12)
Wtr = Wtr / global_scale; Wva = Wva / global_scale
Utr = np.log10(Dtr); Uva = np.log10(Dva)
print(f"global_scale={global_scale:.4f}  train u range [{Utr.min():.2f},{Utr.max():.2f}]", flush=True)

json.dump(dict(n_train=args.n_train, n_val=args.n_val, global_scale=global_scale,
               dist_range_train=[float(Dtr.min()), float(Dtr.max())]),
          open(OUT/"data_manifest.json","w"), indent=2)

# ---------------- model: RaptorFM backbone + regression head ----------------
# Fix (diagnose_transfer.py finding): h_cls alone is RMSNorm'd at every transformer block,
# which structurally strips absolute amplitude. A plain linear probe on h_cls got Spearman
# ~0.45 in-distribution on synthetic but near-zero/negative on every real held-out session --
# meanwhile raw log_rms (computed by the tokenizer, never previously fed to the head) got
# Spearman -0.89 on session 979_S4 and -0.64 on mini5_S3, far stronger than any learned
# embedding. The encoder also already computes h_phys, a dedicated scale-preserving branch
# ("NO saturating nonlinearities, NO bias -- preserves scale") that was likewise never used
# downstream. Feed the head all three: h_cls (learned/normalized structure), h_phys (learned,
# scale-preserving), and log_rms (raw physical amplitude) -- so the head can use whichever
# actually transfers instead of being forced through the RMSNorm bottleneck alone.
class RangePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = RAPTORFMTokenizer(d_model=D, d_physical=64, patch=8, stride=8, mask_ratio=0.0, n_heads=6)
        self.enc = RAPTORFMEncoder(d_model=D, n_heads=6, d_ff=768, n_layers=6, d_physical=64)
        self.head = nn.Sequential(nn.Linear(D * 2 + 1, 128), nn.ReLU(), nn.Linear(128, 1))
    def _backbone(self, x):
        t = self.tok(x, None)
        t["i_masked"] = t["i_t"]; t["q_masked"] = t["q_t"]
        t["mask_i"] = torch.zeros_like(t["mask_i"]); t["mask_q"] = torch.zeros_like(t["mask_q"])
        o = self.enc(t)
        return torch.cat([o["h_cls"], o["h_phys"], o["log_rms"]], dim=-1)
    def forward(self, x, use_checkpoint=False):  # x: [B,4096,1,2]
        if use_checkpoint and self.training:
            feat = ckpt.checkpoint(self._backbone, x, use_reentrant=False)
        else:
            feat = self._backbone(x)
        u_pred = self.head(feat).squeeze(-1)
        return u_pred

model = RangePredictor().to(DEVICE)
n_params = sum(p_.numel() for p_ in model.parameters())
print(f"Params: {n_params:,} ({n_params/1e6:.2f}M)", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

def to_tensor(W):
    return torch.from_numpy(W).unsqueeze(2).float()  # [B,4096,1,2]

def run_epoch(W, U, train, bs):
    n = len(W)
    idx = np.random.permutation(n) if train else np.arange(n)
    model.train() if train else model.eval()
    tot_loss = 0.0; n_batches = 0
    preds = np.empty(n); trues = np.empty(n)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for s in range(0, n, bs):
            b_idx = idx[s:s+bs]
            xb = to_tensor(W[b_idx]).to(DEVICE)
            yb = torch.tensor(U[b_idx], dtype=torch.float32, device=DEVICE)
            amp_ctx = torch.amp.autocast('cuda') if USE_AMP else torch.autocast('cpu', enabled=False)
            with amp_ctx:
                pred = model(xb, use_checkpoint=train)
                loss = F.huber_loss(pred, yb, delta=1.0)
            if train:
                opt.zero_grad()
                if scaler: scaler.scale(loss).backward(); scaler.unscale_(opt)
                else: loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler: scaler.step(opt); scaler.update()
                else: opt.step()
            tot_loss += float(loss) * len(b_idx); n_batches += len(b_idx)
            preds[b_idx] = pred.detach().cpu().numpy(); trues[b_idx] = U[b_idx]
    return tot_loss / n_batches, preds, trues

print("\n=== SANITY: one forward+backward step ===", flush=True)
xb = to_tensor(Wtr[:4]).to(DEVICE); yb = torch.tensor(Utr[:4], dtype=torch.float32, device=DEVICE)
pred = model(xb); loss = F.huber_loss(pred, yb)
opt.zero_grad(); loss.backward()
gn = sum(pp.grad.norm().item()**2 for pp in model.head.parameters() if pp.grad is not None)**0.5
print(f"sanity loss={float(loss):.3f} head_grad_norm={gn:.2e}", flush=True)
assert gn > 0, "no gradient reaching the head"
opt.zero_grad()
print("SANITY PASS\n", flush=True)

print("=== TRAIN (pure synthetic) ===", flush=True)
log = []
best_val_rho = -1
for ep in range(args.epochs):
    t0 = time.time()
    train_loss, _, _ = run_epoch(Wtr, Utr, True, args.batch_size)
    val_loss, vpred, vtrue = run_epoch(Wva, Uva, False, args.batch_size)
    rho = spearmanr(vpred, vtrue).statistic
    r2 = r2_score(vtrue, vpred)
    mae_dex = np.mean(np.abs(vpred - vtrue))
    mae_m = np.mean(np.abs(10**vpred - 10**vtrue))
    row = dict(epoch=ep, train_loss=round(train_loss,4), val_loss=round(val_loss,4),
               val_spearman=round(float(rho),4), val_r2=round(float(r2),4),
               val_mae_dex=round(float(mae_dex),3), val_mae_m=round(float(mae_m),1),
               time_s=round(time.time()-t0,1))
    log.append(row); print(json.dumps(row), flush=True)
    with open(OUT/"training_log.json","w") as f: json.dump(log, f, indent=2)
    if rho > best_val_rho:
        best_val_rho = rho
        torch.save(model.state_dict(), OUT/"best.pt")

print("\nDONE. Best val Spearman:", best_val_rho, flush=True)
json.dump(dict(best_val_spearman=float(best_val_rho), n_params=n_params, args=vars(args)),
          open(OUT/"summary.json","w"), indent=2)
