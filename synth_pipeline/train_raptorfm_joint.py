#!/usr/bin/env python3
"""Joint masked-reconstruction (SSL) + supervised range regression, using the REAL, fully
assembled RAPTORFM class (Raptor/src/models/raptorfm.py) -- tokenizer + encoder + decoder +
fusion + range head, all wired together, with the inter-channel residual/norm bug fixed today.

Why this exists: every training script run this session (train_synth.py, train_frozen_adapter.py,
train_bitfit_adapter.py) was a hand-rolled partial reimplementation of this architecture that
duplicated the [h_cls,h_phys,log_rms] fusion by hand and NEVER exercised the model's own
masked-reconstruction decoder -- mask_ratio was 0.0 in every run today. This script runs the
architecture's actual intended self-supervised objective (per arXiv:2608.05793 "Radio-FM",
Channel-Independent masking at the paper's own ablation-identified optimal ratio, 60%) for the
first time, JOINTLY with the supervised range label where available (SupMAE-style).

Key idea this unlocks: masked reconstruction needs no distance label, so real IQ windows from
ALL 7 sessions can be used for the reconstruction loss with ZERO of the session/time confound
risk that sank every earlier real-data mixing attempt (that confound only exists in the
distance-regression signal, not in reconstructing the waveform itself). Per batch:
  - synthetic samples (torchsig-generated, confound-free by construction): both losses
  - real samples (no label used): reconstruction loss only

Caveat, stated plainly: this pretrains on ALL 7 real sessions' windows at once (not held out per
downstream session), so a session-recognition shortcut could in principle leak into the
representation even without ever seeing the distance label. If downstream results look
promising, redo with the target held-out session excluded from this pool before trusting the
result -- this is a documented simplification, not a validated final protocol.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys, json, time, argparse, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/naveen/Desktop/Learned representation/Raptor")
from simulate import load_cached_dataset
from real_data import load_sessions, real_windows_for_sessions
from src.models.raptorfm import RAPTORFM

p = argparse.ArgumentParser()
p.add_argument('--dataset_name', type=str, default='torchsig_v3_noagc')
p.add_argument('--epochs', type=int, default=15)
p.add_argument('--batch_size', type=int, default=48)
p.add_argument('--real_frac', type=float, default=0.35, help="fraction of each batch drawn from unlabeled real windows")
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--weight_decay', type=float, default=0.05, help="paper value; was 1e-4 in every earlier run")
p.add_argument('--warmup_frac', type=float, default=0.03, help="paper: 3% linear warmup then cosine decay")
p.add_argument('--mask_ratio', type=float, default=0.6, help="paper's ablation-identified optimum")
p.add_argument('--recon_weight', type=float, default=1.0)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out_name', type=str, default='results_raptorfm_joint_v1')
p.add_argument('--exclude_sessions', type=str, default='', help="comma-separated session ids to "
                "exclude from the REAL reconstruction pool -- required for a genuine unseen-"
                "drone/session eval, since SSL exposure to a session's raw IQ (even without its "
                "distance label) could otherwise leak session-recognition shortcuts downstream")
args = p.parse_args()

D = 192
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
OUT = Path(f"/home/naveen/Desktop/Learned representation/experiments/synth_pipeline/{args.out_name}")
OUT.mkdir(parents=True, exist_ok=True)
print(f"DEVICE={DEVICE} AMP={USE_AMP} OUT={OUT}", flush=True)
torch.manual_seed(args.seed); np.random.seed(args.seed)

# ---------------- data ----------------
print(f"Loading synthetic dataset '{args.dataset_name}'...", flush=True)
Wsyn_tr, Dsyn_tr, _, _, manifest = load_cached_dataset(args.dataset_name, "train")
Wsyn_va, Dsyn_va, _, _, _ = load_cached_dataset(args.dataset_name, "val")
Usyn_tr = np.log10(Dsyn_tr); Usyn_va = np.log10(Dsyn_va)
syn_scale = float(np.sqrt(np.mean(Wsyn_tr[:200] ** 2)) + 1e-12)
Wsyn_tr = (Wsyn_tr / syn_scale).astype(np.float32)
Wsyn_va = (Wsyn_va / syn_scale).astype(np.float32)
print(f"synthetic: train={len(Wsyn_tr)} val={len(Wsyn_va)}", flush=True)

captures = load_sessions()
all_sessions = sorted(captures['session_id'].unique())
excluded = [s.strip() for s in args.exclude_sessions.split(',') if s.strip()]
pool_sessions = [s for s in all_sessions if s not in excluded]
print(f"Loading real sessions' windows for the unlabeled reconstruction pool "
      f"(no distance label used from these -- see caveat in module docstring). "
      f"Excluded: {excluded or 'none'}", flush=True)
Wreal_raw, Dreal, _, _, sids_real = real_windows_for_sessions(pool_sessions, max_windows_per_packet=2, seed=args.seed)
real_scale = float(np.sqrt(np.mean(Wreal_raw[:200] ** 2)) + 1e-12)
Wreal = (Wreal_raw / real_scale).astype(np.float32)
print(f"real (unlabeled pool): {len(Wreal)} windows across {len(pool_sessions)} sessions", flush=True)

json.dump(dict(dataset_name=args.dataset_name, syn_scale=syn_scale, real_scale=real_scale,
               n_synth_train=len(Wsyn_tr), n_real_pool=len(Wreal), args=vars(args)),
          open(OUT / "data_manifest.json", "w"), indent=2)

# ---------------- model ----------------
model = RAPTORFM(d_model=D, n_heads=6, d_ff=768, n_layers=6, d_physical=64,
                  patch=8, stride=8, mask_ratio=args.mask_ratio, mode="C").to(DEVICE)
n_params = sum(pp.numel() for pp in model.parameters())
print(f"Params: {n_params:,} ({n_params/1e6:.2f}M)  mask_ratio={args.mask_ratio}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

steps_per_epoch = math.ceil(len(Wsyn_tr) / args.batch_size)
total_steps = steps_per_epoch * args.epochs
warmup_steps = max(1, int(total_steps * args.warmup_frac))

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * prog))

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def to_tensor(W):
    return torch.from_numpy(W).unsqueeze(2).float()


def recon_loss(out):
    # per-ELEMENT masked MSE -- normalize by (masked tokens * D), not just masked tokens,
    # otherwise this is inflated by a factor of D (192) relative to loss_range's scale, which
    # would let reconstruction totally dominate the combined gradient.
    D_ = out['dec_i'].shape[-1]
    mi = out['mask_i'].unsqueeze(-1).float()
    mq = out['mask_q'].unsqueeze(-1).float()
    li = ((out['dec_i'] - out['i_t']) ** 2 * mi).sum() / (mi.sum() * D_).clamp(min=1)
    lq = ((out['dec_q'] - out['q_t']) ** 2 * mq).sum() / (mq.sum() * D_).clamp(min=1)
    return (li + lq) / 2


def run_train_step(xb_syn, yb_syn, xb_real):
    model.train()
    xb = torch.cat([xb_syn, xb_real], dim=0) if xb_real is not None else xb_syn
    amp_ctx = torch.amp.autocast('cuda') if USE_AMP else torch.autocast('cpu', enabled=False)
    with amp_ctx:
        out = model(xb)
        n_syn = len(xb_syn)
        range_pred_syn = out['range_pred'][:n_syn]
        loss_range = F.huber_loss(range_pred_syn, yb_syn, delta=1.0)
        loss_recon = recon_loss(out)
        loss = loss_range + args.recon_weight * loss_recon
    opt.zero_grad()
    if scaler:
        scaler.scale(loss).backward(); scaler.unscale_(opt)
    else:
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if scaler:
        scaler.step(opt); scaler.update()
    else:
        opt.step()
    sched.step()
    return float(loss_range), float(loss_recon)


def run_eval_synth_val(bs=64):
    model.eval()
    preds = np.empty(len(Wsyn_va))
    with torch.no_grad():
        for s in range(0, len(Wsyn_va), bs):
            xb = to_tensor(Wsyn_va[s:s+bs]).to(DEVICE)
            amp_ctx = torch.amp.autocast('cuda') if USE_AMP else torch.autocast('cpu', enabled=False)
            with amp_ctx:
                out = model.encode(xb) if False else model(xb)  # full forward (decoder harmless here)
            preds[s:s+bs] = out['range_pred'].float().cpu().numpy()
    rho = float(spearmanr(preds, Usyn_va).statistic)
    r2 = float(r2_score(Usyn_va, preds))
    return dict(spearman=rho, r2=r2)


print("\n=== SANITY: one forward+backward step ===", flush=True)
xb_s = to_tensor(Wsyn_tr[:4]).to(DEVICE); yb_s = torch.tensor(Usyn_tr[:4], dtype=torch.float32, device=DEVICE)
xb_r = to_tensor(Wreal[:4]).to(DEVICE)
lr_, lc_ = run_train_step(xb_s, yb_s, xb_r)
print(f"sanity loss_range={lr_:.3f} loss_recon={lc_:.3f}", flush=True)
print("SANITY PASS\n", flush=True)

print("=== TRAIN (joint SSL + supervised) ===", flush=True)
rng = np.random.default_rng(args.seed)
log = []
best_val_rho = -1
n_real_per_batch = max(1, int(args.batch_size * args.real_frac))
n_syn_per_batch = args.batch_size

for ep in range(args.epochs):
    t0 = time.time()
    idx_syn = rng.permutation(len(Wsyn_tr))
    tot_range, tot_recon, n_batches = 0.0, 0.0, 0
    for s in range(0, len(idx_syn), n_syn_per_batch):
        b_syn = idx_syn[s:s+n_syn_per_batch]
        b_real = rng.integers(0, len(Wreal), size=n_real_per_batch)
        xb_syn = to_tensor(Wsyn_tr[b_syn]).to(DEVICE)
        yb_syn = torch.tensor(Usyn_tr[b_syn], dtype=torch.float32, device=DEVICE)
        xb_real = to_tensor(Wreal[b_real]).to(DEVICE)
        lr_, lc_ = run_train_step(xb_syn, yb_syn, xb_real)
        tot_range += lr_ * len(b_syn); tot_recon += lc_ * len(b_syn); n_batches += len(b_syn)
    val_metrics = run_eval_synth_val()
    row = dict(epoch=ep, train_loss_range=round(tot_range/n_batches, 4), train_loss_recon=round(tot_recon/n_batches, 4),
               lr=round(sched.get_last_lr()[0], 6), val_synth=val_metrics, time_s=round(time.time()-t0, 1))
    log.append(row); print(json.dumps(row), flush=True)
    with open(OUT / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    if val_metrics['spearman'] > best_val_rho:
        best_val_rho = val_metrics['spearman']
        torch.save(model.state_dict(), OUT / "best.pt")

print(f"\nDONE. Best synth val Spearman: {best_val_rho:.4f}", flush=True)
json.dump(dict(best_val_spearman=best_val_rho, n_params=n_params, args=vars(args)),
          open(OUT / "summary.json", "w"), indent=2)
