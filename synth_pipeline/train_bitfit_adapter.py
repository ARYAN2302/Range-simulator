#!/usr/bin/env python3
"""Clean, controlled re-test of "does synthetic pretraining transfer" -- BitFit adapter with
session-level early stopping, on top of the AGC-fixed generator.

This fixes every methodology gap identified in earlier sweeps:
  1. Generator: uses a dataset generated with torchsig's DigitalAGC/CoarseGainChange EXCLUDED
     (see simulate.py's exclude_agc fix) -- those transforms are designed to remove exactly the
     amplitude-vs-range mapping this pipeline needs to teach.
  2. Seeding: both the random-control backbone init AND all training randomness are seeded
     identically per condition (train_frozen_adapter.py's random control was unseeded and its
     own results varied run-to-run by as much as the effect being measured).
  3. Validation: SESSION-level (leave-one-session-out) early stopping, not window-level KFold --
     one of the 6 real training sessions is held out as an inner validation session; the epoch
     is chosen by inner-val Spearman, never by the true held-out test session.
  4. Adapter: BitFit (Zaken et al. 2021) instead of a frozen-linear-probe or linear-residual-
     correction -- only bias parameters and the tiny RMSNorm/LayerScale scale parameters are
     unfrozen (plus a fresh head), so the backbone cannot grow new session-specific filters,
     only shift/rescale existing ones. This is the adapter deep_research_brief_6 recommended
     over full fine-tuning or a frozen probe for small, confound-prone real data.

Compares pretrained-synthetic vs. random-control backbones under identical treatment, plus the
log_rms-only linear baseline (the bar every prior experiment has struggled to beat).
"""
import sys, json, argparse, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/naveen/Desktop/Learned representation/Raptor")
from real_data import load_sessions, real_windows_for_sessions
from src.models.raptorfm_tokenizer import RAPTORFMTokenizer, RMSNorm, LayerScale
from src.models.raptorfm_encoder import RAPTORFMEncoder

p = argparse.ArgumentParser()
p.add_argument('--real_holdout_session', type=str, required=True)
p.add_argument('--pretrained_ckpt', type=str, required=True)
p.add_argument('--epochs', type=int, default=8)
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out_tag', type=str, default='bitfit')
args = p.parse_args()

D = 192
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
OUT = Path(f"/home/naveen/Desktop/Learned representation/experiments/synth_pipeline/results_{args.out_tag}_{args.real_holdout_session}")
OUT.mkdir(parents=True, exist_ok=True)
print(f"DEVICE={DEVICE} holdout={args.real_holdout_session} out={OUT}", flush=True)


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

    def forward(self, x, use_checkpoint=False):
        feat = ckpt.checkpoint(self._backbone, x, use_reentrant=False) if (use_checkpoint and self.training) else self._backbone(x)
        return self.head(feat).squeeze(-1)


def build_model(pretrained):
    m = RangePredictor().to(DEVICE)
    if pretrained:
        sd = torch.load(args.pretrained_ckpt, map_location=DEVICE, weights_only=True)
        bb_sd = {k: v for k, v in sd.items() if k.startswith('tok.') or k.startswith('enc.')}
        m.load_state_dict(bb_sd, strict=False)  # head intentionally left fresh/random
        print("Loaded PRETRAINED backbone weights (head reinitialized fresh).", flush=True)
    else:
        print("Using RANDOM (untrained, seeded) backbone -- control condition.", flush=True)
    return m


def apply_bitfit(model):
    """Freeze everything, then unfreeze: (a) all Linear/Conv bias parameters in tok+enc,
    (b) RMSNorm.weight and LayerScale.gamma (the only scale-type params in this architecture),
    (c) the head (fresh, fully trainable). Cannot create new session-specific filters -- only
    shift biases or rescale existing channels."""
    n_total = 0; n_trainable = 0
    for p_ in model.parameters():
        p_.requires_grad_(False)
        n_total += p_.numel()
    for name, module in model.tok.named_modules():
        if isinstance(module, (RMSNorm, LayerScale)):
            for p_ in module.parameters():
                p_.requires_grad_(True); n_trainable += p_.numel()
        for pname, p_ in module.named_parameters(recurse=False):
            if pname == 'bias':
                p_.requires_grad_(True); n_trainable += p_.numel()
    for name, module in model.enc.named_modules():
        if isinstance(module, (RMSNorm, LayerScale)):
            for p_ in module.parameters():
                p_.requires_grad_(True); n_trainable += p_.numel()
        for pname, p_ in module.named_parameters(recurse=False):
            if pname == 'bias':
                p_.requires_grad_(True); n_trainable += p_.numel()
    for p_ in model.head.parameters():
        p_.requires_grad_(True); n_trainable += p_.numel()
    print(f"BitFit: {n_trainable:,}/{n_total:,} params trainable ({100*n_trainable/n_total:.2f}%)", flush=True)
    return model


def to_tensor(W):
    return torch.from_numpy(W).unsqueeze(2).float()


def run_eval(model, W, U, bs=64):
    model.eval()
    preds = np.empty(len(W))
    with torch.no_grad():
        for s in range(0, len(W), bs):
            xb = to_tensor(W[s:s+bs]).to(DEVICE)
            preds[s:s+bs] = model(xb, use_checkpoint=False).cpu().numpy()
    rho = float(spearmanr(preds, U).statistic)
    r2 = float(r2_score(U, preds))
    mae_m = float(np.mean(np.abs(10**preds - 10**U)))
    return dict(spearman=rho, r2=r2, mae_m=mae_m), preds


def train_condition(tag, pretrained, Winner_tr, Uinner_tr, Winner_val, Uinner_val, Wtest, Utest):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = build_model(pretrained)
    apply_bitfit(model)
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    log = []
    best = (-np.inf, None, None)  # (inner_val_spearman, epoch, state_dict)
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        model.train()
        idx = rng.permutation(len(Winner_tr))
        tot_loss = 0.0
        for s in range(0, len(idx), args.batch_size):
            b = idx[s:s+args.batch_size]
            xb = to_tensor(Winner_tr[b]).to(DEVICE)
            yb = torch.tensor(Uinner_tr[b], dtype=torch.float32, device=DEVICE)
            amp_ctx = torch.amp.autocast('cuda') if USE_AMP else torch.autocast('cpu', enabled=False)
            with amp_ctx:
                pred = model(xb, use_checkpoint=True)
                loss = F.huber_loss(pred, yb, delta=1.0)
            opt.zero_grad()
            if scaler: scaler.scale(loss).backward(); scaler.unscale_(opt)
            else: loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if scaler: scaler.step(opt); scaler.update()
            else: opt.step()
            tot_loss += float(loss) * len(b)
        inner_val_metrics, _ = run_eval(model, Winner_val, Uinner_val)
        row = dict(epoch=ep, train_loss=round(tot_loss/len(idx), 4), inner_val=inner_val_metrics)
        log.append(row); print(f"[{tag}] {json.dumps(row)}", flush=True)
        if inner_val_metrics['spearman'] > best[0]:
            best = (inner_val_metrics['spearman'], ep, copy.deepcopy(model.state_dict()))

    # reload best-by-inner-val-only state, then evaluate ONCE on the true held-out test session
    model.load_state_dict(best[2])
    test_metrics, test_preds = run_eval(model, Wtest, Utest)
    print(f"[{tag}] SELECTED epoch={best[1]} (inner_val_spearman={best[0]:.4f}) -> "
          f"test_spearman={test_metrics['spearman']:.4f} test_r2={test_metrics['r2']:.4f} "
          f"test_mae_m={test_metrics['mae_m']:.1f}", flush=True)
    return dict(log=log, selected_epoch=best[1], inner_val_spearman=best[0], test=test_metrics)


def main():
    captures = load_sessions()
    all_sessions = sorted(captures['session_id'].unique())
    train_sessions = [s for s in all_sessions if s != args.real_holdout_session]
    inner_val_session = sorted(train_sessions)[0]
    inner_train_sessions = [s for s in train_sessions if s != inner_val_session]
    print(f"Held out (true test): {args.real_holdout_session}", flush=True)
    print(f"Inner val (epoch selection only): {inner_val_session}", flush=True)
    print(f"Inner train: {inner_train_sessions}", flush=True)

    Wtr_raw, Dtr, _, _, _ = real_windows_for_sessions(inner_train_sessions, max_windows_per_packet=3, seed=args.seed)
    Wval_raw, Dval, _, _, _ = real_windows_for_sessions([inner_val_session], max_windows_per_packet=4, seed=args.seed)
    Wte_raw, Dte, _, _, _ = real_windows_for_sessions([args.real_holdout_session], max_windows_per_packet=8, seed=args.seed)
    print(f"inner_train={len(Wtr_raw)} inner_val={len(Wval_raw)} test={len(Wte_raw)}", flush=True)

    scale = float(np.sqrt(np.mean(Wtr_raw[:200] ** 2)) + 1e-12)
    Wtr = (Wtr_raw / scale).astype(np.float32)
    Wval = (Wval_raw / scale).astype(np.float32)
    Wte = (Wte_raw / scale).astype(np.float32)
    Utr = np.log10(Dtr); Uval = np.log10(Dval); Ute = np.log10(Dte)

    # log_rms-only baseline, for reference (fit on ALL 6 training sessions -- it's not being
    # epoch-selected so no leakage concern from using the full pool here)
    print("\n=== log_rms-only baseline (reference bar) ===", flush=True)
    tmp = RangePredictor().to(DEVICE); tmp.eval()
    with torch.no_grad():
        def get_logrms(W, bs=64):
            out = np.empty((len(W), 1), dtype=np.float32)
            for s in range(0, len(W), bs):
                xb = to_tensor(W[s:s+bs]).to(DEVICE)
                t = tmp.tok(xb, None)
                out[s:s+bs] = t["log_rms"].cpu().numpy()
            return out
        lrms_tr = get_logrms(Wtr); lrms_te = get_logrms(Wte)
    base_reg = Ridge(alpha=10.0).fit(lrms_tr, Utr)
    base_pred = base_reg.predict(lrms_te)
    base_rho = float(spearmanr(base_pred, Ute).statistic)
    base_r2 = float(r2_score(Ute, base_pred))
    print(f"log_rms baseline: test_spearman={base_rho:.4f} test_r2={base_r2:.4f}", flush=True)
    del tmp

    results = {'log_rms_baseline': dict(spearman=base_rho, r2=base_r2)}
    for tag, pretrained in [("pretrained_synthetic", True), ("random_control", False)]:
        print(f"\n=== {tag} (BitFit) ===", flush=True)
        results[tag] = train_condition(tag, pretrained, Wtr, Utr, Wval, Uval, Wte, Ute)

    json.dump(results, open(OUT / "results.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(f"log_rms_baseline: spearman={base_rho:.4f} r2={base_r2:.4f}")
    for tag in ["pretrained_synthetic", "random_control"]:
        t = results[tag]['test']
        print(f"{tag:22s} spearman={t['spearman']:.4f} r2={t['r2']:.4f} mae_m={t['mae_m']:.1f} "
              f"(selected epoch {results[tag]['selected_epoch']}, inner_val_rho={results[tag]['inner_val_spearman']:.4f})")


if __name__ == "__main__":
    main()
