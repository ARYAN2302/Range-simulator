#!/usr/bin/env python3
"""BitFit adapter evaluation for the REAL, fully-assembled RAPTORFM class (not a hand-rolled
reimplementation) -- same rigor as train_bitfit_adapter.py (session-level validation, seeded
random control, log_rms reference baseline), applied to the joint SSL+supervised checkpoint
from train_raptorfm_joint.py. This is the test of whether genuine masked-reconstruction
exposure to real IQ data (not just supervised regression on synthetic labels) produces a
representation that beats a random-backbone control -- something pure supervised pretraining
did not achieve today under the same protocol.
"""
import sys, json, argparse, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/naveen/Desktop/Learned representation/Raptor")
from real_data import load_sessions, real_windows_for_sessions
from src.models.raptorfm_tokenizer import RMSNorm, LayerScale
from src.models.raptorfm import RAPTORFM

p = argparse.ArgumentParser()
p.add_argument('--real_holdout_session', type=str, required=True)
p.add_argument('--pretrained_ckpt', type=str, required=True)
p.add_argument('--epochs', type=int, default=8)
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out_tag', type=str, default='raptorfm_bitfit')
args = p.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
OUT = Path(f"/home/naveen/Desktop/Learned representation/experiments/synth_pipeline/results_{args.out_tag}_{args.real_holdout_session}")
OUT.mkdir(parents=True, exist_ok=True)
print(f"DEVICE={DEVICE} holdout={args.real_holdout_session} out={OUT}", flush=True)


def build_model(pretrained):
    m = RAPTORFM(d_model=192, n_heads=6, d_ff=768, n_layers=6, d_physical=64,
                 patch=8, stride=8, mask_ratio=0.6, mode="C").to(DEVICE)
    if pretrained:
        sd = torch.load(args.pretrained_ckpt, map_location=DEVICE, weights_only=True)
        m.load_state_dict(sd, strict=True)
        print("Loaded PRETRAINED (joint SSL+supervised) weights.", flush=True)
    else:
        print("Using RANDOM (untrained, seeded) backbone -- control condition.", flush=True)
    return m


def apply_bitfit(model):
    """Freeze everything, unfreeze bias/RMSNorm/LayerScale in tokenizer+encoder (cannot grow
    new session-specific filters), plus fully retrain fusion + range_head (fresh downstream
    head). Decoder is irrelevant downstream and left frozen/unused."""
    n_total = 0; n_trainable = 0
    for p_ in model.parameters():
        p_.requires_grad_(False); n_total += p_.numel()
    for sub in [model.tokenizer, model.encoder]:
        for name, module in sub.named_modules():
            if isinstance(module, (RMSNorm, LayerScale)):
                for p_ in module.parameters():
                    p_.requires_grad_(True); n_trainable += p_.numel()
            for pname, p_ in module.named_parameters(recurse=False):
                if pname == 'bias':
                    p_.requires_grad_(True); n_trainable += p_.numel()
    for p_ in model.fusion.parameters():
        p_.requires_grad_(True); n_trainable += p_.numel()
    for p_ in model.range_head.parameters():
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
            preds[s:s+bs] = model(xb)['range_pred'].float().cpu().numpy()
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
    best = (-np.inf, None, None)
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
                pred = model(xb)['range_pred']
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

    print("\n=== log_rms-only baseline (reference bar) ===", flush=True)
    tmp = RAPTORFM(d_model=192, n_heads=6, d_ff=768, n_layers=6, d_physical=64,
                    patch=8, stride=8, mask_ratio=0.6, mode="C").to(DEVICE)
    tmp.eval()
    with torch.no_grad():
        def get_logrms(W, bs=64):
            out = np.empty((len(W), 1), dtype=np.float32)
            for s in range(0, len(W), bs):
                xb = to_tensor(W[s:s+bs]).to(DEVICE)
                t = tmp.tokenizer(xb, None)
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
    for tag, pretrained in [("pretrained_ssl_joint", True), ("random_control", False)]:
        print(f"\n=== {tag} (BitFit) ===", flush=True)
        results[tag] = train_condition(tag, pretrained, Wtr, Utr, Wval, Uval, Wte, Ute)

    json.dump(results, open(OUT / "results.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(f"log_rms_baseline: spearman={base_rho:.4f} r2={base_r2:.4f}")
    for tag in ["pretrained_ssl_joint", "random_control"]:
        t = results[tag]['test']
        print(f"{tag:22s} spearman={t['spearman']:.4f} r2={t['r2']:.4f} mae_m={t['mae_m']:.1f} "
              f"(selected epoch {results[tag]['selected_epoch']}, inner_val_rho={results[tag]['inner_val_spearman']:.4f})")


if __name__ == "__main__":
    main()
