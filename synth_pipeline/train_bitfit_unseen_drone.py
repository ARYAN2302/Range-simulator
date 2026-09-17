#!/usr/bin/env python3
"""BitFit adaptation for a genuine UNSEEN-DRONE evaluation: trained entirely on one drone's
sessions, evaluated on a completely different drone's sessions that were never touched at any
stage (not in synthetic waveform sourcing, not in SSL reconstruction, not in this adaptation
step). Companion to train_bitfit_raptorfm.py, which does session-level (same-drone) holdout;
this does drone-level holdout, which is the stronger generalization claim the project actually
wants for "unseen drone" detection.
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
p.add_argument('--train_sessions', type=str, required=True, help="comma-separated, one drone's sessions")
p.add_argument('--inner_val_session', type=str, required=True, help="one of train_sessions, held out for epoch selection")
p.add_argument('--eval_sessions', type=str, required=True, help="comma-separated, a DIFFERENT drone's sessions")
p.add_argument('--pretrained_ckpt', type=str, required=True)
p.add_argument('--epochs', type=int, default=8)
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out_tag', type=str, default='unseen_drone')
args = p.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
OUT = Path(f"/home/naveen/Desktop/Learned representation/experiments/synth_pipeline/results_{args.out_tag}")
OUT.mkdir(parents=True, exist_ok=True)
print(f"DEVICE={DEVICE} out={OUT}", flush=True)

train_sessions = [s.strip() for s in args.train_sessions.split(',')]
eval_sessions = [s.strip() for s in args.eval_sessions.split(',')]
inner_train_sessions = [s for s in train_sessions if s != args.inner_val_session]
print(f"Inner train: {inner_train_sessions}", flush=True)
print(f"Inner val (epoch selection only): {args.inner_val_session}", flush=True)
print(f"UNSEEN-DRONE eval sessions (never touched at any stage): {eval_sessions}", flush=True)


def build_model(pretrained):
    m = RAPTORFM(d_model=192, n_heads=6, d_ff=768, n_layers=6, d_physical=64,
                 patch=8, stride=8, mask_ratio=0.6, mode="C").to(DEVICE)
    if pretrained:
        sd = torch.load(args.pretrained_ckpt, map_location=DEVICE, weights_only=True)
        m.load_state_dict(sd, strict=True)
        print("Loaded PRETRAINED (drone-held-out joint SSL+supervised) weights.", flush=True)
    else:
        print("Using RANDOM (untrained, seeded) backbone -- control condition.", flush=True)
    return m


def apply_bitfit(model):
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


def train_condition(tag, pretrained, Winner_tr, Uinner_tr, Winner_val, Uinner_val):
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
    print(f"[{tag}] SELECTED epoch={best[1]} (inner_val_spearman={best[0]:.4f})", flush=True)
    torch.save(model.state_dict(), OUT / f"{tag}_adapted.pt")
    return model, dict(log=log, selected_epoch=best[1], inner_val_spearman=best[0])


def main():
    Wtr_raw, Dtr, _, _, _ = real_windows_for_sessions(inner_train_sessions, max_windows_per_packet=3, seed=args.seed)
    Wval_raw, Dval, _, _, _ = real_windows_for_sessions([args.inner_val_session], max_windows_per_packet=4, seed=args.seed)
    print(f"inner_train={len(Wtr_raw)} inner_val={len(Wval_raw)}", flush=True)

    scale = float(np.sqrt(np.mean(Wtr_raw[:200] ** 2)) + 1e-12)
    Wtr = (Wtr_raw / scale).astype(np.float32)
    Wval = (Wval_raw / scale).astype(np.float32)
    Utr = np.log10(Dtr); Uval = np.log10(Dval)

    results = {'scale': scale, 'eval': {}}
    models = {}
    for tag, pretrained in [("pretrained_ssl_joint", True), ("random_control", False)]:
        print(f"\n=== {tag} (BitFit, drone-held-out) ===", flush=True)
        model, r = train_condition(tag, pretrained, Wtr, Utr, Wval, Uval)
        results[tag] = r
        models[tag] = model

    print("\n=== UNSEEN-DRONE EVALUATION ===", flush=True)
    for sess in eval_sessions:
        Wte_raw, Dte, _, _, _ = real_windows_for_sessions([sess], max_windows_per_packet=8, seed=args.seed)
        Wte = (Wte_raw / scale).astype(np.float32)
        Ute = np.log10(Dte)

        # log_rms-only baseline for this session (fit on the SAME train drone's pool)
        tmp = models['random_control']  # log_rms is backbone-independent; any instance works
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

        sess_result = {'log_rms_baseline': dict(spearman=base_rho, r2=base_r2)}
        for tag in ["pretrained_ssl_joint", "random_control"]:
            m, _ = run_eval(models[tag], Wte, Ute)
            sess_result[tag] = m
            print(f"[{sess}] {tag}: spearman={m['spearman']:.4f} r2={m['r2']:.4f} mae_m={m['mae_m']:.1f}", flush=True)
        print(f"[{sess}] log_rms_baseline: spearman={base_rho:.4f} r2={base_r2:.4f}", flush=True)
        results['eval'][sess] = sess_result

    json.dump(results, open(OUT / "results.json", "w"), indent=2)
    print("\n=== SUMMARY (unseen drone) ===")
    for sess, r in results['eval'].items():
        print(f"{sess:10s} log_rms={r['log_rms_baseline']['spearman']:.3f}  "
              f"pretrained={r['pretrained_ssl_joint']['spearman']:.3f}  "
              f"random={r['random_control']['spearman']:.3f}")


if __name__ == "__main__":
    main()
