"""
train_ablations.py  —  All ablation experiments in one script.
Runs sequentially (or pick individual ones):
  A. Depth ablation    : 2 / 4 / 6 / 8 residual blocks
  B. FFT ablation      : with FFT (2D) vs without FFT (1D)
  C. Augmentation      : phase augment on vs off
  D. Attention variant : base ResNet vs ResNet + SE attention
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import build_datasets, compute_pos_weight
from model import DeepMonModel, DeepMonNoFFT
from lsig_decode import compute_tx_time_mae, compute_length_mae

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bit_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return (
        (preds == targets).float().mean().item(),
        (preds == targets).all(dim=1).float().mean().item(),
        preds,
    )


def run_epoch(model, loader, criterion, optimizer=None, collect_preds=False):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = total_ba = total_ex = 0.0
    n = 0
    bit_correct = bit_total = None
    all_preds = []
    all_gts   = []

    for batch in loader:
        x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss   = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        ba, ex, preds = bit_metrics(logits, y)
        total_ba += ba; total_ex += ex; n += 1

        correct = (preds == y).sum(dim=0).detach().cpu().numpy()
        total   = np.full(y.shape[1], y.shape[0])
        bit_correct = correct if bit_correct is None else bit_correct + correct
        bit_total   = total   if bit_total   is None else bit_total   + total

        if collect_preds:
            all_preds.append(torch.sigmoid(logits).detach().cpu())
            all_gts.append(y.detach().cpu())

    per_bit = bit_correct / np.maximum(bit_total, 1)
    res = (total_loss / len(loader.dataset), total_ba / n, total_ex / n, per_bit)

    if collect_preds:
        return res, torch.cat(all_preds), torch.cat(all_gts)
    return res


def train_model(model, cfg, dl_train, dl_val, dl_test, train_recs, output_dir, label):
    os.makedirs(output_dir, exist_ok=True)
    pos_weight = compute_pos_weight(train_recs).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    best_val_exact = -1.0
    best_path = os.path.join(output_dir, "best_model.pt")

    for epoch in range(1, cfg.num_epochs + 1):
        tr = run_epoch(model, dl_train, criterion, optimizer)
        va = run_epoch(model, dl_val,   criterion)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{label}] Epoch {epoch:03d} | "
                  f"Train Exact {tr[2]:.4f} | Val Exact {va[2]:.4f}")

        if va[2] > best_val_exact:
            best_val_exact = va[2]
            torch.save(model.state_dict(), best_path)

    # Test
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    (te_loss, te_ba, te_ex, per_bit), all_preds, all_gts = run_epoch(
        model, dl_test, criterion, collect_preds=True
    )

    tx_mae = compute_tx_time_mae(all_preds.numpy(), all_gts.numpy())
    l_mae  = compute_length_mae(all_preds.numpy(),  all_gts.numpy())

    print(f"\n  [{label}] TEST → BitAcc={te_ba:.4f}  Exact={te_ex:.4f}  "
          f"TxMAE={tx_mae['mae_ms']:.4f}ms  LenMAE={l_mae['mae_bytes']:.1f}B")

    return {
        "label": label,
        "bit_acc": te_ba,
        "exact_match": te_ex,
        "tx_mae_ms": tx_mae["mae_ms"],
        "len_mae_bytes": l_mae["mae_bytes"],
    }


# ════════════════════════════════════════════════════════
#  A. Depth ablation
# ════════════════════════════════════════════════════════
def ablation_depth(cfg, dl_train, dl_val, dl_test, train_recs):
    print(f"\n{'='*60}")
    print("ABLATION A: Network Depth (2 / 4 / 6 / 8 blocks)")
    print(f"{'='*60}")
    results = []
    for n_blocks in [2, 4, 6, 8]:
        model  = DeepMonModel(cfg.num_bits, cfg.num_channels, n_blocks).to(DEVICE)
        params = sum(p.numel() for p in model.parameters())
        label  = f"depth_{n_blocks}blocks"
        print(f"\n  blocks={n_blocks}, params={params:,}")
        r = train_model(model, cfg, dl_train, dl_val, dl_test, train_recs,
                        f"outputs_{label}", label)
        r["params"] = params
        results.append(r)

    print(f"\n{'─'*70}")
    print(f"{'Blocks':>8} {'Params':>10} {'BitAcc':>10} {'Exact':>10} {'TxMAE(ms)':>12} {'LenMAE(B)':>12}")
    for r in results:
        print(f"{r['label'].split('_')[1]:>8} {r['params']:>10,} "
              f"{r['bit_acc']:>10.4f} {r['exact_match']:>10.4f} "
              f"{r['tx_mae_ms']:>12.4f} {r['len_mae_bytes']:>12.1f}")
    return results


# ════════════════════════════════════════════════════════
#  B. FFT ablation
# ════════════════════════════════════════════════════════
def ablation_fft(cfg):
    print(f"\n{'='*60}")
    print("ABLATION B: FFT vs No-FFT")
    print(f"{'='*60}")
    results = []

    for no_fft in [False, True]:
        label = "NoFFT" if no_fft else "WithFFT"
        ds_train, ds_val, ds_test, train_recs = build_datasets(cfg, no_fft=no_fft)
        dl_train = DataLoader(ds_train, cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
        dl_val   = DataLoader(ds_val,   cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        dl_test  = DataLoader(ds_test,  cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

        if no_fft:
            model = DeepMonNoFFT(cfg.num_bits, cfg.num_channels, cfg.num_res_blocks).to(DEVICE)
        else:
            model = DeepMonModel(cfg.num_bits, cfg.num_channels, cfg.num_res_blocks).to(DEVICE)

        params = sum(p.numel() for p in model.parameters())
        print(f"\n  {label}, params={params:,}")
        r = train_model(model, cfg, dl_train, dl_val, dl_test, train_recs,
                        f"outputs_{label}", label)
        r["params"] = params
        results.append(r)

    print(f"\n{'─'*70}")
    print(f"{'Method':>10} {'Params':>10} {'BitAcc':>10} {'Exact':>10} {'TxMAE(ms)':>12}")
    for r in results:
        print(f"{r['label']:>10} {r['params']:>10,} "
              f"{r['bit_acc']:>10.4f} {r['exact_match']:>10.4f} {r['tx_mae_ms']:>12.4f}")
    return results


# ════════════════════════════════════════════════════════
#  C. Augmentation ablation
# ════════════════════════════════════════════════════════
def ablation_augment():
    print(f"\n{'='*60}")
    print("ABLATION C: Phase Augmentation On vs Off")
    print(f"{'='*60}")
    results = []

    for use_aug in [True, False]:
        cfg = Config()
        cfg.use_phase_augment = use_aug
        label = "AugOn" if use_aug else "AugOff"

        ds_train, ds_val, ds_test, train_recs = build_datasets(cfg)
        dl_train = DataLoader(ds_train, cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
        dl_val   = DataLoader(ds_val,   cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        dl_test  = DataLoader(ds_test,  cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

        model = DeepMonModel(cfg.num_bits, cfg.num_channels, cfg.num_res_blocks).to(DEVICE)
        print(f"\n  {label}")
        r = train_model(model, cfg, dl_train, dl_val, dl_test, train_recs,
                        f"outputs_{label}", label)
        results.append(r)

    print(f"\n{'─'*55}")
    print(f"{'Augment':>10} {'BitAcc':>10} {'Exact':>10} {'TxMAE(ms)':>12}")
    for r in results:
        print(f"{r['label']:>10} {r['bit_acc']:>10.4f} "
              f"{r['exact_match']:>10.4f} {r['tx_mae_ms']:>12.4f}")
    return results


# ════════════════════════════════════════════════════════
#  D. Attention ablation
# ════════════════════════════════════════════════════════
def ablation_attention(cfg, dl_train, dl_val, dl_test, train_recs):
    print(f"\n{'='*60}")
    print("ABLATION D: Base ResNet vs ResNet + SE Attention")
    print(f"{'='*60}")
    results = []

    for use_attn in [False, True]:
        label = "Attention" if use_attn else "BaseResNet"
        model = DeepMonModel(cfg.num_bits, cfg.num_channels, cfg.num_res_blocks,
                             use_attention=use_attn).to(DEVICE)
        params = sum(p.numel() for p in model.parameters())
        print(f"\n  {label}, params={params:,}")
        r = train_model(model, cfg, dl_train, dl_val, dl_test, train_recs,
                        f"outputs_{label}", label)
        r["params"] = params
        results.append(r)

    print(f"\n{'─'*70}")
    print(f"{'Model':>12} {'Params':>10} {'BitAcc':>10} {'Exact':>10} {'TxMAE(ms)':>12}")
    for r in results:
        print(f"{r['label']:>12} {r['params']:>10,} "
              f"{r['bit_acc']:>10.4f} {r['exact_match']:>10.4f} {r['tx_mae_ms']:>12.4f}")
    return results


# ════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════
def main():
    set_seed(42)
    print(f"Device: {DEVICE}")

    cfg = Config()

    # Shared data loaders (used by depth & attention ablations)
    ds_train, ds_val, ds_test, train_recs = build_datasets(cfg)
    dl_train = DataLoader(ds_train, cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    ablation_depth(cfg, dl_train, dl_val, dl_test, train_recs)
    ablation_fft(cfg)
    ablation_augment()
    ablation_attention(cfg, dl_train, dl_val, dl_test, train_recs)

    print("\n✓ All ablations complete.")


if __name__ == "__main__":
    main()
