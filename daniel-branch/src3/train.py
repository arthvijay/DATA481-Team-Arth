"""
train.py  —  DeepMon Stage 2: L-SIG bit decoding
Fixes vs src2:
  - SiLU activation (≈SeLU from paper) instead of ReLU     [Gap 1]
  - Reports Tx Time MAE and Length MAE at test time         [Q2 post-proc]
  - Cleaner single-file, no duplicated run_epoch across files
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import build_datasets, compute_pos_weight
from model import DeepMonModel
from lsig_decode import compute_tx_time_mae, compute_length_mae

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bit_metrics(logits, targets, threshold=0.5):
    probs  = torch.sigmoid(logits)
    preds  = (probs >= threshold).float()
    bit_acc    = (preds == targets).float().mean().item()
    exact_match = (preds == targets).all(dim=1).float().mean().item()
    return bit_acc, exact_match, preds


def run_epoch(model, loader, criterion, optimizer=None, collect_preds=False):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = total_ba = total_ex = 0.0
    n_batches  = 0
    bit_correct = bit_total = None

    all_preds = []
    all_gts   = []

    for batch in loader:
        x = batch["x"].to(DEVICE)
        y = batch["y"].to(DEVICE)

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
        total_ba += ba
        total_ex += ex
        n_batches += 1

        correct = (preds == y).sum(dim=0).detach().cpu().numpy()
        total   = np.full(y.shape[1], y.shape[0])
        bit_correct = correct if bit_correct is None else bit_correct + correct
        bit_total   = total   if bit_total   is None else bit_total   + total

        if collect_preds:
            all_preds.append(torch.sigmoid(logits).detach().cpu())
            all_gts.append(y.detach().cpu())

    per_bit = bit_correct / np.maximum(bit_total, 1)

    result = (
        total_loss / len(loader.dataset),
        total_ba   / n_batches,
        total_ex   / n_batches,
        per_bit,
    )

    if collect_preds:
        return result, torch.cat(all_preds), torch.cat(all_gts)
    return result


def main():
    cfg = Config()
    set_seed(cfg.random_seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Config: {cfg}")

    # ── Data ────────────────────────────────────────────
    ds_train, ds_val, ds_test, train_recs = build_datasets(cfg)
    dl_train = DataLoader(ds_train, cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # ── Model ───────────────────────────────────────────
    model = DeepMonModel(
        num_bits=cfg.num_bits,
        num_channels=cfg.num_channels,
        num_res_blocks=cfg.num_res_blocks,
        use_attention=False,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Loss, optimizer, scheduler ──────────────────────
    pos_weight = compute_pos_weight(train_recs).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    # ── Training loop ───────────────────────────────────
    best_val_exact = -1.0
    best_path = os.path.join(cfg.output_dir, "best_model.pt")

    for epoch in range(1, cfg.num_epochs + 1):
        tr = run_epoch(model, dl_train, criterion, optimizer)
        va = run_epoch(model, dl_val,   criterion)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{cfg.num_epochs} | "
            f"Train Loss {tr[0]:.4f} BitAcc {tr[1]:.4f} Exact {tr[2]:.4f} | "
            f"Val Loss {va[0]:.4f} BitAcc {va[1]:.4f} Exact {va[2]:.4f}"
        )

        if va[2] > best_val_exact:
            best_val_exact = va[2]
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved (val exact={va[2]:.4f})")

    # ── Test ────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    (te_loss, te_ba, te_ex, per_bit), all_preds, all_gts = run_epoch(
        model, dl_test, criterion, collect_preds=True
    )

    # Bit-level metrics
    print(f"\n{'='*55}")
    print("TEST RESULTS")
    print(f"{'='*55}")
    print(f"Test Loss        : {te_loss:.4f}")
    print(f"Test Bit Acc     : {te_ba:.4f}")
    print(f"Test Exact Match : {te_ex:.4f}")

    print("\nPer-bit accuracy:")
    for i, acc in enumerate(per_bit):
        bar = "█" * int(acc * 20)
        print(f"  bit[{i:02d}]: {acc:.4f} {bar}")

    # ── Post-processing: Tx Time & Length MAE ───────────
    tx_mae = compute_tx_time_mae(all_preds.numpy(), all_gts.numpy())
    l_mae  = compute_length_mae(all_preds.numpy(),  all_gts.numpy())

    print(f"\n{'='*55}")
    print("PHYSICAL PROPERTY DECODING  (post-processed from bits)")
    print(f"{'='*55}")
    print(f"Tx Time MAE : {tx_mae['mae_us']:.3f} μs  =  {tx_mae['mae_ms']:.4f} ms")
    print(f"  (Paper @ 3MHz: 0.077 ms — our target)")
    print(f"  Valid samples : {tx_mae['n_valid']} / {tx_mae['n_valid'] + tx_mae['n_skipped']}")
    print(f"Length  MAE : {l_mae['mae_bytes']:.2f} bytes  =  {l_mae['mae_kb']:.4f} kB")
    print(f"  Valid samples : {l_mae['n_valid']}")

    print(f"\nResults saved to: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
