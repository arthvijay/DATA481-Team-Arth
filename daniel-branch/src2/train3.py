#No-FFT 1D ResNet
import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import compute_pos_weight
from dataset_nofft import build_datasets_nofft
from model_1d import DeepMonNoFFT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bit_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    bit_acc = (preds == targets).float().mean().item()
    exact   = (preds == targets).all(dim=1).float().mean().item()
    return bit_acc, exact, preds


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_ba, total_ex, n = 0.0, 0.0, 0
    bit_correct = bit_total = None

    for batch in loader:
        x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
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

    return (
        total_loss / len(loader.dataset),
        total_ba / n, total_ex / n,
        bit_correct / np.maximum(bit_total, 1),
    )


def main():
    cfg = Config()
    cfg.output_dir = "outputs_nofft"    # 独立输出目录
    set_seed(cfg.random_seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"[No-FFT Ablation] Device: {DEVICE}")
    print("Ablation: Raw time-domain waveform (NO FFT preprocessing)")

    ds_train, ds_val, ds_test, train_recs = build_datasets_nofft(cfg)
    dl_train = DataLoader(ds_train, cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers)
    dl_val   = DataLoader(ds_val,   cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    dl_test  = DataLoader(ds_test,  cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # 验证输入shape
    sample = next(iter(dl_train))
    print(f"Input shape: {sample['x'].shape}")  # 应该是 [64, 2, 768]

    model = DeepMonNoFFT(
        num_bits=cfg.num_bits,
        num_channels=cfg.num_channels,
        num_res_blocks=cfg.num_res_blocks,
        input_length=cfg.n_symbols * cfg.n_fft,
    ).to(DEVICE)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    pos_weight = compute_pos_weight(train_recs).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs
    )

    best_val_exact = -1.0
    best_path = os.path.join(cfg.output_dir, "best_model.pt")

    for epoch in range(1, cfg.num_epochs + 1):
        tr = run_epoch(model, dl_train, criterion, optimizer)
        va = run_epoch(model, dl_val,   criterion)
        scheduler.step()

        print(f"[No-FFT] Epoch {epoch:03d}/{cfg.num_epochs} | "
              f"Train Loss {tr[0]:.4f} BitAcc {tr[1]:.4f} Exact {tr[2]:.4f} | "
              f"Val Loss {va[0]:.4f} BitAcc {va[1]:.4f} Exact {va[2]:.4f}")

        if va[2] > best_val_exact:
            best_val_exact = va[2]
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best saved (val exact={va[2]:.4f})")

    # 测试
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    te = run_epoch(model, dl_test, criterion)

    print(f"\n{'='*50}")
    print(f"[No-FFT] TEST RESULTS")
    print(f"{'='*50}")
    print(f"Test Loss       : {te[0]:.4f}")
    print(f"Test Bit Acc    : {te[1]:.4f}")
    print(f"Test Exact Match: {te[2]:.4f}")
    print("\nPer-bit accuracy:")
    for i, acc in enumerate(te[3]):
        bar = "█" * int(acc * 20)
        print(f"  bit[{i:02d}]: {acc:.4f} {bar}")

    print(f"\n[Summary for ablation table]")
    print(f"  No-FFT  BitAcc={te[1]:.4f}  Exact={te[2]:.4f}")
    print(f"  (Compare with FFT version in outputs_deepmon/)")


if __name__ == "__main__":
    main()