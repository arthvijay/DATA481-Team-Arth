#DeepMon
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import build_datasets, compute_pos_weight
from model import DeepMonModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bit_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    bit_acc = (preds == targets).float().mean().item()
    exact_match = (preds == targets).all(dim=1).float().mean().item()
    return bit_acc, exact_match, preds


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_bit_acc = 0.0
    total_exact = 0.0
    n_batches = 0

    bit_correct = None
    bit_total = None

    for batch in loader:
        x = batch["x"].to(DEVICE)
        y = batch["y"].to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        bit_acc, exact, preds = bit_metrics(logits, y)
        total_bit_acc += bit_acc
        total_exact += exact
        n_batches += 1

        correct = (preds == y).sum(dim=0).detach().cpu().numpy()
        total = np.full(y.shape[1], y.shape[0])
        if bit_correct is None:
            bit_correct = correct
            bit_total = total
        else:
            bit_correct += correct
            bit_total += total

    return (
        total_loss / len(loader.dataset),
        total_bit_acc / n_batches,
        total_exact / n_batches,
        bit_correct / np.maximum(bit_total, 1),
    )


def plot_and_save(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.title("BCE Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_bit_acc"], label="Train")
    plt.plot(epochs, history["val_bit_acc"], label="Val")
    plt.title("Bit Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_exact"], label="Train")
    plt.plot(epochs, history["val_exact"], label="Val")
    plt.title("Exact Match")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()


def main():
    cfg = Config()
    set_seed(cfg.random_seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.expanduser("~/logs"), exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Config: {cfg}")

    # 数据
    ds_train, ds_val, ds_test, train_recs = build_datasets(cfg)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size,
                          shuffle=True, num_workers=cfg.num_workers)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size,
                        shuffle=False, num_workers=cfg.num_workers)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size,
                         shuffle=False, num_workers=cfg.num_workers)

    # 模型
    model = DeepMonModel(
        num_bits=cfg.num_bits,
        num_channels=cfg.num_channels,
        num_res_blocks=cfg.num_res_blocks,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss：pos_weight处理不平衡
    pos_weight = compute_pos_weight(train_recs).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer + 学习率调度
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs
    )

    # 训练
    best_val_exact = -1.0
    best_path = os.path.join(cfg.output_dir, "best_model.pt")

    history = {k: [] for k in [
        "train_loss", "val_loss",
        "train_bit_acc", "val_bit_acc",
        "train_exact", "val_exact",
    ]}

    for epoch in range(1, cfg.num_epochs + 1):
        tr_loss, tr_ba, tr_ex, _ = run_epoch(model, dl_train, criterion, optimizer)
        va_loss, va_ba, va_ex, _ = run_epoch(model, dl_val, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_bit_acc"].append(tr_ba)
        history["val_bit_acc"].append(va_ba)
        history["train_exact"].append(tr_ex)
        history["val_exact"].append(va_ex)

        print(
            f"Epoch {epoch:03d}/{cfg.num_epochs} | "
            f"Train Loss {tr_loss:.4f} BitAcc {tr_ba:.4f} Exact {tr_ex:.4f} | "
            f"Val Loss {va_loss:.4f} BitAcc {va_ba:.4f} Exact {va_ex:.4f}"
        )

        if va_ex > best_val_exact:
            best_val_exact = va_ex
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved (val exact={va_ex:.4f})")

    # 测试
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    te_loss, te_ba, te_ex, per_bit = run_epoch(model, dl_test, criterion)

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Test Loss       : {te_loss:.4f}")
    print(f"Test Bit Acc    : {te_ba:.4f}")
    print(f"Test Exact Match: {te_ex:.4f}")
    print("\nPer-bit accuracy:")
    for i, acc in enumerate(per_bit):
        bar = "█" * int(acc * 20)
        print(f"  bit[{i:02d}]: {acc:.4f} {bar}")

    plot_and_save(history, cfg.output_dir)
    print(f"\nResults saved to: {cfg.output_dir}/")


if __name__ == "__main__":
    main()