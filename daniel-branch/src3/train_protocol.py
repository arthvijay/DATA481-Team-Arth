"""
train_protocol.py  —  DeepMon Stage 1: Protocol Classification
==============================================================
Classifies Wi-Fi protocol from [2, 12, 64] L-SIG spectrogram.
Trained SEPARATELY from Stage 2 (as in the paper).

Classes:
  0 = Non-HT  (802.11a / legacy)
  1 = HT      (802.11n)
  2 = VHT     (802.11ac)

This fixes Gap 2: Stage 1 was completely missing from src2.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import Config
from dataset import discover_records, PacketRecord, load_waveform, \
                    normalize, apply_phase_augment, wave_to_2d_spectrogram
from model import ProtocolClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROTOCOL_MAP = {"Non-HT": 0, "HT": 1, "VHT": 2}
PROTOCOL_NAMES = ["Non-HT (802.11a)", "HT (802.11n)", "VHT (802.11ac)"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Protocol dataset (all protocols, no LSIG filter) ───
class ProtocolDataset(Dataset):
    def __init__(self, records: list, cfg: Config, augment: bool = False):
        # Keep all records that have a known protocol
        self.records = [r for r in records if r.protocol in PROTOCOL_MAP]
        self.cfg     = cfg
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        wave = load_waveform(rec.bin_path)

        if self.augment and self.cfg.use_phase_augment:
            wave = apply_phase_augment(wave)

        wave = normalize(wave)
        x    = wave_to_2d_spectrogram(wave, self.cfg.n_fft, self.cfg.n_symbols)
        y    = PROTOCOL_MAP[rec.protocol]   # integer class label

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long),
        }


def build_protocol_datasets(cfg: Config):
    print("Discovering records (all protocols)...")
    train_all = discover_records(cfg.root_dir, "Train")
    test_recs = discover_records(cfg.root_dir, "Test")

    # No protocol filter — use everything with a known protocol
    train_all = [r for r in train_all if r.protocol in PROTOCOL_MAP]
    test_recs = [r for r in test_recs if r.protocol in PROTOCOL_MAP]

    # Print protocol distribution
    for name, recs in [("Train", train_all), ("Test", test_recs)]:
        from collections import Counter
        counts = Counter(r.protocol for r in recs)
        print(f"\n{name} protocol distribution:")
        for prot, n in sorted(counts.items()):
            print(f"  {prot:8s}: {n:,}")

    train_recs, val_recs = train_test_split(
        train_all, test_size=cfg.val_split, random_state=cfg.random_seed,
        stratify=[r.protocol for r in train_all]   # stratified split
    )

    ds_train = ProtocolDataset(train_recs, cfg, augment=True)
    ds_val   = ProtocolDataset(val_recs,   cfg, augment=False)
    ds_test  = ProtocolDataset(test_recs,  cfg, augment=False)

    return ds_train, ds_val, ds_test


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = correct = total = 0
    all_preds = []
    all_labels = []

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
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


def main():
    cfg = Config()
    cfg.output_dir = "outputs_protocol"
    set_seed(cfg.random_seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"[Stage 1 — Protocol Classifier] Device: {DEVICE}")

    ds_train, ds_val, ds_test = build_protocol_datasets(cfg)
    dl_train = DataLoader(ds_train, cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = ProtocolClassifier(
        num_classes=3,
        num_channels=cfg.num_channels,
        num_res_blocks=cfg.num_res_blocks,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    best_val_acc = -1.0
    best_path = os.path.join(cfg.output_dir, "best_model.pt")

    for epoch in range(1, cfg.num_epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, dl_train, criterion, optimizer)
        va_loss, va_acc, _, _ = run_epoch(model, dl_val,   criterion)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{cfg.num_epochs} | "
            f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
            f"Val Loss {va_loss:.4f} Acc {va_acc:.4f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best saved (val acc={va_acc:.4f})")

    # ── Test ────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    te_loss, te_acc, te_preds, te_labels = run_epoch(model, dl_test, criterion)

    print(f"\n{'='*55}")
    print("TEST RESULTS — Protocol Classification")
    print(f"{'='*55}")
    print(f"Test Loss     : {te_loss:.4f}")
    print(f"Test Accuracy : {te_acc:.4f}")
    print(f"\n(Paper @ 5MHz: 99.17% accuracy)")
    print()
    print(classification_report(
        te_labels, te_preds,
        target_names=["Non-HT", "HT", "VHT"],
        digits=4,
    ))

    print("Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(te_labels, te_preds)
    header = f"{'':>10}" + "".join(f"{n:>10}" for n in ["Non-HT", "HT", "VHT"])
    print(header)
    for i, row in enumerate(cm):
        print(f"{'  '+['Non-HT','HT','VHT'][i]:>10}" + "".join(f"{v:>10}" for v in row))

    print(f"\nModel saved to: {cfg.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
