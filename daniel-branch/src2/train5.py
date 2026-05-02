#Sui Ji Xiang Wei Pian Yi
import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import build_datasets, compute_pos_weight
from model import DeepMonModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bit_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return (preds == targets).float().mean().item(), \
           (preds == targets).all(dim=1).float().mean().item(), preds


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_ba, total_ex, n = 0.0, 0.0, 0.0, 0
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
        total = np.full(y.shape[1], y.shape[0])
        bit_correct = correct if bit_correct is None else bit_correct + correct
        bit_total   = total   if bit_total   is None else bit_total   + total

    return (total_loss / len(loader.dataset),
            total_ba / n, total_ex / n,
            bit_correct / np.maximum(bit_total, 1))


def run_experiment(use_augment: bool, cfg: Config):
    cfg.use_phase_augment = use_augment
    label = "WithAugment" if use_augment else "NoAugment"
    output_dir = f"outputs_aug_{label}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"[Augment Ablation] use_phase_augment={use_augment}")
    print(f"{'='*55}")

    ds_train, ds_val, ds_test, train_recs = build_datasets(cfg)
    dl_train = DataLoader(ds_train, cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers)
    dl_val   = DataLoader(ds_val,   cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers)
    dl_test  = DataLoader(ds_test,  cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers)

    model = DeepMonModel(
        num_bits=cfg.num_bits,
        num_channels=cfg.num_channels,
        num_res_blocks=cfg.num_res_blocks,
    ).to(DEVICE)

    pos_weight = compute_pos_weight(train_recs).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(),
                                  lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg.num_epochs)

    best_val_exact = -1.0
    best_path = os.path.join(output_dir, "best_model.pt")

    for epoch in range(1, cfg.num_epochs + 1):
        tr = run_epoch(model, dl_train, criterion, optimizer)
        va = run_epoch(model, dl_val,   criterion)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{label}] Epoch {epoch:03d}/{cfg.num_epochs} | "
                  f"Train Exact {tr[2]:.4f} | Val Exact {va[2]:.4f}")

        if va[2] > best_val_exact:
            best_val_exact = va[2]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    te = run_epoch(model, dl_test, criterion)
    print(f"\n[{label}] TEST | BitAcc={te[1]:.4f}  Exact={te[2]:.4f}")
    return {"augment": use_augment, "bit_acc": te[1], "exact_match": te[2]}


def main():
    set_seed(42)
    print(f"Device: {DEVICE}")

    results = []
    for use_aug in [True, False]:
        cfg = Config()   # 每次新建，避免状态污染
        r = run_experiment(use_aug, cfg)
        results.append(r)

    print(f"\n{'='*55}")
    print("AUGMENTATION ABLATION SUMMARY")
    print(f"{'='*55}")
    print(f"{'Augment':>12} {'BitAcc':>10} {'ExactMatch':>12}")
    print("-" * 38)
    for r in results:
        aug_label = "Yes" if r["augment"] else "No"
        print(f"{aug_label:>12} {r['bit_acc']:>10.4f} {r['exact_match']:>12.4f}")


if __name__ == "__main__":
    main()
