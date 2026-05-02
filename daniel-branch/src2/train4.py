#2/4/6/8 ResBlocks Ablation
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


def run_experiment(num_blocks: int, cfg: Config,
                   dl_train, dl_val, dl_test, train_recs):

    output_dir = f"outputs_depth_{num_blocks}blocks"
    os.makedirs(output_dir, exist_ok=True)

    model = DeepMonModel(
        num_bits=cfg.num_bits,
        num_channels=cfg.num_channels,
        num_res_blocks=num_blocks,     # ← 唯一变量
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*55}")
    print(f"[Depth Ablation] num_res_blocks={num_blocks}, params={params:,}")
    print(f"{'='*55}")

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
            print(f"  Epoch {epoch:03d}/{cfg.num_epochs} | "
                  f"Train BitAcc {tr[1]:.4f} Exact {tr[2]:.4f} | "
                  f"Val BitAcc {va[1]:.4f} Exact {va[2]:.4f}")

        if va[2] > best_val_exact:
            best_val_exact = va[2]
            torch.save(model.state_dict(), best_path)

    # 测试
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    te = run_epoch(model, dl_test, criterion)

    print(f"\n[blocks={num_blocks}] TEST | "
          f"BitAcc={te[1]:.4f}  Exact={te[2]:.4f}")
    return {"blocks": num_blocks, "params": params,
            "bit_acc": te[1], "exact_match": te[2]}


def main():
    cfg = Config()
    set_seed(cfg.random_seed)

    print(f"Device: {DEVICE}")
    print("Loading data once, reusing for all depth experiments...")

    ds_train, ds_val, ds_test, train_recs = build_datasets(cfg)
    dl_train = DataLoader(ds_train, cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers)
    dl_val   = DataLoader(ds_val,   cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers)
    dl_test  = DataLoader(ds_test,  cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers)

    results = []
    for num_blocks in [2, 4, 6, 8]:   # ← 四个深度
        r = run_experiment(num_blocks, cfg, dl_train, dl_val,
                           dl_test, train_recs)
        results.append(r)

    # 汇总表
    print(f"\n{'='*55}")
    print("DEPTH ABLATION SUMMARY")
    print(f"{'='*55}")
    print(f"{'Blocks':>8} {'Params':>10} {'BitAcc':>10} {'ExactMatch':>12}")
    print("-" * 45)
    for r in results:
        print(f"{r['blocks']:>8} {r['params']:>10,} "
              f"{r['bit_acc']:>10.4f} {r['exact_match']:>12.4f}")


if __name__ == "__main__":
    main()
