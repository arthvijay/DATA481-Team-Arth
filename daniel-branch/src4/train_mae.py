import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import build_datasets, compute_pos_weight
from model_mae import IQMaskedAutoencoder
from lsig_decode import compute_tx_time_mae, compute_length_mae

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def bit_metrics(logits, targets):
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == targets).float().mean().item(), \
           (preds == targets).all(dim=1).float().mean().item(), preds

def pretrain(model, dl_train, n_epochs=50, lr=1e-3):
    print("\n=== Stage A: MAE Pre-training ===")
    os.makedirs("outputs_mae", exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        model.train()
        total = 0
        for batch in dl_train:
            x = batch["x"].to(DEVICE)
            opt.zero_grad()
            loss = model.forward_pretrain(x)
            loss.backward()
            opt.step()
            total += loss.item()
        sch.step()
        avg = total / len(dl_train)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Pretrain Epoch {epoch:03d}/{n_epochs} | Recon Loss {avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), "outputs_mae/pretrained_encoder.pt")

    print(f"  Pre-training done. Best recon loss: {best_loss:.4f}")

def finetune(model, dl_train, dl_val, dl_test, train_recs,
             n_epochs=100, lr=1e-3, freeze_encoder=True, tag="frozen"):
    print(f"\n=== Stage B: Fine-tuning (freeze={freeze_encoder}) ===")
    os.makedirs("outputs_mae", exist_ok=True)

    if freeze_encoder:
        for name, p in model.named_parameters():
            if 'lsig_head' not in name:
                p.requires_grad = False
    else:
        for p in model.parameters():
            p.requires_grad = True

    pos_weight = compute_pos_weight(train_recs).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_exact = -1
    best_path = f"outputs_mae/best_{tag}.pt"

    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch in dl_train:
            x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
            opt.zero_grad()
            loss = criterion(model.forward_finetune(x), y)
            loss.backward()
            opt.step()
        sch.step()

        model.eval()
        va_ba = va_ex = va_n = 0
        with torch.no_grad():
            for batch in dl_val:
                x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
                logits = model.forward_finetune(x)
                ba, ex, _ = bit_metrics(logits, y)
                va_ba += ba; va_ex += ex; va_n += 1
        va_ba /= va_n; va_ex /= va_n

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Finetune [{tag}] Epoch {epoch:03d}/{n_epochs} | "
                  f"Val BitAcc {va_ba:.4f} Exact {va_ex:.4f}")

        if va_ex > best_exact:
            best_exact = va_ex
            torch.save(model.state_dict(), best_path)
            print(f"    ✓ Best saved (val exact={va_ex:.4f})")

    # Test
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch in dl_test:
            x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
            all_preds.append(torch.sigmoid(model.forward_finetune(x)).cpu())
            all_gts.append(y.cpu())
    all_preds = torch.cat(all_preds)
    all_gts   = torch.cat(all_gts)

    hard = (all_preds >= 0.5).float()
    te_ex = (hard == all_gts).all(dim=1).float().mean().item()
    te_ba = (hard == all_gts).float().mean().item()
    tx_mae = compute_tx_time_mae(all_preds.numpy(), all_gts.numpy())
    l_mae  = compute_length_mae(all_preds.numpy(),  all_gts.numpy())

    print(f"\n  [{tag}] TEST | BitAcc={te_ba:.4f} ExactMatch={te_ex:.4f} "
          f"TxMAE={tx_mae['mae_ms']:.4f}ms LenMAE={l_mae['mae_bytes']:.1f}B")
    return te_ex

def main():
    set_seed(42)
    cfg = Config()
    os.makedirs("outputs_mae", exist_ok=True)

    print(f"Device: {DEVICE}")

    ds_train, ds_val, ds_test, train_recs = build_datasets(cfg, no_fft=True)
    dl_train = DataLoader(ds_train, cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True)

    model = IQMaskedAutoencoder(
        seq_len=768, patch_size=16, embed_dim=128,
        encoder_depth=4, decoder_depth=2, num_heads=4,
        mask_ratio=0.5, num_bits=24,
    ).to(DEVICE)
    print(f"MAE parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Stage A: pre-train
    pretrain(model, dl_train, n_epochs=50, lr=1e-3)

    # Stage B-1: frozen encoder
    model.load_state_dict(torch.load("outputs_mae/pretrained_encoder.pt",
                                     map_location=DEVICE))
    ex_frozen = finetune(model, dl_train, dl_val, dl_test, train_recs,
                         n_epochs=100, freeze_encoder=True, tag="frozen")

    # Stage B-2: full fine-tune
    model.load_state_dict(torch.load("outputs_mae/pretrained_encoder.pt",
                                     map_location=DEVICE))
    ex_full = finetune(model, dl_train, dl_val, dl_test, train_recs,
                       n_epochs=100, freeze_encoder=False, tag="full")

    print(f"\n{'='*50}")
    print(f"FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"MAE frozen encoder finetune : ExactMatch={ex_frozen:.4f}")
    print(f"MAE full finetune           : ExactMatch={ex_full:.4f}")
    print(f"DeepMon baseline            : ExactMatch=~0.657")

if __name__ == "__main__":
    main()
