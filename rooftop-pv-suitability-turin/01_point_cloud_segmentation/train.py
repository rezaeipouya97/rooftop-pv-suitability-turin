#!/usr/bin/env python3
"""
train.py - Training loop for TorchSparse UNet on LAS point clouds.

Features:
  - TorchSparse backend (conv_mode=2 for segmentation)
  - Mixed precision (AMP) for ~2x speedup
  - Gradient clipping for sparse conv stability
  - HAG feature via --dtm
  - 10 input channels: xyz_rel + rgb + normals + HAG

Usage:
    python train.py \
        --train_files /path/a.las /path/b.las \
        --val_files   /path/c.las \
        --dtm         /path/DTM_1m.tif \
        --out_dir     ~/ts_runs/exp1 \
        --reject_minority 0.3
"""

import os, sys, time, argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

import torchsparse

from dataset import LASCropDataset, torchsparse_collate_fn, compute_class_weights
from model import TorchSparseUNet


CLASS_NAMES = ["ground", "building", "tree"]


def compute_iou(preds, targets, num_classes=3):
    ious = []
    for c in range(num_classes):
        inter = ((preds == c) & (targets == c)).sum()
        union = ((preds == c) | (targets == c)).sum()
        ious.append(inter / union if union > 0 else float("nan"))
    return ious


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=3, use_amp=True):
    model.eval()
    total_loss, n = 0.0, 0
    all_p, all_t = [], []
    for batched_st, blabels in loader:
        with autocast(enabled=use_amp):
            batched_st = batched_st.to(device)
            out = model(batched_st)
            loss = criterion(out.F, blabels.to(device))
        total_loss += loss.item(); n += 1
        all_p.append(out.F.argmax(1).cpu().numpy())
        all_t.append(blabels.numpy())
    all_p = np.concatenate(all_p)
    all_t = np.concatenate(all_t)
    ious  = compute_iou(all_p, all_t, num_classes)
    return total_loss / max(n, 1), ious, float(np.nanmean(ious))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--train_files", nargs="+", required=True)
    pa.add_argument("--val_files",   nargs="+", required=True)
    pa.add_argument("--dtm",         required=True, help="DTM GeoTIFF for HAG")
    pa.add_argument("--out_dir",     default="~/ts_runs/exp1")
    pa.add_argument("--voxel_size",  type=float, default=0.20)
    pa.add_argument("--crop_size",   type=float, default=20.0)
    pa.add_argument("--batch_size",  type=int,   default=4)
    pa.add_argument("--lr",          type=float, default=1e-3)
    pa.add_argument("--epochs",      type=int,   default=80)
    pa.add_argument("--samples_per_epoch", type=int, default=500)
    pa.add_argument("--val_samples", type=int,   default=100)
    pa.add_argument("--max_points",  type=int,   default=800_000)
    pa.add_argument("--num_workers", type=int,   default=4)
    pa.add_argument("--num_classes", type=int,   default=3)
    pa.add_argument("--seed",        type=int,   default=42)
    pa.add_argument("--cache_size",  type=int,   default=2)
    pa.add_argument("--no_amp",      action="store_true")
    pa.add_argument("--grad_clip",   type=float, default=10.0)

    pa.add_argument("--file_sampling", choices=["uniform","proportional","manual_weights"],
                    default="uniform")
    pa.add_argument("--file_weights", nargs="*", type=float, default=None)

    pa.add_argument("--reject_minority", type=float, default=0.0)
    pa.add_argument("--reject_class",    type=int,   default=2)
    pa.add_argument("--reject_min_frac", type=float, default=0.05)
    args = pa.parse_args()

    use_amp = (not args.no_amp) and torch.cuda.is_available()

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}  AMP: {use_amp}")
    print(f"TorchSparse version: {torchsparse.__version__}\n")

    # ---- Class weights ----
    cw = compute_class_weights(args.train_files, args.num_classes)

    # ---- Datasets ----
    train_ds = LASCropDataset(
        file_paths=args.train_files, dtm_path=args.dtm,
        voxel_size=args.voxel_size, crop_size=args.crop_size,
        samples_per_epoch=args.samples_per_epoch, augment=True,
        max_points=args.max_points, cache_size=args.cache_size,
        file_sampling=args.file_sampling, file_weights=args.file_weights,
        reject_minority=args.reject_minority, reject_class=args.reject_class,
        reject_min_frac=args.reject_min_frac,
    )
    val_ds = LASCropDataset(
        file_paths=args.val_files, dtm_path=args.dtm,
        voxel_size=args.voxel_size, crop_size=args.crop_size,
        samples_per_epoch=args.val_samples, augment=False,
        max_points=args.max_points, cache_size=args.cache_size,
        file_sampling="uniform",
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=torchsparse_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=torchsparse_collate_fn,
        pin_memory=True,
    )

    # ---- Model (10 channels: xyz_rel + rgb + normals + HAG) ----
    model = TorchSparseUNet(in_channels=10, num_classes=args.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=cw.to(device))
    scaler = GradScaler(enabled=use_amp)

    # ---- Training ----
    best_miou = 0.0
    log = []
    ckpt_path = out / "best_model.pth"
    log_path  = out / "train_log.json"

    with open(out / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        t0 = time.time()

        for batched_st, blabels in train_loader:
            with autocast(enabled=use_amp):
                batched_st = batched_st.to(device)
                out_t = model(batched_st)
                loss = criterion(out_t.F, blabels.to(device))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item(); nb += 1

        scheduler.step()
        tr_loss = ep_loss / max(nb, 1)
        dt = time.time() - t0

        # Validate
        v_loss, ious, miou = validate(model, val_loader, criterion, device,
                                       args.num_classes, use_amp=use_amp)

        iou_s = "  ".join(
            f"{CLASS_NAMES[i]}={ious[i]:.4f}" if not np.isnan(ious[i])
            else f"{CLASS_NAMES[i]}=N/A"
            for i in range(args.num_classes)
        )
        print(f"[{epoch:03d}/{args.epochs}] "
              f"trn={tr_loss:.4f} val={v_loss:.4f} mIoU={miou:.4f}  "
              f"{iou_s}  lr={scheduler.get_last_lr()[0]:.1e}  {dt:.0f}s")

        log.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": v_loss,
                     "miou": miou, "ious": dict(zip(CLASS_NAMES, ious))})

        if miou > best_miou:
            best_miou = miou
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(),
                         "miou": miou, "args": vars(args)}, ckpt_path)
            print(f"  -> saved best (mIoU={miou:.4f})")

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Best mIoU={best_miou:.4f}  Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
