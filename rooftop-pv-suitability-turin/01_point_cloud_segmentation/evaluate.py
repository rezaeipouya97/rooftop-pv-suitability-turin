#!/usr/bin/env python3
"""
evaluate.py - Generate confusion matrix + full metrics from predicted LAS.

Produces thesis-ready output:
  - Confusion matrix (absolute counts + percentages)
  - Per-class: Precision, Recall, F1, IoU
  - Overall Accuracy, mIoU

Usage (on a tile with ground truth labels):
    python evaluate.py \
        --input /path/to/predicted.las \
        --label_field label \
        --pred_field predicted_label

Or run inference first, then evaluate:
    python evaluate.py \
        --input /path/to/tile_with_gt.las \
        --checkpoint ~/ts_runs/exp1/best_model.pth \
        --dtm /path/to/DTM_1m.tif
"""

import argparse
import numpy as np
import laspy


CLASS_NAMES = ["ground", "building", "tree"]
NUM_CLASSES = 3


def confusion_matrix(gt, pred, num_classes):
    """Build NxN confusion matrix. Rows = ground truth, Cols = predicted."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = ((gt == i) & (pred == j)).sum()
    return cm


def compute_metrics(cm):
    """From confusion matrix, compute per-class and overall metrics."""
    num_classes = cm.shape[0]
    metrics = {}

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        metrics[c] = {
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Precision": precision, "Recall": recall, "F1": f1, "IoU": iou,
        }

    total = cm.sum()
    correct = np.trace(cm)
    overall_acc = correct / total if total > 0 else 0.0
    miou = np.mean([metrics[c]["IoU"] for c in range(num_classes)])

    return metrics, overall_acc, miou


def print_results(cm, metrics, overall_acc, miou):
    """Print thesis-ready tables."""
    n = NUM_CLASSES

    # --- Confusion Matrix (counts) ---
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (counts)")
    print("Rows = Ground Truth, Columns = Predicted")
    print("=" * 70)
    header = f"{'':>12s}" + "".join(f"{CLASS_NAMES[j]:>14s}" for j in range(n)) + f"{'Total':>14s}"
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{CLASS_NAMES[i]:>12s}"
        for j in range(n):
            row += f"{cm[i, j]:>14,}"
        row += f"{cm[i, :].sum():>14,}"
        print(row)
    totals = f"{'Total':>12s}"
    for j in range(n):
        totals += f"{cm[:, j].sum():>14,}"
    totals += f"{cm.sum():>14,}"
    print("-" * len(header))
    print(totals)

    # --- Confusion Matrix (percentages per row) ---
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (% of ground truth class)")
    print("=" * 70)
    header = f"{'':>12s}" + "".join(f"{CLASS_NAMES[j]:>14s}" for j in range(n))
    print(header)
    print("-" * len(header))
    for i in range(n):
        row_sum = cm[i, :].sum()
        row = f"{CLASS_NAMES[i]:>12s}"
        for j in range(n):
            pct = 100.0 * cm[i, j] / row_sum if row_sum > 0 else 0
            row += f"{pct:>13.1f}%"
        print(row)

    # --- Per-class metrics ---
    print("\n" + "=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)
    header = f"{'Class':>12s}{'Precision':>12s}{'Recall':>12s}{'F1':>12s}{'IoU':>12s}{'Support':>14s}"
    print(header)
    print("-" * len(header))
    for c in range(n):
        m = metrics[c]
        support = cm[c, :].sum()
        print(f"{CLASS_NAMES[c]:>12s}"
              f"{m['Precision']:>12.4f}"
              f"{m['Recall']:>12.4f}"
              f"{m['F1']:>12.4f}"
              f"{m['IoU']:>12.4f}"
              f"{support:>14,}")

    print("-" * len(header))
    print(f"\n  Overall Accuracy: {overall_acc:.4f}")
    print(f"  Mean IoU (mIoU):  {miou:.4f}")
    print(f"  Total points:     {cm.sum():,}")
    print("=" * 70)


def main():
    pa = argparse.ArgumentParser(description="Evaluate predictions vs ground truth")
    pa.add_argument("--input", required=True, help="LAS file with both GT and predictions")
    pa.add_argument("--label_field", default="label", help="Ground truth field name")
    pa.add_argument("--pred_field", default="predicted_label", help="Prediction field name")

    # Optional: run inference on-the-fly if no predictions exist yet
    pa.add_argument("--checkpoint", default=None, help="If set, run inference first")
    pa.add_argument("--dtm", default=None, help="DTM for HAG (needed if --checkpoint)")
    pa.add_argument("--voxel_size", type=float, default=0.20)
    pa.add_argument("--crop_size", type=float, default=20.0)
    pa.add_argument("--stride", type=float, default=10.0)
    pa.add_argument("--max_points", type=int, default=800_000)
    args = pa.parse_args()

    print(f"Loading {args.input}")
    las = laspy.read(args.input)

    # Get ground truth
    gt = np.asarray(getattr(las, args.label_field), dtype=np.int64)
    print(f"  {len(gt):,} points")
    print(f"  GT classes: {np.unique(gt).tolist()}")

    # Get predictions
    if args.checkpoint is not None:
        # Run inference on this tile
        print(f"\nRunning inference with checkpoint: {args.checkpoint}")
        import torch
        import rasterio
        from torchsparse import SparseTensor
        from torchsparse.utils.quantize import sparse_quantize
        from model import TorchSparseUNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TorchSparseUNet(in_channels=10, num_classes=3).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"  Model from epoch {ckpt['epoch']}, mIoU={ckpt['miou']:.4f}")

        # Load DTM
        assert args.dtm is not None, "--dtm required when using --checkpoint"
        with rasterio.open(args.dtm) as src:
            dtm_data = src.read(1)
            dtm_transform = src.transform
            dtm_nodata = src.nodata

        # Extract features
        xyz = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)
        r = np.asarray(las.red, dtype=np.float32)
        g = np.asarray(las.green, dtype=np.float32)
        b = np.asarray(las.blue, dtype=np.float32)
        mx = max(r.max(), g.max(), b.max(), 1.0)
        sc = 65535.0 if mx > 255 else 255.0
        rgb = np.clip(np.stack([r, g, b], axis=-1) / sc, 0, 1)

        normals = np.zeros((len(xyz), 3), dtype=np.float32)
        for nx_n, ny_n, nz_n in [('NormalX','NormalY','NormalZ'),('nx','ny','nz')]:
            try:
                normals = np.stack([
                    np.asarray(las[nx_n], dtype=np.float32),
                    np.asarray(las[ny_n], dtype=np.float32),
                    np.asarray(las[nz_n], dtype=np.float32),
                ], axis=-1)
                break
            except Exception:
                continue

        # HAG
        rows, cols = rasterio.transform.rowcol(dtm_transform, xyz[:,0], xyz[:,1])
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        h, w = dtm_data.shape
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        dtm_z = np.full(len(xyz), np.nan, dtype=np.float64)
        dtm_z[valid] = dtm_data[rows[valid], cols[valid]]
        if dtm_nodata is not None:
            dtm_z[dtm_z == dtm_nodata] = np.nan
        hag = np.nan_to_num(xyz[:, 2] - dtm_z, nan=0.0).astype(np.float32)

        # Sliding window inference
        N = len(xyz)
        vote_counts = np.zeros((N, 3), dtype=np.int32)
        half = args.crop_size / 2.0
        xy_min = xyz[:, :2].min(axis=0)
        xy_max = xyz[:, :2].max(axis=0)

        xs = np.arange(xy_min[0] + half, xy_max[0] - half + args.stride, args.stride)
        ys = np.arange(xy_min[1] + half, xy_max[1] - half + args.stride, args.stride)
        if len(xs) == 0: xs = np.array([(xy_min[0]+xy_max[0])/2])
        if len(ys) == 0: ys = np.array([(xy_min[1]+xy_max[1])/2])

        print(f"  {len(xs)*len(ys)} crops")

        with torch.no_grad():
            for cx in xs:
                for cy in ys:
                    mask = (
                        (xyz[:,0] >= cx-half) & (xyz[:,0] < cx+half) &
                        (xyz[:,1] >= cy-half) & (xyz[:,1] < cy+half)
                    )
                    idx = np.where(mask)[0]
                    if len(idx) < 100: continue
                    if len(idx) > args.max_points:
                        idx = np.random.choice(idx, args.max_points, replace=False)

                    sub_xyz = xyz[idx].astype(np.float32)
                    xyz_rel = (sub_xyz - sub_xyz.mean(axis=0)).astype(np.float32)
                    coords = np.floor(xyz_rel / args.voxel_size).astype(np.int32)
                    feats = np.concatenate([xyz_rel, rgb[idx], normals[idx],
                                            hag[idx, np.newaxis]], axis=-1).astype(np.float32)

                    coords_q, u_idx, inv_idx = sparse_quantize(
                        coords, voxel_size=1, return_index=True, return_inverse=True)
                    if isinstance(u_idx, torch.Tensor): u_idx = u_idx.numpy()
                    if isinstance(inv_idx, torch.Tensor): inv_idx = inv_idx.numpy()
                    u_idx = np.asarray(u_idx, dtype=np.int64)
                    inv_idx = np.asarray(inv_idx, dtype=np.int64)

                    c = torch.from_numpy(coords[u_idx]).int()
                    f = torch.from_numpy(feats[u_idx]).float()
                    # TorchSparse 2.0.0b: coords are [x, y, z, batch] — batch LAST
                    batch_col = torch.zeros(c.shape[0], 1, dtype=torch.int32)
                    coords_4d = torch.cat([c, batch_col], dim=1)

                    st = SparseTensor(coords=coords_4d.to(device), feats=f.to(device))
                    out = model(st)
                    vp = out.F.argmax(dim=1).cpu().numpy()
                    pp = vp[inv_idx]

                    for ci in range(3):
                        cm = pp == ci
                        if cm.any():
                            np.add.at(vote_counts[:, ci], idx[cm], 1)

        covered = vote_counts.sum(axis=1) > 0
        pred = np.zeros(N, dtype=np.int64)
        pred[covered] = vote_counts[covered].argmax(axis=1)
        print(f"  Covered: {covered.sum():,} / {N:,}")

    else:
        pred = np.asarray(getattr(las, args.pred_field), dtype=np.int64)
        print(f"  Pred classes: {np.unique(pred).tolist()}")

    # Filter to valid labels only (0, 1, 2)
    valid_mask = (gt >= 0) & (gt < NUM_CLASSES) & (pred >= 0) & (pred < NUM_CLASSES)
    gt = gt[valid_mask]
    pred = pred[valid_mask]
    print(f"  Valid points for evaluation: {len(gt):,}")

    # Compute
    cm = confusion_matrix(gt, pred, NUM_CLASSES)
    metrics, overall_acc, miou = compute_metrics(cm)
    print_results(cm, metrics, overall_acc, miou)


if __name__ == "__main__":
    main()
