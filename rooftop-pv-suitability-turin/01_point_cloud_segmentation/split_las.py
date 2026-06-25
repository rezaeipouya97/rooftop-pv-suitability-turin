#!/usr/bin/env python3
"""
split_las.py - Spatially split a LAS file into two halves (train/val).

Splits along the longer axis (X or Y) at the midpoint.

Usage:
    python split_las.py \
        --input /path/to/park.las \
        --out_train /path/to/park_train.las \
        --out_val /path/to/park_val.las
"""

import argparse
import numpy as np
import laspy


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--input", required=True)
    pa.add_argument("--out_train", required=True)
    pa.add_argument("--out_val", required=True)
    pa.add_argument("--ratio", type=float, default=0.5,
                    help="Fraction for training (0.5 = 50/50 split)")
    args = pa.parse_args()

    print(f"Reading {args.input}")
    las = laspy.read(args.input)
    N = len(las.x)
    print(f"  {N:,} points")

    xyz = np.stack([las.x, las.y, las.z], axis=-1)
    x_range = xyz[:, 0].max() - xyz[:, 0].min()
    y_range = xyz[:, 1].max() - xyz[:, 1].min()

    # Split along the longer axis
    if x_range >= y_range:
        axis = 0
        axis_name = "X"
    else:
        axis = 1
        axis_name = "Y"

    vals = xyz[:, axis]
    split_val = np.quantile(vals, args.ratio)

    train_mask = vals <= split_val
    val_mask   = vals > split_val

    print(f"  Splitting along {axis_name} axis at {split_val:.1f}")
    print(f"  Train: {train_mask.sum():,} points ({100*train_mask.sum()/N:.1f}%)")
    print(f"  Val:   {val_mask.sum():,} points ({100*val_mask.sum()/N:.1f}%)")

    # Check class distribution in both halves
    try:
        lbl = np.asarray(las.label, dtype=np.int64)
        names = {0: "ground", 1: "building", 2: "tree"}
        print(f"\n  Class distribution:")
        print(f"  {'':>12s}  {'Train':>12s}  {'Val':>12s}")
        for c in sorted(np.unique(lbl)):
            n_tr = (lbl[train_mask] == c).sum()
            n_va = (lbl[val_mask] == c).sum()
            name = names.get(c, str(c))
            print(f"  {name:>12s}  {n_tr:>12,}  {n_va:>12,}")
    except AttributeError:
        pass

    # Write
    print(f"\nWriting {args.out_train}")
    las_train = las[train_mask]
    las_train.write(args.out_train)

    print(f"Writing {args.out_val}")
    las_val = las[val_mask]
    las_val.write(args.out_val)

    print("Done.")


if __name__ == "__main__":
    main()
