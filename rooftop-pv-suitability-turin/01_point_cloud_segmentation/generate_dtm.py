#!/usr/bin/env python3
"""
generate_dtm.py - Generate a DTM from raw LAS using Cloth Simulation Filter.

No ML dependency. Requires: pip install cloth-simulation-filter laspy rasterio scipy numpy

Usage:
    python generate_dtm.py \
        --input   /path/to/big.las \
        --output  /path/to/DTM_1m.tif \
        --ground_las /path/to/ground.las \
        --cloth_resolution 1.0 \
        --rigidness 2 \
        --class_threshold 0.5 \
        --max_iterations 500 \
        --subsample 0.2
"""

import argparse, time
from pathlib import Path

import numpy as np
import laspy
import CSF


def main():
    pa = argparse.ArgumentParser(description="Generate DTM via CSF")
    pa.add_argument("--input",   required=True)
    pa.add_argument("--output",  required=True)
    pa.add_argument("--ground_las", default=None)
    pa.add_argument("--resolution", type=float, default=1.0)
    pa.add_argument("--cloth_resolution", type=float, default=1.0)
    pa.add_argument("--rigidness", type=int, default=2)
    pa.add_argument("--class_threshold", type=float, default=0.5)
    pa.add_argument("--max_iterations", type=int, default=500)
    pa.add_argument("--subsample", type=float, default=1.0)
    args = pa.parse_args()

    t0 = time.time()

    print(f"Reading {args.input}")
    las = laspy.read(args.input)
    xyz = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)
    N = xyz.shape[0]
    print(f"  {N:,} points")

    if args.subsample < 1.0:
        n_sub = int(N * args.subsample)
        sub_idx = np.random.choice(N, n_sub, replace=False)
        xyz_csf = xyz[sub_idx]
        print(f"  Subsampled to {n_sub:,} ({args.subsample*100:.0f}%) for CSF")
    else:
        xyz_csf = xyz
        sub_idx = None

    print(f"Running CSF (cloth_res={args.cloth_resolution}, rigidness={args.rigidness})")
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = args.cloth_resolution
    csf.params.rigidness = args.rigidness
    csf.params.class_threshold = args.class_threshold
    csf.params.interations = args.max_iterations
    csf.params.time_step = 0.65

    csf.setPointCloud(xyz_csf)
    ground_idx = CSF.VecInt()
    non_ground_idx = CSF.VecInt()
    csf.do_filtering(ground_idx, non_ground_idx)

    ground_idx = np.array(ground_idx)
    print(f"  CSF: {len(ground_idx):,} ground points ({100*len(ground_idx)/len(xyz_csf):.1f}%)")

    if sub_idx is not None:
        ground_global = sub_idx[ground_idx]
    else:
        ground_global = ground_idx

    ground_xyz = xyz[ground_global]

    if args.ground_las:
        print(f"  Saving ground points: {args.ground_las}")
        gpath = Path(args.ground_las)
        gpath.parent.mkdir(parents=True, exist_ok=True)
        header = laspy.LasHeader(point_format=0, version="1.2")
        header.offsets = np.floor(ground_xyz.min(axis=0))
        header.scales = np.array([0.001, 0.001, 0.001])
        out_las = laspy.LasData(header)
        out_las.x = ground_xyz[:, 0]
        out_las.y = ground_xyz[:, 1]
        out_las.z = ground_xyz[:, 2]
        out_las.write(str(gpath))

    print(f"Rasterising to {args.resolution}m DTM")
    res = args.resolution
    x_min, y_min = ground_xyz[:, 0].min(), ground_xyz[:, 1].min()
    x_max, y_max = ground_xyz[:, 0].max(), ground_xyz[:, 1].max()

    cols = int(np.ceil((x_max - x_min) / res)) + 1
    rows = int(np.ceil((y_max - y_min) / res)) + 1
    print(f"  Grid: {cols} x {rows}")

    col_idx = np.floor((ground_xyz[:, 0] - x_min) / res).astype(np.int64)
    row_idx = np.floor((y_max - ground_xyz[:, 1]) / res).astype(np.int64)
    col_idx = np.clip(col_idx, 0, cols - 1)
    row_idx = np.clip(row_idx, 0, rows - 1)

    dtm = np.full((rows, cols), np.nan, dtype=np.float64)
    sum_grid   = np.zeros((rows, cols), dtype=np.float64)
    count_grid = np.zeros((rows, cols), dtype=np.int64)
    np.add.at(sum_grid,   (row_idx, col_idx), ground_xyz[:, 2])
    np.add.at(count_grid, (row_idx, col_idx), 1)

    valid = count_grid > 0
    dtm[valid] = sum_grid[valid] / count_grid[valid]

    from scipy.ndimage import distance_transform_edt
    nan_mask = np.isnan(dtm)
    if nan_mask.any():
        print(f"  Filling {nan_mask.sum():,} empty cells with NN interpolation")
        indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        dtm = dtm[tuple(indices)]

    import rasterio
    from rasterio.transform import from_bounds

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)

    with rasterio.open(
        str(out_path), 'w', driver='GTiff',
        height=rows, width=cols, count=1, dtype='float32',
        crs=None, transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(dtm.astype(np.float32), 1)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  DTM: {out_path}  Shape: {rows}x{cols}  Res: {res}m")
    print(f"  Z range: [{np.nanmin(dtm):.1f}, {np.nanmax(dtm):.1f}]")


if __name__ == "__main__":
    main()
