"""
inference_big_v2.py - Spatially-tiled inference for large LAS using TorchSparse.

Key design:
  - Reads entire LAS ONCE into memmaps (pass 1)
  - Tiles slice from memmaps (pass 2) - no re-reading the file
  - HAG as 10th feature channel
  - Vectorized vote accumulation
"""

import argparse, time
from pathlib import Path

import numpy as np
import laspy
import rasterio
import torch

import torchsparse
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

from model import TorchSparseUNet


CLASS_NAMES = ["ground", "building", "tree"]


def read_normals(las_chunk):
    for nx_name, ny_name, nz_name in [
        ('NormalX', 'NormalY', 'NormalZ'),
        ('normal x', 'normal y', 'normal z'),
        ('normalx', 'normaly', 'normalz'),
        ('nx', 'ny', 'nz'),
    ]:
        try:
            return np.stack([
                np.asarray(las_chunk[nx_name], dtype=np.float32),
                np.asarray(las_chunk[ny_name], dtype=np.float32),
                np.asarray(las_chunk[nz_name], dtype=np.float32),
            ], axis=-1)
        except Exception:
            continue
    return None


def read_rgb(las_chunk):
    r = np.asarray(las_chunk.red,   dtype=np.float32)
    g = np.asarray(las_chunk.green, dtype=np.float32)
    b = np.asarray(las_chunk.blue,  dtype=np.float32)
    max_val = max(r.max(), g.max(), b.max(), 1.0)
    scale = 65535.0 if max_val > 255.0 else 255.0
    rgb = np.stack([r, g, b], axis=-1) / scale
    return np.clip(rgb, 0.0, 1.0)


def sample_dtm(dtm_data, dtm_transform, dtm_nodata, x, y, z):
    rows, cols = rasterio.transform.rowcol(dtm_transform, x, y)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    h, w = dtm_data.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    dtm_z = np.full(len(x), np.nan, dtype=np.float64)
    dtm_z[valid] = dtm_data[rows[valid], cols[valid]]
    if dtm_nodata is not None:
        dtm_z[dtm_z == dtm_nodata] = np.nan
    hag = z - dtm_z
    return np.nan_to_num(hag, nan=0.0).astype(np.float32)


@torch.no_grad()
def predict_crop(model, xyz, rgb, normals, hag, voxel_size, device):
    """Run inference on a single crop. 10-channel TorchSparse input."""
    xyz_rel = (xyz - xyz.mean(axis=0)).astype(np.float32)
    coords = np.floor(xyz_rel / voxel_size).astype(np.int32)
    feats  = np.concatenate([xyz_rel, rgb, normals, hag[:, np.newaxis]], axis=-1).astype(np.float32)

    # TorchSparse quantize
    coords_q, unique_idx, inverse_idx = sparse_quantize(
        coords, voxel_size=1,
        return_index=True, return_inverse=True,
    )

    if isinstance(unique_idx, torch.Tensor): unique_idx = unique_idx.numpy()
    if isinstance(inverse_idx, torch.Tensor): inverse_idx = inverse_idx.numpy()
    unique_idx  = np.asarray(unique_idx,  dtype=np.int64)
    inverse_idx = np.asarray(inverse_idx, dtype=np.int64)

    # Build SparseTensor with batch dim (single sample, batch=0)
    c = torch.from_numpy(coords[unique_idx]).int()
    f = torch.from_numpy(feats[unique_idx]).float()

    # TorchSparse 2.0.0b: coords are [x, y, z, batch] — batch LAST
    batch_col = torch.zeros(c.shape[0], 1, dtype=torch.int32)
    coords_4d = torch.cat([c, batch_col], dim=1)

    st = SparseTensor(
        coords=coords_4d.to(device),
        feats=f.to(device),
    )
    out = model(st)

    voxel_preds = out.F.argmax(dim=1).cpu().numpy()
    return voxel_preds[inverse_idx]


def main():
    pa = argparse.ArgumentParser(description="TorchSparse tiled inference for large LAS")
    pa.add_argument("--checkpoint", required=True)
    pa.add_argument("--input",     required=True)
    pa.add_argument("--output",    required=True)
    pa.add_argument("--dtm",       required=True, help="DTM GeoTIFF for HAG")
    pa.add_argument("--voxel_size", type=float, default=0.20)
    pa.add_argument("--crop_size",  type=float, default=20.0)
    pa.add_argument("--stride",     type=float, default=10.0)
    pa.add_argument("--max_points", type=int,   default=800_000)
    pa.add_argument("--tile_size",  type=float, default=100.0)
    pa.add_argument("--chunk_size", type=int,   default=10_000_000)
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load model ----
    model = TorchSparseUNet(in_channels=10, num_classes=3).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}, mIoU={ckpt['miou']:.4f}")

    # ---- Load DTM ----
    print(f"Loading DTM: {args.dtm}")
    with rasterio.open(args.dtm) as src:
        dtm_data = src.read(1)
        dtm_transform = src.transform
        dtm_nodata = src.nodata

    # ---- Point count ----
    with laspy.open(args.input) as reader:
        total_points = reader.header.point_count
    print(f"Input: {args.input}  ({total_points:,} points)")

    out_dir = Path(args.output).expanduser().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # PASS 1: Read ALL data into memmaps (ONE file read)
    # ================================================================
    print(f"\n--- Pass 1: Reading all data into memmaps ---")
    t0 = time.time()

    xy_file  = out_dir / "_xy_temp.npy"
    z_file   = out_dir / "_z_temp.npy"
    rgb_file = out_dir / "_rgb_temp.npy"
    nrm_file = out_dir / "_nrm_temp.npy"

    xy_all  = np.memmap(str(xy_file),  dtype=np.float64, mode='w+', shape=(total_points, 2))
    z_all   = np.memmap(str(z_file),   dtype=np.float64, mode='w+', shape=(total_points,))
    rgb_all = np.memmap(str(rgb_file), dtype=np.float32, mode='w+', shape=(total_points, 3))
    nrm_all = np.memmap(str(nrm_file), dtype=np.float32, mode='w+', shape=(total_points, 3))

    offset = 0
    with laspy.open(args.input) as reader:
        for chunk in reader.chunk_iterator(args.chunk_size):
            n = len(chunk.x)
            xy_all[offset:offset+n, 0] = chunk.x
            xy_all[offset:offset+n, 1] = chunk.y
            z_all[offset:offset+n]     = chunk.z
            rgb_all[offset:offset+n]   = read_rgb(chunk)
            nrm_c = read_normals(chunk)
            if nrm_c is None:
                nrm_all[offset:offset+n] = 0.0
            else:
                nrm_all[offset:offset+n] = nrm_c
            offset += n
            print(f"  Read {offset:,} / {total_points:,}", end='\r')

    xy_min = np.array([xy_all[:, 0].min(), xy_all[:, 1].min()])
    xy_max = np.array([xy_all[:, 0].max(), xy_all[:, 1].max()])
    print(f"\n  XY: [{xy_min[0]:.1f},{xy_min[1]:.1f}] to [{xy_max[0]:.1f},{xy_max[1]:.1f}]")
    print(f"  Pass 1: {time.time()-t0:.0f}s")

    # ---- Tile grid ----
    tile_size = args.tile_size
    overlap = args.crop_size
    n_tiles_x = int(np.ceil((xy_max[0] - xy_min[0]) / tile_size))
    n_tiles_y = int(np.ceil((xy_max[1] - xy_min[1]) / tile_size))
    total_tiles = n_tiles_x * n_tiles_y
    print(f"  Tiles: {n_tiles_x} x {n_tiles_y} = {total_tiles}")

    # Votes
    vote_file = out_dir / "_votes_temp.npy"
    votes = np.memmap(str(vote_file), dtype=np.int16, mode='w+', shape=(total_points, 3))
    votes[:] = 0

    # ================================================================
    # PASS 2: Tile inference (slicing from memmaps, no re-read)
    # ================================================================
    print(f"\n--- Pass 2: Tile inference ---")
    t1 = time.time()
    total_crops = 0

    for ti in range(n_tiles_x):
        for tj in range(n_tiles_y):
            tile_num = ti * n_tiles_y + tj + 1

            tx_min = xy_min[0] + ti * tile_size - overlap
            tx_max = xy_min[0] + (ti + 1) * tile_size + overlap
            ty_min = xy_min[1] + tj * tile_size - overlap
            ty_max = xy_min[1] + (tj + 1) * tile_size + overlap

            mask = (
                (xy_all[:, 0] >= tx_min) & (xy_all[:, 0] < tx_max) &
                (xy_all[:, 1] >= ty_min) & (xy_all[:, 1] < ty_max)
            )
            tile_indices = np.where(mask)[0]
            if len(tile_indices) < 100:
                continue

            tile_xy  = np.array(xy_all[tile_indices])
            tile_z   = np.array(z_all[tile_indices])
            tile_xyz = np.column_stack([tile_xy, tile_z]).astype(np.float32)
            tile_rgb = np.array(rgb_all[tile_indices])
            tile_nrm = np.array(nrm_all[tile_indices])
            tile_hag = sample_dtm(dtm_data, dtm_transform, dtm_nodata,
                                  tile_xy[:, 0], tile_xy[:, 1], tile_z)

            print(f"  Tile {tile_num}/{total_tiles}: {len(tile_indices):,} pts", end="")

            half = args.crop_size / 2.0
            core_xmin = xy_min[0] + ti * tile_size
            core_xmax = xy_min[0] + (ti + 1) * tile_size
            core_ymin = xy_min[1] + tj * tile_size
            core_ymax = xy_min[1] + (tj + 1) * tile_size

            xs = np.arange(core_xmin + half, core_xmax - half + args.stride, args.stride)
            ys = np.arange(core_ymin + half, core_ymax - half + args.stride, args.stride)
            if len(xs) == 0: xs = np.array([(core_xmin + core_xmax) / 2])
            if len(ys) == 0: ys = np.array([(core_ymin + core_ymax) / 2])

            tile_crops = 0
            for cx in xs:
                for cy in ys:
                    cmask = (
                        (tile_xyz[:, 0] >= cx - half) & (tile_xyz[:, 0] < cx + half) &
                        (tile_xyz[:, 1] >= cy - half) & (tile_xyz[:, 1] < cy + half)
                    )
                    crop_local = np.where(cmask)[0]
                    if len(crop_local) < 100:
                        continue
                    if len(crop_local) > args.max_points:
                        crop_local = np.random.choice(crop_local, args.max_points, replace=False)

                    preds = predict_crop(
                        model, tile_xyz[crop_local], tile_rgb[crop_local],
                        tile_nrm[crop_local], tile_hag[crop_local],
                        args.voxel_size, device,
                    )

                    g_idx = tile_indices[crop_local]
                    for c in range(3):
                        c_mask = preds == c
                        if c_mask.any():
                            np.add.at(votes[:, c], g_idx[c_mask], 1)

                    tile_crops += 1

            total_crops += tile_crops
            elapsed = time.time() - t1
            rate = tile_num / max(elapsed, 1)
            eta = (total_tiles - tile_num) / max(rate, 0.001)
            print(f"  | {tile_crops} crops | ETA: {eta/60:.0f}min")

    # ================================================================
    # Majority vote
    # ================================================================
    print(f"\n--- Final predictions ---")
    final_preds = np.zeros(total_points, dtype=np.uint8)
    batch = 10_000_000
    covered = 0
    for s in range(0, total_points, batch):
        e = min(s + batch, total_points)
        v = np.array(votes[s:e])
        has_votes = v.sum(axis=1) > 0
        covered += has_votes.sum()
        final_preds[s:e][has_votes] = v[has_votes].argmax(axis=1).astype(np.uint8)

    uncovered = total_points - covered
    print(f"  Covered: {covered:,}  Uncovered: {uncovered:,} ({100*uncovered/total_points:.2f}%)")

    # ================================================================
    # Write output
    # ================================================================
    print(f"\n--- Writing output ---")
    out_path = Path(args.output).expanduser()

    with laspy.open(args.input) as reader:
        header = reader.header
        new_header = laspy.LasHeader(
            point_format=header.point_format, version=header.version)
        new_header.offsets = header.offsets
        new_header.scales  = header.scales
        new_header.vlrs = header.vlrs
        new_header.add_extra_dim(laspy.ExtraBytesParams(
            name="predicted_label", type=np.uint8,
            description="TorchSparse prediction"))

        with laspy.open(str(out_path), mode='w', header=new_header) as writer:
            w_offset = 0
            for chunk in reader.chunk_iterator(args.chunk_size):
                n = len(chunk.x)
                new_points = laspy.ScaleAwarePointRecord.zeros(n, header=new_header)
                for dim in chunk.point_format.dimension_names:
                    if dim in new_header.point_format.dimension_names:
                        new_points[dim] = chunk[dim]
                new_points.predicted_label = final_preds[w_offset:w_offset+n]
                writer.write_points(new_points)
                w_offset += n
                print(f"  Written {w_offset:,} / {total_points:,}", end='\r')

    # Cleanup
    del votes, xy_all, z_all, rgb_all, nrm_all
    for f in [vote_file, xy_file, z_file, rgb_file, nrm_file]:
        if f.exists():
            f.unlink()

    size_gb = out_path.stat().st_size / 1e9
    total_time = (time.time() - t0) / 60
    print(f"\n\n  Done: {out_path} ({size_gb:.2f} GB)")
    print(f"  Total crops: {total_crops:,}  Time: {total_time:.1f} min")
    for c in range(3):
        n = (final_preds == c).sum()
        print(f"  {CLASS_NAMES[c]:10s}: {n:>12,} ({100*n/total_points:.1f}%)")


if __name__ == "__main__":
    main()
