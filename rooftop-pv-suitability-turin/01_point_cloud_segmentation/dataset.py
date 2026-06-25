"""
dataset.py - Lazy-loading LAS dataset for TorchSparse training.

Features per point (10 channels):
  xyz_rel(3) + rgb(3) + normals(3) + HAG(1)

Returns (SparseTensor, labels_tensor) per sample.
Use torchsparse_collate_fn for DataLoader.
"""

import os, math, random
from typing import List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import laspy
import rasterio
import torch
from torch.utils.data import Dataset

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate


# ---------------------------------------------------------------------------
# DTM sampler
# ---------------------------------------------------------------------------

class DTMSampler:
    """Load a GeoTIFF DTM once, sample HAG at any (x, y) coordinates."""

    def __init__(self, dtm_path: str):
        with rasterio.open(dtm_path) as src:
            self.data = src.read(1)
            self.transform = src.transform
            self.nodata = src.nodata
        print(f"[DTM] Loaded {dtm_path}  shape={self.data.shape}")

    def sample(self, x, y, z):
        """Return height-above-ground. NaN replaced with 0."""
        rows, cols = rasterio.transform.rowcol(self.transform, x, y)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        h, w = self.data.shape
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        dtm_z = np.full(len(x), np.nan, dtype=np.float64)
        dtm_z[valid] = self.data[rows[valid], cols[valid]]
        if self.nodata is not None:
            dtm_z[dtm_z == self.nodata] = np.nan
        hag = z - dtm_z
        return np.nan_to_num(hag, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# LAS file cache
# ---------------------------------------------------------------------------

class LASCache:
    def __init__(self, maxsize=2):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._maxsize = maxsize

    def get(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        data = _load_las(path)
        self._cache[path] = data
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return data


def _load_las(path):
    las = laspy.read(path)
    xyz = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)

    r = np.asarray(las.red,   dtype=np.float32)
    g = np.asarray(las.green, dtype=np.float32)
    b = np.asarray(las.blue,  dtype=np.float32)
    max_val = max(r.max(), g.max(), b.max())
    scale = 65535.0 if max_val > 255.0 else 255.0
    rgb = np.stack([r, g, b], axis=-1) / scale
    rgb = np.clip(rgb, 0.0, 1.0)

    # Try multiple normal field names (handles different LAS exporters)
    normals = None
    for nx_n, ny_n, nz_n in [
        ('NormalX', 'NormalY', 'NormalZ'),
        ('normal x', 'normal y', 'normal z'),
        ('normalx', 'normaly', 'normalz'),
        ('nx', 'ny', 'nz'),
    ]:
        try:
            normals = np.stack([
                np.asarray(las[nx_n], dtype=np.float32),
                np.asarray(las[ny_n], dtype=np.float32),
                np.asarray(las[nz_n], dtype=np.float32),
            ], axis=-1)
            break
        except Exception:
            continue
    if normals is None:
        print(f"  [WARN] No normals in {path} - using zeros")
        normals = np.zeros((xyz.shape[0], 3), dtype=np.float32)

    labels = np.asarray(las.label, dtype=np.int64)
    xy_min = xyz[:, :2].min(axis=0)
    xy_max = xyz[:, :2].max(axis=0)

    return {
        "xyz": xyz, "rgb": rgb, "normals": normals, "labels": labels,
        "xy_min": xy_min, "xy_max": xy_max,
    }


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(file_paths, num_classes=3, log_smoothing=True):
    counts = np.zeros(num_classes, dtype=np.float64)
    for p in file_paths:
        las = laspy.read(p)
        lbl = np.asarray(las.label, dtype=np.int64)
        for c in range(num_classes):
            counts[c] += (lbl == c).sum()
        del las, lbl
    freq = counts / counts.sum()
    if log_smoothing:
        w = 1.0 / np.log(1.02 + freq)
    else:
        w = counts.sum() / (num_classes * counts + 1e-6)
    w = w / w.sum() * num_classes
    print(f"[class_weights] counts={counts.astype(int).tolist()}  weights={w.round(4).tolist()}")
    return torch.tensor(w, dtype=torch.float32)


def _get_file_npoints(paths):
    counts = []
    for p in paths:
        with laspy.open(p) as f:
            counts.append(f.header.point_count)
    return counts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LASCropDataset(Dataset):
    """
    Virtual-epoch dataset for TorchSparse.
    Returns (SparseTensor, labels_tensor) per sample.
    """

    def __init__(
        self,
        file_paths: List[str],
        dtm_path: str,
        voxel_size: float = 0.20,
        crop_size: float = 20.0,
        samples_per_epoch: int = 500,
        augment: bool = False,
        min_points: int = 1024,
        max_points: int = 800_000,
        cache_size: int = 2,
        file_sampling: str = "uniform",
        file_weights: Optional[List[float]] = None,
        reject_minority: float = 0.0,
        reject_class: int = 2,
        reject_min_frac: float = 0.05,
    ):
        super().__init__()
        self.file_paths        = file_paths
        self.voxel_size        = voxel_size
        self.crop_size         = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.augment           = augment
        self.min_points        = min_points
        self.max_points        = max_points
        self.reject_minority   = reject_minority
        self.reject_class      = reject_class
        self.reject_min_frac   = reject_min_frac

        self._cache = LASCache(maxsize=cache_size)
        self._dtm = DTMSampler(dtm_path)

        n = len(file_paths)
        if file_sampling == "proportional":
            counts = _get_file_npoints(file_paths)
            total  = sum(counts)
            self._file_weights = [c / total for c in counts]
        elif file_sampling == "manual_weights":
            assert file_weights is not None and len(file_weights) == n
            s = sum(file_weights)
            self._file_weights = [w / s for w in file_weights]
        else:
            self._file_weights = [1.0 / n] * n

        print(f"[dataset] {n} files, sampling={file_sampling}")

    def __len__(self):
        return self.samples_per_epoch

    def _pick_file(self):
        return random.choices(self.file_paths, weights=self._file_weights, k=1)[0]

    def _random_crop(self, require_minority=False):
        for _ in range(50):
            path  = self._pick_file()
            cloud = self._cache.get(path)
            xyz   = cloud["xyz"]
            xy_min, xy_max = cloud["xy_min"], cloud["xy_max"]

            cx = random.uniform(xy_min[0], xy_max[0])
            cy = random.uniform(xy_min[1], xy_max[1])
            half = self.crop_size / 2.0
            mask = (
                (xyz[:, 0] >= cx - half) & (xyz[:, 0] < cx + half) &
                (xyz[:, 1] >= cy - half) & (xyz[:, 1] < cy + half)
            )
            idx = np.where(mask)[0]
            if idx.shape[0] < self.min_points:
                continue

            if require_minority:
                lbl = cloud["labels"][idx]
                frac = (lbl == self.reject_class).sum() / lbl.shape[0]
                if frac < self.reject_min_frac:
                    continue

            if idx.shape[0] > self.max_points:
                idx = np.random.choice(idx, self.max_points, replace=False)

            sub_xyz = xyz[idx]
            hag = self._dtm.sample(sub_xyz[:, 0], sub_xyz[:, 1], sub_xyz[:, 2])

            return (
                sub_xyz.astype(np.float32),
                cloud["rgb"][idx],
                cloud["normals"][idx],
                hag,
                cloud["labels"][idx],
            )

        # Fallback
        cloud = self._cache.get(self.file_paths[0])
        n = min(self.max_points, cloud["xyz"].shape[0])
        idx = np.arange(n)
        sub_xyz = cloud["xyz"][idx]
        hag = self._dtm.sample(sub_xyz[:, 0], sub_xyz[:, 1], sub_xyz[:, 2])
        return (
            sub_xyz.astype(np.float32), cloud["rgb"][idx],
            cloud["normals"][idx], hag, cloud["labels"][idx],
        )

    @staticmethod
    def _augment_points(xyz):
        theta = random.uniform(0, 2 * math.pi)
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        xyz = xyz @ R.T
        scale = random.uniform(0.9, 1.1)
        xyz *= scale
        xyz += np.random.normal(0, 0.005, size=xyz.shape).astype(np.float32)
        return xyz

    @staticmethod
    def _augment_color(rgb):
        rgb = rgb + random.uniform(-0.1, 0.1)
        mean = rgb.mean()
        rgb = (rgb - mean) * random.uniform(0.8, 1.2) + mean
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx):
        require_minority = (
            self.reject_minority > 0 and random.random() < self.reject_minority
        )

        xyz, rgb, normals, hag, labels = self._random_crop(require_minority)

        xyz_rel = (xyz - xyz.mean(axis=0)).astype(np.float32)

        if self.augment:
            xyz_rel = self._augment_points(xyz_rel)
            rgb = self._augment_color(rgb)

        # Quantise to voxel grid (N x 3 integer coords)
        coords = np.floor(xyz_rel / self.voxel_size).astype(np.int32)

        # 10-channel features
        feats = np.concatenate([
            xyz_rel, rgb, normals, hag[:, np.newaxis],
        ], axis=-1).astype(np.float32)

        # TorchSparse sparse_quantize: returns (unique_coords, unique_idx)
        # Build N×3 coords, sparse_collate will prepend batch column
        coords_ts, unique_idx = sparse_quantize(
            coords, voxel_size=1,  # already quantised, so voxel_size=1
            return_index=True,
        )

        if isinstance(unique_idx, torch.Tensor):
            unique_idx = unique_idx.numpy()
        unique_idx = np.asarray(unique_idx, dtype=np.int64)

        feats_q  = torch.from_numpy(feats[unique_idx]).float()
        labels_q = torch.from_numpy(labels[unique_idx]).long()

        # SparseTensor with N×3 coords (no batch dim yet - sparse_collate adds it)
        st = SparseTensor(
            coords=torch.from_numpy(coords[unique_idx]).int(),
            feats=feats_q,
        )

        return st, labels_q


# ---------------------------------------------------------------------------
# Collation for DataLoader
# ---------------------------------------------------------------------------

def torchsparse_collate_fn(batch):
    """Collate list of (SparseTensor, labels) into batched tensors.
    sparse_collate prepends a batch-index column to coords."""
    sparse_tensors, labels_list = zip(*batch)
    batched_st = sparse_collate(list(sparse_tensors))
    batched_labels = torch.cat(list(labels_list), dim=0)
    return batched_st, batched_labels
