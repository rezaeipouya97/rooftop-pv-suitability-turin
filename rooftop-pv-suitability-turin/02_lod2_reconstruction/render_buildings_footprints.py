#!/usr/bin/env python3
"""
render_buildings_footprints.py
Top-down plan view of the BUILDING points (predicted_label == 1) from the
classified cloud, with the BDTRE footprint outlines drawn on top.

Designed for the huge ~654M-point file: reads in chunks, never loads it all.

Run on the cluster:
    conda activate /home/prezaei/mink_env   # (already active in your shell)
    python render_buildings_footprints.py \
        --las /mnt/beegfs-compat/prezaei/Minkowski_Project/results/big_pred_exp2_rescued.las \
        --footprints /PATH/TO/footprints.shp \
        --output buildings_footprints.png \
        --resolution 0.5

If you don't have/need the footprints, omit --footprints and it will just
render the building points.
"""

import argparse, time
import numpy as np
import laspy

BUILDING_CLASS = 1
PRED_FIELD = "predicted_label"


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--las", required=True)
    pa.add_argument("--footprints", default=None, help="Shapefile or GeoJSON of footprints (optional)")
    pa.add_argument("--output", default="buildings_footprints.png")
    pa.add_argument("--resolution", type=float, default=0.5, help="grid cell size in metres")
    pa.add_argument("--pred_field", default=PRED_FIELD)
    pa.add_argument("--chunk", type=int, default=10_000_000)
    pa.add_argument("--xmin", type=float, default=None, help="optional zoom window")
    pa.add_argument("--xmax", type=float, default=None)
    pa.add_argument("--ymin", type=float, default=None)
    pa.add_argument("--ymax", type=float, default=None)
    args = pa.parse_args()

    t0 = time.time()
    # optional zoom window
    zoom = all(v is not None for v in [args.xmin, args.xmax, args.ymin, args.ymax])

    def in_zoom(bx, by):
        if not zoom:
            return np.ones(len(bx), dtype=bool)
        return (bx >= args.xmin) & (bx <= args.xmax) & (by >= args.ymin) & (by <= args.ymax)

    # ---- pass 1: find XY extent of building points ----
    print("Pass 1: scanning building-point extent...")
    x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
    n_bld = 0
    with laspy.open(args.las) as reader:
        total = reader.header.point_count
        for chunk in reader.chunk_iterator(args.chunk):
            pred = np.asarray(chunk[args.pred_field], dtype=np.int32)
            m = pred == BUILDING_CLASS
            if not m.any():
                continue
            bx = np.asarray(chunk.x, dtype=np.float64)[m]
            by = np.asarray(chunk.y, dtype=np.float64)[m]
            zm = in_zoom(bx, by)
            bx, by = bx[zm], by[zm]
            if len(bx) == 0:
                continue
            x_min, x_max = min(x_min, bx.min()), max(x_max, bx.max())
            y_min, y_max = min(y_min, by.min()), max(y_max, by.max())
            n_bld += len(bx)
            print(f"  building pts so far: {n_bld:,}", end="\r")
    print(f"\n  building points: {n_bld:,}")
    print(f"  extent X[{x_min:.1f},{x_max:.1f}] Y[{y_min:.1f},{y_max:.1f}]")

    # ---- build raster grid ----
    res = args.resolution
    cols = int(np.ceil((x_max - x_min) / res)) + 1
    rows = int(np.ceil((y_max - y_min) / res)) + 1
    grid = np.zeros((rows, cols), dtype=np.uint32)
    print(f"  grid {cols} x {rows}")

    # ---- pass 2: fill grid with building-point density ----
    print("Pass 2: rasterising building points...")
    with laspy.open(args.las) as reader:
        for chunk in reader.chunk_iterator(args.chunk):
            pred = np.asarray(chunk[args.pred_field], dtype=np.int32)
            m = pred == BUILDING_CLASS
            if not m.any():
                continue
            bx = np.asarray(chunk.x, dtype=np.float64)[m]
            by = np.asarray(chunk.y, dtype=np.float64)[m]
            zm = in_zoom(bx, by)
            bx, by = bx[zm], by[zm]
            if len(bx) == 0:
                continue
            ci = np.clip(((bx - x_min) / res).astype(np.int64), 0, cols - 1)
            ri = np.clip(((y_max - by) / res).astype(np.int64), 0, rows - 1)
            np.add.at(grid, (ri, ci), 1)

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(14, 14 * rows / cols), dpi=220)
    ax.set_facecolor("white")

    # building points as a soft slate fill (not harsh black)
    occupied = grid > 0
    cmap = ListedColormap([(1, 1, 1, 0), (0.30, 0.36, 0.44, 0.95)])  # transparent / slate
    ax.imshow(occupied, cmap=cmap, extent=[x_min, x_max, y_min, y_max],
              origin="upper", interpolation="nearest")

    # ---- overlay footprints if provided ----
    if args.footprints:
        try:
            import geopandas as gpd
            print(f"Overlaying footprints: {args.footprints}")
            gdf = gpd.read_file(args.footprints)
            if zoom:
                gdf = gdf.cx[args.xmin:args.xmax, args.ymin:args.ymax]
            gdf.boundary.plot(ax=ax, edgecolor="#C0392B", linewidth=0.8)
        except Exception as e:
            print(f"  [WARN] could not draw footprints: {e}")

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("Classified building points (slate) with BDTRE footprint outlines (red)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {args.output}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
