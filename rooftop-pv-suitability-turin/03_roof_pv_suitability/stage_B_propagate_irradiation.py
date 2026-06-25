#!/usr/bin/env python3
"""
stage_B_propagate_irradiation.py
================================

Propagate the per-polygon annual solar irradiation values (SI_Ann) from
master_file.shp onto the LOD2.2 roof faces via spatial intersection.

The SI_Ann column was computed by the author, by running r.sun on a 1 m
building DSM (PVGIS diffuse/global irradiation, Meteonorm Linke turbidity,
albedo 0.25 for the dense urban centre). This script does NOT recompute
irradiation; it spatially attributes those already-computed values onto our
LOD2.2 face geometry.

For each LOD2.2 face:
    si_ann_kwh_m2_yr =
        Σ(SI_Ann_polygon × intersection_area) / Σ(intersection_area)

i.e. an area-weighted mean of SI_Ann across all master_file polygons that
overlap this face. If no overlap exists, si_ann is NaN.

Input
-----
  face_gpkg     lod22_face_deviation_dsm.gpkg
  master_shp    master_file.shp (must contain SI_Ann column)

Output
------
  face_gpkg with new columns:
      si_ann_kwh_m2_yr        area-weighted SI_Ann from master_file
      si_coverage_pct         % of face area that had master_file coverage

Usage
-----
  python stage_B_propagate_irradiation.py \\
      --face-gpkg   /path/to/lod22_face_deviation_dsm.gpkg \\
      --master      /path/to/master_file.shp \\
      --out         /path/to/lod22_face_with_irradiation.gpkg
"""
import argparse
import functools
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
print = functools.partial(print, flush=True)  # noqa: A001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-gpkg", required=True,
                    help="lod22_face_deviation_dsm.gpkg")
    ap.add_argument("--master", required=True,
                    help="master_file.shp (must have SI_Ann column)")
    ap.add_argument("--out", required=True,
                    help="output GPKG (will overwrite if exists)")
    ap.add_argument("--si-column", default="SI_Ann",
                    help="irradiation column name in master file "
                         "(default: 'SI_Ann')")
    ap.add_argument("--simplify-master", type=float, default=0.10,
                    help="Douglas-Peucker tolerance (m) for master polygons "
                         "before intersecting. 0 = disabled. Speeds the "
                         "intersection 5-20x without affecting results at "
                         "face scale. (default: 0.10)")
    args = ap.parse_args()

    print(f"[Stage B] propagating {args.si_column} onto LOD2.2 faces")
    t0 = time.time()

    print(f"  reading {args.face_gpkg}...")
    faces = gpd.read_file(args.face_gpkg)
    print(f"  → {len(faces):,} faces, CRS {faces.crs}")
    if "building_i" not in faces.columns or "face_idx" not in faces.columns:
        sys.exit("ERROR: face GPKG must have building_i and face_idx columns")

    print(f"  reading {args.master}...")
    master = gpd.read_file(args.master)
    print(f"  → {len(master):,} polygons, CRS {master.crs}")
    if args.si_column not in master.columns:
        sys.exit(f"ERROR: '{args.si_column}' column not found in master. "
                 f"Columns: {list(master.columns)}")

    # CRS alignment
    if master.crs is None:
        master = master.set_crs(epsg=32632)
    if faces.crs is None:
        faces = faces.set_crs(epsg=32632)
    if master.crs != faces.crs:
        print(f"  reprojecting master {master.crs} → {faces.crs}")
        master = master.to_crs(faces.crs)
    print(f"  [{time.time()-t0:.1f}s]")

    # Filter out invalid / null SI values
    master = master[master[args.si_column].notna()].copy()
    master = master[master.geometry.is_valid & ~master.geometry.is_empty].copy()
    master = master.reset_index(drop=True)
    print(f"  master after dropping nulls/invalids: {len(master):,}")

    # Simplify master polygons — drops most rasterisation noise without
    # affecting area-weighted means at face scale
    if args.simplify_master > 0:
        print(f"  simplifying master at {args.simplify_master} m tolerance...")
        master["geometry"] = master.geometry.simplify(
            args.simplify_master, preserve_topology=True
        )
        master = master[
            master.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ]
        master = master[~master.geometry.is_empty].copy().reset_index(drop=True)
        print(f"  → {len(master):,} polygons after simplify "
              f"[{time.time()-t0:.1f}s]")

    # Build spatial index on master
    print(f"  building STRtree on master polygons...")
    t_idx = time.time()
    master_geoms = master.geometry.values
    tree = STRtree(master_geoms)
    master_si = master[args.si_column].astype(float).values
    print(f"  → indexed in {time.time()-t_idx:.1f}s")

    # Per-face aggregation
    n_faces = len(faces)
    si_numerator = np.zeros(n_faces)
    coverage_area = np.zeros(n_faces)
    n_overlaps = np.zeros(n_faces, dtype=int)

    print(f"  intersecting {n_faces:,} faces with master polygons...")
    t_loop = time.time()
    last_report = t_loop
    face_geoms = faces.geometry.values
    for i in range(n_faces):
        fg = face_geoms[i]
        if fg is None or fg.is_empty:
            continue
        cands = tree.query(fg)
        if len(cands) == 0:
            continue
        for j in cands:
            mg = master_geoms[j]
            if not fg.intersects(mg):
                continue
            inter = fg.intersection(mg)
            if inter.is_empty:
                continue
            a = inter.area
            if a <= 0:
                continue
            si_numerator[i] += master_si[j] * a
            coverage_area[i] += a
            n_overlaps[i] += 1

        now = time.time()
        if now - last_report > 10:
            rate = (i + 1) / (now - t_loop)
            eta = (n_faces - i - 1) / max(rate, 1e-6)
            print(f"    {i+1:,}/{n_faces:,}  ({rate:.0f}/s, ETA {eta:.0f}s)")
            last_report = now
    print(f"  intersection done [{time.time()-t_loop:.1f}s]")

    # Compute area-weighted mean SI per face
    si_wmean = np.where(coverage_area > 0, si_numerator / coverage_area, np.nan)
    face_area = faces["area_m2"].astype(float).values
    coverage_pct = np.where(face_area > 0,
                            100.0 * coverage_area / face_area,
                            0.0)

    # Attach columns and write
    faces["si_ann_kwh_m2_yr"]  = np.round(si_wmean, 2)
    faces["si_coverage_pct"]   = np.round(coverage_pct, 2)
    faces["si_n_overlaps"]     = n_overlaps

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # GPKG reserves 'fid'; strip if present
    faces_w = faces.drop(columns=[c for c in ("fid", "fid_") if c in faces.columns])
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    faces_w.to_file(tmp, driver="GPKG")
    if out_path.exists():
        out_path.unlink()
    tmp.rename(out_path)

    print(f"\n=== Stage B summary ===")
    n_covered = int((coverage_pct > 50).sum())
    print(f"  faces processed:                     {n_faces:,}")
    print(f"  faces with >50% SI coverage:         {n_covered:,}  "
          f"({100*n_covered/n_faces:.1f}%)")
    print(f"  faces with no SI overlap:            "
          f"{int((n_overlaps == 0).sum()):,}")
    valid = ~np.isnan(si_wmean)
    if valid.any():
        print(f"  SI_Ann (kWh/m²/yr) on covered faces:")
        print(f"    min:     {np.nanmin(si_wmean):.1f}")
        print(f"    median:  {np.nanmedian(si_wmean):.1f}")
        print(f"    mean:    {np.nanmean(si_wmean):.1f}")
        print(f"    max:     {np.nanmax(si_wmean):.1f}")
    print(f"  → wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)  "
          f"[total {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
