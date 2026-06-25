#!/usr/bin/env python3
"""
stage_B2_monthly_irradiation.py
================================

Propagate the 12 per-polygon MONTHLY solar irradiation values
(SI_Jan, SI_Feb, ..., SI_Dec) from master_file.shp onto the LOD2.2
roof faces via the same area-weighted spatial intersection that Stage B
used for the annual SI_Ann.

Why this exists
---------------
Stage B propagated only SI_Ann (annual) onto the faces. For the hourly
SCI/SSI analysis (Stages H-L) we need monthly granularity per face so
we can scale a PVGIS-derived hourly *shape* to match our DSM-shadowed
*magnitude* month by month. That preserves the OUR-data dominance for
absolute production while letting PVGIS provide the diurnal distribution.

For each LOD2.2 face, for each month m ∈ {Jan, ..., Dec}:

    si_<m>_kwh_m2_mo =
        Σ(SI_<m>_polygon × intersection_area) / Σ(intersection_area)

i.e. area-weighted mean of SI_<m> across all master_file polygons
overlapping this face. If no overlap exists, all 12 values are NaN.

Input
-----
  face-gpkg   path to lod22_face_four_layers.gpkg
              (must have building_i, face_idx, area_m2, geometry)
  master      path to master_file.shp (must have SI_Jan..SI_Dec)

Output
------
  out-csv     CSV (one row per face) with columns:
                building_i, face_idx,
                si_jan_kwh_m2_mo, si_feb_..., ..., si_dec_...,
                si_monthly_coverage_pct, si_monthly_n_overlaps

Usage
-----
  python stage_B2_monthly_irradiation.py \\
      --face-gpkg $OUT/lod22_face_four_layers.gpkg \\
      --master    "$NEW/master_file.shp" \\
      --out-csv   $OUT/face_monthly_irradiation.csv
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

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SOURCE_COLS = [f"SI_{m}" for m in MONTHS]      # in master_file.shp
TARGET_COLS = [f"si_{m.lower()}_kwh_m2_mo" for m in MONTHS]  # in output CSV


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--face-gpkg", required=True,
                    help="lod22_face_four_layers.gpkg (Stage C output)")
    ap.add_argument("--master", required=True,
                    help="master_file.shp (must have SI_Jan..SI_Dec)")
    ap.add_argument("--out-csv", required=True,
                    help="output CSV (will overwrite if exists)")
    ap.add_argument("--simplify-master", type=float, default=0.10,
                    help="Douglas-Peucker tolerance (m) for master polygons "
                         "before intersecting. 0 = disabled. (default: 0.10)")
    args = ap.parse_args()

    print(f"[Stage B2] propagating SI_Jan..SI_Dec onto LOD2.2 faces")
    t0 = time.time()

    # ── 1. Load faces ────────────────────────────────────────────────
    print(f"  reading {args.face_gpkg}...")
    faces = gpd.read_file(args.face_gpkg)
    print(f"  → {len(faces):,} faces, CRS {faces.crs}")
    for col in ("building_i", "face_idx", "area_m2"):
        if col not in faces.columns:
            sys.exit(f"ERROR: face GPKG must have '{col}' column")

    # ── 2. Load master and check columns ─────────────────────────────
    print(f"  reading {args.master}...")
    master = gpd.read_file(args.master)
    print(f"  → {len(master):,} polygons, CRS {master.crs}")
    missing = [c for c in SOURCE_COLS if c not in master.columns]
    if missing:
        sys.exit(f"ERROR: missing monthly columns in master: {missing}\n"
                 f"  Available: {[c for c in master.columns if 'SI_' in c]}")

    # ── 3. CRS alignment ─────────────────────────────────────────────
    if master.crs is None:
        master = master.set_crs(epsg=32632)
    if faces.crs is None:
        faces = faces.set_crs(epsg=32632)
    if master.crs != faces.crs:
        print(f"  reprojecting master {master.crs} → {faces.crs}")
        master = master.to_crs(faces.crs)
    print(f"  [{time.time()-t0:.1f}s]")

    # ── 4. Filter master for valid geometry + at least one non-null SI ──
    # Treat a polygon as usable if ANY of SI_Jan..SI_Dec is non-null
    any_si = master[SOURCE_COLS].notna().any(axis=1)
    master = master[any_si].copy()
    master = master[master.geometry.is_valid & ~master.geometry.is_empty].copy()
    master = master.reset_index(drop=True)
    print(f"  master after dropping nulls/invalids: {len(master):,}")

    # ── 5. Simplify master polygons ──────────────────────────────────
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

    # ── 6. Build STRtree spatial index ──────────────────────────────
    print(f"  building STRtree on master polygons...")
    t_idx = time.time()
    master_geoms = master.geometry.values
    tree = STRtree(master_geoms)
    # Pre-extract all 12 monthly columns as a (n_master, 12) ndarray
    master_si = master[SOURCE_COLS].astype(float).values
    # Fill NaN with 0 in the numerator — we still divide by coverage_area
    # built from the same rows, so the area-weighted mean stays correct
    # as long as we only sum where SI is finite. Simpler: use nansum
    # per pair, but pre-fill 0 + a mask is cleaner.
    valid_mask = ~np.isnan(master_si)   # (n_master, 12)
    master_si_filled = np.where(valid_mask, master_si, 0.0)
    print(f"  → indexed in {time.time()-t_idx:.1f}s")

    # ── 7. Per-face area-weighted aggregation, 12 columns at once ───
    n_faces = len(faces)
    # numerator and coverage arrays — one per month (12 columns)
    si_num = np.zeros((n_faces, 12))         # Σ(SI_m × inter_area) per face
    cov_area = np.zeros((n_faces, 12))       # Σ(inter_area) where SI_m valid
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
            # add this polygon's contribution to all 12 months
            si_num[i]   += master_si_filled[j] * a   # uses 0 for NaN
            cov_area[i] += valid_mask[j].astype(float) * a   # only count valid months
            n_overlaps[i] += 1

        now = time.time()
        if now - last_report > 10:
            rate = (i + 1) / (now - t_loop)
            eta = (n_faces - i - 1) / max(rate, 1e-6)
            print(f"    {i+1:,}/{n_faces:,}  ({rate:.0f}/s, ETA {eta:.0f}s)")
            last_report = now
    print(f"  intersection done [{time.time()-t_loop:.1f}s]")

    # ── 8. Compute area-weighted means ───────────────────────────────
    si_wmean = np.where(cov_area > 0, si_num / cov_area, np.nan)   # (n, 12)
    face_area = faces["area_m2"].astype(float).values
    # Coverage % uses any-month coverage as the reference (any month that
    # has at least one valid overlap counts toward face's coverage)
    any_cov = cov_area.max(axis=1)   # if any month has coverage, face is covered
    coverage_pct = np.where(face_area > 0,
                            100.0 * any_cov / face_area,
                            0.0)

    # ── 9. Build the output DataFrame ────────────────────────────────
    out = pd.DataFrame({
        "building_i": faces["building_i"].values,
        "face_idx":   faces["face_idx"].values,
    })
    for k, col in enumerate(TARGET_COLS):
        out[col] = np.round(si_wmean[:, k], 3)
    out["si_monthly_coverage_pct"] = np.round(coverage_pct, 2)
    out["si_monthly_n_overlaps"]   = n_overlaps

    # ── 10. Write CSV ────────────────────────────────────────────────
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6

    # ── 11. Summary ──────────────────────────────────────────────────
    print(f"\n=== Stage B2 summary ===")
    n_covered = int((coverage_pct > 50).sum())
    print(f"  faces processed:                     {n_faces:,}")
    print(f"  faces with >50% coverage:            {n_covered:,}  "
          f"({100*n_covered/n_faces:.1f}%)")
    print(f"  faces with no overlap:               "
          f"{int((n_overlaps == 0).sum()):,}")

    # Per-month summary: median across all covered faces
    print(f"\n  Monthly SI (kWh/m²/month), median across covered faces:")
    for k, m in enumerate(MONTHS):
        col_vals = si_wmean[:, k]
        valid = ~np.isnan(col_vals)
        if valid.any():
            med = np.nanmedian(col_vals)
            print(f"    {m}: {med:6.2f}")

    # Sanity check: sum of monthly medians vs the annual SI we already have
    monthly_sum_median = np.nansum([np.nanmedian(si_wmean[:, k])
                                    for k in range(12)])
    print(f"\n  Σ (monthly medians) = {monthly_sum_median:.1f} kWh/m²/yr")
    if "si_ann_kwh_m2_yr" in faces.columns:
        annual_median = faces["si_ann_kwh_m2_yr"].median()
        print(f"  median si_ann_kwh_m2_yr (from Stage B) = {annual_median:.1f}")
        print(f"  ratio (monthly sum / annual): "
              f"{monthly_sum_median / max(annual_median, 1e-6):.3f}  "
              "(expected ≈ 1.0)")

    print(f"\n  → wrote {out_path}  ({size_mb:.1f} MB)  "
          f"[total {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
