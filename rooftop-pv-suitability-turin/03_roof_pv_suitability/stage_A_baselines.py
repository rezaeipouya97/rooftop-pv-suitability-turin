#!/usr/bin/env python3
"""
stage_A_baselines.py
====================

Build the per-building baselines table by joining two sources:
  1. The Mutani 0.35 coefficient method  ─►  footprint × 0.35
  2. The roofer-based LOD1.3 / LOD2.2 deviation pipeline output
     (lod_comparison_dsm.csv) which already contains lod13_effective_m2,
     lod22_effective_m2, and area_gain_lod2_m2.

The roofer LOD1.3 number is a *real measurement*, not a coefficient guess:
  lod13_effective_m2 = lod13_total_m2 × (1 − lod13_feature_pct_50cm / 100)
where lod13 is a flat-top extrusion at rf_h_roof_50p and the feature_pct
measures how much of the DSM deviates from that flat plane. For pitched
roofs, this deviation is huge (e.g. 80%+) because the flat plane is a
poor model. Hence LOD1.3 effective is small.

The roofer LOD2.2 number is similar but the LOD2.2 model is a planar-faceted
mesh, so the feature_pct only catches *real* obstructions (chimneys,
dormers, MVS noise), not the slope of the roof itself.

The thesis story is:
  Mutani 0.35 coef    →  national-scale crude reference
  LOD1.3 effective    →  measured flat-top baseline (still over-counts
                          because every real pitched roof reports as 80%+
                          "obstructed" relative to its flat top)
  LOD2.2 effective    →  measured faceted baseline (Layer 1)
  Layer 2 suitable    →  classifier-corrected (our new contribution)
  Layer 3 kWh/yr      →  × SI_Ann (no PR, no working-area)

Inputs
------
  --comparison-csv  /path/to/lod_comparison_dsm.csv   (Stage 4 output)
  --footprint       /path/to/foot_print.shp           (BDTRE)

Output
------
  baselines.csv
    building_i, footprint_area_m2, pv_area_lod1_coef,
    lod13_total_m2, lod13_effective_m2, lod13_feature_pct_50cm,
    lod22_total_m2, lod22_effective_m2, lod22_correction_m2,
    lod22_feature_pct_50cm, area_gain_lod2_m2, ratio_lod2_lod1,
    rf_roof_type, rf_h_roof_50p, rf_rmse_lod22

Usage
-----
  python stage_A_baselines.py \\
      --comparison-csv /mnt/.../lod_comparison_dsm.csv \\
      --footprint      /mnt/.../foot_print.shp \\
      --out            /mnt/.../baselines.csv
"""
import argparse
import functools
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

print = functools.partial(print, flush=True)  # noqa: A001


LOD1_COEFFICIENT = 0.35  # Italian-national rooftop PV reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-csv", required=True,
                    help="lod_comparison_dsm.csv from the deviation pipeline")
    ap.add_argument("--footprint", required=True,
                    help="BDTRE building footprints (.shp or .gpkg)")
    ap.add_argument("--out", required=True,
                    help="output CSV path")
    ap.add_argument("--coefficient", type=float, default=LOD1_COEFFICIENT,
                    help=f"Mutani LOD1 coefficient (default {LOD1_COEFFICIENT})")
    args = ap.parse_args()

    print(f"[Stage A] building baselines table")
    print(f"  Mutani coefficient: {args.coefficient}")
    t0 = time.time()

    # 1. roofer comparison CSV
    print(f"  reading {args.comparison_csv}...")
    cmp = pd.read_csv(args.comparison_csv)
    print(f"  → {len(cmp):,} rows, columns: {list(cmp.columns)}")
    if "building_i" not in cmp.columns:
        sys.exit("ERROR: comparison CSV has no 'building_i' column")
    needed = ["lod13_total_m2", "lod13_effective_m2", "lod13_feature_pct_50cm",
              "lod22_total_m2", "lod22_effective_m2", "lod22_correction_m2",
              "lod22_feature_pct_50cm", "area_gain_lod2_m2", "ratio_lod2_lod1"]
    missing = [c for c in needed if c not in cmp.columns]
    if missing:
        sys.exit(f"ERROR: comparison CSV missing columns: {missing}")

    # 2. footprint shapefile for the coefficient and for joining
    print(f"  reading {args.footprint}...")
    fp = gpd.read_file(args.footprint)
    print(f"  → {len(fp):,} footprints, CRS {fp.crs}")

    # Normalise building_i (shapefile truncation)
    if "building_i" not in fp.columns:
        for cand in ("building_id", "cod_obj", "id_edif", "local_id", "gid"):
            if cand in fp.columns:
                fp = fp.rename(columns={cand: "building_i"})
                print(f"  renamed {cand!r} → 'building_i'")
                break
        else:
            sys.exit(f"ERROR: no building_i column in footprint. "
                     f"Columns: {list(fp.columns)}")

    # Reproject to UTM 32N if needed for area
    if fp.crs is None:
        fp = fp.set_crs(epsg=32632)
        print(f"  [WARN] no CRS — assumed EPSG:32632")
    elif fp.crs.is_geographic:
        print(f"  reprojecting {fp.crs} → EPSG:32632")
        fp = fp.to_crs(epsg=32632)

    fp["footprint_area_m2"] = fp.geometry.area
    fp["pv_area_lod1_coef"] = fp["footprint_area_m2"] * args.coefficient
    fp_df = fp[["building_i", "footprint_area_m2", "pv_area_lod1_coef"]]

    # 3. join: outer so we preserve everything from both sides
    out = fp_df.merge(cmp, on="building_i", how="outer", indicator=True)
    n_both     = int((out["_merge"] == "both").sum())
    n_fp_only  = int((out["_merge"] == "left_only").sum())
    n_cmp_only = int((out["_merge"] == "right_only").sum())
    out = out.drop(columns=["_merge"])
    out = out.sort_values("building_i").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.4f")

    print(f"\n=== Stage A summary ===")
    print(f"  buildings in footprint shapefile:        {len(fp_df):,}")
    print(f"  buildings in comparison CSV:             {len(cmp):,}")
    print(f"  joined (both sources):                   {n_both:,}")
    print(f"  in footprint only (no LOD2 reconstr.):   {n_fp_only:,}")
    print(f"  in CSV only (no footprint match):        {n_cmp_only:,}")
    print(f"")
    print(f"  Σ footprint_area_m2:                     "
          f"{out['footprint_area_m2'].sum():>14,.0f} m²")
    print(f"  Σ pv_area_lod1_coef  (Mutani 0.35):      "
          f"{out['pv_area_lod1_coef'].sum():>14,.0f} m²")
    print(f"  Σ lod13_effective_m2 (roofer flat-top):  "
          f"{out['lod13_effective_m2'].sum():>14,.0f} m²")
    print(f"  Σ lod22_effective_m2 (roofer LOD2.2):    "
          f"{out['lod22_effective_m2'].sum():>14,.0f} m²")
    print(f"")
    print(f"  → wrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB)  "
          f"[total {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
