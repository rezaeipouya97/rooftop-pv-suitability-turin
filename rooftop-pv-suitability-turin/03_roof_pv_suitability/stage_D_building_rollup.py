#!/usr/bin/env python3
"""
stage_D_building_rollup.py
==========================

Aggregate the per-face 3-layer cascade (Stage C) into per-building totals,
joined with the baselines table (Stage A) so a single CSV/GPKG holds the
complete comparison cascade.

Per building columns (after this stage):
  building_i
  footprint_area_m2          BDTRE footprint
  pv_area_lod1_coef          Mutani 0.35 × footprint
  lod13_total_m2             roofer LOD1.3 gross
  lod13_effective_m2         roofer LOD1.3 after feature_pct correction
  lod22_total_m2             roofer LOD2.2 gross
  lod22_effective_m2         roofer LOD2.2 after feature_pct correction (~= layer1)
  layer1_available_m2        Σ face layer1 (should match lod22_effective_m2)
  layer2_suitable_m2         Σ face layer2 (classifier)
  layer3_kwh_yr              Σ face layer3 (× SI_Ann)
  n_faces                    count of LOD2.2 faces
  faces_with_si              count with valid SI_Ann coverage
  area_gain_lod2_m2          LOD2_eff − LOD1_eff (from roofer)
  ratio_lod2_lod1            LOD2_eff / LOD1_eff
  coef_vs_layer2_ratio       pv_area_lod1_coef / layer2_suitable_m2
  coef_vs_lod13_ratio        pv_area_lod1_coef / lod13_effective_m2
  rf_roof_type, rf_h_roof_50p, rf_rmse_lod22 (from roofer)

Usage
-----
  python stage_D_building_rollup.py \\
      --face-layers   /path/to/lod22_face_four_layers.gpkg \\
      --baselines     /path/to/baselines.csv \\
      --footprint     /path/to/foot_print.shp \\
      --out-csv       /path/to/buildings_cascade.csv \\
      --out-gpkg      /path/to/buildings_cascade.gpkg
"""
import argparse
import functools
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)  # noqa: A001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-layers", required=True,
                    help="Stage C output GPKG (per face)")
    ap.add_argument("--baselines", required=True,
                    help="Stage A output CSV (baselines)")
    ap.add_argument("--footprint", required=True,
                    help="BDTRE footprint shapefile for GPKG geometry")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-gpkg", required=True)
    args = ap.parse_args()

    print(f"[Stage D] per-building rollup + baselines join")
    t0 = time.time()

    # ----- 1. per-face → per-building sum -----
    print(f"  reading face layers...")
    faces = gpd.read_file(args.face_layers)
    print(f"  → {len(faces):,} faces  [{time.time()-t0:.1f}s]")
    needed = ("building_i", "area_m2", "layer1_available_m2",
              "layer2_suitable_m2", "layer3_kwh_yr", "si_ann_kwh_m2_yr")
    for c in needed:
        if c not in faces.columns:
            sys.exit(f"ERROR: missing column '{c}' in face GPKG")

    grp = faces.groupby("building_i")
    bld = grp.agg(
        n_faces             = ("face_idx",            "size"),
        total_face_area_m2  = ("area_m2",             "sum"),
        layer1_available_m2 = ("layer1_available_m2", "sum"),
        layer2_suitable_m2  = ("layer2_suitable_m2",  "sum"),
        layer3_kwh_yr       = ("layer3_kwh_yr",       "sum"),
        faces_with_si       = ("si_ann_kwh_m2_yr",
                                lambda s: int(s.notna().sum())),
    ).reset_index()
    for c in ("total_face_area_m2", "layer1_available_m2",
              "layer2_suitable_m2"):
        bld[c] = bld[c].round(2)
    bld["layer3_kwh_yr"] = bld["layer3_kwh_yr"].round(1)
    print(f"  rolled up to {len(bld):,} buildings")

    # ----- 2. join with baselines -----
    print(f"  reading baselines {args.baselines}...")
    base = pd.read_csv(args.baselines)
    print(f"  → {len(base):,} rows in baselines, "
          f"columns: {list(base.columns)[:6]}...")

    bld = base.merge(bld, on="building_i", how="outer")

    # Fill missing per-face aggregates with 0 (buildings without LOD2 faces)
    for c in ("n_faces", "total_face_area_m2", "layer1_available_m2",
              "layer2_suitable_m2", "layer3_kwh_yr", "faces_with_si"):
        if c in bld.columns:
            bld[c] = bld[c].fillna(0)
    bld["n_faces"]       = bld["n_faces"].astype(int)
    bld["faces_with_si"] = bld["faces_with_si"].astype(int)

    # ----- 3. derived comparison ratios -----
    bld["coef_vs_layer2_ratio"] = np.where(
        bld["layer2_suitable_m2"] > 0,
        bld["pv_area_lod1_coef"] / bld["layer2_suitable_m2"],
        np.nan,
    )
    bld["coef_vs_lod13_ratio"] = np.where(
        bld["lod13_effective_m2"] > 0,
        bld["pv_area_lod1_coef"] / bld["lod13_effective_m2"],
        np.nan,
    )

    # Cross-check: layer1_available should match lod22_effective_m2 from
    # the roofer pipeline. If they diverge dramatically that's a methodology
    # alarm — the two are the same quantity computed two different ways:
    # roofer Stage 3 area-weights face-level percentages; we sum face-level
    # (area × (1 − pct/100)).
    has_both = (bld["lod22_effective_m2"].notna() &
                (bld["layer1_available_m2"] > 0))
    if has_both.any():
        diff_pct = (
            100 * (bld.loc[has_both, "layer1_available_m2"]
                   - bld.loc[has_both, "lod22_effective_m2"])
            / bld.loc[has_both, "lod22_effective_m2"]
        )
        print(f"\n  cross-check: layer1_available_m2 vs lod22_effective_m2:")
        print(f"    median % diff: {diff_pct.median():+.2f}%")
        print(f"    p25 / p75    : {diff_pct.quantile(0.25):+.2f}% / "
              f"{diff_pct.quantile(0.75):+.2f}%")
        if abs(diff_pct.median()) > 5:
            print(f"  [WARN] median diff > 5%. Check Stage C formula "
                  f"or area-weighting convention.")

    # Round derived ratios for readability
    for c in ("coef_vs_layer2_ratio", "coef_vs_lod13_ratio"):
        bld[c] = bld[c].round(3)

    # ----- 4. write CSV -----
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    bld.to_csv(out_csv, index=False, float_format="%.2f")
    print(f"\n  wrote {out_csv}  ({out_csv.stat().st_size/1024:.1f} KB)")

    # ----- 5. write GPKG with footprint geometry for QGIS -----
    print(f"  reading footprint geometry...")
    fp = gpd.read_file(args.footprint)
    if "building_i" not in fp.columns:
        for cand in ("building_id", "cod_obj", "id_edif"):
            if cand in fp.columns:
                fp = fp.rename(columns={cand: "building_i"})
                break
    if fp.crs is None:
        fp = fp.set_crs(epsg=32632)
    fp = fp[["building_i", "geometry"]]

    gdf = fp.merge(bld, on="building_i", how="right")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=fp.crs)

    out_gpkg = Path(args.out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf_w = gdf.drop(columns=[c for c in ("fid", "fid_") if c in gdf.columns])
    tmp = out_gpkg.with_suffix(out_gpkg.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    gdf_w.to_file(tmp, driver="GPKG")
    if out_gpkg.exists():
        out_gpkg.unlink()
    tmp.rename(out_gpkg)
    print(f"  wrote {out_gpkg}  ({out_gpkg.stat().st_size/1e6:.1f} MB)")

    # ----- 6. headline summary -----
    has_lod2 = bld[bld["n_faces"] > 0]
    print(f"\n=== Stage D summary ===")
    print(f"  buildings total:                         {len(bld):,}")
    print(f"  with LOD2.2 reconstruction + classifier: {len(has_lod2):,}")
    print(f"")
    print(f"  Σ across LOD2-reconstructed buildings (m²):")
    print(f"    pv_area_lod1_coef  (Mutani 0.35):    "
          f"{has_lod2['pv_area_lod1_coef'].sum():>14,.0f}")
    print(f"    lod13_effective    (roofer flat):    "
          f"{has_lod2['lod13_effective_m2'].sum():>14,.0f}")
    print(f"    lod22_effective    (roofer faceted): "
          f"{has_lod2['lod22_effective_m2'].sum():>14,.0f}")
    print(f"    layer1_available   (our sum):        "
          f"{has_lod2['layer1_available_m2'].sum():>14,.0f}")
    print(f"    layer2_suitable    (classifier):     "
          f"{has_lod2['layer2_suitable_m2'].sum():>14,.0f}")
    print(f"    layer3_kwh_yr:                       "
          f"{has_lod2['layer3_kwh_yr'].sum():>14,.0f} kWh/yr")

    r_l2  = has_lod2["coef_vs_layer2_ratio"].dropna()
    r_l13 = has_lod2["coef_vs_lod13_ratio"].dropna()
    if len(r_l2):
        print(f"\n  building-level coef_vs_layer2_ratio  (Mutani / classifier):")
        print(f"    median: {r_l2.median():.2f}×    p25..p75: "
              f"[{r_l2.quantile(0.25):.2f}, {r_l2.quantile(0.75):.2f}]")
    if len(r_l13):
        print(f"  building-level coef_vs_lod13_ratio  (Mutani / roofer flat):")
        print(f"    median: {r_l13.median():.2f}×    p25..p75: "
              f"[{r_l13.quantile(0.25):.2f}, {r_l13.quantile(0.75):.2f}]")
    print(f"  [total {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
