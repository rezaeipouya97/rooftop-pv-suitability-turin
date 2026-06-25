#!/usr/bin/env python3
"""
stage_F_augment_viewer.py
=========================

Augment the existing viewer JSONs with the cascade attributes computed in
Stages A-D so the web viewer can colour and inspect by those metrics.

Reads:
  - buildings_3d.json          existing viewer JSON, 1,396 buildings
  - faces_3d_pv.json           existing viewer JSON, 5,434 faces with pv_*
  - buildings_cascade.csv      Stage D output
  - lod22_face_four_layers.gpkg  Stage C output

Writes:
  - buildings_3d_cascade.json  same shape as buildings_3d.json + new props
  - faces_3d_cascade.json      same shape as faces_3d_pv.json + new props

The new building properties:
  pv_area_lod1_coef            Mutani 0.35 × footprint
  layer1_available_m2          Σ per-face Layer 1
  layer2_suitable_m2           Σ per-face Layer 2 (classifier)
  layer3_kwh_yr                Σ per-face Layer 3
  coef_vs_layer2_ratio         Mutani / Layer 2 (the thesis headline)
  coef_vs_lod13_ratio          Mutani / roofer flat
  footprint_area_m2            BDTRE footprint

The new face properties:
  si_ann_kwh_m2_yr             propagated from master_file
  si_coverage_pct              fraction of face covered by master_file
  layer1_available_m2          area × (1 − feature_pct_50cm/100)
  layer2_suitable_m2           pv_good_area_m2 (from merge)
  layer3_kwh_yr                Layer 2 × si_ann

Existing JSON keys are preserved. Files are written compact.

Usage
-----
  python stage_F_augment_viewer.py \\
      --buildings-json /path/to/buildings_3d.json \\
      --faces-json     /path/to/faces_3d_pv.json \\
      --buildings-csv  /path/to/buildings_cascade.csv \\
      --face-gpkg      /path/to/lod22_face_four_layers.gpkg \\
      --out-dir        /path/to/viewer/data/
"""
import argparse
import functools
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)  # noqa: A001


# columns to lift from the per-building CSV onto buildings_3d.json features
BUILDING_COLS = [
    "footprint_area_m2",
    "pv_area_lod1_coef",
    "lod13_total_m2",
    "lod13_effective_m2",
    "layer1_available_m2",
    "layer2_suitable_m2",
    "layer3_kwh_yr",
    "coef_vs_layer2_ratio",
    "coef_vs_lod13_ratio",
    "n_faces",
    "faces_with_si",
]

# columns to lift from the per-face GPKG onto faces_3d_pv.json features
FACE_COLS = [
    "si_ann_kwh_m2_yr",
    "si_coverage_pct",
    "si_n_overlaps",
    "layer1_available_m2",
    "layer2_suitable_m2",
    "layer3_kwh_yr",
]


def jsonable(v):
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings-json", required=True,
                    help="existing buildings_3d.json from the deviation viewer")
    ap.add_argument("--faces-json", required=True,
                    help="existing faces_3d_pv.json from the merge step")
    ap.add_argument("--buildings-csv", required=True,
                    help="Stage D buildings_cascade.csv")
    ap.add_argument("--face-gpkg", required=True,
                    help="Stage C lod22_face_four_layers.gpkg")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------- buildings -------------------
    print(f"[1/2] augmenting buildings...")
    with open(args.buildings_json) as f:
        bjson = json.load(f)
    print(f"  loaded {len(bjson['features']):,} building features")

    bcsv = pd.read_csv(args.buildings_csv)
    print(f"  loaded {len(bcsv):,} rows from buildings_cascade.csv")

    # build a building_i → dict-of-cascade-attrs lookup
    bcsv_cols = [c for c in BUILDING_COLS if c in bcsv.columns]
    missing = [c for c in BUILDING_COLS if c not in bcsv.columns]
    if missing:
        print(f"  [WARN] missing columns from buildings CSV: {missing}")
    blook = {}
    for _, row in bcsv.iterrows():
        attrs = {}
        for c in bcsv_cols:
            attrs[c] = jsonable(row[c])
        blook[row["building_i"]] = attrs

    n_matched = 0
    n_unmatched = 0
    for feat in bjson["features"]:
        bi = feat["properties"].get("building_i")
        attrs = blook.get(bi)
        if attrs:
            feat["properties"].update(attrs)
            n_matched += 1
        else:
            n_unmatched += 1
            # fill nulls so the viewer's code can rely on the keys existing
            for c in bcsv_cols:
                feat["properties"][c] = None
    print(f"  matched {n_matched:,}/{len(bjson['features']):,} buildings")
    if n_unmatched:
        print(f"  [WARN] {n_unmatched:,} buildings had no cascade row")

    out_b = out_dir / "buildings_3d_cascade.json"
    with open(out_b, "w") as f:
        json.dump(bjson, f, separators=(",", ":"))
    print(f"  → wrote {out_b}  ({out_b.stat().st_size/1e6:.2f} MB)")

    # ------------------- faces -------------------
    print(f"\n[2/2] augmenting faces...")
    with open(args.faces_json) as f:
        fjson = json.load(f)
    print(f"  loaded {len(fjson['features']):,} face features")

    fg = gpd.read_file(args.face_gpkg)
    print(f"  loaded {len(fg):,} faces from {Path(args.face_gpkg).name}")

    fg_cols = [c for c in FACE_COLS if c in fg.columns]
    missing = [c for c in FACE_COLS if c not in fg.columns]
    if missing:
        print(f"  [WARN] missing columns from face GPKG: {missing}")

    # build a (building_i, face_idx) → dict-of-cascade-attrs lookup
    flook = {}
    for _, row in fg.iterrows():
        attrs = {}
        for c in fg_cols:
            attrs[c] = jsonable(row[c])
        key = (row["building_i"], int(row["face_idx"]))
        flook[key] = attrs

    n_matched = 0
    n_unmatched = 0
    for feat in fjson["features"]:
        bi  = feat["properties"].get("building_i")
        fi  = feat["properties"].get("face_idx")
        if fi is None:
            n_unmatched += 1
            continue
        key = (bi, int(fi))
        attrs = flook.get(key)
        if attrs:
            feat["properties"].update(attrs)
            n_matched += 1
        else:
            n_unmatched += 1
            for c in fg_cols:
                feat["properties"][c] = None
    print(f"  matched {n_matched:,}/{len(fjson['features']):,} faces")
    if n_unmatched:
        print(f"  [WARN] {n_unmatched:,} faces had no cascade row")

    out_f = out_dir / "faces_3d_cascade.json"
    with open(out_f, "w") as f:
        json.dump(fjson, f, separators=(",", ":"))
    print(f"  → wrote {out_f}  ({out_f.stat().st_size/1e6:.2f} MB)")

    print(f"\n[total {time.time()-t0:.1f}s]")
    print(f"\nDeploy by copying these two files into the viewer's data/ dir:")
    print(f"  cp {out_b} <viewer>/data/")
    print(f"  cp {out_f} <viewer>/data/")
    print(f"or just point the viewer's DATA paths at the new filenames.")


if __name__ == "__main__":
    main()
