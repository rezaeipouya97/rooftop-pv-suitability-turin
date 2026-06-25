#!/usr/bin/env python3
"""
stage_C_face_layers.py
======================

Compute the three-layer area cascade per LOD2.2 face.

Inputs are the outputs of Stage B (faces with SI_Ann attached) and the
existing merge_pv_into_faces.py output (faces with pv_good_area_m2 etc.).
Either pass the merged GPKG as --face-pv-gpkg, or pass both the irradiation
GPKG and the PV-attributes CSV from the merge script.

Per face:

  layer1_available_m2 = area_m2 × (1 − feature_pct_50cm / 100)
                        — the LOD2-modelability area; existing methodology

  layer2_suitable_m2  = pv_good_area_m2
                        — from our classifier overlay; the new contribution

  layer3_kwh_yr       = layer2_suitable_m2 × si_ann_kwh_m2_yr
                        — gross annual irradiation on PV-suitable area.
                          Honest headline upper bound. NO performance ratio,
                          NO working-area factor — those need supervisor-
                          cited coefficients and are not applied here.

Output: lod22_face_four_layers.gpkg with the three layer columns added.

Usage
-----
Two-input form:
    python stage_C_face_layers.py \\
        --face-irrad-gpkg /path/to/lod22_face_with_irradiation.gpkg \\
        --face-pv-csv     /path/to/face_pv_attributes.csv \\
        --out             /path/to/lod22_face_four_layers.gpkg

One-input form (faces already have both si_ann_kwh_m2_yr and pv_* columns):
    python stage_C_face_layers.py \\
        --face-irrad-gpkg /path/to/lod22_face_with_irradiation_and_pv.gpkg \\
        --out             /path/to/lod22_face_four_layers.gpkg
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
    ap.add_argument("--face-irrad-gpkg", required=True,
                    help="output of Stage B (with si_ann_kwh_m2_yr column)")
    ap.add_argument("--face-pv-csv", default=None,
                    help="face_pv_attributes.csv from merge_pv_into_faces.py "
                         "(optional if the GPKG already has pv_* columns)")
    ap.add_argument("--out", required=True,
                    help="output GPKG path")
    args = ap.parse_args()

    print(f"[Stage C] computing 3-layer cascade per face")
    t0 = time.time()

    faces = gpd.read_file(args.face_irrad_gpkg)
    print(f"  faces: {len(faces):,}, columns: {list(faces.columns)}")

    needed = ("area_m2", "feature_pct_50cm", "si_ann_kwh_m2_yr",
              "building_i", "face_idx")
    for c in needed:
        if c not in faces.columns:
            sys.exit(f"ERROR: missing column '{c}' in face GPKG")

    # Attach pv_* attributes if not already present
    if "pv_good_area_m2" not in faces.columns:
        if not args.face_pv_csv:
            sys.exit("ERROR: GPKG has no 'pv_good_area_m2' column and no "
                     "--face-pv-csv provided")
        print(f"  reading face PV CSV {args.face_pv_csv}...")
        pv = pd.read_csv(args.face_pv_csv)
        before = len(faces)
        faces = faces.merge(pv, on=["building_i", "face_idx"], how="left")
        print(f"  joined PV CSV: {len(faces):,} rows (started {before:,})")
        # any unmatched → fill with zero PV area (no classifier evidence)
        if "pv_good_area_m2" not in faces.columns:
            sys.exit("ERROR: CSV did not provide pv_good_area_m2 column")
        faces["pv_good_area_m2"] = faces["pv_good_area_m2"].fillna(0)

    # Compute layers
    area = faces["area_m2"].astype(float)
    feat = faces["feature_pct_50cm"].astype(float).fillna(0)
    pv   = faces["pv_good_area_m2"].astype(float).fillna(0)
    si   = faces["si_ann_kwh_m2_yr"].astype(float)

    faces["layer1_available_m2"] = (area * (1 - feat / 100.0)).round(2)
    faces["layer2_suitable_m2"]  = pv.round(2)
    # layer3 only where we have a valid SI value
    faces["layer3_kwh_yr"] = (pv * si).round(1)

    # Sanity: cap layer2 at area (the classifier overlay shouldn't exceed face
    # area in theory, but spatial intersection rounding could).
    overshoot = (faces["layer2_suitable_m2"] > area + 0.1).sum()
    if overshoot:
        print(f"  [WARN] {overshoot:,} faces have pv_good > area_m2; "
              f"clamping")
        faces["layer2_suitable_m2"] = np.minimum(faces["layer2_suitable_m2"],
                                                  area)

    # Write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faces_w = faces.drop(
        columns=[c for c in ("fid", "fid_") if c in faces.columns]
    )
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    faces_w.to_file(tmp, driver="GPKG")
    if out_path.exists():
        out_path.unlink()
    tmp.rename(out_path)

    print(f"\n=== Stage C summary ===")
    print(f"  faces:                          {len(faces):,}")
    print(f"  Σ area_m2 (gross LOD2.2):       "
          f"{area.sum():>14,.0f} m²")
    print(f"  Σ layer1_available_m2:          "
          f"{faces['layer1_available_m2'].sum():>14,.0f} m²")
    print(f"  Σ layer2_suitable_m2:           "
          f"{faces['layer2_suitable_m2'].sum():>14,.0f} m²")
    valid_kwh = faces["layer3_kwh_yr"].dropna()
    if len(valid_kwh):
        print(f"  Σ layer3_kwh_yr (covered):      "
              f"{valid_kwh.sum():>14,.0f} kWh/yr  "
              f"(on {len(valid_kwh):,}/{len(faces):,} faces)")
    print(f"  → wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)  "
          f"[total {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
