#!/usr/bin/env python3
"""
merge_pv_into_faces.py
======================

Aggregate per-polygon classifier predictions (predictions.gpkg) onto the
LOD2.2 per-face deviation analysis (lod22_face_deviation_dsm.gpkg) and the
matching viewer JSON (faces_3d.json), producing PV-suitability attributes
on every LOD2.2 face.

The LOD2.2 face is the unit of decision. For every face:

  pv_good_area_m2    = sum of intersection-area with all pred=1 polygons
  pv_bad_area_m2     = sum with pred=2
  pv_obs_area_m2     = sum with pred=3
  pv_classified_area_m2 = sum of the three above
  pv_good_pct        = 100 * pv_good_area_m2 / face.area_m2
  pv_obs_pct         = 100 * pv_obs_area_m2 / face.area_m2
  pv_coverage_pct    = 100 * pv_classified_area_m2 / face.area_m2
  pv_p_good_wmean    = area-weighted mean of p_good across overlapping pred=1
  pv_n_overlaps      = count of intersecting classifier polygons
  pv_verdict         = string (see thresholds at top of compute_verdict())

This script is idempotent: re-running it overwrites the *_pv outputs.

Outputs (under --out-dir):
  lod22_face_deviation_dsm_pv.gpkg   # GPKG with new pv_* columns
  faces_3d_pv.json                    # viewer JSON with new pv_* properties
  face_pv_attributes.csv              # per-face aggregates for reference

Usage:
  python merge_pv_into_faces.py \
    --predictions    /mnt/.../predictions.gpkg \
    --face-gpkg      /mnt/.../lod22_face_deviation_dsm.gpkg \
    --faces-json     /mnt/.../faces_3d.json \
    --out-dir        /mnt/.../viewer_pv/data/
"""
import argparse
import functools
import json
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


# --------------------------------------------------------------------------
# Verdict thresholds — TUNABLE; see compute_verdict()
# --------------------------------------------------------------------------
VERDICT_THRESHOLDS = {
    "pv_suitable_min_good_pct":          60.0,
    "pv_suitable_max_obs_pct":           20.0,
    "pv_suitable_max_slope_deg":         60.0,    # exclude near-vertical faces
    "partial_min_good_pct":              30.0,
    "partial_max_obs_pct":               35.0,
    "obstructed_min_obs_pct":            30.0,
    "no_coverage_max_pct":               20.0,    # below this → classifier had no opinion
}


def compute_verdict(row) -> str:
    """Categorical PV verdict per face from the percentages."""
    if row["pv_coverage_pct"] < VERDICT_THRESHOLDS["no_coverage_max_pct"]:
        return "no-coverage"
    if row["pv_obs_pct"] > VERDICT_THRESHOLDS["obstructed_min_obs_pct"]:
        return "obstructed"
    is_flat_or_pitched = (
        bool(row.get("is_flat", False))
        or float(row.get("slope_deg", 0.0))
          <= VERDICT_THRESHOLDS["pv_suitable_max_slope_deg"]
    )
    if (row["pv_good_pct"] >= VERDICT_THRESHOLDS["pv_suitable_min_good_pct"]
        and row["pv_obs_pct"] <= VERDICT_THRESHOLDS["pv_suitable_max_obs_pct"]
        and is_flat_or_pitched):
        return "pv-suitable"
    if (row["pv_good_pct"] >= VERDICT_THRESHOLDS["partial_min_good_pct"]
        and row["pv_obs_pct"] <= VERDICT_THRESHOLDS["partial_max_obs_pct"]
        and is_flat_or_pitched):
        return "partially-suitable"
    return "unsuitable"


def aggregate_per_face(faces: gpd.GeoDataFrame,
                       polys: gpd.GeoDataFrame) -> pd.DataFrame:
    """For every face row, compute the PV aggregates from intersecting polys.

    Uses a shapely STRtree spatial index on the classifier polygons so each
    face query is O(log n) candidate retrieval, then exact intersection on
    those candidates only.
    """
    print(f"  building STRtree on {len(polys):,} classifier polygons...")
    t0 = time.time()
    poly_geoms = polys.geometry.values  # numpy array of shapely geometries
    tree = STRtree(poly_geoms)
    print(f"    done [{time.time()-t0:.1f}s]")

    poly_pred  = polys["pred"].values.astype(int)
    poly_pgood = (polys["p_good"].values
                  if "p_good" in polys.columns
                  else np.zeros(len(polys)))

    # Output arrays
    n_faces = len(faces)
    good_area = np.zeros(n_faces)
    bad_area  = np.zeros(n_faces)
    obs_area  = np.zeros(n_faces)
    pgood_weight_sum = np.zeros(n_faces)
    pgood_value_sum  = np.zeros(n_faces)
    n_overlaps = np.zeros(n_faces, dtype=int)

    print(f"  intersecting {n_faces:,} faces with classifier polygons...")
    t0 = time.time()
    face_geoms = faces.geometry.values
    last_report = t0
    for i in range(n_faces):
        face_g = face_geoms[i]
        if face_g is None or face_g.is_empty:
            continue
        candidate_idxs = tree.query(face_g)  # shapely 2.x returns indices
        if len(candidate_idxs) == 0:
            continue
        for j in candidate_idxs:
            pg = poly_geoms[j]
            if pg is None or pg.is_empty:
                continue
            # Quick reject on bounding box already handled by tree.query;
            # now do exact intersects + intersection-area.
            if not face_g.intersects(pg):
                continue
            inter = face_g.intersection(pg)
            if inter.is_empty:
                continue
            a = inter.area
            if a <= 0:
                continue
            cls = poly_pred[j]
            if cls == 1:
                good_area[i] += a
                pgood_weight_sum[i] += a
                pgood_value_sum[i] += a * poly_pgood[j]
            elif cls == 2:
                bad_area[i] += a
            elif cls == 3:
                obs_area[i] += a
            n_overlaps[i] += 1

        now = time.time()
        if now - last_report > 10:
            done = i + 1
            rate = done / (now - t0)
            eta = (n_faces - done) / max(rate, 1e-6)
            print(f"    {done:,}/{n_faces:,}  ({rate:.0f}/s, ETA {eta:.0f}s)")
            last_report = now
    print(f"    done [{time.time()-t0:.1f}s]")

    classified_area = good_area + bad_area + obs_area
    face_area = faces["area_m2"].values.astype(float)
    safe_area = np.maximum(face_area, 1e-6)
    safe_weights = np.maximum(pgood_weight_sum, 1e-9)

    df = pd.DataFrame({
        "building_i": faces["building_i"].values,
        "face_idx":   faces["face_idx"].values,
        "pv_good_area_m2":       np.round(good_area, 3),
        "pv_bad_area_m2":        np.round(bad_area,  3),
        "pv_obs_area_m2":        np.round(obs_area,  3),
        "pv_classified_area_m2": np.round(classified_area, 3),
        "pv_good_pct":           np.round(100 * good_area       / safe_area, 2),
        "pv_bad_pct":            np.round(100 * bad_area        / safe_area, 2),
        "pv_obs_pct":            np.round(100 * obs_area        / safe_area, 2),
        "pv_coverage_pct":       np.round(100 * classified_area / safe_area, 2),
        "pv_p_good_wmean":       np.round(pgood_value_sum / safe_weights, 4),
        "pv_n_overlaps":         n_overlaps,
    })
    # zero weighted-mean for faces with no pred=1 overlap
    df.loc[pgood_weight_sum == 0, "pv_p_good_wmean"] = 0.0
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="predictions.gpkg from the classifier")
    ap.add_argument("--face-gpkg", required=True,
                    help="lod22_face_deviation_dsm.gpkg")
    ap.add_argument("--faces-json", required=True,
                    help="existing faces_3d.json to enrich")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--simplify-polys", type=float, default=0.10,
                    help="Douglas-Peucker tolerance (m) for classifier "
                         "polygons before intersection. 0 = disable. "
                         "0.10 m is invisible at LOD2.2 face scale and "
                         "speeds intersection 5–20×. (default: 0.10)")
    ap.add_argument("--filter-noise", action="store_true", default=True,
                    help="Drop classifier polygons with area < 0.5 m² or "
                         "vertex-density > 1000 verts/m² before intersecting. "
                         "These are pixel-noise speckles. (default: on)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # 1. Load and sanity-check inputs
    # ----------------------------------------------------------------------
    print(f"[1/5] loading inputs...")
    t0 = time.time()
    faces = gpd.read_file(args.face_gpkg)
    print(f"  faces:        {len(faces):,} rows, columns: {list(faces.columns)}")
    print(f"  faces crs:    {faces.crs}")
    for c in ("building_i", "face_idx", "area_m2", "geometry"):
        if c not in faces.columns:
            sys.exit(f"ERROR: face GPKG missing column '{c}'")

    polys = gpd.read_file(args.predictions)
    print(f"  polys:        {len(polys):,} rows, columns: {list(polys.columns)[:8]}...")
    print(f"  polys crs:    {polys.crs}")
    for c in ("pred", "geometry"):
        if c not in polys.columns:
            sys.exit(f"ERROR: predictions GPKG missing column '{c}'")

    # Make sure both layers are in the same projected CRS (EPSG:32632 expected)
    if faces.crs is None:
        faces = faces.set_crs(epsg=32632)
        print(f"  [WARN] faces had no CRS; assumed EPSG:32632")
    if polys.crs is None:
        polys = polys.set_crs(epsg=32632)
        print(f"  [WARN] polys had no CRS; assumed EPSG:32632")
    if polys.crs != faces.crs:
        print(f"  reprojecting polys {polys.crs} → {faces.crs}")
        polys = polys.to_crs(faces.crs)
    print(f"  [{time.time()-t0:.1f}s]")

    # ----------------------------------------------------------------------
    # 2. Prepare classifier polygons: filter noise, simplify
    # ----------------------------------------------------------------------
    print(f"\n[2/5] preparing classifier polygons...")
    t0 = time.time()
    before = len(polys)
    if args.filter_noise:
        # area filter
        polys["_area_m2"] = polys.geometry.area
        polys = polys[polys["_area_m2"] >= 0.5].copy()
        # vertex density filter (drops extreme pixel-noise blobs)
        def _vc(g):
            if g is None or g.is_empty:
                return 0
            if g.geom_type == "Polygon":
                return len(g.exterior.coords) + sum(len(r.coords) for r in g.interiors)
            return sum(len(p.exterior.coords) + sum(len(r.coords) for r in p.interiors)
                       for p in g.geoms)
        polys["_vd"] = polys.geometry.apply(_vc) / np.maximum(polys["_area_m2"], 1e-6)
        polys = polys[polys["_vd"] <= 1000.0].copy()
        polys = polys.drop(columns=["_area_m2", "_vd"])
        print(f"  filter_noise: {len(polys):,}/{before:,} survived")

    if args.simplify_polys > 0 and len(polys) > 0:
        print(f"  simplifying at {args.simplify_polys} m tolerance...")
        polys["geometry"] = polys.geometry.simplify(
            args.simplify_polys, preserve_topology=True
        )
        # drop anything that simplified to empty / non-polygon
        polys = polys[polys.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        polys = polys[~polys.geometry.is_empty].copy()

    polys = polys.reset_index(drop=True)
    print(f"  → {len(polys):,} polygons ready  [{time.time()-t0:.1f}s]")

    # ----------------------------------------------------------------------
    # 3. Compute per-face aggregates via spatial intersection
    # ----------------------------------------------------------------------
    print(f"\n[3/5] computing per-face aggregates...")
    t0 = time.time()
    agg = aggregate_per_face(faces, polys)
    print(f"  [{time.time()-t0:.1f}s]")

    # ----------------------------------------------------------------------
    # 4. Compute verdicts + summary, write enriched GPKG and CSV
    # ----------------------------------------------------------------------
    print(f"\n[4/5] computing verdicts + writing GPKG/CSV...")
    enriched = faces.merge(agg, on=["building_i", "face_idx"], how="left")

    # Fill any unmatched faces (shouldn't be any) with zeros
    for col in ("pv_good_area_m2", "pv_bad_area_m2", "pv_obs_area_m2",
                "pv_classified_area_m2", "pv_good_pct", "pv_bad_pct",
                "pv_obs_pct", "pv_coverage_pct", "pv_p_good_wmean",
                "pv_n_overlaps"):
        if col in enriched.columns:
            enriched[col] = enriched[col].fillna(0)

    # Verdict
    enriched["pv_verdict"] = enriched.apply(compute_verdict, axis=1)
    print(f"  verdict counts:")
    print("    " + enriched["pv_verdict"].value_counts().to_string()
                       .replace("\n", "\n    "))

    # Write enriched GPKG (atomic-ish)
    out_gpkg = out_dir / "lod22_face_deviation_dsm_pv.gpkg"
    tmp_gpkg = out_gpkg.with_suffix(".gpkg.tmp")
    if tmp_gpkg.exists():
        tmp_gpkg.unlink()
    # GPKG reserves 'fid'; strip if present
    enriched_w = enriched.drop(
        columns=[c for c in ("fid", "fid_") if c in enriched.columns]
    )
    enriched_w.to_file(tmp_gpkg, driver="GPKG")
    if out_gpkg.exists():
        out_gpkg.unlink()
    tmp_gpkg.rename(out_gpkg)
    print(f"  wrote {out_gpkg} ({out_gpkg.stat().st_size/1e6:.1f} MB)")

    # CSV of just the PV attributes (lightweight)
    out_csv = out_dir / "face_pv_attributes.csv"
    csv_cols = ["building_i", "face_idx"] + [
        c for c in enriched.columns if c.startswith("pv_")
    ]
    enriched[csv_cols].to_csv(out_csv, index=False)
    print(f"  wrote {out_csv} ({out_csv.stat().st_size/1024:.1f} KB)")

    # ----------------------------------------------------------------------
    # 5. Patch faces_3d.json with PV attributes
    # ----------------------------------------------------------------------
    print(f"\n[5/5] patching faces_3d.json...")
    t0 = time.time()
    with open(args.faces_json) as f:
        fc = json.load(f)
    n_in = len(fc["features"])
    print(f"  loaded {n_in:,} features from {args.faces_json}")

    # Build a (building_i, face_idx) → pv attributes lookup
    lookup = {}
    pv_cols = [c for c in enriched.columns if c.startswith("pv_")]
    for _, row in enriched.iterrows():
        key = (row["building_i"], int(row["face_idx"]))
        attrs = {}
        for c in pv_cols:
            v = row[c]
            # JSON-serialise numpy types
            if isinstance(v, (np.integer,)):
                attrs[c] = int(v)
            elif isinstance(v, (np.floating,)):
                attrs[c] = (None if (np.isnan(v) or np.isinf(v))
                            else float(v))
            elif isinstance(v, (np.bool_,)):
                attrs[c] = bool(v)
            else:
                attrs[c] = v
        lookup[key] = attrs

    n_merged = 0
    n_missing = 0
    for feat in fc["features"]:
        p = feat.get("properties", {})
        key = (p.get("building_i"), int(p.get("face_idx", -1)))
        if key in lookup:
            p.update(lookup[key])
            n_merged += 1
        else:
            n_missing += 1
            # still add empty pv_* fields so the viewer can rely on them
            for c in pv_cols:
                p[c] = None
    print(f"  merged: {n_merged:,}/{n_in:,}")
    if n_missing:
        print(f"  [WARN] {n_missing:,} features had no matching GPKG row")

    out_json = out_dir / "faces_3d_pv.json"
    with open(out_json, "w") as f:
        json.dump(fc, f, separators=(",", ":"))  # compact
    print(f"  wrote {out_json} ({out_json.stat().st_size/1e6:.1f} MB) "
          f"[{time.time()-t0:.1f}s]")

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    pv_suitable_area = enriched.loc[
        enriched["pv_verdict"] == "pv-suitable", "area_m2"
    ].sum()
    partial_area = enriched.loc[
        enriched["pv_verdict"] == "partially-suitable", "area_m2"
    ].sum()
    total_area = enriched["area_m2"].sum()

    print(f"\n=== Summary ===")
    print(f"  Faces processed:           {len(enriched):,}")
    print(f"  Total LOD2.2 face area:    {total_area:,.0f} m²")
    print(f"  PV-suitable faces:         {(enriched['pv_verdict']=='pv-suitable').sum():,}  "
          f"({pv_suitable_area:,.0f} m², "
          f"{100*pv_suitable_area/max(total_area,1):.1f}%)")
    print(f"  Partially-suitable faces:  {(enriched['pv_verdict']=='partially-suitable').sum():,}  "
          f"({partial_area:,.0f} m², "
          f"{100*partial_area/max(total_area,1):.1f}%)")
    print(f"  Obstructed faces:          {(enriched['pv_verdict']=='obstructed').sum():,}")
    print(f"  Unsuitable faces:          {(enriched['pv_verdict']=='unsuitable').sum():,}")
    print(f"  No-coverage faces:         {(enriched['pv_verdict']=='no-coverage').sum():,}")


if __name__ == "__main__":
    main()
