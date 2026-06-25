#!/usr/bin/env python3
"""
Stage G — Residential filter and family count
===============================================

Adds residential / family-count attributes to buildings_cascade.csv, following
the Mutani exercise sheet methodology (PDF: 2025_Exercise 2 - Solar Energy).

Method
------
1. Read master_file.shp; group segments by building_i; propagate the first
   non-null `User` value to every segment of the building. (Only one segment
   per building carries User in the source data.)
2. Map User strings → is_residential (boolean). Include both pure
   ("residenziale", "abitativa") and mixed-use ("residenziale e ...")
   categories — rationale: the ISTAT census block FAM21 field counts only
   actual resident families, so mixed-use shop/apartment buildings are not
   inflated by the inclusion.
3. Drop garages / anomalies: footprint_area_m2 < 40 OR rf_h_roof_50p < 3.
4. Spatially join each remaining building's footprint centroid to the
   matching ISTAT 2021 census section (PRO_COM=1272 = Torino).
5. Per census block, disaggregate FAM21 to buildings proportionally to
   their volume:
       N_families_b = FAM21_block × V_b / Σ V_b_in_block
   where V_b = footprint_area_m2 × rf_h_roof_50p.
6. Write new columns to buildings_cascade.csv: User, is_residential,
   n_floors, building_volume_m3, sez21_id, fam21_block, n_families_building.

Run
---
    python stage_G_residential_and_families.py \\
        --buildings-csv $OUT/buildings_cascade.csv \\
        --master-shp    "$NEW/master_file.shp" \\
        --footprint-shp "$NEW/foot_print.shp" \\
        --istat-shp     "$BASE/data/istat_2021/R01_21_WGS84.shp" \\
        --out-csv       $OUT/buildings_cascade.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────

# User values that count as residential. Include mixed-use because FAM21
# already counts only actual residents.
RESIDENTIAL_USER_VALUES = {
    "residenziale",
    "abitativa",
    "residenziale e commerciale",
    "residenziale e produttivo",
    "residenziale e ufficio pubblico",
}

PRO_COM_TORINO = 1272      # ISTAT comune code for Torino
MIN_FOOTPRINT_M2 = 40.0    # Mutani PDF: drop garages
MIN_BUILDING_HEIGHT_M = 3.0  # Mutani PDF: drop height anomalies
FLOOR_HEIGHT_M = 3.0       # used only to derive n_floors for reporting

# ─── Helpers ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buildings-csv", required=True, type=Path,
                   help="Stage D output: per-building cascade CSV")
    p.add_argument("--master-shp", required=True, type=Path,
                   help="master_file.shp — source of the User column")
    p.add_argument("--footprint-shp", required=True, type=Path,
                   help="foot_print.shp — building footprint geometries")
    p.add_argument("--istat-shp", required=True, type=Path,
                   help="ISTAT 2021 census sections shapefile (R01_21_WGS84.shp)")
    p.add_argument("--out-csv", required=True, type=Path,
                   help="Output CSV (can be same as --buildings-csv to overwrite)")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[stage_G] {msg}", flush=True)


def normalise_user(v) -> str:
    """Strip + lowercase a User string, NaN → empty string."""
    if pd.isna(v):
        return ""
    return str(v).strip().lower()


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # 1 ── Read buildings cascade CSV ────────────────────────────────────
    log(f"reading {args.buildings_csv}")
    buildings = pd.read_csv(args.buildings_csv)
    log(f"  {len(buildings):,} buildings in input")

    # Confirm essential columns
    needed = {"building_i", "footprint_area_m2", "rf_h_roof_50p"}
    missing = needed - set(buildings.columns)
    if missing:
        log(f"ERROR: missing columns in buildings_csv: {missing}")
        return 1

    # 2 ── Propagate User from master_file to buildings ──────────────────
    log(f"reading master_file User column from {args.master_shp}")
    mf = gpd.read_file(args.master_shp,
                       columns=["building_i", "User"],
                       ignore_geometry=True)
    log(f"  {len(mf):,} master_file segments")
    mf["User"] = mf["User"].map(normalise_user)
    mf_with_user = mf[mf["User"] != ""].copy()
    log(f"  {len(mf_with_user):,} segments have a non-empty User value")

    # Take first non-null User per building_i
    user_per_building = (mf_with_user
                        .groupby("building_i")["User"]
                        .first()
                        .reset_index())
    log(f"  {len(user_per_building):,} buildings have a User value via propagation")

    buildings = buildings.merge(user_per_building, on="building_i", how="left")
    n_user_missing = buildings["User"].isna().sum()
    log(f"  {n_user_missing:,} buildings have no User after propagation "
        f"(treated as non-residential)")
    buildings["User"] = buildings["User"].fillna("")

    # 3 ── is_residential flag ───────────────────────────────────────────
    buildings["is_residential"] = buildings["User"].isin(RESIDENTIAL_USER_VALUES)
    n_res_before_drop = int(buildings["is_residential"].sum())
    log(f"  residential by User: {n_res_before_drop:,} / {len(buildings):,}")

    # Apply Mutani PDF drop rules: footprint < 40 OR height < 3
    too_small = buildings["footprint_area_m2"] < MIN_FOOTPRINT_M2
    too_short = buildings["rf_h_roof_50p"] < MIN_BUILDING_HEIGHT_M
    drop_mask = too_small | too_short
    log(f"  drop rules: {int(too_small.sum()):,} too small "
        f"(<{MIN_FOOTPRINT_M2} m²), {int(too_short.sum()):,} too short "
        f"(<{MIN_BUILDING_HEIGHT_M} m)")
    buildings.loc[drop_mask, "is_residential"] = False
    log(f"  residential after drops: {int(buildings['is_residential'].sum()):,}")

    # 4 ── Building volume + floor count ─────────────────────────────────
    buildings["building_volume_m3"] = (buildings["footprint_area_m2"]
                                       * buildings["rf_h_roof_50p"])
    buildings["n_floors"] = np.maximum(
        1,
        np.round(buildings["rf_h_roof_50p"] / FLOOR_HEIGHT_M).astype("Int64")
    )

    # 5 ── Spatial join to ISTAT census blocks ───────────────────────────
    log(f"reading footprint geometries from {args.footprint_shp}")
    fp = gpd.read_file(args.footprint_shp,
                       columns=["building_i"])
    log(f"  {len(fp):,} footprint polygons read; CRS={fp.crs}")

    # Use centroid for the join (avoids edge cases where footprints span
    # two adjacent census blocks)
    fp["centroid"] = fp.geometry.centroid
    fp_centroids = gpd.GeoDataFrame(
        fp[["building_i"]].copy(),
        geometry=fp["centroid"].values,
        crs=fp.crs,
    )

    log(f"reading ISTAT census sections from {args.istat_shp}")
    istat = gpd.read_file(args.istat_shp,
                          columns=["PRO_COM", "SEZ21_ID", "FAM21",
                                   "POP21", "EDI21"])
    log(f"  {len(istat):,} ISTAT sections (all Piedmont)")
    istat = istat[istat["PRO_COM"] == PRO_COM_TORINO].copy()
    log(f"  {len(istat):,} sections in Torino (PRO_COM={PRO_COM_TORINO})")

    # Reproject footprints to the ISTAT CRS if needed (both should be UTM 32N)
    if fp_centroids.crs != istat.crs:
        log(f"  reprojecting footprints from {fp_centroids.crs} to {istat.crs}")
        fp_centroids = fp_centroids.to_crs(istat.crs)

    log("spatial-joining each footprint centroid to its ISTAT section")
    fp_with_sez = gpd.sjoin(
        fp_centroids,
        istat[["SEZ21_ID", "FAM21", "POP21", "geometry"]],
        how="left",
        predicate="within",
    )
    n_no_match = fp_with_sez["SEZ21_ID"].isna().sum()
    log(f"  {n_no_match:,} footprints did NOT match any ISTAT section "
        f"(likely outside Torino or in a roadway)")

    # Drop the spatial-join helper column (index_right) before merging
    fp_with_sez = fp_with_sez.drop(columns=["index_right", "geometry"],
                                   errors="ignore")
    fp_with_sez = pd.DataFrame(fp_with_sez)  # plain DataFrame, drop geometry

    # 6 ── Disaggregate FAM21 per block by volume ────────────────────────
    log("disaggregating FAM21 per census block proportionally to volume")
    buildings = buildings.merge(
        fp_with_sez[["building_i", "SEZ21_ID", "FAM21", "POP21"]],
        on="building_i", how="left",
    )
    buildings = buildings.rename(columns={
        "SEZ21_ID": "sez21_id",
        "FAM21": "fam21_block",
        "POP21": "pop21_block",
    })

    # Only residential buildings contribute to and receive a share
    res_mask = buildings["is_residential"]
    res = buildings.loc[res_mask].copy()

    # Sum residential building volume per census block
    vol_per_block = (res.groupby("sez21_id")["building_volume_m3"]
                       .sum()
                       .rename("block_volume_m3"))
    log(f"  {len(vol_per_block):,} census blocks have at least one "
        f"residential building")

    res = res.merge(vol_per_block, left_on="sez21_id",
                    right_index=True, how="left")

    # families allocated to this building (PDF: people, but we use FAM21 directly)
    res["n_families_building"] = np.where(
        (res["block_volume_m3"] > 0) & res["fam21_block"].notna(),
        res["fam21_block"] * res["building_volume_m3"] / res["block_volume_m3"],
        np.nan,
    )

    # population follows the same disaggregation (informational, not used downstream)
    res["n_people_building"] = np.where(
        (res["block_volume_m3"] > 0) & res["pop21_block"].notna(),
        res["pop21_block"] * res["building_volume_m3"] / res["block_volume_m3"],
        np.nan,
    )

    # Round families to integer (you can't have 1.4 families)
    res["n_families_building"] = res["n_families_building"].round().astype("Int64")
    res["n_people_building"] = res["n_people_building"].round().astype("Int64")

    # Merge back
    buildings = buildings.merge(
        res[["building_i", "block_volume_m3", "n_families_building",
             "n_people_building"]],
        on="building_i", how="left",
    )

    # 7 ── Validation summary ────────────────────────────────────────────
    n_res = int(buildings["is_residential"].sum())
    n_with_fam = int((buildings["n_families_building"].fillna(0) > 0).sum())
    total_fam = int(buildings["n_families_building"].fillna(0).sum())
    total_ppl = int(buildings["n_people_building"].fillna(0).sum())
    log("─" * 60)
    log("Stage G summary")
    log(f"  total buildings:                  {len(buildings):,}")
    log(f"  residential after filters:        {n_res:,}")
    log(f"  residential with n_families > 0:  {n_with_fam:,}")
    log(f"  total families estimated:         {total_fam:,}")
    log(f"  total people estimated:           {total_ppl:,}")
    if n_with_fam > 0:
        med_fam = buildings.loc[buildings["n_families_building"] > 0,
                                "n_families_building"].median()
        log(f"  median n_families per residential building: {med_fam}")
    log("─" * 60)

    # 8 ── Write output ──────────────────────────────────────────────────
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    buildings.to_csv(args.out_csv, index=False)
    log(f"wrote {args.out_csv}")
    log(f"  added columns: User, is_residential, n_floors, "
        f"building_volume_m3, sez21_id, fam21_block, pop21_block, "
        f"block_volume_m3, n_families_building, n_people_building")

    return 0


if __name__ == "__main__":
    sys.exit(main())
