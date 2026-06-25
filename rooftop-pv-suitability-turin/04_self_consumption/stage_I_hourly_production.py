#!/usr/bin/env python3
"""
Stage I — Hourly PV production per building
============================================

Combines:
  - per-face hourly tilted irradiance from Stage H (W/m², already DSM-scaled
    against our monthly SI_<month> values, with orientation tag)
  - per-face PV-suitable area from Stage C (layer2_suitable_m2)
  - efficiency + performance ratio constants

into a per-building 8,760-hour production timeseries.

Formula (Mutani notes)
----------------------
    P_PV[h] = PR · H_se[h] · (f · A_pv) · η

where:
    PR    = performance ratio (default 0.80)
    H_se  = global irradiance on the inclined plane (W/m²) at hour h
            (from Stage H — PVGIS shape × our DSM monthly magnitude)
    f     = active-area fraction (frame factor, default 0.80). Accounts for the
            inactive panel frame only. We do NOT use the more common 0.60, which
            also bundles a reduction for roof obstructions/unusable area — that
            reduction is already made, face by face, inside the Layer 2 classifier
            area, so applying 0.60 on top would remove the unusable area twice.
            (Per Prof. Mutani's note on using the active surface.) The thesis
            (Chapter 4) results use f = 0.80.
    A_pv  = layer2_suitable_m2 (classifier-corrected PV area, m²)
    η     = module efficiency = 0.24 (premium monocrystalline, Mutani spec)

The orientation tag from Stage H is carried through so Stage L can filter
to "south-facing only" or report SCI by orientation.

Output
------
Parquet: building_hourly_production.parquet
    building_i, hour_of_year, datetime, pv_kwh

Optional: --face-output also writes per-face hourly production so we can
make per-orientation aggregations in Stage L.

Run
---
    python stage_I_hourly_production.py \\
        --hourly-irrad   $OUT/face_hourly_irradiation.parquet \\
        --face-layers    $OUT/lod22_face_four_layers.gpkg \\
        --out            $OUT/building_hourly_production.parquet \\
        --face-output    $OUT/face_hourly_production.parquet \\
        --pr 0.80 --efficiency 0.24 --active-fraction 0.80
"""
from __future__ import annotations
import argparse
import functools
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)  # noqa: A001


DEFAULT_PR = 0.80
DEFAULT_EFFICIENCY = 0.24
DEFAULT_ACTIVE_FRACTION = 0.80  # frame factor (active panel surface); see module docstring


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hourly-irrad", required=True, type=Path,
                   help="Stage H output: face_hourly_irradiation.parquet")
    p.add_argument("--face-layers", required=True, type=Path,
                   help="Stage C output: lod22_face_four_layers.gpkg")
    p.add_argument("--out", required=True, type=Path,
                   help="Per-building hourly production parquet")
    p.add_argument("--face-output", type=Path, default=None,
                   help="Optional: also write per-face hourly production "
                        "(carries orientation tag, used by Stage L for "
                        "per-orientation aggregation)")
    p.add_argument("--pr", type=float, default=DEFAULT_PR,
                   help=f"Performance ratio (default {DEFAULT_PR})")
    p.add_argument("--efficiency", type=float, default=DEFAULT_EFFICIENCY,
                   help=f"Module efficiency η (default {DEFAULT_EFFICIENCY})")
    p.add_argument("--active-fraction", type=float,
                   default=DEFAULT_ACTIVE_FRACTION,
                   help=f"Active-area fraction f / frame factor applied to the "
                        f"PV-suitable area (default {DEFAULT_ACTIVE_FRACTION}; "
                        f"the Chapter 4 thesis results use 0.80). Do not set 0.60 "
                        f"— that double-counts the Layer 2 obstruction removal.")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[stage_I] {msg}")


def main() -> int:
    args = parse_args()
    log(f"PR = {args.pr}, η = {args.efficiency}, f (active fraction) = {args.active_fraction}")

    # 1 ── Load hourly irradiation ────────────────────────────────────────
    log(f"reading {args.hourly_irrad}")
    cols = ["building_i", "face_idx", "orientation",
            "hour_of_year", "datetime", "ghi_tilted_wm2"]
    irrad = pd.read_parquet(args.hourly_irrad, columns=cols)
    log(f"  {len(irrad):,} face × hour rows")
    log(f"  {irrad['building_i'].nunique():,} unique buildings, "
        f"{irrad.groupby(['building_i', 'face_idx']).ngroups:,} unique faces")
    log(f"  orientations: {sorted(irrad['orientation'].unique().tolist())}")

    # 2 ── Load per-face PV area ─────────────────────────────────────────
    log(f"reading {args.face_layers}")
    faces = gpd.read_file(args.face_layers, ignore_geometry=True)
    log(f"  {len(faces):,} faces in face-layers")

    needed = {"building_i", "face_idx", "layer2_suitable_m2"}
    if not needed.issubset(set(faces.columns)):
        log(f"ERROR: missing columns: {needed - set(faces.columns)}")
        return 1

    face_areas = faces[["building_i", "face_idx",
                        "layer2_suitable_m2"]].copy()
    face_areas["layer2_suitable_m2"] = (face_areas["layer2_suitable_m2"]
                                        .fillna(0.0))
    log(f"  total layer2_suitable_m2: "
        f"{face_areas['layer2_suitable_m2'].sum():,.0f} m²")

    # 3 ── Join area onto irradiance ─────────────────────────────────────
    log("joining irradiance with per-face PV area")
    df = irrad.merge(face_areas, on=["building_i", "face_idx"], how="left")
    df["layer2_suitable_m2"] = df["layer2_suitable_m2"].fillna(0.0)

    # 4 ── Apply the production formula ──────────────────────────────────
    log("computing hourly production per face: P = PR · H · (f · A) · η")
    df["pv_face_kwh"] = (args.pr
                         * df["ghi_tilted_wm2"]
                         * (args.active_fraction * df["layer2_suitable_m2"])
                         * args.efficiency) / 1000.0

    # 5 ── Aggregate per building × hour ─────────────────────────────────
    log("summing per building × hour")
    bld_hourly = (df.groupby(["building_i", "hour_of_year", "datetime"],
                             observed=True)["pv_face_kwh"]
                    .sum()
                    .reset_index()
                    .rename(columns={"pv_face_kwh": "pv_kwh"}))
    log(f"  {len(bld_hourly):,} building × hour rows")

    # 6 ── Sanity check ──────────────────────────────────────────────────
    annual = bld_hourly.groupby("building_i")["pv_kwh"].sum()
    log("─" * 60)
    log("Stage I summary")
    log(f"  total city-wide gross annual PV: {annual.sum()/1e6:,.2f} GWh/yr")
    log(f"  buildings with PV > 0:           "
        f"{int((annual > 0).sum()):,} / {len(annual):,}")
    if int((annual > 0).sum()) > 0:
        nz = annual[annual > 0]
        log(f"  median per-building annual PV:   {nz.median():,.0f} kWh/yr")
        log(f"  min / max:                       "
            f"{nz.min():.0f} / {nz.max():,.0f} kWh/yr")

    # By orientation
    by_ori = df.groupby("orientation")["pv_face_kwh"].sum()
    log(f"  annual PV by orientation (GWh/yr):")
    for ori, kwh in by_ori.sort_values(ascending=False).items():
        log(f"    {ori:5s}: {kwh/1e6:6.2f}")
    log("─" * 60)

    # 7 ── Write parquet(s) ──────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bld_hourly = bld_hourly[["building_i", "hour_of_year", "datetime", "pv_kwh"]]
    log(f"writing {args.out}")
    bld_hourly.to_parquet(args.out, engine="pyarrow", compression="snappy")
    log(f"  wrote {args.out.stat().st_size / 1e6:.1f} MB")

    if args.face_output is not None:
        log(f"writing per-face hourly production {args.face_output}")
        face_out = df[["building_i", "face_idx", "orientation",
                       "hour_of_year", "datetime", "pv_face_kwh"]].copy()
        face_out = face_out.rename(columns={"pv_face_kwh": "pv_kwh"})
        face_out.to_parquet(args.face_output, engine="pyarrow",
                            compression="snappy")
        log(f"  wrote {args.face_output.stat().st_size / 1e6:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
