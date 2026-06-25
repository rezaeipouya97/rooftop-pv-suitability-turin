#!/usr/bin/env python3
"""
Stage H — PVGIS hourly tilted irradiation, scaled to our DSM monthly totals
============================================================================

For each LOD2.2 face, build an 8,760-hour profile of tilted-plane irradiance
(W/m²) by combining two data sources:

  1. PVGIS v5.2 hourly TMY profile → provides the time-of-day distribution
     (when energy arrives during each day, per orientation)
  2. Our DSM-shadowed monthly SI_<month> values → provide the absolute
     magnitude (how much energy actually reaches each face after
     neighbour-building shading, computed from the 5 cm DSM)

For each face f in each month m:

    scale_f,m = SI_<m>_face  /  PVGIS_monthly_total[orientation(f), tilt(f), m]
    H_face[h] = PVGIS_hourly[orientation(f), tilt(f), h]  ×  scale_f,month(h)

The hourly *shape* therefore matches PVGIS physics (sun position, clear-sky);
the absolute monthly *magnitude* matches our DSM-measured shadowing.

This is what makes the SCI/SSI comparison vs Usta-Mutani 2025 meaningful:
they used PVGIS unshaded; we use our shaded monthly totals.

Orientation bucketing
---------------------
Faces are classified by their azimuth into 7 named orientations, all
queried at tilt bins {15°, 30°, 45°, 60°}. 7 × 4 = 28 PVGIS calls.
Flat faces (slope < 5°) get a standard 30°-tilt S-facing install.

  S    : azimuth 150°–210°  (true south, ±30°)
  SE   : azimuth  90°–150°
  SW   : azimuth 210°–270°
  E    : azimuth  45°– 90°
  W    : azimuth 270°–315°
  N    : azimuth 315°–360° or 0°–45°  (penalty case)
  flat : (slope < 5°, install at 30° S)

The orientation tag is written back per face so Stage L can filter
("south-facing only" = orientation ∈ {S, SE, SW, flat}).

Output
------
Parquet: face_hourly_irradiation.parquet with columns
    building_i, face_idx, orientation, slope_deg, azimuth_deg,
    tilt_bin, hour_of_year, datetime, ghi_tilted_wm2

Run
---
    python stage_H_pvgis_hourly.py \\
        --face-layers     $OUT/lod22_face_four_layers.gpkg \\
        --monthly-irrad   $OUT/face_monthly_irradiation.csv \\
        --out             $OUT/face_hourly_irradiation.parquet \\
        --lat 45.07 --lon 7.69
"""
from __future__ import annotations
import argparse
import functools
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

print = functools.partial(print, flush=True)  # noqa: A001

# ─── Configuration ────────────────────────────────────────────────────────

PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
TURIN_LAT = 45.07
TURIN_LON = 7.69
TMY_YEAR_START = 2020
TMY_YEAR_END = 2020

# Tilt bins (degrees). Each face's slope is snapped to the nearest one.
TILT_BINS = [15, 30, 45, 60]

# Flat-roof override: if slope < threshold, install at 30° S
FLAT_ROOF_SLOPE_THRESHOLD_DEG = 5.0
FLAT_ROOF_INSTALL_TILT_DEG = 30.0
FLAT_ROOF_ORIENTATION_TAG = "flat"

# Orientation tags (compass-azimuth ranges, uphill direction)
# Order matters: each face matches the first range it falls into.
ORIENTATION_RANGES = [
    # (tag,  azimuth_min,  azimuth_max,  representative_az_for_pvgis_query)
    ("S",   150.0, 210.0, 180.0),
    ("SE",   90.0, 150.0, 135.0),
    ("SW",  210.0, 270.0, 225.0),
    ("E",    45.0,  90.0,  90.0),
    ("W",   270.0, 315.0, 270.0),
    # N wraps around 360°/0°
    ("N",   315.0, 360.0,   0.0),  # also covers 0-45 below
]
N_FALLBACK_RANGE = (0.0, 45.0)   # second N range
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHLY_COL_NAMES = [f"si_{m.lower()}_kwh_m2_mo" for m in MONTHS]

# Polite delay between PVGIS calls
PVGIS_DELAY_S = 1.0


# ─── Helpers ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--face-layers", required=True, type=Path,
                   help="Stage C output: lod22_face_four_layers.gpkg")
    p.add_argument("--monthly-irrad", required=True, type=Path,
                   help="Stage B2 output: face_monthly_irradiation.csv "
                        "(per-face DSM-shadowed monthly totals)")
    p.add_argument("--out", required=True, type=Path,
                   help="Output parquet")
    p.add_argument("--lat", type=float, default=TURIN_LAT,
                   help=f"Latitude for PVGIS (default {TURIN_LAT})")
    p.add_argument("--lon", type=float, default=TURIN_LON,
                   help=f"Longitude for PVGIS (default {TURIN_LON})")
    p.add_argument("--delay", type=float, default=PVGIS_DELAY_S,
                   help="Seconds to wait between PVGIS calls")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[stage_H] {msg}")


def classify_orientation(slope_deg: float, azimuth_deg: float) -> tuple[str, float]:
    """
    Return (orientation_tag, representative_azimuth_deg) for a face.
    representative_azimuth_deg is what we send to PVGIS for that bucket.
    """
    if slope_deg < FLAT_ROOF_SLOPE_THRESHOLD_DEG:
        return (FLAT_ROOF_ORIENTATION_TAG, 180.0)
    az = azimuth_deg % 360.0
    for tag, lo, hi, rep_az in ORIENTATION_RANGES:
        if lo <= az < hi:
            return (tag, rep_az)
    # N second range (0–45°)
    if N_FALLBACK_RANGE[0] <= az < N_FALLBACK_RANGE[1]:
        return ("N", 0.0)
    # safety fallback
    return ("S", 180.0)


def classify_tilt(slope_deg: float, orientation_tag: str) -> int:
    """Snap face's tilt to nearest TILT_BINS value."""
    if orientation_tag == FLAT_ROOF_ORIENTATION_TAG:
        return int(FLAT_ROOF_INSTALL_TILT_DEG)
    # Snap to nearest bin
    diffs = [abs(slope_deg - t) for t in TILT_BINS]
    return TILT_BINS[int(np.argmin(diffs))]


def compass_to_pvgis_azimuth(az_compass_deg: float) -> float:
    """Convert compass az (0=N, 90=E, 180=S) to PVGIS (0=S, -90=E, +90=W)."""
    pv = az_compass_deg - 180.0
    if pv > 180.0:
        pv -= 360.0
    elif pv < -180.0:
        pv += 360.0
    return pv


def fetch_pvgis_hourly(lat: float, lon: float,
                       tilt_deg: float, azimuth_compass_deg: float,
                       session: requests.Session,
                       timeout: int = 60) -> pd.DataFrame:
    """Call PVGIS seriescalc API. Returns DataFrame [datetime, ghi_tilted_wm2]."""
    az_pv = compass_to_pvgis_azimuth(azimuth_compass_deg)
    params = {
        "lat": lat, "lon": lon,
        "startyear": TMY_YEAR_START, "endyear": TMY_YEAR_END,
        "angle": tilt_deg, "aspect": az_pv,
        "outputformat": "json", "components": 0, "browser": 0,
    }
    r = session.get(PVGIS_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    hourly = data["outputs"]["hourly"]
    df = pd.DataFrame(hourly)
    df = df.rename(columns={"G(i)": "ghi_tilted_wm2", "time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d:%H%M")
    return df[["datetime", "ghi_tilted_wm2"]]


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # 1 ── Load faces + classify ─────────────────────────────────────────
    log(f"reading {args.face_layers}")
    faces = gpd.read_file(args.face_layers, ignore_geometry=True)
    log(f"  {len(faces):,} faces in input")
    needed = {"building_i", "face_idx", "slope_deg", "azimuth_deg"}
    missing = needed - set(faces.columns)
    if missing:
        log(f"ERROR: missing columns in face-layers: {missing}")
        return 1

    log("classifying faces by orientation + tilt bin")
    orientations = []
    tilts = []
    rep_azs = []
    for _, row in faces.iterrows():
        slope = float(row["slope_deg"]) if pd.notna(row["slope_deg"]) else 30.0
        az    = float(row["azimuth_deg"]) if pd.notna(row["azimuth_deg"]) else 180.0
        ori, rep_az = classify_orientation(slope, az)
        tilt = classify_tilt(slope, ori)
        orientations.append(ori)
        tilts.append(tilt)
        rep_azs.append(rep_az)
    faces["orientation"] = orientations
    faces["tilt_bin"] = tilts
    faces["_pvgis_az"] = rep_azs

    log("  orientation counts:")
    for ori, n in faces["orientation"].value_counts().items():
        log(f"    {ori:5s}: {n:>5,} faces")
    log("  tilt-bin counts:")
    for t, n in faces["tilt_bin"].value_counts().sort_index().items():
        log(f"    {t}°: {n:>5,} faces")

    # 2 ── Load DSM monthly per-face totals ──────────────────────────────
    log(f"reading {args.monthly_irrad}")
    monthly = pd.read_csv(args.monthly_irrad)
    log(f"  {len(monthly):,} rows in monthly CSV")
    miss = [c for c in MONTHLY_COL_NAMES if c not in monthly.columns]
    if miss:
        log(f"ERROR: monthly CSV missing columns: {miss}")
        return 1

    faces = faces.merge(monthly[["building_i", "face_idx"]
                                + MONTHLY_COL_NAMES],
                        on=["building_i", "face_idx"], how="left")
    n_no_monthly = faces[MONTHLY_COL_NAMES].isna().all(axis=1).sum()
    log(f"  {n_no_monthly:,} faces have no DSM monthly data "
        f"(will get raw PVGIS, unscaled)")

    # 3 ── Identify unique (orientation, tilt) bins → PVGIS calls ────────
    unique_bins = (faces[["orientation", "tilt_bin", "_pvgis_az"]]
                   .drop_duplicates()
                   .sort_values(["orientation", "tilt_bin"])
                   .reset_index(drop=True))
    log(f"  {len(unique_bins):,} unique (orientation, tilt) bins → PVGIS calls")

    # 4 ── Call PVGIS per bin ────────────────────────────────────────────
    log(f"PVGIS endpoint: {PVGIS_URL}")
    log(f"location: lat={args.lat}, lon={args.lon}")
    session = requests.Session()
    bin_profiles: dict[tuple[str, int], pd.DataFrame] = {}
    bin_monthly_pvgis: dict[tuple[str, int], np.ndarray] = {}
    t0 = time.time()
    n_fails = 0

    for i, row in unique_bins.iterrows():
        key = (row["orientation"], int(row["tilt_bin"]))
        rep_az = float(row["_pvgis_az"])
        tilt = int(row["tilt_bin"])
        try:
            df = fetch_pvgis_hourly(args.lat, args.lon, tilt, rep_az,
                                    session=session)
            bin_profiles[key] = df
            # Compute monthly totals from this profile (kWh/m²/month)
            df["_month"] = df["datetime"].dt.month
            monthly_kwh = (df.groupby("_month")["ghi_tilted_wm2"]
                             .sum() / 1000.0).values   # 12 values
            bin_monthly_pvgis[key] = monthly_kwh
            elapsed = time.time() - t0
            log(f"  bin {i+1:>2}/{len(unique_bins)}: orientation={key[0]:5s} "
                f"tilt={tilt:>2}° (compass_az={rep_az:>4.0f}°) → "
                f"{len(df):,} hours, annual={monthly_kwh.sum():.0f} kWh/m²/yr "
                f"[t={elapsed:.0f}s]")
        except Exception as e:
            log(f"  bin {i+1:>2}/{len(unique_bins)}: FAILED — "
                f"{type(e).__name__}: {e}")
            n_fails += 1
        time.sleep(args.delay)

    if n_fails > 0:
        log(f"WARNING: {n_fails}/{len(unique_bins)} PVGIS calls failed; "
            "affected faces will get NaN irradiance")

    # 5 ── Expand faces × hours, apply monthly scaling ───────────────────
    log("expanding bin profiles to per-face per-hour rows + monthly scaling")
    out_chunks: list[pd.DataFrame] = []

    for key, prof in bin_profiles.items():
        ori, tilt = key
        face_subset = faces.loc[(faces["orientation"] == ori)
                                & (faces["tilt_bin"] == tilt)].copy()
        n_faces = len(face_subset)
        if n_faces == 0:
            continue

        pvgis_monthly_for_bin = bin_monthly_pvgis[key]  # 12 values

        # For each face: compute 12 scale factors (one per month)
        # face_monthly[k] = SI_<month_k+1>_kwh_m2_mo  for k = 0..11
        face_monthly = face_subset[MONTHLY_COL_NAMES].values  # (n, 12)

        # scale_f,m = face_monthly[f,m] / pvgis_monthly_for_bin[m]
        # If face_monthly is NaN → scale 1.0 (fall back to raw PVGIS)
        # If pvgis_monthly is 0 (sun never reaches it that month, very unlikely
        #  at these latitudes) → scale 1.0 to avoid divide-by-zero
        denom = np.where(pvgis_monthly_for_bin > 1e-6,
                         pvgis_monthly_for_bin, 1.0)
        scale = np.where(np.isnan(face_monthly),
                         1.0,
                         face_monthly / denom)
        # clip pathological scales (sometimes SI_<m> > PVGIS for partially-
        # measured shaded faces; cap at 1.5x to keep things sane)
        scale = np.clip(scale, 0.0, 1.5)

        # Build per-face hourly profile by expanding the bin profile
        # We need: for each face × each hour, hourly_value × scale[month_of_hour]
        # Use month index of each hour to broadcast
        prof_month_idx = (prof["datetime"].dt.month - 1).values  # (8760,)
        prof_hourly_vals = prof["ghi_tilted_wm2"].values         # (8760,)

        n_hours = len(prof_hourly_vals)
        # build the (n_faces × n_hours) scaled matrix in one shot
        # scaled[f, h] = prof_hourly_vals[h] * scale[f, prof_month_idx[h]]
        # do via fancy indexing: scale[:, prof_month_idx] → (n_faces, n_hours)
        scaled = prof_hourly_vals[None, :] * scale[:, prof_month_idx]

        # Flatten to long format
        bld_arr  = np.repeat(face_subset["building_i"].values, n_hours)
        fidx_arr = np.repeat(face_subset["face_idx"].values, n_hours)
        slope_arr = np.repeat(face_subset["slope_deg"].values, n_hours)
        az_arr    = np.repeat(face_subset["azimuth_deg"].values, n_hours)
        hour_arr  = np.tile(np.arange(n_hours), n_faces)
        dt_arr    = np.tile(prof["datetime"].values, n_faces)

        chunk = pd.DataFrame({
            "building_i":      bld_arr,
            "face_idx":        fidx_arr,
            "orientation":     ori,
            "slope_deg":       slope_arr,
            "azimuth_deg":     az_arr,
            "tilt_bin":        tilt,
            "hour_of_year":    hour_arr,
            "datetime":        dt_arr,
            "ghi_tilted_wm2":  scaled.flatten(),
        })
        out_chunks.append(chunk)

    if not out_chunks:
        log("ERROR: no faces survived binning + PVGIS fetch — nothing to write")
        return 2

    log(f"concatenating {len(out_chunks):,} bin chunks")
    out = pd.concat(out_chunks, ignore_index=True)
    log(f"  total rows: {len(out):,}")

    # 6 ── Write parquet ─────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing {args.out}")
    out.to_parquet(args.out, engine="pyarrow", compression="snappy")
    log(f"  wrote {args.out.stat().st_size / 1e6:.1f} MB")

    # 7 ── Sanity check ──────────────────────────────────────────────────
    annual_kwh_m2 = (out.groupby(["building_i", "face_idx"])
                     ["ghi_tilted_wm2"].sum() / 1000.0)
    log("─" * 60)
    log("Stage H summary")
    log(f"  faces with hourly profile:       {annual_kwh_m2.shape[0]:,}")
    log(f"  median annual irradiance:        "
        f"{annual_kwh_m2.median():.0f} kWh/m²/yr")
    log(f"  min / max annual irradiance:     "
        f"{annual_kwh_m2.min():.0f} / {annual_kwh_m2.max():.0f} kWh/m²/yr")
    log(f"  median annual SI_Ann (DSM ref):  "
        f"{faces.get('si_ann_kwh_m2_yr', pd.Series([np.nan])).median():.0f} kWh/m²/yr"
        if "si_ann_kwh_m2_yr" in faces.columns else
        "  (no si_ann_kwh_m2_yr in face GPKG for cross-check)")
    log("─" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
