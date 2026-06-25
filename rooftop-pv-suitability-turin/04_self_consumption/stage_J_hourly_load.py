#!/usr/bin/env python3
"""
Stage J — Hourly residential electric load (ARERA measured data)
=================================================================

Builds 8,760-hour load profiles per residential building using
real measured data from ARERA's "Prelievo medio orario" report
(Italian regulatory authority for energy):

  https://reporting.arera.it/SASVisualAnalytics/...

Filtered to:
  - Residenza = Residente   (resident, not vacation properties)
  - Three day types: Sabato (Sat), Domenica (Sun), Giorno feriale (weekday)
  - 24 hourly values for each day type, in kWh/customer/hour

Methodology
-----------
1. For each hour of the year, look up the matching day type:
     Monday-Friday → Giorno feriale
     Saturday      → Sabato
     Sunday        → Domenica
2. Use the corresponding hourly weight w[hour]
3. Build a 8,760-value array of unscaled weights, sum it
4. Normalise to a per-family annual reference (default: 2,700 kWh/year,
   ARERA cliente tipo). User can override via CLI.
5. Multiply by n_families per building.

Why we normalise instead of using ARERA's daily totals directly
--------------------------------------------------------------
ARERA reports the AVERAGE across all residential customers, which is
about 1,030 kWh/year/customer (after recent efficiency improvements).
The Usta-Mutani 2025 paper uses 2,700 kWh/year/family (cliente tipo
reference — 4-person household, 3 kW connection) for SCI/SSI
benchmarking. We default to 2,700 for fair comparison against their
published PV scenario (Usta-Mutani 2025, Table 9, Scenario 9: SCI ~63.12%, SSI ~55.47%), but the --annual-kwh-per-family flag lets
you run scenarios with the ARERA-measured 1,030 number too.

Output
------
Parquet: building_hourly_load.parquet
    building_i, hour_of_year, datetime, load_kwh

Run
---
    python stage_J_hourly_load.py \\
        --buildings-csv     $OUT/buildings_cascade.csv \\
        --datetime-source   $OUT/face_hourly_irradiation.parquet \\
        --out               $OUT/building_hourly_load.parquet \\
        --annual-kwh-per-family 2700
"""
from __future__ import annotations
import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)  # noqa: A001


# ─── ARERA hourly weights — Torino residenti ──────────────────────────────
# Extracted from ARERA "Prelievo medio orario (kWh)" dashboard.
# Source: https://reporting.arera.it/SASVisualAnalytics/...
# Columns: hour H1..H24 (ARERA convention: H1 = 00:00-01:00, H24 = 23:00-00:00)
# Values are kWh per residential customer per hour.

# H1..H24 → index 0..23 (Python convention)
ARERA_WEEKDAY = np.array([
    0.0960, 0.0809, 0.0732, 0.0697, 0.0696, 0.0752, 0.0923, 0.1121,  # H1-H8
    0.1127, 0.1111, 0.1090, 0.1125, 0.1199, 0.1208, 0.1169, 0.1135,  # H9-H16
    0.1144, 0.1262, 0.1493, 0.1748, 0.1778, 0.1680, 0.1458, 0.1198,  # H17-H24
])
ARERA_SATURDAY = np.array([
    0.1005, 0.0849, 0.0762, 0.0719, 0.0708, 0.0745, 0.0841, 0.1021,
    0.1222, 0.1335, 0.1344, 0.1376, 0.1462, 0.1466, 0.1424, 0.1350,
    0.1332, 0.1409, 0.1586, 0.1779, 0.1728, 0.1585, 0.1401, 0.1208,
])
ARERA_SUNDAY = np.array([
    0.1032, 0.0883, 0.0775, 0.0737, 0.0718, 0.0739, 0.0804, 0.0940,
    0.1159, 0.1352, 0.1448, 0.1528, 0.1566, 0.1459, 0.1379, 0.1306,
    0.1290, 0.1369, 0.1551, 0.1753, 0.1754, 0.1645, 0.1437, 0.1191,
])


# ─── Monthly modulation (placeholder until ARERA monthly is extracted) ────
# ARERA dashboard has a "Prelievo medio mensile" chart but it was not yet
# extracted. Until then we use a small monthly modulation reflecting the
# canonical Italian residential pattern (winter peak from lighting, summer
# peak from AC). Annual mean is normalised to 1.0 so it never changes
# the annual total.

NORMALISED_MONTHLY_FACTORS = np.array([
    1.10,  # Jan
    1.08,  # Feb
    1.00,  # Mar
    0.92,  # Apr
    0.88,  # May
    0.95,  # Jun
    1.10,  # Jul (AC season)
    1.08,  # Aug
    0.95,  # Sep
    0.92,  # Oct
    1.00,  # Nov
    1.10,  # Dec
])
NORMALISED_MONTHLY_FACTORS /= NORMALISED_MONTHLY_FACTORS.mean()


# ─── Helpers ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buildings-csv", required=True, type=Path,
                   help="Stage G output: per-building CSV with n_families_building")
    p.add_argument("--datetime-source", required=True, type=Path,
                   help="Parquet with the canonical 8760-hour datetime index "
                        "(use Stage H's face_hourly_irradiation.parquet)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--annual-kwh-per-family", type=float, default=2700.0,
                   help="Annual kWh per family to normalise the profile to. "
                        "Default 2700 (ARERA cliente tipo / Usta-Mutani 2025 ref). "
                        "Use 1030 for ARERA measured average per customer.")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[stage_J] {msg}")


def build_yearly_profile_kwh(annual_kwh: float,
                             hours: pd.DatetimeIndex) -> np.ndarray:
    """
    Build an 8,760-value array of hourly kWh whose sum equals annual_kwh,
    using ARERA day-type hourly shapes × monthly modulation.

    Day-type mapping:
      Mon-Fri → ARERA_WEEKDAY
      Saturday → ARERA_SATURDAY
      Sunday → ARERA_SUNDAY
    """
    hour_of_day = hours.hour.values     # 0..23
    weekday = hours.weekday.values      # 0=Mon, 5=Sat, 6=Sun
    month_idx = hours.month.values - 1  # 0..11

    # Pick ARERA hourly weight per (day_type, hour_of_day)
    weights = np.empty(len(hours), dtype=float)
    is_sat = weekday == 5
    is_sun = weekday == 6
    is_wd  = ~(is_sat | is_sun)
    weights[is_wd]  = ARERA_WEEKDAY[hour_of_day[is_wd]]
    weights[is_sat] = ARERA_SATURDAY[hour_of_day[is_sat]]
    weights[is_sun] = ARERA_SUNDAY[hour_of_day[is_sun]]

    # Apply monthly modulation
    weights *= NORMALISED_MONTHLY_FACTORS[month_idx]

    # Normalise so the annual sum equals annual_kwh
    scaled = weights * (annual_kwh / weights.sum())
    return scaled


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    log(f"annual reference: {args.annual_kwh_per_family} kWh/family/year")
    log("hourly shape source: ARERA 'Prelievo medio orario' "
        "(Residente, 3 day types)")

    # 1 ── Load buildings + n_families ───────────────────────────────────
    log(f"reading {args.buildings_csv}")
    bld = pd.read_csv(args.buildings_csv)
    log(f"  {len(bld):,} buildings in input")

    if "n_families_building" not in bld.columns:
        log("ERROR: buildings_csv missing n_families_building (run Stage G first)")
        return 1
    if "is_residential" not in bld.columns:
        log("ERROR: buildings_csv missing is_residential")
        return 1

    res = bld.loc[bld["is_residential"].fillna(False)
                  & bld["n_families_building"].fillna(0).gt(0),
                  ["building_i", "n_families_building"]].copy()
    log(f"  {len(res):,} residential buildings with n_families > 0")
    log(f"  total families: {int(res['n_families_building'].sum()):,}")

    # 2 ── Get canonical 8760-hour datetime index ────────────────────────
    log(f"reading datetime index from {args.datetime_source}")
    src = pd.read_parquet(args.datetime_source,
                          columns=["building_i", "hour_of_year", "datetime"])
    canon = (src.drop_duplicates("hour_of_year")
                .sort_values("hour_of_year")
                .reset_index(drop=True))
    log(f"  canonical timeline has {len(canon):,} hours")
    if len(canon) not in (8760, 8784):
        log(f"  WARNING: expected 8760 or 8784 hours, got {len(canon)}")
    dt_index = pd.DatetimeIndex(canon["datetime"])

    # 3 ── Build the per-family yearly load profile ──────────────────────
    log("building per-family yearly load profile from ARERA shapes")
    one_family = build_yearly_profile_kwh(args.annual_kwh_per_family, dt_index)

    # Diagnostics: report the implied daily totals per day-type
    daily_sums = pd.Series(one_family).groupby(
        np.repeat(np.arange(len(one_family) // 24 + 1), 24)[:len(one_family)]
    ).sum()
    avg_daily = daily_sums.mean()
    log(f"  annual total per family: {one_family.sum():,.0f} kWh "
        f"(should equal {args.annual_kwh_per_family})")
    log(f"  daily-avg per family:    {avg_daily:.2f} kWh/day")
    log(f"  hourly peak (per fam):   {one_family.max():.4f} kWh @ "
        f"hour-of-day {one_family.argmax() % 24}")
    log(f"  hourly trough (per fam): {one_family.min():.4f} kWh @ "
        f"hour-of-day {one_family.argmin() % 24}")

    # 4 ── Multiply by n_families per building, chunk-wise ───────────────
    log("scaling by n_families and assembling per-building × hour rows")
    nfam = res["n_families_building"].astype(float).values
    bldids = res["building_i"].values

    chunk_size = 500
    out_chunks: list[pd.DataFrame] = []
    for start in range(0, len(res), chunk_size):
        end = min(start + chunk_size, len(res))
        nf_chunk = nfam[start:end][:, None]                 # (chunk, 1)
        load_matrix = nf_chunk * one_family[None, :]        # (chunk, 8760)

        c = pd.DataFrame({
            "building_i":   np.repeat(bldids[start:end], len(dt_index)),
            "hour_of_year": np.tile(canon["hour_of_year"].values, end - start),
            "datetime":     np.tile(dt_index.values, end - start),
            "load_kwh":     load_matrix.flatten(),
        })
        out_chunks.append(c)
        if (start // chunk_size) % 4 == 0:
            log(f"  built rows for buildings {start:,}..{end:,}")

    out = pd.concat(out_chunks, ignore_index=True)
    log(f"  total rows: {len(out):,}")

    # 5 ── Sanity check + write ──────────────────────────────────────────
    annual_per_bld = out.groupby("building_i")["load_kwh"].sum()
    expected = (res.set_index("building_i")["n_families_building"].astype(float)
                * args.annual_kwh_per_family)
    diff = (annual_per_bld - expected).abs()
    max_diff = diff.max() if len(diff) else 0.0
    log("─" * 60)
    log("Stage J summary")
    log(f"  city-wide annual residential load: "
        f"{annual_per_bld.sum() / 1e6:,.2f} GWh/yr")
    log(f"  max per-building deviation from expected: {max_diff:.2f} kWh "
        "(should be ≈ 0)")
    log(f"  buildings with load > 0:           {len(annual_per_bld):,}")
    log("─" * 60)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing {args.out}")
    out = out[["building_i", "hour_of_year", "datetime", "load_kwh"]]
    out.to_parquet(args.out, engine="pyarrow", compression="snappy")
    log(f"  wrote {args.out.stat().st_size / 1e6:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
