#!/usr/bin/env python3
"""
Stage K — Hourly self-consumption + annual SCI, SSI
====================================================

Combines per-building hourly PV production (Stage I) with hourly load
(Stage J) to compute:

  For each hour h, per building b:
      SC[b,h]      = min(P_PV[b,h], L[b,h])       (self-consumed)
      export[b,h]  = P_PV[b,h] - SC[b,h]          (sold to grid)
      import[b,h]  = L[b,h] - SC[b,h]             (bought from grid)

  Annual aggregates per building:
      annual_pv_kwh      = Σ_h P_PV
      annual_load_kwh    = Σ_h L
      annual_sc_kwh      = Σ_h SC
      annual_export_kwh  = Σ_h export
      annual_import_kwh  = Σ_h import
      sci_pct            = 100 · annual_sc / annual_pv
      ssi_pct            = 100 · annual_sc / annual_load

Definitions match Todeschi-Mutani 2021 and Usta-Mutani 2025.

Outputs
-------
1) Adds 7 new columns to buildings_cascade.csv:
       annual_pv_kwh, annual_load_kwh, annual_sc_kwh,
       annual_export_kwh, annual_import_kwh, sci_pct, ssi_pct
2) Optional: writes building_hourly_balance.parquet (full hourly time-series)
   for downstream Stage L plots.

Run
---
    python stage_K_self_consumption.py \\
        --pv-parquet     $OUT/building_hourly_production.parquet \\
        --load-parquet   $OUT/building_hourly_load.parquet \\
        --buildings-csv  $OUT/buildings_cascade.csv \\
        --out-csv        $OUT/buildings_cascade.csv \\
        --out-hourly     $OUT/building_hourly_balance.parquet
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Helpers ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pv-parquet", required=True, type=Path,
                   help="Stage I output: building_hourly_production.parquet")
    p.add_argument("--load-parquet", required=True, type=Path,
                   help="Stage J output: building_hourly_load.parquet")
    p.add_argument("--buildings-csv", required=True, type=Path,
                   help="Stage G+ output: per-building cascade CSV")
    p.add_argument("--out-csv", required=True, type=Path,
                   help="Output CSV (can be same as --buildings-csv)")
    p.add_argument("--out-hourly", type=Path, default=None,
                   help="Optional: write full hourly balance parquet here "
                        "(used by Stage L for daily/monthly figures)")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[stage_K] {msg}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # 1 ── Load hourly PV + load ──────────────────────────────────────────
    log(f"reading PV: {args.pv_parquet}")
    pv = pd.read_parquet(args.pv_parquet,
                         columns=["building_i", "hour_of_year",
                                  "datetime", "pv_kwh"])
    log(f"  {len(pv):,} rows, {pv['building_i'].nunique():,} buildings")

    log(f"reading load: {args.load_parquet}")
    ld = pd.read_parquet(args.load_parquet,
                         columns=["building_i", "hour_of_year",
                                  "datetime", "load_kwh"])
    log(f"  {len(ld):,} rows, {ld['building_i'].nunique():,} buildings")

    # 2 ── Outer join so we have rows for every (building, hour) that has
    # either PV or load. Buildings with no load (non-residential) won't have
    # a load row, so their load_kwh will be NaN → fill 0 (no consumption).
    # Buildings with no PV (no Layer 2 area) similarly get 0 production.
    log("joining PV and load on (building_i, hour_of_year, datetime)")
    bal = pv.merge(ld, on=["building_i", "hour_of_year", "datetime"],
                   how="outer")
    bal["pv_kwh"] = bal["pv_kwh"].fillna(0.0)
    bal["load_kwh"] = bal["load_kwh"].fillna(0.0)
    log(f"  {len(bal):,} rows after join")
    log(f"  {bal['building_i'].nunique():,} unique buildings")

    # 3 ── Hour-by-hour self-consumption ─────────────────────────────────
    log("computing hour-by-hour self-consumption, import, export")
    bal["sc_kwh"]     = np.minimum(bal["pv_kwh"], bal["load_kwh"])
    bal["export_kwh"] = bal["pv_kwh"] - bal["sc_kwh"]
    bal["import_kwh"] = bal["load_kwh"] - bal["sc_kwh"]

    # 4 ── Annual aggregates per building ────────────────────────────────
    log("aggregating per building")
    agg = (bal.groupby("building_i", observed=True)
              .agg(annual_pv_kwh     = ("pv_kwh",     "sum"),
                   annual_load_kwh   = ("load_kwh",   "sum"),
                   annual_sc_kwh     = ("sc_kwh",     "sum"),
                   annual_export_kwh = ("export_kwh", "sum"),
                   annual_import_kwh = ("import_kwh", "sum"))
              .reset_index())

    # SCI = sc / pv;  SSI = sc / load  — only where the denominator > 0
    agg["sci_pct"] = np.where(agg["annual_pv_kwh"] > 0,
                              100.0 * agg["annual_sc_kwh"]
                                    / agg["annual_pv_kwh"],
                              np.nan)
    agg["ssi_pct"] = np.where(agg["annual_load_kwh"] > 0,
                              100.0 * agg["annual_sc_kwh"]
                                    / agg["annual_load_kwh"],
                              np.nan)

    log(f"  {len(agg):,} buildings have an annual aggregate")
    log(f"  with PV > 0:   {int((agg['annual_pv_kwh'] > 0).sum()):,}")
    log(f"  with load > 0: {int((agg['annual_load_kwh'] > 0).sum()):,}")
    log(f"  with both > 0: "
        f"{int(((agg['annual_pv_kwh'] > 0) & (agg['annual_load_kwh'] > 0)).sum()):,}")

    # 5 ── Merge new columns into buildings_cascade.csv ──────────────────
    log(f"reading existing {args.buildings_csv}")
    bld = pd.read_csv(args.buildings_csv)

    # Drop any pre-existing versions of our new columns to avoid suffix mess
    drop_cols = [c for c in agg.columns if c != "building_i" and c in bld.columns]
    if drop_cols:
        log(f"  dropping pre-existing columns to refresh: {drop_cols}")
        bld = bld.drop(columns=drop_cols)

    bld = bld.merge(agg, on="building_i", how="left")
    log(f"  added 7 new columns to {len(bld):,} buildings")

    # 6 ── City-wide validation summary ──────────────────────────────────
    res = bld[(bld.get("is_residential", False) == True)
              & (bld["annual_pv_kwh"].fillna(0) > 0)
              & (bld["annual_load_kwh"].fillna(0) > 0)].copy()
    if len(res) == 0:
        log("WARNING: no residential building has both PV and load > 0")
    else:
        city_pv     = res["annual_pv_kwh"].sum()
        city_load   = res["annual_load_kwh"].sum()
        city_sc     = res["annual_sc_kwh"].sum()
        city_sci    = 100.0 * city_sc / city_pv if city_pv > 0 else 0.0
        city_ssi    = 100.0 * city_sc / city_load if city_load > 0 else 0.0
        med_sci     = res["sci_pct"].median()
        med_ssi     = res["ssi_pct"].median()

        log("─" * 60)
        log("Stage K summary — residential, both PV>0 and load>0")
        log(f"  buildings in scope:          {len(res):,}")
        log(f"  city-wide annual PV:         {city_pv/1e6:,.2f} GWh/yr")
        log(f"  city-wide annual load:       {city_load/1e6:,.2f} GWh/yr")
        log(f"  city-wide annual SC:         {city_sc/1e6:,.2f} GWh/yr")
        log(f"  ratio PV/load (sizing):      {city_pv/city_load:.2f}×")
        log(f"  city-wide SCI:               {city_sci:5.1f}%")
        log(f"  city-wide SSI:               {city_ssi:5.1f}%")
        log(f"  median per-building SCI:     {med_sci:5.1f}%")
        log(f"  median per-building SSI:     {med_ssi:5.1f}%")
        log(f"  ▶ Usta-Mutani 2025 PV reference (Table 9 Sc.9, eta 23%): SCI≈63.12%, SSI≈55.47%  [10/12 was their STC heating, not PV]")
        log("─" * 60)

    # 7 ── Write outputs ─────────────────────────────────────────────────
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    bld.to_csv(args.out_csv, index=False)
    log(f"wrote {args.out_csv}")

    if args.out_hourly is not None:
        log(f"writing hourly balance parquet: {args.out_hourly}")
        bal = bal[["building_i", "hour_of_year", "datetime",
                   "pv_kwh", "load_kwh", "sc_kwh",
                   "export_kwh", "import_kwh"]]
        bal.to_parquet(args.out_hourly,
                       engine="pyarrow", compression="snappy")
        log(f"  wrote {args.out_hourly.stat().st_size / 1e6:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
