#!/usr/bin/env python3
"""
Stage L — SCI/SSI comparison report + figures (v2)
==================================================

Produces figures for the supervisor meeting:

  fig_L0_monthly_irradiation.png        — the chart Mutani drew on page 1
                                          (monthly kWh/m²/month by orientation)
  fig_L1_sci_histogram.png              — per-building SCI histogram + reference
  fig_L2_ssi_histogram.png              — per-building SSI histogram + reference
  fig_L3_duck_curve_typical_day.png     — THE chart Mutani drew on page 2
                                          (hourly PV vs load, summer vs winter)
  fig_L4_sci_by_orientation.png         — SCI breakdown by face orientation
  fig_L5_AvsB_summary.png               — scenario A (all faces) vs B (south only)

Plus written outputs:
  sci_ssi_summary.csv                   — headline numbers, both scenarios
  comparison_report.md                  — readable Markdown summary

Scenarios
---------
A — "all faces": every face's PV counts, regardless of orientation
B — "south-facing only": only faces with orientation ∈ {S, SE, SW, flat}

Both scenarios are computed automatically. The PV side of B is the subset
of A with non-south faces zeroed out.

Run
---
    python stage_L_comparison_report.py \\
        --buildings-csv      $OUT/buildings_cascade.csv \\
        --hourly-balance     $OUT/building_hourly_balance.parquet \\
        --face-production    $OUT/face_hourly_production.parquet \\
        --face-monthly-irrad $OUT/face_monthly_irradiation.csv \\
        --face-layers        $OUT/lod22_face_four_layers.gpkg \\
        --out-dir            $OUT/figures_sci_ssi/
"""
from __future__ import annotations
import argparse
import functools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

print = functools.partial(print, flush=True)  # noqa: A001


# ─── Constants ────────────────────────────────────────────────────────────

USTA_MUTANI_2025_SCI = 63.12   # PV electricity, residential whole-city (Usta-Mutani 2025, Table 9, Scenario 9, eta 23%)
USTA_MUTANI_2025_SSI = 55.47   # PV electricity, residential whole-city (Usta-Mutani 2025, Table 9, Scenario 9, eta 23%)

SOUTH_FACING_ORIENTATIONS = {"S", "SE", "SW", "flat"}

# Months for the monthly chart
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHLY_COL_NAMES = [f"si_{m.lower()}_kwh_m2_mo" for m in MONTHS]

# Plot palette
PALETTE = {
    "pv":      "#D9A93F",   # gold
    "load":    "#7FA9D6",   # blue
    "sc":      "#5BA85B",   # green
    "ref":     "#C46060",   # red
    "hist":    "#1E3A5F",   # navy
    "muted":   "#777777",
}
ORIENTATION_COLORS = {
    "S":    "#D9A93F",   # gold
    "SE":   "#F7B267",   # orange
    "SW":   "#C46060",   # red
    "E":    "#A285C9",   # purple
    "W":    "#8DA9C4",   # blue-grey
    "N":    "#7BB07B",   # green
    "flat": "#B0B0B0",   # grey
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buildings-csv",      required=True, type=Path)
    p.add_argument("--hourly-balance",     required=True, type=Path,
                   help="Stage K hourly balance parquet")
    p.add_argument("--face-production",    required=True, type=Path,
                   help="Stage I per-face hourly production parquet")
    p.add_argument("--face-monthly-irrad", required=True, type=Path,
                   help="Stage B2 per-face monthly irradiation CSV")
    p.add_argument("--face-layers",        required=True, type=Path,
                   help="Stage C face_four_layers.gpkg (for face geometry + orientation linkage)")
    p.add_argument("--out-dir",            required=True, type=Path)
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[stage_L] {msg}")


def setup_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=PALETTE["hist"], pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─── fig_L0: monthly irradiation by orientation (Mutani page 1) ───────────

def fig_monthly_irradiation(face_monthly: pd.DataFrame,
                            face_orient: pd.DataFrame,
                            out_path: Path) -> None:
    """
    The chart Mutani drew on page 1 of her notes: 12 months on x-axis,
    one line per orientation, showing average kWh/m²/month per face.
    """
    log("  fig_L0: monthly irradiation by orientation")
    # Join monthly + orientation
    df = face_monthly.merge(face_orient, on=["building_i", "face_idx"],
                            how="inner")
    log(f"    {len(df):,} faces with both monthly + orientation")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot one curve per orientation, ordered S → flat
    plotted_orientations = []
    order = ["S", "SE", "SW", "E", "W", "N", "flat"]
    for ori in order:
        sub = df[df["orientation"] == ori]
        if len(sub) == 0:
            continue
        # Mean across all faces with this orientation, per month
        monthly_means = sub[MONTHLY_COL_NAMES].mean(axis=0).values
        if np.isnan(monthly_means).all():
            continue
        n_faces = len(sub)
        ax.plot(MONTHS, monthly_means,
                color=ORIENTATION_COLORS.get(ori, PALETTE["muted"]),
                linewidth=2.2, marker="o", markersize=5,
                label=f"{ori}  (n={n_faces})")
        plotted_orientations.append(ori)

    setup_axes(ax,
               "Monthly solar irradiation by roof orientation (DSM-measured)",
               "Month",
               "Average kWh / m² / month per face")
    ax.legend(frameon=False, loc="upper right", fontsize=10,
              title="Orientation")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"    wrote {out_path.name}  (orientations plotted: {plotted_orientations})")


# ─── fig_L1: SCI histogram + reference line ──────────────────────────────

def fig_sci_histogram(df_res: pd.DataFrame, out_path: Path,
                      scenario_label: str) -> None:
    sci = df_res["sci_pct"].dropna()
    if len(sci) == 0:
        log(f"  fig_L1: no SCI data for {scenario_label}, skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(sci, bins=40, color=PALETTE["hist"], alpha=0.85, edgecolor="white")
    ax.axvline(USTA_MUTANI_2025_SCI, color=PALETTE["ref"], linewidth=2.2,
               linestyle="--",
               label=f"Usta-Mutani 2025 PV, whole-city (SCI = {USTA_MUTANI_2025_SCI}%)")
    ax.axvline(sci.median(), color=PALETTE["sc"], linewidth=2.0,
               label=f"Our median (SCI = {sci.median():.1f}%)")
    setup_axes(ax,
               f"Self-Consumption Index per residential building — {scenario_label}",
               "SCI (%)  =  100 × annual self-consumed PV / annual PV production",
               "Number of buildings")
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"  wrote {out_path.name}")


# ─── fig_L2: SSI histogram + reference line ──────────────────────────────

def fig_ssi_histogram(df_res: pd.DataFrame, out_path: Path,
                      scenario_label: str) -> None:
    ssi = df_res["ssi_pct"].dropna()
    if len(ssi) == 0:
        log(f"  fig_L2: no SSI data for {scenario_label}, skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ssi, bins=40, color=PALETTE["sc"], alpha=0.85, edgecolor="white")
    ax.axvline(USTA_MUTANI_2025_SSI, color=PALETTE["ref"], linewidth=2.2,
               linestyle="--",
               label=f"Usta-Mutani 2025 PV, whole-city (SSI = {USTA_MUTANI_2025_SSI}%)")
    ax.axvline(ssi.median(), color=PALETTE["hist"], linewidth=2.0,
               label=f"Our median (SSI = {ssi.median():.1f}%)")
    setup_axes(ax,
               f"Self-Sufficiency Index per residential building — {scenario_label}",
               "SSI (%)  =  100 × annual self-consumed PV / annual load",
               "Number of buildings")
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"  wrote {out_path.name}")


# ─── fig_L3: duck curves (Mutani page 2) ──────────────────────────────────

def fig_duck_curve_typical_day(bal: pd.DataFrame, res_ids: set,
                               out_path: Path,
                               scenario_label: str) -> None:
    """
    Hourly PV vs load for typical summer day and typical winter day.
    Two panels side by side. Self-consumption (overlap) shaded green.
    """
    log(f"  fig_L3: duck-curve for {scenario_label}")
    bal_res = bal[bal["building_i"].isin(res_ids)].copy()
    if len(bal_res) == 0:
        log(f"  fig_L3: no residential hourly data, skipping")
        return
    bal_res["month"] = bal_res["datetime"].dt.month
    bal_res["hour"] = bal_res["datetime"].dt.hour

    # Summer = July (month 7); winter = January (month 1)
    panels = [(1, "Typical winter day (January)"),
              (7, "Typical summer day (July)")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (month, title) in zip(axes, panels):
        sub = bal_res[bal_res["month"] == month]
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        # Average across all hours of that month
        monthly_hourly = (sub.groupby("hour")[["pv_kwh", "load_kwh", "sc_kwh"]]
                            .mean()
                            .reset_index())
        ax.fill_between(monthly_hourly["hour"], 0, monthly_hourly["pv_kwh"],
                        color=PALETTE["pv"], alpha=0.50,
                        label="PV production")
        ax.plot(monthly_hourly["hour"], monthly_hourly["load_kwh"],
                color=PALETTE["load"], linewidth=2.4, label="Load (consumption)")
        ax.fill_between(monthly_hourly["hour"], 0, monthly_hourly["sc_kwh"],
                        color=PALETTE["sc"], alpha=0.65,
                        label="Self-consumed (overlap)")
        setup_axes(ax, title,
                   "Hour of day",
                   "Average kWh / hour  (per building)")
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlim(0, 23)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"City-wide average hourly PV vs load — {scenario_label}",
                 fontsize=14, fontweight="bold",
                 color=PALETTE["hist"], y=1.07)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"  wrote {out_path.name}")


# ─── fig_L4: SCI breakdown by face orientation ─────────────────────────────

def fig_sci_by_orientation(face_prod: pd.DataFrame,
                           bal: pd.DataFrame,
                           bld: pd.DataFrame,
                           out_path: Path) -> None:
    """
    For each face orientation, compute its contribution to the city-wide
    total and its associated SCI when only that orientation produces.
    """
    log("  fig_L4: SCI/PV breakdown by face orientation")

    # Sum production per orientation
    by_ori = (face_prod.groupby("orientation")["pv_kwh"]
              .sum()
              .sort_values(ascending=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: annual PV by orientation (bar)
    ax = axes[0]
    orientations = by_ori.index.tolist()
    values_gwh = (by_ori.values / 1e6)
    colors = [ORIENTATION_COLORS.get(o, PALETTE["muted"]) for o in orientations]
    ax.bar(orientations, values_gwh, color=colors, alpha=0.85,
           edgecolor=PALETTE["hist"])
    for i, v in enumerate(values_gwh):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom",
                fontsize=10, color=PALETTE["hist"])
    setup_axes(ax,
               "Annual gross PV production by face orientation",
               "Face orientation",
               "Annual PV (GWh/yr)")
    ax.set_ylim(bottom=0)

    # Panel 2: share of total
    ax = axes[1]
    total = by_ori.sum()
    pct = (by_ori / total) * 100.0
    ax.barh(orientations, pct.values, color=colors, alpha=0.85,
            edgecolor=PALETTE["hist"])
    for i, v in enumerate(pct.values):
        ax.text(v, i, f"  {v:.1f}%", va="center",
                fontsize=10, color=PALETTE["hist"])
    setup_axes(ax,
               "Share of city-wide PV total by orientation",
               "Share of total annual PV (%)",
               "Face orientation")
    ax.invert_yaxis()
    ax.set_xlim(0, max(pct.values) * 1.18)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"  wrote {out_path.name}")


# ─── fig_L5: scenario A vs B summary ──────────────────────────────────────

def fig_a_vs_b_summary(summary_a: dict, summary_b: dict,
                       out_path: Path) -> None:
    """Side-by-side bar comparison of the two scenarios."""
    log("  fig_L5: scenario A vs B summary")
    metrics = [
        ("PV (GWh)",  "city_pv_gwh"),
        ("Load (GWh)", "city_load_gwh"),
        ("SC (GWh)",  "city_sc_gwh"),
        ("SCI (%)",   "city_sci_pct"),
        ("SSI (%)",   "city_ssi_pct"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for ax, (label, key) in zip(axes, metrics):
        a = summary_a.get(key, 0) or 0
        b = summary_b.get(key, 0) or 0
        bars = ax.bar(["all faces", "south only"], [a, b],
                      color=[PALETTE["hist"], PALETTE["sc"]],
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, [a, b]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height(), f"{v:.1f}",
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                    color=PALETTE["hist"])
        # reference line if it's an index metric
        if "sci" in key:
            ax.axhline(USTA_MUTANI_2025_SCI, color=PALETTE["ref"],
                       linestyle="--", linewidth=1.5, alpha=0.6,
                       label=f"ref {USTA_MUTANI_2025_SCI}%")
            ax.legend(frameon=False, fontsize=9, loc="upper right")
        elif "ssi" in key:
            ax.axhline(USTA_MUTANI_2025_SSI, color=PALETTE["ref"],
                       linestyle="--", linewidth=1.5, alpha=0.6,
                       label=f"ref {USTA_MUTANI_2025_SSI}%")
            ax.legend(frameon=False, fontsize=9, loc="upper right")
        ax.set_title(label, fontsize=11, fontweight="bold",
                     color=PALETTE["hist"])
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)

    fig.suptitle("Scenario A (all faces) vs B (south-facing only)",
                 fontsize=14, fontweight="bold",
                 color=PALETTE["hist"], y=1.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"  wrote {out_path.name}")


# ─── Summary writers ──────────────────────────────────────────────────────

def compute_summary(df_res: pd.DataFrame) -> dict:
    """Compute headline numbers for one scenario."""
    if len(df_res) == 0:
        return dict(n_residential_buildings=0,
                    n_families_total=0,
                    city_pv_gwh=0.0, city_load_gwh=0.0, city_sc_gwh=0.0,
                    city_sci_pct=None, city_ssi_pct=None,
                    median_sci_pct=None, median_ssi_pct=None,
                    p25_sci_pct=None, p75_sci_pct=None,
                    p25_ssi_pct=None, p75_ssi_pct=None,
                    usta_mutani_2025_sci=USTA_MUTANI_2025_SCI,
                    usta_mutani_2025_ssi=USTA_MUTANI_2025_SSI)

    city_pv   = df_res["annual_pv_kwh"].sum()
    city_load = df_res["annual_load_kwh"].sum()
    city_sc   = df_res["annual_sc_kwh"].sum()
    return dict(
        n_residential_buildings = int(len(df_res)),
        n_families_total        = int(df_res["n_families_building"].fillna(0).sum()),
        city_pv_gwh             = round(city_pv / 1e6, 3),
        city_load_gwh           = round(city_load / 1e6, 3),
        city_sc_gwh             = round(city_sc / 1e6, 3),
        city_sci_pct            = round(100.0 * city_sc / city_pv, 2) if city_pv > 0 else None,
        city_ssi_pct            = round(100.0 * city_sc / city_load, 2) if city_load > 0 else None,
        median_sci_pct          = round(float(df_res["sci_pct"].median()), 2),
        median_ssi_pct          = round(float(df_res["ssi_pct"].median()), 2),
        p25_sci_pct             = round(float(df_res["sci_pct"].quantile(0.25)), 2),
        p75_sci_pct             = round(float(df_res["sci_pct"].quantile(0.75)), 2),
        p25_ssi_pct             = round(float(df_res["ssi_pct"].quantile(0.25)), 2),
        p75_ssi_pct             = round(float(df_res["ssi_pct"].quantile(0.75)), 2),
        usta_mutani_2025_sci    = USTA_MUTANI_2025_SCI,
        usta_mutani_2025_ssi    = USTA_MUTANI_2025_SSI,
    )


def write_comparison_markdown(summary_a: dict, summary_b: dict,
                              out_path: Path) -> None:
    body = f"""# SCI / SSI comparison report — Turin centro storico

Two scenarios reported:
- **A — all faces**: every face's PV counts toward the building total
- **B — south-facing only**: only faces with orientation S/SE/SW/flat contribute

## Methodology

- **Hourly irradiance**: PVGIS v5.2 TMY hourly profile (Turin, ~45.07°N)
  × per-month rescale to match our DSM-measured `SI_<month>` totals.
  Faces shaded by neighbouring buildings get lower production than PVGIS
  would suggest on its own.
- **PV production**: `P = PR · H · A · η` with PR=0.80, η=0.24,
  A=`layer2_suitable_m2` (classifier-corrected). Per-face, per-hour.
- **Load**: ARERA "Prelievo medio orario" Italian residential hourly data
  (three day-types: weekday, Saturday, Sunday). Normalised to
  2,700 kWh/family/year (ARERA cliente tipo, matches Usta-Mutani 2025).
- **Families per building**: ISTAT 2021 FAM21 disaggregated per
  census block by building volume.

## Headline comparison

| Quantity                      | A (all faces)              | B (south only)             |
|-------------------------------|----------------------------|----------------------------|
| Residential buildings         | {summary_a['n_residential_buildings']:,} | {summary_b['n_residential_buildings']:,} |
| Total families                | {summary_a['n_families_total']:,}        | {summary_b['n_families_total']:,}        |
| Annual PV (GWh/yr)            | {summary_a['city_pv_gwh']:.2f}           | {summary_b['city_pv_gwh']:.2f}           |
| Annual load (GWh/yr)          | {summary_a['city_load_gwh']:.2f}         | {summary_b['city_load_gwh']:.2f}         |
| Annual SC (GWh/yr)            | {summary_a['city_sc_gwh']:.2f}           | {summary_b['city_sc_gwh']:.2f}           |
| **City-wide SCI**             | **{summary_a['city_sci_pct']}%**         | **{summary_b['city_sci_pct']}%**         |
| **City-wide SSI**             | **{summary_a['city_ssi_pct']}%**         | **{summary_b['city_ssi_pct']}%**         |
| Median per-building SCI       | {summary_a['median_sci_pct']}% (IQR {summary_a['p25_sci_pct']}–{summary_a['p75_sci_pct']}) | {summary_b['median_sci_pct']}% (IQR {summary_b['p25_sci_pct']}–{summary_b['p75_sci_pct']}) |
| Median per-building SSI       | {summary_a['median_ssi_pct']}% (IQR {summary_a['p25_ssi_pct']}–{summary_a['p75_ssi_pct']}) | {summary_b['median_ssi_pct']}% (IQR {summary_b['p25_ssi_pct']}–{summary_b['p75_ssi_pct']}) |

## Comparison vs Usta-Mutani 2025

Their PV electricity scenario (Usta-Mutani 2025, Table 9, Scenario 9, eta 23%), whole-city residential:
- SCI ≈ {USTA_MUTANI_2025_SCI}%
- SSI ≈ {USTA_MUTANI_2025_SSI}%

| Index | A (all) delta | B (south) delta |
|-------|---------------|-----------------|
| SCI   | {(summary_a['city_sci_pct'] or 0) - USTA_MUTANI_2025_SCI:+.1f} pp | {(summary_b['city_sci_pct'] or 0) - USTA_MUTANI_2025_SCI:+.1f} pp |
| SSI   | {(summary_a['city_ssi_pct'] or 0) - USTA_MUTANI_2025_SSI:+.1f} pp | {(summary_b['city_ssi_pct'] or 0) - USTA_MUTANI_2025_SSI:+.1f} pp |

## Caveats

- Monthly load modulation uses placeholder factors (Stage J docstring).
  When ARERA monthly chart is extracted, replace
  `NORMALISED_MONTHLY_FACTORS` and re-run from Stage J.
- Heritage exclusion (D.Lgs. 42/2004) not applied to Layer 2.
"""
    out_path.write_text(body)


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1 ── Load per-building summary ─────────────────────────────────────
    log(f"reading {args.buildings_csv}")
    bld = pd.read_csv(args.buildings_csv)
    needed = {"is_residential", "sci_pct", "ssi_pct",
              "annual_pv_kwh", "annual_load_kwh", "annual_sc_kwh",
              "n_families_building"}
    missing = needed - set(bld.columns)
    if missing:
        log(f"ERROR: buildings_csv missing columns {missing} (run Stage K first)")
        return 1

    res_a = bld[bld["is_residential"].fillna(False)
                & bld["sci_pct"].notna()
                & bld["ssi_pct"].notna()].copy()
    log(f"  {len(res_a):,} residential buildings with valid SCI/SSI")
    if len(res_a) == 0:
        log("ERROR: no residential buildings with valid indices")
        return 2

    # 2 ── Load hourly balance, per-face production, monthly irrad ───────
    log(f"reading {args.hourly_balance}")
    bal = pd.read_parquet(args.hourly_balance,
                          columns=["building_i", "datetime",
                                   "pv_kwh", "load_kwh", "sc_kwh"])
    log(f"  {len(bal):,} hourly rows")

    log(f"reading {args.face_production}")
    face_prod = pd.read_parquet(args.face_production,
                                columns=["building_i", "face_idx",
                                         "orientation", "datetime",
                                         "pv_kwh"])
    log(f"  {len(face_prod):,} face × hour rows")

    log(f"reading {args.face_monthly_irrad}")
    face_monthly = pd.read_csv(args.face_monthly_irrad)
    log(f"  {len(face_monthly):,} faces with monthly DSM data")

    # Per-face orientation (one row per face)
    face_orient = (face_prod[["building_i", "face_idx", "orientation"]]
                   .drop_duplicates()
                   .reset_index(drop=True))
    log(f"  {len(face_orient):,} unique faces with orientation tag")

    res_ids = set(res_a["building_i"])

    # 3 ── Scenario A — figures ──────────────────────────────────────────
    log("─" * 60)
    log("SCENARIO A — all faces")
    log("─" * 60)
    dir_a = args.out_dir / "all_faces"
    dir_a.mkdir(exist_ok=True)
    fig_monthly_irradiation(face_monthly, face_orient,
                            dir_a / "fig_L0_monthly_irradiation.png")
    fig_sci_histogram(res_a, dir_a / "fig_L1_sci_histogram.png",
                      "scenario A: all faces")
    fig_ssi_histogram(res_a, dir_a / "fig_L2_ssi_histogram.png",
                      "scenario A: all faces")
    fig_duck_curve_typical_day(bal, res_ids,
                               dir_a / "fig_L3_duck_curve_typical_day.png",
                               "scenario A: all faces")
    fig_sci_by_orientation(face_prod, bal, bld,
                           dir_a / "fig_L4_sci_by_orientation.png")
    summary_a = compute_summary(res_a)
    pd.DataFrame([summary_a]).to_csv(dir_a / "sci_ssi_summary.csv",
                                     index=False)

    # 4 ── Scenario B — south-facing only ────────────────────────────────
    log("─" * 60)
    log("SCENARIO B — south-facing only (S, SE, SW, flat)")
    log("─" * 60)
    # Filter face_prod to south-facing
    south_face_keys = face_orient[
        face_orient["orientation"].isin(SOUTH_FACING_ORIENTATIONS)
    ][["building_i", "face_idx"]]
    log(f"  south-facing faces: {len(south_face_keys):,}")

    south_prod = face_prod.merge(south_face_keys,
                                 on=["building_i", "face_idx"],
                                 how="inner")
    log(f"  south-facing hourly rows: {len(south_prod):,}")

    # Recompute per-building hourly PV in scenario B
    log("  re-aggregating hourly PV with south-only filter")
    bld_pv_b = (south_prod.groupby(["building_i", "datetime"], observed=True)
                ["pv_kwh"].sum().reset_index())

    # Re-join with original load (load doesn't change)
    log("  re-computing self-consumption with new PV side")
    load = bal[["building_i", "datetime", "load_kwh"]]
    bal_b = bld_pv_b.merge(load, on=["building_i", "datetime"], how="outer")
    bal_b["pv_kwh"] = bal_b["pv_kwh"].fillna(0.0)
    bal_b["load_kwh"] = bal_b["load_kwh"].fillna(0.0)
    bal_b["sc_kwh"] = np.minimum(bal_b["pv_kwh"], bal_b["load_kwh"])

    # Per-building annual sums
    agg_b = (bal_b.groupby("building_i", observed=True)
                .agg(annual_pv_kwh=("pv_kwh", "sum"),
                     annual_load_kwh=("load_kwh", "sum"),
                     annual_sc_kwh=("sc_kwh", "sum"))
                .reset_index())
    agg_b["sci_pct"] = np.where(agg_b["annual_pv_kwh"] > 0,
                                100.0 * agg_b["annual_sc_kwh"]
                                       / agg_b["annual_pv_kwh"],
                                np.nan)
    agg_b["ssi_pct"] = np.where(agg_b["annual_load_kwh"] > 0,
                                100.0 * agg_b["annual_sc_kwh"]
                                       / agg_b["annual_load_kwh"],
                                np.nan)

    # Merge with building-level (residential + n_families)
    res_b = (bld[bld["is_residential"].fillna(False)
                 & bld["building_i"].isin(res_ids)]
             [["building_i", "n_families_building"]]
             .merge(agg_b, on="building_i", how="left"))
    res_b = res_b.dropna(subset=["sci_pct", "ssi_pct"])
    log(f"  {len(res_b):,} residential buildings in scenario B")

    dir_b = args.out_dir / "south_only"
    dir_b.mkdir(exist_ok=True)
    fig_sci_histogram(res_b, dir_b / "fig_L1_sci_histogram.png",
                      "scenario B: south only")
    fig_ssi_histogram(res_b, dir_b / "fig_L2_ssi_histogram.png",
                      "scenario B: south only")
    fig_duck_curve_typical_day(bal_b, res_ids,
                               dir_b / "fig_L3_duck_curve_typical_day.png",
                               "scenario B: south only")
    summary_b = compute_summary(res_b)
    pd.DataFrame([summary_b]).to_csv(dir_b / "sci_ssi_summary.csv",
                                     index=False)

    # 5 ── Cross-scenario comparison ─────────────────────────────────────
    log("─" * 60)
    log("CROSS-SCENARIO comparison figure + report")
    log("─" * 60)
    fig_a_vs_b_summary(summary_a, summary_b,
                       args.out_dir / "fig_L5_AvsB_summary.png")
    write_comparison_markdown(summary_a, summary_b,
                              args.out_dir / "comparison_report.md")

    # 6 ── Console summary ────────────────────────────────────────────────
    log("─" * 60)
    log("HEADLINES")
    log(f"  Scenario A (all faces) ── SCI {summary_a['city_sci_pct']}% · "
        f"SSI {summary_a['city_ssi_pct']}%")
    log(f"  Scenario B (south only) ─ SCI {summary_b['city_sci_pct']}% · "
        f"SSI {summary_b['city_ssi_pct']}%")
    log(f"  Reference (Usta-Mutani) ─ SCI {USTA_MUTANI_2025_SCI}% · "
        f"SSI {USTA_MUTANI_2025_SSI}%")
    log("─" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
