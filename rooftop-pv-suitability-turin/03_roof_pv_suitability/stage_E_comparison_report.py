#!/usr/bin/env python3
"""
stage_E_comparison_report.py
============================

Headline comparison: the LOD1 coefficient method (Mutani 0.35) vs. the
roofer-derived LOD1.3 effective area vs. the roofer LOD2.2 effective area
(layer1) vs. our classifier-corrected layer2 suitable area.

Reads the Stage D rollup and produces:
  - comparison_report.csv  one row per building with all comparison columns
  - 5 PNG figures for the thesis
  - text summary printed to stdout for the supervisor meeting

The figures:
  fig1_cascade_total.png       — 5-bar cascade across all buildings:
                                  Mutani coef → roofer LOD1.3 eff →
                                  roofer LOD2.2 gross → layer1 → layer2.
                                  Shows the dropoff.
  fig2_coef_overestimate_hist.png
                                — histogram of (coef / layer2) ratio per
                                  building. Where coefficient method is
                                  most wrong.
  fig3_scatter_footprint_vs_layer2.png
                                — scatter: footprint vs layer2_suitable,
                                  coloured by rf_roof_type. The 0.35 line
                                  drawn for reference.
  fig4_by_roof_type.png        — bars: median coef-vs-layer2 ratio by
                                  rf_roof_type.
  fig5_kwh_per_m2_footprint.png
                                — histogram: layer3_kwh_yr / footprint per
                                  building. PV intensity distribution.

Usage
-----
  python stage_E_comparison_report.py \\
      --building-csv  /path/to/buildings_cascade.csv \\
      --face-gpkg     /path/to/lod22_face_four_layers.gpkg \\
      --out-dir       /path/to/comparison/
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


def make_figures(bld: pd.DataFrame, face: gpd.GeoDataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "legend.fontsize": 9})

    has_lod2 = bld[bld["n_faces"] > 0].copy()

    # --- fig1: 5-bar cascade totals ---
    labels = [
        "Mutani 0.35\n(footprint × 0.35)",
        "LOD1.3 effective\n(roofer flat-top)",
        "LOD2.2 gross\n(roofer faceted)",
        "Layer 1\navailable\n(after feature_pct)",
        "Layer 2\nsuitable\n(classifier)",
    ]
    totals = [
        has_lod2["pv_area_lod1_coef"].sum(),
        has_lod2["lod13_effective_m2"].sum(),
        has_lod2["lod22_total_m2"].sum(),
        has_lod2["layer1_available_m2"].sum(),
        has_lod2["layer2_suitable_m2"].sum(),
    ]
    colors = ["#7FA9D6", "#A1C7E6", "#888888", "#F7B267", "#5BA85B"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, totals, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, totals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}\nm²",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total area across all buildings (m²)")
    ax.set_title(f"Per-building PV area cascade — "
                 f"{len(has_lod2):,} buildings with LOD2.2 reconstruction")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(totals) * 1.18)
    fig.tight_layout()
    p = out_dir / "fig1_cascade_total.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")

    # --- fig2: coef vs layer2 ratio histogram ---
    ratios = has_lod2["coef_vs_layer2_ratio"].dropna().clip(0, 10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=60, color="#6c757d", edgecolor="white", linewidth=0.5)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5,
               label="ratio = 1 (coefficient matches our number)")
    med = ratios.median()
    ax.axvline(med, color="#5BA85B", linestyle="-", linewidth=1.5,
               label=f"median = {med:.2f}×")
    ax.set_xlabel("Mutani 0.35 coefficient PV-area  /  Layer 2 suitable area")
    ax.set_ylabel("Number of buildings")
    ax.set_title("Where the 0.35 coefficient over- or under-estimates per building")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = out_dir / "fig2_coef_overestimate_hist.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")

    # --- fig3: scatter footprint vs layer2 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    if "rf_roof_type" in has_lod2.columns:
        for rtype, g in has_lod2.groupby("rf_roof_type"):
            ax.scatter(g["footprint_area_m2"], g["layer2_suitable_m2"],
                       s=12, alpha=0.55, label=str(rtype))
        ax.legend(title="rf_roof_type", loc="upper left")
    else:
        ax.scatter(has_lod2["footprint_area_m2"],
                   has_lod2["layer2_suitable_m2"], s=12, alpha=0.55)
    xmax = has_lod2["footprint_area_m2"].max()
    ax.plot([0, xmax], [0, 0.35 * xmax], "r--", linewidth=1,
            label="0.35 × footprint (Mutani coef)")
    ax.set_xlabel("Footprint area (m²)")
    ax.set_ylabel("Layer 2 suitable area (m²)  — classifier")
    ax.set_title("Footprint vs. classifier-suitable area: where the linear coefficient fails")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = out_dir / "fig3_scatter_footprint_vs_layer2.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")

    # --- fig4: by roof type ---
    if "rf_roof_type" in has_lod2.columns:
        by = has_lod2.groupby("rf_roof_type")["coef_vs_layer2_ratio"].agg(
            ["count", "median", "mean"]
        ).sort_values("median", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(by.index.astype(str), by["median"],
                color="#7FA9D6", edgecolor="black", linewidth=0.5)
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2)
        for i, (idx, row) in enumerate(by.iterrows()):
            ax.text(row["median"], i,
                    f"  n={int(row['count'])}, med={row['median']:.2f}×",
                    va="center", fontsize=9)
        ax.set_xlabel("Median (Mutani coef / Layer 2 suitable)")
        ax.set_ylabel("rf_roof_type")
        ax.set_title("Coefficient method bias by roof type")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        p = out_dir / "fig4_by_roof_type.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p}")

    # --- fig5: kWh per m² footprint distribution ---
    has_kwh = has_lod2[(has_lod2["layer3_kwh_yr"] > 0)
                       & (has_lod2["footprint_area_m2"] > 0)].copy()
    if len(has_kwh):
        has_kwh["kwh_per_m2_footprint"] = (
            has_kwh["layer3_kwh_yr"] / has_kwh["footprint_area_m2"]
        )
        v = has_kwh["kwh_per_m2_footprint"].clip(
            0, has_kwh["kwh_per_m2_footprint"].quantile(0.99)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(v, bins=60, color="#F7B267", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Layer 3 kWh/yr per m² of footprint")
        ax.set_ylabel("Number of buildings")
        ax.set_title(f"PV intensity per building "
                     f"({len(has_kwh):,} buildings with SI coverage)")
        ax.axvline(v.median(), color="#5BA85B", linestyle="-",
                   linewidth=1.5, label=f"median = {v.median():.1f}")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        p = out_dir / "fig5_kwh_per_m2_footprint.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building-csv", required=True,
                    help="Stage D buildings_cascade.csv")
    ap.add_argument("--face-gpkg", required=True,
                    help="Stage C face GPKG (for rf_roof_type if missing)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    print(f"[Stage E] comparison report + figures")
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  reading {args.building_csv}...")
    bld = pd.read_csv(args.building_csv)
    print(f"  → {len(bld):,} buildings")

    if "rf_roof_type" not in bld.columns:
        print(f"  pulling rf_roof_type from face GPKG...")
        face = gpd.read_file(args.face_gpkg)
        if "rf_roof_type" in face.columns:
            rt = face.groupby("building_i")["rf_roof_type"].first().reset_index()
            bld = bld.merge(rt, on="building_i", how="left")
    else:
        face = gpd.read_file(args.face_gpkg)

    cols = ["building_i", "footprint_area_m2",
            "pv_area_lod1_coef",
            "lod13_total_m2", "lod13_effective_m2", "lod13_feature_pct_50cm",
            "lod22_total_m2", "lod22_effective_m2", "lod22_feature_pct_50cm",
            "area_gain_lod2_m2", "ratio_lod2_lod1",
            "n_faces", "total_face_area_m2",
            "layer1_available_m2", "layer2_suitable_m2", "layer3_kwh_yr",
            "coef_vs_layer2_ratio", "coef_vs_lod13_ratio",
            "faces_with_si", "rf_roof_type", "rf_rmse_lod22", "rf_h_roof_50p"]
    cols = [c for c in cols if c in bld.columns]
    out_csv = out_dir / "comparison_report.csv"
    bld[cols].to_csv(out_csv, index=False, float_format="%.2f")
    print(f"  wrote {out_csv}")

    has_lod2 = bld[bld["n_faces"] > 0]
    print(f"\n=== Stage E headline summary ===")
    print(f"  buildings with LOD2.2 + classifier:    {len(has_lod2):,}/{len(bld):,}")
    print(f"")
    print(f"  Σ city-wide (m², on LOD2-reconstructed buildings):")
    print(f"    pv_area_lod1_coef  (Mutani 0.35):    "
          f"{has_lod2['pv_area_lod1_coef'].sum():>14,.0f}")
    print(f"    lod13_effective    (roofer flat):    "
          f"{has_lod2['lod13_effective_m2'].sum():>14,.0f}")
    print(f"    lod22_total        (roofer gross):   "
          f"{has_lod2['lod22_total_m2'].sum():>14,.0f}")
    print(f"    layer1_available   (LOD2 corrected): "
          f"{has_lod2['layer1_available_m2'].sum():>14,.0f}")
    print(f"    layer2_suitable    (classifier):     "
          f"{has_lod2['layer2_suitable_m2'].sum():>14,.0f}")
    print(f"    layer3_kwh_yr:                       "
          f"{has_lod2['layer3_kwh_yr'].sum():>14,.0f} kWh/yr")

    coef_total = has_lod2["pv_area_lod1_coef"].sum()
    lod13_total = has_lod2["lod13_effective_m2"].sum()
    layer2_total = has_lod2["layer2_suitable_m2"].sum()
    if layer2_total > 0:
        gap_l2 = (coef_total - layer2_total) / layer2_total * 100
        sign = "over" if gap_l2 > 0 else "under"
        print(f"\n  → city-wide: Mutani 0.35 {sign}-estimates Layer 2 "
              f"(classifier-suitable) by {abs(gap_l2):.1f}%")
    if lod13_total > 0:
        gap_l13 = (coef_total - lod13_total) / lod13_total * 100
        sign = "over" if gap_l13 > 0 else "under"
        print(f"  → city-wide: Mutani 0.35 {sign}-estimates LOD1.3 effective "
              f"(roofer flat-top) by {abs(gap_l13):.1f}%")

    r_l2 = has_lod2["coef_vs_layer2_ratio"].dropna()
    r_l13 = has_lod2["coef_vs_lod13_ratio"].dropna()
    if len(r_l2):
        print(f"\n  building-level Mutani/Layer2 ratio:")
        print(f"    median: {r_l2.median():.2f}×    "
              f"p25..p75: [{r_l2.quantile(0.25):.2f}, "
              f"{r_l2.quantile(0.75):.2f}]")
        print(f"    fraction over-estimating (>1.2×):  "
              f"{100*(r_l2 > 1.2).mean():.1f}%")
        print(f"    fraction under-estimating (<0.8×): "
              f"{100*(r_l2 < 0.8).mean():.1f}%")
    if len(r_l13):
        print(f"\n  building-level Mutani/LOD1.3-effective ratio:")
        print(f"    median: {r_l13.median():.2f}×    "
              f"p25..p75: [{r_l13.quantile(0.25):.2f}, "
              f"{r_l13.quantile(0.75):.2f}]")

    print(f"\n  making figures...")
    make_figures(bld, face, out_dir)

    print(f"\n[total {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
