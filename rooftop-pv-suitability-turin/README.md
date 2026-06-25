# Rooftop PV Suitability — Historic Centre of Turin

Code for the MSc thesis *"From Aerial Photogrammetry to Per-Building Rooftop PV
Suitability for the Historic Centre of Turin"* (Politecnico di Torino).

The project builds a full pipeline from aerial photogrammetric imagery to a
per-building assessment of rooftop photovoltaic (PV) suitability and
self-consumption, for the dense historic centre (*centro storico*) of Turin.

> **Author:** Pouya Rezaei — Politecnico di Torino
> **Supervisor:** Prof. Antonia Spanò · **Co-supervisors:** Prof. Giacomo Patrucco, Prof. Guglielmina Mutani

---

## Pipeline overview

| Stage | Folder | What it does |
|---|---|---|
| 1 | [`01_point_cloud_segmentation/`](01_point_cloud_segmentation/) | Classify a ~654 M-point photogrammetric cloud into ground / building / tree with a sparse-convolution deep-learning model (TorchSparse), clean it, and extract building footprints. |
| 2 | [`02_lod2_reconstruction/`](02_lod2_reconstruction/) | Reconstruct LOD1.3 and LOD2.2 3D building models from the classified points + footprints with **geoflow-roofer**. |
| 3 | [`03_roof_pv_suitability/`](03_roof_pv_suitability/) | Find the usable roof area per building (3-layer area cascade), compare it against the Italian national 0.35 coefficient, and drive an interactive 3D viewer. |
| 4 | [`04_self_consumption/`](04_self_consumption/) | Hourly PV production vs residential demand per building → Self-Consumption (SCI) and Self-Sufficiency (SSI) indices, compared against the Usta-Mutani (2025) PV scenario. |

The stages run in order; each consumes the previous stage's output.

---

## Repository structure

```
rooftop-pv-suitability-turin/
├── 01_point_cloud_segmentation/   # classify the point cloud (ground/building/tree)
├── 02_lod2_reconstruction/        # LOD1.3 / LOD2.2 reconstruction (geoflow-roofer)
├── 03_roof_pv_suitability/        # usable-area cascade + 0.35-coef comparison + viewer
└── 04_self_consumption/           # hourly SCI / SSI analysis
```

Each folder has its own `README.md` with the script-by-script run order.

---

## Headline results

- **Point-cloud segmentation:** ≈ 98.5 % overall accuracy, ≈ 95.9 % mIoU on the balanced validation tile.
- **Reconstruction:** 1,396 buildings reconstructed (LOD1.3 / LOD2.2) from 1,510 footprints.
- **Usable roof area:** the national 0.35 coefficient **over-estimates** the classifier-suitable area by **+44.5 %** city-wide and is unreliable per building — close for pitched roofs, very wrong for flat roofs.
- **Self-consumption (Chapter 4, all-faces scenario, frame factor f = 0.80):**

  | Quantity (residential) | Value |
  |---|---|
  | Annual PV | **11.05 GWh/yr** |
  | PV / load ratio | **0.52×** |
  | City-wide SCI | **51.76 %** |
  | City-wide SSI | **26.76 %** |

  The historic centre is **PV-undersized**: scarce production relative to demand
  keeps per-building self-consumption high but caps self-sufficiency. This sits
  below the whole-city Usta-Mutani (2025) PV reference (SCI 63.12 % / SSI 55.47 %) —
  which is the finding, not a validation failure.

---

## PV production: active-area fraction (frame factor)

The per-face PV production follows

```
P = PR · H · (f · A) · η
```

with `PR = 0.80` (performance ratio), `η = 0.24` (module efficiency),
`A = layer2_suitable_m2` (the classifier-cleaned per-face area), and **`f = 0.80`**,
the active-area fraction.

`f = 0.80` is the panel **frame factor only**. We do *not* apply the more common
0.60 coefficient, because 0.60 also bundles a reduction for roof obstructions and
unusable area — and that reduction is already made, face by face, inside the
Layer 2 classifier area. Applying 0.60 on top of Layer 2 would remove the unusable
area twice. This follows Prof. Mutani's note on using the active surface in the PV
formula.

The thesis results (Chapter 4) are produced with `f = 0.80`, which is the default
of `--active-fraction` in `04_self_consumption/stage_I_hourly_production.py`.

---

## Important note on data

This repository contains **code only**. The input data (aerial imagery, point
clouds, DTM/DSM, cadastral and BDTRE building footprints, census sections) is
owned by the Municipality of Turin / Regione Piemonte / ISTAT and is **not
redistributable**. The scripts are provided for transparency and reproducibility
of the method, not as a runnable package without that data. A `.gitignore` blocks
data and model-weight files so they are never committed.

### Steps performed upstream / in other tools (not in this repo)

- the XGBoost roof-segment **classifier training** script (the pipeline consumes
  its output, `predictions.gpkg`);
- the QGIS/GRASS roof-segmentation step (see the TikZ diagram
  `03_roof_pv_suitability/segmentation_workflow.tex`);
- the r.sun irradiation run that produces `SI_Ann` (stage B only *propagates* it).

---

## Main tools

Python · PyTorch · TorchSparse · laspy · rasterio · NumPy · SciPy · scikit-learn ·
XGBoost · GeoPandas · Shapely · pyarrow · Matplotlib · CSF (Cloth Simulation
Filter) · geoflow-roofer · GRASS/QGIS · PVGIS · MapLibre + deck.gl (viewer)

---

## License

Released under the [MIT License](LICENSE). The license covers the **code** in this
repository only; it does not grant any rights over the underlying municipal /
regional / national datasets, which are not included.
