# 03 — Finding the usable roof area

Pipeline that turns the classified roof segments into a per-building usable-area
result and the LOD comparison, plus the interactive 3D viewer.

## Inputs produced upstream (not in this folder)

- **Roof segmentation** (`i.segment` -> polygonize -> ~94,000 segments by
  orientation): done in QGIS/GRASS upstream. See the TikZ diagram
  `segmentation_workflow.tex` for the workflow.
- **Roof-segment classifier** (XGBoost, 3 classes — good / obstruction / window —
  trained on 259 hand labels): the training script lives outside this folder. The
  pipeline below consumes its output, `predictions.gpkg`.
- **Annual roof irradiation** `SI_Ann` (and the 12 monthly `SI_<month>` rasters):
  computed from a 1 m building DSM run through **r.sun** (PVGIS diffuse/global,
  Meteonorm Linke turbidity, albedo 0.25 for the dense urban centre). `stage_B`
  *propagates* these already-computed values onto the LOD2.2 faces; the r.sun run
  itself is a separate upstream step.

## Run order

```
predictions.gpkg
   |
   v
merge_pv_into_faces.py          # classifier predictions -> per-face good/bad/obstruction m2
stage_A_baselines.py            # per-building baselines (footprint, 0.35 coef, LOD1.3)
stage_B_propagate_irradiation.py    # attribute annual SI_Ann to faces
stage_B2_monthly_irradiation.py     # attribute the 12 monthly SI values to faces
stage_C_face_layers.py          # 3-layer usable-area cascade per LOD2.2 face
stage_D_building_rollup.py      # roll faces up to per-building totals -> buildings_cascade.csv
stage_E_comparison_report.py    # 0.35 coefficient vs classifier-suitable comparison
stage_F_augment_viewer.py       # attach cascade numbers to the viewer JSONs
web_viewer/                     # MapLibre + deck.gl 3D viewer
```

## Key result

The national **0.35 coefficient** (footprint x 0.35) **over-estimates** the
classifier-suitable roof area by **+44.5 %** city-wide. Per building it is
unreliable in both directions: close to correct for pitched roofs (which dominate
the centre) but heavily over-estimating the few flat roofs. The agreement at city
scale is therefore misleading.

## Diagram

`segmentation_workflow.tex` — a compact, colour-coded TikZ diagram of the
segmentation workflow. Needs `\usepackage{tikz}` and
`\usetikzlibrary{positioning, arrows.meta}` in the preamble. (The node styles are
named `joinbox` / `outbox` on purpose — `join` and `out` are reserved TikZ
keywords.)

## Viewer

`web_viewer/` runs locally:

```bash
cd web_viewer
python -m http.server 8000
# then open http://localhost:8000/
```

It needs the data JSONs (`buildings_3d_cascade.json`, `faces_3d_cascade.json`,
`walls_3d.json`, `feature_points.json`) in a `data/` subfolder — produced by
`stage_F`. In the LOD2.2 view the walls do not render (a display limitation only;
the analysis is on the roof faces).
