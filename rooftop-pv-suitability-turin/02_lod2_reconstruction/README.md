# 02 — LOD2.2 building reconstruction

The 3D building models (LOD1.3 and LOD2.2) were reconstructed with
**geoflow-roofer** (TU Delft 3D geoinformation group), run from a Singularity/
Apptainer container on the cluster. Roofer is a separate tool, so this folder
contains the **configuration** used to drive it plus the helper scripts written
around it — not roofer itself.

## Files

- **`roofer_config_v2.toml`** — the actual roofer configuration used for the final
  reconstruction. Roofer's defaults are tuned for LiDAR; these values were
  re-tuned for the denser, noisier photogrammetric input. Key changes from the
  LiDAR defaults: `complexity-factor` 0.888 → 0.45 (avoid false roof planes from
  noise), `plane-detect-epsilon` 0.30 → 0.55 m (tolerate point scatter),
  `lod11-fallback-planes` 900 → 5000 (stop complex roofs collapsing to a flat box).

- **`render_buildings_footprints.py`** — renders a top-down plan view of the
  classified building points with the BDTRE footprint outlines on top (the figure
  used to show the reconstruction inputs). Reads the huge LAS in chunks.

## Inputs / outputs

- Inputs: classified building points (from stage 01) + BDTRE building footprints.
- Outputs: LOD1.3 and LOD2.2 CityJSON models (1,396 buildings reconstructed from
  1,510 footprints; complex/noisy roofs fall back from LOD2.2 to LOD1.3).

> Data files (point clouds, footprints, CityJSON outputs) are not included — see
> the main repository README.
