# 04 — Rooftop PV self-consumption

Six scripts (G through L) that extend the per-building usable-area cascade with an
hourly self-consumption analysis: for residential buildings, do we produce PV when
we need to consume?

## What it does

Implements the methodology from Prof. Mutani's *"2025 — Exercise 2: Solar Energy"*
exercise, applied to the historic centre of Turin, and compares the result against
the **PV-electricity** scenario of Usta-Mutani (2025).

> **Reference note.** The comparison uses Usta-Mutani's **PV-electricity** scenario
> — **SCI 63.12 % / SSI 55.47 %** (their Table 9, Scenario 9, eta = 23 %, the row
> closest to this work's eta = 0.24). Their 10 % / 12 % figures are a *solar-thermal
> heating* scenario and are **not** the right comparison for PV.

| Indicator | Formula | Meaning |
|---|---|---|
| **SCI** (Self-Consumption Index) | 100 x sum(SC) / sum(PV) | Of all PV produced, what % was used on-site? |
| **SSI** (Self-Sufficiency Index) | 100 x sum(SC) / sum(Load) | Of all consumption, what % came from PV? |
| **SC** (self-consumption per hour) | min(PV[h], Load[h]) | Hour-by-hour overlap |

## Run order

```
G -> H -> I -> J -> K -> L
```

## Scripts (brief)

- **`stage_G_residential_and_families.py`** — residential filter (pure + mixed
  use), drop garages/anomalies (footprint < 40 m2 OR height < 3 m), spatial join
  to ISTAT 2021 census sections (PRO_COM = 1272), disaggregate FAM21 per block by
  building volume.
- **`stage_H_pvgis_hourly.py`** — per-face hourly tilted-plane irradiance from
  PVGIS v5.2 (TMY), binned by slope x azimuth; flat faces -> 30 deg / south.
- **`stage_I_hourly_production.py`** — `P = PR * H * (f * A) * eta` per face per
  hour (PR = 0.80, eta = 0.24, **f = 0.80** active-area fraction,
  A = `layer2_suitable_m2`), summed per building. See the frame-factor note below.
- **`stage_J_hourly_load.py`** — 8,760-h load per building from the ARERA
  *"Prelievo medio orario"* hourly shape (weekday/Sat/Sun) x monthly modulation,
  normalised to 2,700 kWh/family/yr, scaled by `n_families`.
- **`stage_K_self_consumption.py`** — hour-by-hour SC = min(PV, Load); annual
  SCI/SSI per building; writes columns + an hourly-balance parquet.
- **`stage_L_comparison_report.py`** — 5 figures (300 dpi), `sci_ssi_summary.csv`,
  `comparison_report.md`. Two scenarios: A = all faces, B = south-facing only.

## Frame factor (active-area fraction, f = 0.80)

Production uses `P = PR * H * (f * A) * eta` with **f = 0.80**, the panel frame
factor only. We do **not** use the more common 0.60 coefficient: 0.60 also bundles
a reduction for roof obstructions and unusable area, and that reduction is already
applied face by face inside the Layer 2 classifier area — applying 0.60 on top
would remove the unusable area twice. (Per Prof. Mutani's note on using the active
surface.) `f` is the `--active-fraction` argument of `stage_I`, default **0.80**;
the thesis Chapter 4 results use this default.

## Results (thesis run, f = 0.80)

| Quantity | A: all faces | B: south only |
|---|---|---|
| Residential buildings (with PV + load) | 417 | 335 |
| Families | 7,915 | 6,671 |
| Annual PV (GWh/yr) | 11.05 | 7.43 |
| Annual load (GWh/yr) | 21.37 | 18.01 |
| PV / load ratio | 0.52x | 0.41x |
| City-wide SCI | 51.76 % | 55.49 % |
| City-wide SSI | 26.76 % | 22.88 % |
| Median per-building SCI | 70.52 % | 76.77 % |
| Median per-building SSI | 31.66 % | 30.32 % |

Building reconciliation: 1,396 reconstructed -> 882 residential (Stage G filters)
-> 417 with both PV-suitable area and resident load (scenario A) -> 335 (scenario
B, south-only subset).

**Interpretation.** The centro-storico results sit **below** the whole-city
Usta-Mutani PV reference (63.12 % / 55.47 %). This is the finding, not a validation
failure: their figures are whole-city, on larger south/flat roofs with PV well
matched to demand, whereas the historic centre is PV-undersized (PV/load = 0.52x),
dense, and its roofs face mostly intercardinal directions (Roman grid). Scarce
production relative to demand keeps per-building self-consumption high but caps
self-sufficiency. Compare aggregate-to-aggregate (our 51.76 / 26.76 vs their
63.12 / 55.47), not their aggregate against our per-building median; their load
denominator is whole-city (~1000 GWh) vs our ~21 GWh, so treat it as a reference,
not a strict benchmark.

## Honest caveats (carried into the thesis)

- **Monthly load modulation is a placeholder.** The hourly ARERA shape is real
  measured data; the month-to-month factors in `stage_J` are a canonical
  winter/summer pattern, not measured ARERA monthly data. Replace
  `NORMALISED_MONTHLY_FACTORS` and re-run from Stage J when the monthly series is
  available.
- **2,700 kWh/family is a reference choice, not a measurement.** ARERA's measured
  average per customer is ~1,030 kWh/yr; 2,700 (*cliente tipo*) is used for
  comparability with Usta-Mutani and raises the load accordingly.
- Heritage exclusion (D.Lgs. 42/2004) is not applied to Layer 2.
