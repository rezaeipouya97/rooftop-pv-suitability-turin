/* ============================================================================
 * LOD2.2 vs DSM deviation viewer
 * ============================================================================
 * Stack: MapLibre GL JS (basemap) + deck.gl MapboxOverlay (3D layers)
 *
 * Data files (in ./data/):
 *   buildings_3d.json    — GeoJSON FeatureCollection, WGS84 footprints + attrs
 *   faces_3d.json        — GeoJSON-3D, WGS84 lon/lat + ortho_z_m, per-triangle
 *   feature_points.json  — { by_building: { BLD_xxx: [[lon,lat,z,delta], ...] } }
 *
 * State kept on `state` global. UI events update state and call render().
 * ============================================================================ */

const DATA = {
  buildings: 'data/buildings_3d_cascade.json',   // augmented with PV cascade attrs
  faces:     'data/faces_3d_cascade.json',       // augmented with PV cascade attrs
  walls:     'data/walls_3d.json',
  points:    'data/feature_points.json',
};

const TURIN_CENTRE = [7.6862, 45.0703];   // approx centre of centro storico
const INITIAL_ZOOM = 16.0;
const INITIAL_PITCH = 60;
const INITIAL_BEARING = -20;

/* ---------------------------------------------------------------------------
 * Colour ramps
 * ------------------------------------------------------------------------- */

// Sequential blue→yellow→red ramp (sunset-ish), good for percent / RMS metrics
function rampSequential(t) {
  // t in [0, 1]
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [0.00, [ 50,  90, 160]],     // deep blue
    [0.25, [120, 180, 200]],     // light blue
    [0.50, [240, 230, 140]],     // pale yellow
    [0.75, [240, 150,  80]],     // orange
    [1.00, [180,  40,  40]],     // red
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [a, ca] = stops[i], [b, cb] = stops[i + 1];
    if (t <= b) {
      const f = (t - a) / (b - a);
      return [
        Math.round(ca[0] + f * (cb[0] - ca[0])),
        Math.round(ca[1] + f * (cb[1] - ca[1])),
        Math.round(ca[2] + f * (cb[2] - ca[2])),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

// Diverging blue↔white↔red ramp for signed Δ (chimneys=red, recesses=blue)
function rampDiverging(t) {
  // t in [-1, 1]
  if (t > 0) {
    const f = Math.min(1, t);
    return [
      Math.round(255 + f * (200 - 255)),
      Math.round(255 + f * ( 60 - 255)),
      Math.round(255 + f * ( 60 - 255)),
    ];
  } else {
    const f = Math.min(1, -t);
    return [
      Math.round(255 + f * ( 60 - 255)),
      Math.round(255 + f * (110 - 255)),
      Math.round(255 + f * (200 - 255)),
    ];
  }
}

// Ratio diverging ramp — for coef_vs_layer2_ratio and similar.
// t = 1 → white (coefficient matches our number)
// t > 1 → red (coefficient over-estimates), saturating at t = 3
// t < 1 → green (coefficient under-estimates), saturating at t = 0.33
function rampRatio(value) {
  if (value === null || value === undefined || isNaN(value)) {
    return [200, 200, 200];
  }
  // map ratio to a [-1, +1] axis on log scale
  // log(1) = 0  → white
  // log(3) ≈ 1.10  → full red
  // log(1/3) ≈ -1.10 → full green
  const lv = Math.log(Math.max(value, 0.01));
  const t = Math.max(-1, Math.min(1, lv / 1.10));
  if (t > 0) {
    // white → red as t goes 0 → 1
    const f = t;
    return [
      Math.round(255 + f * (200 - 255)),  // 255 → 200
      Math.round(255 + f * ( 60 - 255)),  // 255 → 60
      Math.round(255 + f * ( 60 - 255)),  // 255 → 60
    ];
  } else {
    // white → green as t goes 0 → -1
    const f = -t;
    return [
      Math.round(255 + f * ( 60 - 255)),
      Math.round(255 + f * (160 - 255)),
      Math.round(255 + f * ( 80 - 255)),
    ];
  }
}

/* ---------------------------------------------------------------------------
 * Metric definitions — mapping field → label, range, formatter
 * ------------------------------------------------------------------------- */
const METRICS = {
  coef_vs_layer2_ratio: {
    label: '0.35 coefficient ÷ Layer 2 suitable — per-building error ratio',
    field: 'coef_vs_layer2_ratio',
    perBuilding: true,
    diverging: true,
    fmt:   v => '×' + v.toFixed(2),
    range: [0.33, 3.0],
  },
  layer2_suitable_m2: {
    label: 'Layer 2 suitable m² — classifier-corrected PV area per building',
    field: 'layer2_suitable_m2',
    perBuilding: true,
    fmt:   v => v.toFixed(0) + ' m²',
    range: [0, 800],
  },
  layer3_kwh_yr: {
    label: 'Layer 3 kWh/yr — gross annual irradiation × Layer 2 (per building)',
    field: 'layer3_kwh_yr',
    perBuilding: true,
    fmt:   v => (v >= 1000 ? (v/1000).toFixed(1) + ' MWh/yr' : v.toFixed(0) + ' kWh/yr'),
    range: [0, 1500000],
  },
  si_ann_kwh_m2_yr: {
    label: 'SI_Ann (kWh/m²/yr) — per-face annual irradiation',
    field: 'si_ann_kwh_m2_yr',
    fmt:   v => v.toFixed(0) + ' kWh/m²',
    range: [0, 1800],
  },
  pv_verdict: {
    label: 'PV verdict — classifier result on this face',
    field: 'pv_verdict',
    categorical: true,
    categories: {
      'pv-suitable':         [ 30, 130,  50, 230],
      'partially-suitable':  [140, 190,  90, 230],
      'obstructed':          [180,  60, 180, 230],
      'unsuitable':          [200,  80,  80, 230],
      'no-coverage':         [180, 180, 180, 200],
    },
    fmt: v => v,
    range: [0, 1],
  },
  pv_good_pct: {
    label: 'pv_good_pct (%) — % of face that is PV-usable',
    field: 'pv_good_pct',
    fmt:   v => v.toFixed(1) + '%',
    range: [0, 100],
  },
  pv_obs_pct: {
    label: 'pv_obs_pct (%) — % of face obstructed',
    field: 'pv_obs_pct',
    fmt:   v => v.toFixed(1) + '%',
    range: [0, 100],
  },
  lod22_feature_pct_50cm: {
    label: 'feature_pct_50cm (%)',
    field: 'lod22_feature_pct_50cm',
    fmt:   v => v.toFixed(1) + '%',
    range: [0, 100],
  },
  area_gain_lod2_m2: {
    label: 'area_gain_lod2_m2',
    field: 'area_gain_lod2_m2',
    fmt:   v => v.toFixed(0) + ' m²',
    range: [0, 1500],     // p95-ish from Stage 4 was 655 m²; widen for outliers
  },
  lod22_wt_delta_rms: {
    label: 'wt_delta_rms (m)',
    field: 'lod22_wt_delta_rms',
    fmt:   v => v.toFixed(2) + ' m',
    range: [0, 5],        // p95 ~4.78 from Stage 4
  },
  ratio_lod2_lod1: {
    label: 'ratio_lod2_lod1 (×)',
    field: 'ratio_lod2_lod1',
    fmt:   v => '×' + v.toFixed(2),
    range: [0.5, 4],
  },
};

/* ---------------------------------------------------------------------------
 * State
 * ------------------------------------------------------------------------- */
const state = {
  data: null,                         // loaded JSON files
  selectedBuilding: null,             // building_i string
  colorBy: 'coef_vs_layer2_ratio',
  buildingStyle: 'lod22',             // 'lod22' = per-face roofs (default), 'lod13' = flat prism
  facesByBuilding: null,              // Map<building_i, [features...]>
};

/* ---------------------------------------------------------------------------
 * Map setup
 * ------------------------------------------------------------------------- */
// CartoDB Positron (no labels, retina @2x). Free, no API key.
// Light grey neutral basemap that lets the 3D building colours stand out.
const BASEMAP_STYLE = {
  version: 8,
  sources: {
    'carto-positron': {
      type: 'raster',
      tiles: [
        'https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}@2x.png',
        'https://b.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}@2x.png',
        'https://c.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}@2x.png',
        'https://d.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}@2x.png',
      ],
      tileSize: 256,
      attribution: '&copy; <a href="https://carto.com/attributions">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxzoom: 19,
    },
  },
  layers: [
    { id: 'carto-base', type: 'raster', source: 'carto-positron' },
  ],
};

const map = new maplibregl.Map({
  container: 'map',
  style: BASEMAP_STYLE,
  center: TURIN_CENTRE,
  zoom: INITIAL_ZOOM,
  pitch: INITIAL_PITCH,
  bearing: INITIAL_BEARING,
  antialias: true,
});
map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }));

const overlay = new deck.MapboxOverlay({ interleaved: false, layers: [] });
map.addControl(overlay);

/* ---------------------------------------------------------------------------
 * Boot — load data, build layers, wire UI
 * ------------------------------------------------------------------------- */
async function boot() {
  const [buildings, faces, walls, points] = await Promise.all([
    fetch(DATA.buildings).then(r => r.json()),
    fetch(DATA.faces).then(r => r.json()),
    fetch(DATA.walls).then(r => r.json()),
    fetch(DATA.points).then(r => r.json()),
  ]);

  state.data = { buildings, faces, walls, points };

  // -----------------------------------------------------------------------
  // Z baseline correction.
  //
  // Face/wall Z values are orthometric (m above sea level), e.g. 226–302 m
  // for Turin. deck.gl interprets the 3rd coord as metres above the basemap,
  // so without correction buildings would render 226 m in the air.
  //
  // We subtract the minimum Z across the whole dataset (walls AND faces) so
  // the lowest point sits at z = 0. Use walls' min because they include
  // ground-level vertices, giving the true terrain baseline.
  // -----------------------------------------------------------------------
  let zMin = Infinity;
  for (const w of walls.features) {
    if (w.properties.z_min < zMin) zMin = w.properties.z_min;
  }
  if (!isFinite(zMin)) {
    // Fallback to faces if walls didn't load
    for (const f of faces.features) {
      for (const v of f.geometry.coordinates[0]) {
        if (v.length >= 3 && isFinite(v[2]) && v[2] < zMin) zMin = v[2];
      }
    }
  }
  if (!isFinite(zMin)) zMin = 0;
  state.zBaseline = zMin;
  console.log(`[boot] Z baseline: ${zMin.toFixed(2)} m (subtracted from all heights)`);

  // Apply baseline to faces, walls, and feature points (in-place)
  for (const f of faces.features) {
    for (const v of f.geometry.coordinates[0]) {
      if (v.length >= 3) v[2] -= zMin;
    }
  }
  for (const w of walls.features) {
    for (const v of w.geometry.coordinates[0]) {
      if (v.length >= 3) v[2] -= zMin;
    }
    w.properties.z_min -= zMin;
    w.properties.z_max -= zMin;
  }
  for (const bi in points.by_building) {
    for (const p of points.by_building[bi]) p[2] -= zMin;
  }
  for (const f of buildings.features) {
    const h = f.properties.rf_h_roof_50p;
    if (h && isFinite(h)) f.properties._height_visual = h - zMin;
  }

  // Index faces and walls by building_i for fast lookup
  state.facesByBuilding = new Map();
  for (const f of faces.features) {
    const bi = f.properties.building_i;
    if (!state.facesByBuilding.has(bi)) state.facesByBuilding.set(bi, []);
    state.facesByBuilding.get(bi).push(f);
  }
  state.wallsByBuilding = new Map();
  for (const w of walls.features) {
    const bi = w.properties.building_i;
    if (!state.wallsByBuilding.has(bi)) state.wallsByBuilding.set(bi, []);
    state.wallsByBuilding.get(bi).push(w);
  }

  // Building props lookup — used so walls can colour by building-level metrics
  state.buildingProps = new Map();
  for (const f of buildings.features) {
    state.buildingProps.set(f.properties.building_i, f.properties);
  }

  // Per-building eave height (= minimum Z of all roof faces for that building).
  // We extrude wall prisms up to this height, then the roof faces sit on top.
  state.eaveHeight = new Map();
  for (const [bi, fs] of state.facesByBuilding) {
    let zMin = Infinity;
    for (const f of fs) {
      for (const v of f.geometry.coordinates[0]) {
        if (v.length >= 3 && isFinite(v[2]) && v[2] < zMin) zMin = v[2];
      }
    }
    if (isFinite(zMin)) state.eaveHeight.set(bi, zMin);
  }

  console.log(`[boot] indexed ${state.facesByBuilding.size} buildings (faces), ` +
              `${state.wallsByBuilding.size} buildings (walls), ` +
              `${state.eaveHeight.size} eave heights`);

  // Recentre on data centroid (don't trust the hard-coded TURIN_CENTRE)
  if (buildings.features.length > 0) {
    let sumLon = 0, sumLat = 0, n = 0;
    for (const f of buildings.features) {
      // Get the first [lon, lat] vertex regardless of Polygon vs MultiPolygon
      const t = f.geometry.type;
      let firstVertex = null;
      if (t === 'Polygon') {
        // coordinates: [ring, hole1, ...]; ring: [[lon,lat], ...]
        firstVertex = f.geometry.coordinates[0]?.[0];
      } else if (t === 'MultiPolygon') {
        // coordinates: [poly, ...]; poly: [ring, ...]; ring: [[lon,lat], ...]
        firstVertex = f.geometry.coordinates[0]?.[0]?.[0];
      }
      if (firstVertex && firstVertex.length >= 2 &&
          isFinite(firstVertex[0]) && isFinite(firstVertex[1])) {
        sumLon += firstVertex[0];
        sumLat += firstVertex[1];
        n++;
      }
    }
    if (n > 0) {
      map.flyTo({ center: [sumLon / n, sumLat / n], duration: 0 });
    } else {
      console.warn('[boot] could not compute data centroid; staying at default centre');
    }
  }

  console.log(`[boot] loaded ${buildings.features.length} buildings, ` +
              `${faces.features.length} faces, ${points.n_points} feature points`);

  wireUI();
  render();
  updateLegend();
}

/* ---------------------------------------------------------------------------
 * UI wiring
 * ------------------------------------------------------------------------- */
function wireUI() {
  document.getElementById('colorBy').addEventListener('change', e => {
    state.colorBy = e.target.value;
    render();
    updateLegend();
  });
  document.getElementById('buildingStyle').addEventListener('change', e => {
    state.buildingStyle = e.target.value;
    render();
  });
}

/* ---------------------------------------------------------------------------
 * Layer construction
 * ------------------------------------------------------------------------- */

function buildLayers() {
  const layers = [];
  const isLod22 = state.buildingStyle === 'lod22';
  const m = METRICS[state.colorBy];

  // -----------------------------------------------------------------------
  // Unified colour: every wall and every roof face of a given building
  // gets the SAME colour (when the metric is per-building) or each face
  // uses its own value (when the metric exists at the per-face level too).
  //
  // PV metrics + feature_pct_50cm are face-level; everything else is
  // per-building.
  // -----------------------------------------------------------------------
  const FACE_LEVEL_METRICS = new Set([
    'lod22_feature_pct_50cm',
    'pv_verdict',
    'pv_good_pct',
    'pv_obs_pct',
    'si_ann_kwh_m2_yr',
  ]);
  const useFaceLevel = FACE_LEVEL_METRICS.has(state.colorBy);

  function colorByBuildingI(bi) {
    const props = state.buildingProps.get(bi);
    if (!props) return [180, 180, 180, 230];
    const v = props[m.field];
    if (v === null || v === undefined || isNaN(v)) return [180, 180, 180, 230];
    // diverging ratio (coefficient vs Layer 2 etc.) — uses fixed centre at 1.0
    if (m.diverging) {
      return [...rampRatio(v), 230];
    }
    const [lo, hi] = m.range;
    return [...rampSequential((v - lo) / (hi - lo)), 230];
  }

  function colorFaceFeature(f) {
    if (useFaceLevel) {
      const v = f.properties[m.field];
      if (v === null || v === undefined) return [180, 180, 180, 230];
      // Categorical (e.g. pv_verdict) — look up rgba by string
      if (m.categorical) {
        return m.categories[v] || [180, 180, 180, 200];
      }
      if (m.diverging) {
        return [...rampRatio(v), 230];
      }
      const [lo, hi] = m.range;
      return [...rampSequential((v - lo) / (hi - lo)), 230];
    }
    return colorByBuildingI(f.properties.building_i);
  }

  if (isLod22) {
    // Per-face LOD2.2 roofs — the headline visualisation. Each roof face is
    // rendered at its true 3D position and coloured by its own metric value.
    // No walls (deck.gl's PolygonLayer can't render vertical polygons; for
    // the full mesh with walls, the user opens our enriched CityJSON in
    // ninjaGL via the header link).
    layers.push(new deck.PolygonLayer({
      id: 'roofs-lod22',
      data: state.data.faces.features,
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 100, 100],
      filled: true,
      stroked: true,
      extruded: false,
      _full3d: true,
      getPolygon: f => f.geometry.coordinates[0],
      getFillColor: f => colorFaceFeature(f),
      getLineColor: [40, 40, 40, 220],
      getLineWidth: 0.4,
      lineWidthMinPixels: 0.5,
      onClick: info => {
        if (info.object) {
          selectBuilding(info.object.properties.building_i);
          selectFace(info.object);
        }
      },
      updateTriggers: {
        getFillColor: [state.colorBy],
      },
    }));
  } else {
    // LOD1.3 prism mode — flat-topped extruded blocks for the LOD-comparison
    // story. Same colours as LOD2.2 mode at the building level.
    layers.push(new deck.GeoJsonLayer({
      id: 'buildings-lod13',
      data: state.data.buildings,
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 0, 80],
      extruded: true,
      wireframe: false,
      getFillColor: f => colorByBuildingI(f.properties.building_i),
      getLineColor: [80, 75, 65, 200],
      lineWidthMinPixels: 0.5,
      getElevation: f => {
        const h = f.properties._height_visual;
        return h && isFinite(h) ? h : 8;
      },
      onClick: info => {
        if (info.object) selectBuilding(info.object.properties.building_i);
      },
      updateTriggers: {
        getFillColor: [state.colorBy],
      },
    }));
  }

  return layers;
}

function render() {
  overlay.setProps({ layers: buildLayers() });
}

/* ---------------------------------------------------------------------------
 * Selection
 * ------------------------------------------------------------------------- */
function selectBuilding(building_i) {
  state.selectedBuilding = building_i;
  render();
  populatePanel(building_i);
}

function selectFace(faceFeature) {
  populateFacePanel(faceFeature);
}

/* ---------------------------------------------------------------------------
 * Side panel
 * ------------------------------------------------------------------------- */
function populatePanel(building_i) {
  const f = state.data.buildings.features.find(
    x => x.properties.building_i === building_i);
  if (!f) return;
  const p = f.properties;

  document.getElementById('panel-empty').classList.add('hidden');
  document.getElementById('panel-content').classList.remove('hidden');
  document.getElementById('panel-title').textContent = building_i;

  // Compute building bbox dimensions from its wall and roof vertices, so the
  // user sees building scale immediately. Italian palazzo blocks have a Bound
  // Ratio ~0.3 (footprint << bbox); without a visible scale cue, users will
  // misjudge size from the 3D rendering.
  const walls = state.wallsByBuilding.get(building_i) || [];
  const faces = state.facesByBuilding.get(building_i) || [];
  let lonMin = Infinity, lonMax = -Infinity, latMin = Infinity, latMax = -Infinity;
  for (const w of walls) {
    for (const v of w.geometry.coordinates[0]) {
      if (v[0] < lonMin) lonMin = v[0]; if (v[0] > lonMax) lonMax = v[0];
      if (v[1] < latMin) latMin = v[1]; if (v[1] > latMax) latMax = v[1];
    }
  }
  // Convert lon/lat span to metres at this latitude
  const latMid = (latMin + latMax) / 2;
  const widthM  = (lonMax - lonMin) * 111320 * Math.cos(latMid * Math.PI / 180);
  const heightM = (latMax - latMin) * 110540;
  const bboxM2  = widthM * heightM;
  const bldg_total22 = p.lod22_total_m2 || 0;
  const boundRatio = bboxM2 > 0 ? bldg_total22 / bboxM2 : 0;

  const rt = p.rf_roof_type || '—';
  document.getElementById('panel-rooftype').innerHTML =
      `<strong style="color:#222">${bldg_total22.toFixed(0)} m² built</strong> &nbsp; ` +
      `bbox ${widthM.toFixed(0)}×${heightM.toFixed(0)} m &nbsp; ` +
      `bound ratio ${boundRatio.toFixed(2)}<br>` +
      `<span style="color:#888;font-size:12px">roof type: ${rt} &middot; ` +
      `h_roof_50p: ${(p.rf_h_roof_50p || 0).toFixed(1)} m &middot; ` +
      `${faces.length} roof face${faces.length===1?'':'s'}, ${walls.length} walls</span>`;

  // PV cascade bars (the thesis story)
  // 0.35 coefficient / LOD1.3 eff / Layer 1 = LOD2.2 eff / Layer 2 classifier
  const coef    = p.pv_area_lod1_coef    || 0;
  const eff13   = p.lod13_effective_m2   || 0;
  const layer1  = p.layer1_available_m2  || 0;
  const layer2  = p.layer2_suitable_m2   || 0;
  const kwhyr   = p.layer3_kwh_yr        || 0;
  const cascadeMax = Math.max(coef, eff13, layer1, layer2, 1);
  const ratio   = p.coef_vs_layer2_ratio;

  const cascadeBars = [
    ['lod13',       '0.35 coefficient',   coef,   'm²'],
    ['lod22-gross', 'LOD1.3 effective',   eff13,  'm²'],
    ['lod22-gross', 'Layer 1 available',  layer1, 'm²'],
    ['lod22-eff',   'Layer 2 suitable',   layer2, 'm²'],
  ];
  const cascadeHTML = cascadeBars.map(([cls, lbl, v, u]) => `
    <div class="bar-row ${cls}">
      <span class="label">${lbl}</span>
      <span class="track"><span class="fill" style="width:${(v/cascadeMax*100).toFixed(1)}%"></span></span>
      <span class="val">${v.toFixed(0)} ${u}</span>
    </div>
  `).join('');
  // ratio interpretation line
  let ratioLine = '<span class="muted">—</span>';
  if (ratio !== null && ratio !== undefined && !isNaN(ratio)) {
    const pct = Math.round((ratio - 1) * 100);
    const sign = pct >= 0 ? 'over' : 'under';
    const colour = ratio >= 1 ? '#C46060' : '#5BA85B';
    ratioLine = `<span style="color:${colour};font-weight:600">` +
                `0.35 coefficient ${sign}-estimates Layer 2 by ${Math.abs(pct)}%</span>` +
                ` (×${ratio.toFixed(2)})`;
  }
  // kWh/yr line
  const kwhLine = kwhyr > 0
    ? `<span style="color:#444">Layer 3 gross irradiation: <strong>` +
      (kwhyr >= 1000 ? `${(kwhyr/1000).toFixed(1)} MWh/yr` : `${kwhyr.toFixed(0)} kWh/yr`) +
      `</strong></span>`
    : '';
  const cascadeBlock = document.getElementById('panel-cascade');
  if (cascadeBlock) {
    cascadeBlock.innerHTML =
      cascadeHTML +
      `<div style="margin-top:8px;font-size:12px">${ratioLine}</div>` +
      (kwhLine ? `<div style="margin-top:4px;font-size:12px">${kwhLine}</div>` : '');
  }

  // Quality / trust metrics (so the supervisor can sanity-check the building)
  const rows = [
    ['feature_pct_50cm (LOD2.2)',    p.lod22_feature_pct_50cm, '%'],
    ['rf_rmse_lod22 (geoflow quality)', p.rf_rmse_lod22,        'm'],
    ['n faces (rooftop)',            (state.facesByBuilding.get(building_i) || []).length, ''],
  ];
  document.getElementById('panel-table').innerHTML = rows.map(([k, v, u]) => `
    <tr><td class="k">${k}</td>
        <td class="v">${v == null ? '—' : (typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : v)} ${u}</td></tr>
  `).join('');

  // Reset face card
  document.getElementById('panel-face').innerHTML =
      '<div class="muted">No face selected.</div>';
}

function populateFacePanel(faceFeature) {
  const p = faceFeature.properties;
  document.getElementById('panel-face').innerHTML = `
    <div class="face-title">Face #${p.face_idx}</div>
    <table style="width:100%; font-size:11px; border-collapse:collapse;">
      <tr><td>area</td><td style="text-align:right">${p.area_m2.toFixed(1)} m²</td></tr>
      <tr><td>slope</td><td style="text-align:right">${p.slope_deg.toFixed(1)}°</td></tr>
      <tr><td>azimuth (uphill)</td><td style="text-align:right">${p.azimuth_deg.toFixed(1)}°</td></tr>
      <tr><td>flat?</td><td style="text-align:right">${p.is_flat ? 'yes' : 'no'}</td></tr>
      <tr><td colspan="2" style="padding-top:10px;border-top:1px solid #ddd;font-weight:600;color:#2a4858">PV classifier (this face)</td></tr>
      <tr><td>verdict</td><td style="text-align:right">${p.pv_verdict ?? '—'}</td></tr>
      <tr><td>good %</td><td style="text-align:right">${(p.pv_good_pct ?? 0).toFixed(1)} %</td></tr>
      <tr><td>obstructed %</td><td style="text-align:right">${(p.pv_obs_pct ?? 0).toFixed(1)} %</td></tr>
      <tr><td>coverage %</td><td style="text-align:right">${(p.pv_coverage_pct ?? 0).toFixed(1)} %</td></tr>
      <tr><td colspan="2" style="padding-top:10px;border-top:1px solid #ddd;font-weight:600;color:#2a4858">PV cascade (this face)</td></tr>
      <tr><td>SI_Ann (kWh/m²/yr)</td><td style="text-align:right">${p.si_ann_kwh_m2_yr == null ? '—' : p.si_ann_kwh_m2_yr.toFixed(0)}</td></tr>
      <tr><td>Layer 1 available</td><td style="text-align:right">${p.layer1_available_m2 == null ? '—' : p.layer1_available_m2.toFixed(1) + ' m²'}</td></tr>
      <tr><td>Layer 2 suitable</td><td style="text-align:right">${p.layer2_suitable_m2 == null ? '—' : p.layer2_suitable_m2.toFixed(1) + ' m²'}</td></tr>
      <tr><td>Layer 3 gross kWh/yr</td><td style="text-align:right">${p.layer3_kwh_yr == null ? '—' : p.layer3_kwh_yr.toFixed(0)}</td></tr>
    </table>
  `;
}

/* ---------------------------------------------------------------------------
 * Legend
 * ------------------------------------------------------------------------- */
function updateLegend() {
  const m = METRICS[state.colorBy];
  document.getElementById('legend-title').textContent = m.label;
  const grad = document.getElementById('legend-gradient');

  if (m.categorical) {
    // Stack of coloured rows with their category names
    const rows = Object.entries(m.categories).map(([name, rgba]) => `
      <div style="display:flex;align-items:center;gap:6px;font-size:11px;margin:2px 0;">
        <span style="width:18px;height:12px;background:rgb(${rgba[0]},${rgba[1]},${rgba[2]});border-radius:2px;display:inline-block;flex-shrink:0;"></span>
        <span>${name}</span>
      </div>
    `).join('');
    grad.style.background = 'transparent';
    grad.style.height = 'auto';
    grad.style.width = 'auto';
    grad.innerHTML = rows;
    document.getElementById('legend-min').textContent = '';
    document.getElementById('legend-mid').textContent = '';
    document.getElementById('legend-max').textContent = '';
    return;
  }

  if (m.diverging) {
    // Diverging ratio ramp — green (under) → white (match=1) → red (over)
    grad.innerHTML = '';
    grad.style.height = '12px';
    grad.style.width = '200px';
    const stops = [];
    for (let i = 0; i <= 10; i++) {
      const t = -1 + (2 * i / 10);   // map 0..10 → -1..+1
      // rampRatio takes a ratio; invert log to get the ratio for this t
      const ratio = Math.exp(t * 1.10);
      const [r, g, b] = rampRatio(ratio);
      stops.push(`rgb(${r},${g},${b}) ${i*10}%`);
    }
    grad.style.background = `linear-gradient(90deg, ${stops.join(', ')})`;
    document.getElementById('legend-min').textContent = m.fmt(m.range[0]);
    document.getElementById('legend-mid').textContent = m.fmt(1.0);
    document.getElementById('legend-max').textContent = m.fmt(m.range[1]);
    return;
  }

  // Sequential ramp (original behaviour)
  grad.innerHTML = '';
  grad.style.height = '12px';
  grad.style.width = '200px';
  const stops = [];
  for (let i = 0; i <= 10; i++) {
    const t = i / 10;
    const [r, g, b] = rampSequential(t);
    stops.push(`rgb(${r},${g},${b}) ${i*10}%`);
  }
  grad.style.background = `linear-gradient(90deg, ${stops.join(', ')})`;
  document.getElementById('legend-min').textContent = m.fmt(m.range[0]);
  document.getElementById('legend-mid').textContent =
      m.fmt((m.range[0] + m.range[1]) / 2);
  document.getElementById('legend-max').textContent = m.fmt(m.range[1]);
}

/* ---------------------------------------------------------------------------
 * Boot
 * ------------------------------------------------------------------------- */
map.on('load', () => {
  boot().catch(err => {
    console.error(err);
    document.getElementById('map').innerHTML =
        `<div style="padding:40px;color:#900;font-family:monospace">
          Failed to load viewer data:<br>${err.message}
        </div>`;
  });
});
