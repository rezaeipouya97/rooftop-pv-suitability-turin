#!/usr/bin/env python3
"""
extract_footprints.py - Extract 2D building footprints from classified point cloud.

Pipeline:
  1. Read building points (predicted_label == 1) in chunks
  2. Rasterize to 2D binary occupancy grid (0.5m default)
  3. Morphological closing to fill small gaps in roofs
  4. Connected component labeling → individual buildings
  5. Vectorize each component to polygon (contour tracing)
  6. Simplify polygons (Douglas-Peucker)
  7. Export as GeoJSON + Shapefile

No ML dependency. Needs: laspy, numpy, scipy, rasterio, shapely, fiona

Usage:
    python extract_footprints.py \
        --input /path/to/big_pred_final.las \
        --output /path/to/footprints.geojson \
        --resolution 0.5 \
        --min_area 20
"""

import argparse, time
from pathlib import Path

import numpy as np
import laspy
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_opening, label


def main():
    pa = argparse.ArgumentParser(description="Extract building footprints from classified LAS")
    pa.add_argument("--input", required=True, help="Classified LAS with predicted_label field")
    pa.add_argument("--output", required=True, help="Output GeoJSON path")
    pa.add_argument("--output_shp", default=None, help="Optional: also output as Shapefile")
    pa.add_argument("--pred_field", default="predicted_label")
    pa.add_argument("--building_class", type=int, default=1)
    pa.add_argument("--resolution", type=float, default=0.5,
                    help="Rasterization resolution in metres (default 0.5m)")
    pa.add_argument("--min_area", type=float, default=20.0,
                    help="Minimum building area in m^2 (removes tiny fragments)")
    pa.add_argument("--closing_radius", type=int, default=3,
                    help="Morphological closing radius in pixels (fills roof gaps)")
    pa.add_argument("--opening_radius", type=int, default=2,
                    help="Morphological opening radius (removes thin noise)")
    pa.add_argument("--simplify_tolerance", type=float, default=0.5,
                    help="Douglas-Peucker simplification tolerance in metres")
    pa.add_argument("--chunk_size", type=int, default=10_000_000)
    args = pa.parse_args()

    t0 = time.time()

    # ================================================================
    # Step 1: Read building points XY in chunks (memory efficient)
    # ================================================================
    print(f"Step 1: Reading building points from {args.input}")

    with laspy.open(args.input) as reader:
        total_points = reader.header.point_count
    print(f"  Total points: {total_points:,}")

    # First pass: find XY extent of building points
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    building_count = 0

    with laspy.open(args.input) as reader:
        for chunk in reader.chunk_iterator(args.chunk_size):
            pred = np.asarray(chunk[args.pred_field], dtype=np.int32)
            mask = pred == args.building_class
            if mask.sum() == 0:
                continue
            bx = np.asarray(chunk.x, dtype=np.float64)[mask]
            by = np.asarray(chunk.y, dtype=np.float64)[mask]
            x_min = min(x_min, bx.min())
            x_max = max(x_max, bx.max())
            y_min = min(y_min, by.min())
            y_max = max(y_max, by.max())
            building_count += mask.sum()
            print(f"  Scanned {building_count:,} building points...", end='\r')

    print(f"\n  Building points: {building_count:,} ({100*building_count/total_points:.1f}%)")
    print(f"  XY extent: [{x_min:.1f},{y_min:.1f}] to [{x_max:.1f},{y_max:.1f}]")

    # ================================================================
    # Step 2: Rasterize to binary occupancy grid
    # ================================================================
    res = args.resolution
    cols = int(np.ceil((x_max - x_min) / res)) + 1
    rows = int(np.ceil((y_max - y_min) / res)) + 1
    print(f"\nStep 2: Rasterizing to {cols}x{rows} grid ({res}m resolution)")

    grid = np.zeros((rows, cols), dtype=np.uint8)

    with laspy.open(args.input) as reader:
        for chunk in reader.chunk_iterator(args.chunk_size):
            pred = np.asarray(chunk[args.pred_field], dtype=np.int32)
            mask = pred == args.building_class
            if mask.sum() == 0:
                continue
            bx = np.asarray(chunk.x, dtype=np.float64)[mask]
            by = np.asarray(chunk.y, dtype=np.float64)[mask]

            col_idx = np.floor((bx - x_min) / res).astype(np.int64)
            row_idx = np.floor((y_max - by) / res).astype(np.int64)
            col_idx = np.clip(col_idx, 0, cols - 1)
            row_idx = np.clip(row_idx, 0, rows - 1)

            grid[row_idx, col_idx] = 1

    filled_pct = 100 * grid.sum() / grid.size
    print(f"  Filled cells: {grid.sum():,} ({filled_pct:.1f}%)")

    # ================================================================
    # Step 3: Morphological cleanup
    # ================================================================
    print(f"\nStep 3: Morphological cleanup")

    # Closing: fill small holes in roofs (e.g., courtyards, chimneys)
    if args.closing_radius > 0:
        struct_close = ndimage.generate_binary_structure(2, 1)
        struct_close = ndimage.iterate_structure(struct_close, args.closing_radius)
        grid = binary_closing(grid, structure=struct_close).astype(np.uint8)
        print(f"  After closing (r={args.closing_radius}): {grid.sum():,} cells")

    # Opening: remove thin noise (fences, wires, narrow artifacts)
    if args.opening_radius > 0:
        struct_open = ndimage.generate_binary_structure(2, 1)
        struct_open = ndimage.iterate_structure(struct_open, args.opening_radius)
        grid = binary_opening(grid, structure=struct_open).astype(np.uint8)
        print(f"  After opening (r={args.opening_radius}): {grid.sum():,} cells")

    # ================================================================
    # Step 4: Connected component labeling
    # ================================================================
    print(f"\nStep 4: Connected component labeling")
    labeled, num_features = label(grid)
    print(f"  Raw components: {num_features:,}")

    # Filter by minimum area
    min_pixels = int(args.min_area / (res * res))
    component_sizes = ndimage.sum(grid, labeled, range(1, num_features + 1))
    valid_components = []
    for i, size in enumerate(component_sizes, start=1):
        if size >= min_pixels:
            valid_components.append(i)

    print(f"  After area filter (>{args.min_area}m^2): {len(valid_components):,} buildings")

    # ================================================================
    # Step 5: Vectorize to polygons
    # ================================================================
    print(f"\nStep 5: Vectorizing to polygons")

    try:
        from shapely.geometry import shape, mapping, MultiPolygon
        from shapely.ops import unary_union
        import rasterio
        from rasterio.features import shapes
        from rasterio.transform import from_bounds
    except ImportError as e:
        print(f"  ERROR: {e}")
        print("  Install: pip install shapely rasterio fiona")
        return

    # Create a clean binary mask with only valid components
    clean_grid = np.zeros_like(grid, dtype=np.uint8)
    for comp_id in valid_components:
        clean_grid[labeled == comp_id] = 1

    # Rasterio transform for georeferencing
    transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)

    # Vectorize
    polygons = []
    building_ids = []
    bid = 0

    for geom, value in shapes(clean_grid, mask=clean_grid == 1, transform=transform):
        poly = shape(geom)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.area < args.min_area:
            continue

        # Simplify
        if args.simplify_tolerance > 0:
            poly = poly.simplify(args.simplify_tolerance, preserve_topology=True)

        if poly.is_empty:
            continue

        # Handle MultiPolygon from simplification
        if poly.geom_type == 'MultiPolygon':
            for sub_poly in poly.geoms:
                if sub_poly.area >= args.min_area:
                    bid += 1
                    polygons.append(sub_poly)
                    building_ids.append(bid)
        else:
            bid += 1
            polygons.append(poly)
            building_ids.append(bid)

    print(f"  Final polygons: {len(polygons):,}")

    # Stats
    areas = [p.area for p in polygons]
    print(f"  Area range: {min(areas):.0f} - {max(areas):.0f} m^2")
    print(f"  Total footprint area: {sum(areas):,.0f} m^2")

    # ================================================================
    # Step 6: Export
    # ================================================================
    import json

    # GeoJSON
    print(f"\nStep 6: Writing {args.output}")
    features = []
    for bid, poly in zip(building_ids, polygons):
        features.append({
            "type": "Feature",
            "properties": {
                "building_id": bid,
                "area_m2": round(poly.area, 1),
                "perimeter_m": round(poly.length, 1),
            },
            "geometry": mapping(poly),
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), 'w') as f:
        json.dump(geojson, f)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  GeoJSON: {out_path} ({size_mb:.1f} MB)")

    # Shapefile (optional)
    if args.output_shp:
        try:
            import fiona
            from fiona.crs import from_epsg

            shp_path = Path(args.output_shp)
            shp_path.parent.mkdir(parents=True, exist_ok=True)

            schema = {
                'geometry': 'Polygon',
                'properties': {
                    'building_id': 'int',
                    'area_m2': 'float',
                    'perimeter_m': 'float',
                },
            }

            with fiona.open(str(shp_path), 'w', 'ESRI Shapefile', schema) as dst:
                for bid, poly in zip(building_ids, polygons):
                    if poly.geom_type == 'MultiPolygon':
                        # Write largest polygon only for shapefile
                        poly = max(poly.geoms, key=lambda p: p.area)
                    dst.write({
                        'geometry': mapping(poly),
                        'properties': {
                            'building_id': bid,
                            'area_m2': round(poly.area, 1),
                            'perimeter_m': round(poly.length, 1),
                        },
                    })
            print(f"  Shapefile: {shp_path}")
        except ImportError:
            print("  Shapefile skipped (fiona not installed)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  {len(polygons):,} building footprints extracted")


if __name__ == "__main__":
    main()
