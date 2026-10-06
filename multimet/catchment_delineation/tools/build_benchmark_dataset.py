# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a stratified multi-continent reference watershed benchmark dataset."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

_KM_PER_DEGREE: float = 111.0
_TIER_MICRO_MAX_KM2: float = 100.0
_TIER_SMALL_MAX_KM2: float = 500.0
_TIER_MEDIUM_MAX_KM2: float = 2500.0
_TIER_LARGE_MAX_KM2: float = 10000.0
_MAX_SAMPLES_PER_TIER: int = 45
_TARGET_PER_CONTINENT: int = 200

_ID_CANDIDATES: tuple[str, ...] = (
    'gauge_id',
    'Unnamed: 0',
    'index',
    'id',
    'station_id',
)
_LAT_CANDIDATES: tuple[str, ...] = (
    'latitude',
    'lat',
    'gauge_lat',
    'CARAVAN:gauge_lat',
    'caravan:gauge_lat',
)
_LON_CANDIDATES: tuple[str, ...] = (
    'longitude',
    'lon',
    'gauge_lon',
    'CARAVAN:gauge_lon',
    'caravan:gauge_lon',
)
_AREA_CANDIDATES: tuple[str, ...] = (
    'calculated_drain_area',
    'reference_area_km2',
    'area_km2',
    'area',
)


def compute_geodesic_area(geom: BaseGeometry) -> float:
    """Compute approximate spherical area in km2 for a WGS84 geometry."""
    centroid_lat = float(geom.centroid.y)
    scale = (
        _KM_PER_DEGREE * _KM_PER_DEGREE * math.cos(math.radians(centroid_lat))
    )
    return float(geom.area * scale)


def get_quadrant(lat: float, lon: float) -> str:
    """Return hemisphere quadrant label (NE, NW, SE, SW)."""
    lat_str = 'N' if lat >= 0 else 'S'
    lon_str = 'E' if lon >= 0 else 'W'
    return f'{lat_str}{lon_str}'


def get_size_tier(area_km2: float) -> str:
    """Classify drainage area into 5 standard size tiers."""
    if area_km2 < _TIER_MICRO_MAX_KM2:
        return '1_micro'
    if area_km2 < _TIER_SMALL_MAX_KM2:
        return '2_small'
    if area_km2 < _TIER_MEDIUM_MAX_KM2:
        return '3_medium'
    if area_km2 < _TIER_LARGE_MAX_KM2:
        return '4_large'
    return '5_macro'


def _resolve_shapefiles(shape_inputs: list[str | Path]) -> list[Path]:
    """Expand --shapes files and directories without hidden fallback paths."""
    shp_paths: list[Path] = []
    for raw in shape_inputs:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f'Shapefile path does not exist: {p}')
        if p.is_dir():
            matched = sorted(p.rglob('*_basin_shapes.shp'))
            if not matched:
                raise FileNotFoundError(
                    f'No *_basin_shapes.shp files found in directory: {p}'
                )
            shp_paths.extend(matched)
        elif p.is_file():
            shp_paths.append(p)
        else:
            raise FileNotFoundError(f'Invalid shapefile path: {p}')
    if not shp_paths:
        raise FileNotFoundError('No shapefiles found from --shapes arguments.')
    return shp_paths


def _find_first_col(
    columns: list[str], candidates: tuple[str, ...]
) -> str | None:
    for cand in candidates:
        if cand in columns:
            return cand
    return None


def _load_coords_csv(csv_path: Path) -> pd.DataFrame:
    """Load and normalize a single coordinate CSV table."""
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f'Coordinate CSV does not exist: {csv_path}')
    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    id_col = _find_first_col(cols, _ID_CANDIDATES)
    lat_col = _find_first_col(cols, _LAT_CANDIDATES)
    lon_col = _find_first_col(cols, _LON_CANDIDATES)
    area_col = _find_first_col(cols, _AREA_CANDIDATES)
    if id_col is None or lat_col is None or lon_col is None:
        raise KeyError(
            f'Coordinate CSV {csv_path} must contain ID, latitude, and '
            f'longitude columns. Found: {cols}'
        )

    norm = pd.DataFrame(
        {
            'gauge_id': df[id_col].astype(str),
            'latitude': pd.to_numeric(df[lat_col], errors='coerce'),
            'longitude': pd.to_numeric(df[lon_col], errors='coerce'),
            'calculated_drain_area': (
                pd.to_numeric(df[area_col], errors='coerce')
                if area_col is not None
                else float('nan')
            ),
        }
    ).dropna(subset=['gauge_id', 'latitude', 'longitude'])

    stripped = norm.copy()
    stripped['gauge_id'] = (
        stripped['gauge_id'].str.lower().str.removeprefix('caravan_')
    )
    combined = pd.concat([norm, stripped], ignore_index=True)
    return combined.drop_duplicates(subset=['gauge_id'], keep='first')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Build stratified reference watershed benchmark dataset.'
    )
    parser.add_argument(
        '--shapes',
        nargs='+',
        required=True,
        help=(
            'One or more shapefile paths or directories containing '
            '*_basin_shapes.shp.'
        ),
    )
    parser.add_argument(
        '--coords-csv',
        nargs='*',
        default=[],
        help='Optional explicit path(s) to coordinate CSV files.',
    )
    parser.add_argument(
        '--world-geojson',
        type=Path,
        required=True,
        help='Explicit path to world continents GeoJSON file.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Explicit output file path (.parquet).',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build benchmark parquet from explicit user-supplied input paths."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    shp_paths = _resolve_shapefiles(args.shapes)
    world_path = Path(args.world_geojson).expanduser().resolve()
    if not world_path.exists() or not world_path.is_file():
        raise FileNotFoundError(
            f'World GeoJSON file does not exist: {world_path}'
        )

    csv_paths: list[Path] = []
    for raw_csv in args.coords_csv or []:
        cp = Path(raw_csv).expanduser().resolve()
        if not cp.exists() or not cp.is_file():
            raise FileNotFoundError(f'Coordinate CSV does not exist: {cp}')
        if cp not in csv_paths:
            csv_paths.append(cp)

    coords_frames = [_load_coords_csv(cp) for cp in csv_paths]
    coords_df = (
        pd.concat(coords_frames, ignore_index=True).drop_duplicates(
            subset=['gauge_id'], keep='first'
        )
        if coords_frames
        else pd.DataFrame(
            columns=[
                'gauge_id',
                'latitude',
                'longitude',
                'calculated_drain_area',
            ]
        )
    )

    world = gpd.read_file(world_path)
    out_file = Path(args.output).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    merged_frames: list[pd.DataFrame] = []
    for shp_p in shp_paths:
        shapes_gdf = gpd.read_file(shp_p)
        shp_cols = list(shapes_gdf.columns)
        id_col = _find_first_col(shp_cols, _ID_CANDIDATES)
        if id_col is None:
            raise KeyError(
                f'Shapefile {shp_p} is missing a gauge_id column. Found: {shp_cols}'
            )
        if id_col != 'gauge_id':
            shapes_gdf = shapes_gdf.rename(columns={id_col: 'gauge_id'})
        shapes_gdf['gauge_id'] = shapes_gdf['gauge_id'].astype(str)

        shp_lat_col = _find_first_col(shp_cols, _LAT_CANDIDATES)
        shp_lon_col = _find_first_col(shp_cols, _LON_CANDIDATES)
        shp_area_col = _find_first_col(shp_cols, _AREA_CANDIDATES)

        if not coords_df.empty:
            merged = shapes_gdf[['gauge_id', 'geometry']].merge(
                coords_df, on='gauge_id', how='left'
            )
        else:
            merged = shapes_gdf[['gauge_id', 'geometry']].copy()
            merged['latitude'] = float('nan')
            merged['longitude'] = float('nan')
            merged['calculated_drain_area'] = float('nan')

        if shp_lat_col is not None and shp_lon_col is not None:
            merged['latitude'] = merged['latitude'].fillna(
                pd.to_numeric(shapes_gdf[shp_lat_col], errors='coerce')
            )
            merged['longitude'] = merged['longitude'].fillna(
                pd.to_numeric(shapes_gdf[shp_lon_col], errors='coerce')
            )
        if shp_area_col is not None:
            merged['calculated_drain_area'] = merged[
                'calculated_drain_area'
            ].fillna(pd.to_numeric(shapes_gdf[shp_area_col], errors='coerce'))

        merged = merged.dropna(
            subset=['gauge_id', 'latitude', 'longitude', 'geometry']
        ).copy()
        if merged.empty:
            continue

        missing_area = merged['calculated_drain_area'].isna()
        if missing_area.any():
            merged.loc[missing_area, 'calculated_drain_area'] = merged.loc[
                missing_area, 'geometry'
            ].apply(compute_geodesic_area)

        pts = [
            Point(xy)
            for xy in zip(merged['longitude'], merged['latitude'], strict=True)
        ]
        pts_gdf = gpd.GeoDataFrame(
            merged[['gauge_id']], geometry=pts, crs='EPSG:4326'
        )
        joined = gpd.sjoin(
            pts_gdf,
            world[['continent', 'geometry']],
            how='left',
            predicate='within',
        ).drop_duplicates(subset=['gauge_id'])
        merged = merged.merge(
            joined[['gauge_id', 'continent']], on='gauge_id', how='left'
        ).dropna(subset=['continent'])
        if not merged.empty:
            merged_frames.append(merged)

    export_cols = [
        'gauge_id',
        'continent',
        'hemisphere',
        'size_tier',
        'latitude',
        'longitude',
        'reference_area_km2',
        'geometry_wkt',
    ]
    if not merged_frames:
        raise ValueError(
            'No valid benchmark basins matched the input shapefiles, '
            'coordinates, and world continent polygons.'
        )

    all_basins = pd.concat(merged_frames, ignore_index=True).drop_duplicates(
        subset=['gauge_id'], keep='first'
    )

    continents = [
        'Africa',
        'Asia',
        'Europe',
        'North America',
        'South America',
        'Oceania',
    ]
    cols = [
        'gauge_id',
        'latitude',
        'longitude',
        'calculated_drain_area',
        'continent',
        'geometry',
    ]
    candidates: list[pd.DataFrame] = []

    for cont in continents:
        pool = all_basins[all_basins['continent'] == cont][cols].copy()
        if pool.empty:
            continue
        pool['size_tier'] = pool['calculated_drain_area'].apply(get_size_tier)
        sampled_cont: list[pd.DataFrame] = []
        for tier in sorted(pool['size_tier'].unique()):
            sub = pool[pool['size_tier'] == tier]
            n_take = min(len(sub), _MAX_SAMPLES_PER_TIER)
            sampled_cont.append(sub.sample(n=n_take, random_state=42))

        cont_df = pd.concat(sampled_cont, ignore_index=True)
        if len(cont_df) > _TARGET_PER_CONTINENT:
            cont_df = cont_df.sample(n=_TARGET_PER_CONTINENT, random_state=42)
        elif len(cont_df) < _TARGET_PER_CONTINENT and len(pool) > len(cont_df):
            rem = pool[~pool['gauge_id'].isin(cont_df['gauge_id'])]
            needed = min(_TARGET_PER_CONTINENT - len(cont_df), len(rem))
            cont_df = pd.concat(
                [cont_df, rem.sample(n=needed, random_state=42)],
                ignore_index=True,
            )
        candidates.append(cont_df)

    final_df = pd.concat(candidates, ignore_index=True)
    final_df['hemisphere'] = final_df.apply(
        lambda r: get_quadrant(r['latitude'], r['longitude']), axis=1
    )
    final_df['size_tier'] = final_df['calculated_drain_area'].apply(
        get_size_tier
    )
    final_df['geometry_wkt'] = final_df['geometry'].apply(lambda g: g.wkt)
    final_df['reference_area_km2'] = final_df['calculated_drain_area'].round(2)

    out_df = final_df[export_cols].copy()
    out_df.to_parquet(out_file, index=False)
    sys.stdout.write(
        f'Benchmark dataset created: {len(out_df)} basins -> {out_file}\n'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
