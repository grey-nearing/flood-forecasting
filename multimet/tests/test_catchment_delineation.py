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

"""Unit and integration tests for DEM catchment delineation module."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from multimet.catchment_delineation import (
    RES_DEG,
    TILE_CELLS,
    CatchmentCoverageError,
    DemDelineator,
    is_tile_available,
    latlon_to_tile_key,
    list_available_tiles,
    tile_key_to_filename,
)
from multimet.catchment_delineation.cli import (
    _sanitize_feature_for_export,
    load_coords_from_csv,
    load_coords_from_file,
    main,
    parse_coord_str,
)
from multimet.utils.gcs import is_gcs_path, normalize_gcs_path

_SOUTH_D8: int = 4
_WEST_D8: int = 16
_EAST_D8: int = 1


def _write_synthetic_tile(
    directory: Path, filename: str = 'n40w090.npy'
) -> Path:
    """Create a synthetic 5x5 degree tile in an isolated tmp directory."""
    directory.mkdir(parents=True, exist_ok=True)
    tile_file = directory / filename
    arr = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    for r_c in (381, 1000, 2000):
        for c_c in (1000, 1473, 2000):
            arr[r_c - 1, c_c] = _SOUTH_D8
            arr[r_c - 2, c_c] = _SOUTH_D8
            arr[r_c - 1, c_c + 1] = _WEST_D8
            arr[r_c, c_c - 1] = _EAST_D8
    np.save(tile_file, arr)
    return tile_file


@pytest.mark.unit
def test_tile_key_and_filename() -> None:
    lat_top, lon_left = latlon_to_tile_key(39.6828, -88.7729)
    assert lat_top == 40
    assert lon_left == -90
    assert tile_key_to_filename(lat_top, lon_left) == 'n40w090.npy'

    lat_s, lon_e = latlon_to_tile_key(-12.4, 25.6)
    assert lat_s == -10
    assert lon_e == 25
    assert tile_key_to_filename(lat_s, lon_e) == 's10e025.npy'


@pytest.mark.unit
def test_parse_coord_str() -> None:
    lat, lon = parse_coord_str('39.6828,-88.7729')
    assert pytest.approx(lat, abs=1e-4) == 39.6828
    assert pytest.approx(lon, abs=1e-4) == -88.7729

    lat2, lon2 = parse_coord_str('40.5   -86.2')
    assert pytest.approx(lat2, abs=1e-4) == 40.5
    assert pytest.approx(lon2, abs=1e-4) == -86.2

    with pytest.raises(ValueError, match='Invalid coordinate format'):
        parse_coord_str('invalid_coord')


@pytest.mark.unit
def test_explicit_paths_required() -> None:
    with pytest.raises(ValueError, match='explicit tile source is required'):
        DemDelineator()

    with pytest.raises(ValueError, match='explicit cache_dir is required'):
        DemDelineator(gcs_uri='gs://test-bucket/tiles')


@pytest.mark.unit
def test_load_coords_from_csv(tmp_path: Path) -> None:
    csv_file = tmp_path / 'test_coords.csv'
    with csv_file.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['gauge_id', 'latitude', 'longitude'])
        writer.writerow(['G1', '39.6828', '-88.7729'])
        writer.writerow(['G2', '40.4172', '-86.8858'])

    coords, ids = load_coords_from_csv(csv_file)
    assert len(coords) == 2
    assert ids == ['G1', 'G2']
    assert pytest.approx(coords[0][0], abs=1e-4) == 39.6828
    assert pytest.approx(coords[0][1], abs=1e-4) == -88.7729


@pytest.mark.unit
def test_missing_coordinate_file_raises_not_substituted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing local coordinate files must raise FileNotFoundError."""
    monkeypatch.chdir(tmp_path)
    for missing_name in ('coordinates.csv', 'caravan', 'caravan_coordinates'):
        with pytest.raises(FileNotFoundError, match='does not exist'):
            load_coords_from_file(missing_name)


@pytest.mark.unit
def test_strict_column_resolution(tmp_path: Path) -> None:
    """Column detection must not match substrings like 'y' in survey_year."""
    csv_file = tmp_path / 'survey.csv'
    pd.DataFrame(
        {
            'survey_year': [1999.0],
            'max_flow': [500.0],
            'latitude': [39.6828],
            'longitude': [-88.7729],
        }
    ).to_csv(csv_file, index=False)

    coords, _ = load_coords_from_file(csv_file)
    assert pytest.approx(coords[0][0], abs=1e-4) == 39.6828
    assert pytest.approx(coords[0][1], abs=1e-4) == -88.7729

    bad_csv = tmp_path / 'bbox.csv'
    pd.DataFrame({'xmin': [1.0], 'ymin': [2.0]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match='explicit lat column'):
        load_coords_from_file(bad_csv)

    ambig_csv = tmp_path / 'ambig.csv'
    pd.DataFrame(
        {'lat': [10.0], 'latitude': [39.6828], 'longitude': [-88.7729]}
    ).to_csv(ambig_csv, index=False)
    with pytest.raises(ValueError, match='Ambiguous lat columns'):
        load_coords_from_file(ambig_csv)


@pytest.mark.unit
def test_load_coords_crawls_caravan_attributes_dir(tmp_path: Path) -> None:
    """When given a directory, crawls standard Caravan attributes_other_*.csv."""
    attr_dir = tmp_path / 'attributes' / 'camels'
    attr_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            'gauge_id': ['camels_01013500'],
            'gauge_lat': [39.6828],
            'gauge_lon': [-88.7729],
        }
    ).to_csv(attr_dir / 'attributes_other_camels.csv', index=False)

    coords, ids = load_coords_from_file(tmp_path)
    assert ids == ['camels_01013500']
    assert pytest.approx(coords[0][0], abs=1e-4) == 39.6828


@pytest.mark.unit
def test_synthetic_dem_delineator(tmp_path: Path) -> None:
    tile_arr = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    tile_arr[99, 100] = _SOUTH_D8
    tile_arr[98, 100] = _SOUTH_D8
    tile_arr[99, 101] = _WEST_D8
    np.save(tmp_path / 'n40w090.npy', tile_arr)

    delineator = DemDelineator(tiles_dir=tmp_path)
    out_lat = 40.0 - 100.0 * RES_DEG
    out_lon = -90.0 + 100.0 * RES_DEG

    res = delineator.delineate(lat=out_lat, lon=out_lon, snap_window_cells=1)
    assert res['type'] == 'Feature'
    props = res['properties']
    assert props['upstream_cells_count'] == 4
    assert props['area_km2'] > 0
    assert res['geometry']['type'] in ('Polygon', 'MultiPolygon')
    assert is_tile_available(40, -90, tmp_path)
    assert list_available_tiles(tmp_path) == ['n40w090.npy']


@pytest.mark.unit
def test_cross_tile_delineation_watertight(tmp_path: Path) -> None:
    """Verify seamless watershed traversal across a 5x5 degree tile seam."""
    south = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    north = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    col = 1200
    for row in (0, 1, 2):
        south[row, col] = _SOUTH_D8
    for row in (TILE_CELLS - 1, TILE_CELLS - 2, TILE_CELLS - 3):
        north[row, col] = _SOUTH_D8
    np.save(tmp_path / 'n40w090.npy', south)
    np.save(tmp_path / 'n45w090.npy', north)

    delineator = DemDelineator(tiles_dir=tmp_path)
    lat = 40.0 - 3 * RES_DEG
    lon = -90.0 + col * RES_DEG
    feat = delineator.delineate(lat=lat, lon=lon, snap_window_cells=0)
    assert feat['properties']['upstream_cells_count'] == 7
    assert feat['properties']['tiles_spanned_count'] == 2


@pytest.mark.unit
def test_missing_tile_aborts_in_both_local_and_gcs_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing or failed neighbor tiles must abort in BOTH local and GCS modes."""
    import multimet.catchment_delineation.delineator as cd_del

    south = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    col = 1200
    for row in (0, 1, 2):
        south[row, col] = _SOUTH_D8
    np.save(tmp_path / 'n40w090.npy', south)

    lat = 40.0 - 3 * RES_DEG
    lon = -90.0 + col * RES_DEG

    local_delin = DemDelineator(tiles_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match='n45w090.npy'):
        local_delin.delineate(lat=lat, lon=lon, snap_window_cells=0)

    def _fail_download(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError('Simulated GCS 503 failure')

    monkeypatch.setattr(cd_del, 'download_tile_from_gcs', _fail_download)
    gcs_delin = DemDelineator(
        gcs_uri='gs://test-bucket/tiles', cache_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match='Simulated GCS 503 failure'):
        gcs_delin.delineate(lat=lat, lon=lon, snap_window_cells=0)


@pytest.mark.unit
def test_max_cells_raises_never_truncates(tmp_path: Path) -> None:
    """Exceeding an explicit max_cells cap must raise CatchmentCoverageError."""
    tile_arr = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    for row in range(90, 100):
        tile_arr[row, 100] = _SOUTH_D8
    np.save(tmp_path / 'n40w090.npy', tile_arr)

    delineator = DemDelineator(tiles_dir=tmp_path)
    lat = 40.0 - 100 * RES_DEG
    lon = -90.0 + 100 * RES_DEG

    full_feat = delineator.delineate(lat=lat, lon=lon, snap_window_cells=0)
    assert full_feat['properties']['upstream_cells_count'] == 11

    with pytest.raises(CatchmentCoverageError, match='exceeded max_cells=5'):
        delineator.delineate(lat=lat, lon=lon, snap_window_cells=0, max_cells=5)


@pytest.mark.unit
def test_delineate_batch_missing_in_means_nan_out(tmp_path: Path) -> None:
    """Out-of-coverage or NaN coordinates in a batch produce NaN/None records."""
    _write_synthetic_tile(tmp_path)
    delineator = DemDelineator(tiles_dir=tmp_path)
    coords = [(39.6828, -88.7729), (65.0, -150.0), (float('nan'), -88.0)]
    ids = ['valid_1', 'ooc_north', 'nan_coord']
    fc = delineator.delineate_batch(coords=coords, ids=ids)

    assert len(fc['features']) == 3
    assert fc['features'][0]['geometry'] is not None
    assert fc['features'][0]['properties']['status'] == 'SUCCESS'

    for missing_feat in fc['features'][1:]:
        assert missing_feat['geometry'] is None
        assert math.isnan(missing_feat['properties']['area_km2'])
        assert missing_feat['properties']['status'].startswith('MISSING_DATA')


@pytest.mark.unit
def test_sanitize_feature_rejects_missing_fields_no_zero_fill() -> None:
    """Missing area or outlet coordinates must raise KeyError, never fill 0.0."""
    incomplete_feat = {
        'type': 'Feature',
        'properties': {'catchment_id': 'g1'},
        'geometry': {'type': 'Polygon', 'coordinates': []},
    }
    with pytest.raises(KeyError, match='missing required area'):
        _sanitize_feature_for_export(incomplete_feat)


@pytest.mark.unit
def test_clean_cache_only_removes_newly_downloaded_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--clean-cache must never delete user tiles_dir or pre-existing files."""
    import multimet.catchment_delineation.delineator as cd_del

    user_tiles_dir = tmp_path / 'user_tiles'
    _write_synthetic_tile(user_tiles_dir)
    user_readme = user_tiles_dir / 'USER_NOTES.txt'
    user_readme.write_text('do not delete', encoding='utf-8')

    out_json = tmp_path / 'out_local.geojson'
    rc = main(
        [
            '--lat',
            '39.6828',
            '--lon',
            '-88.7729',
            '--tiles-dir',
            str(user_tiles_dir),
            '--clean-cache',
            '-o',
            str(out_json),
        ]
    )
    assert rc == 0
    assert user_tiles_dir.is_dir()
    assert user_readme.is_file()
    assert (user_tiles_dir / 'n40w090.npy').is_file()

    cache_dir = tmp_path / 'gcs_cache'
    cache_dir.mkdir()
    preexisting = cache_dir / 'keep_me.txt'
    preexisting.write_text('keep', encoding='utf-8')

    def _mock_dl(
        lat_top: int,
        lon_left: int,
        target_dir: str | Path,
        source_uri: str,
        *,
        created_files: set[Path] | None = None,
    ) -> Path:
        del source_uri
        fname = tile_key_to_filename(lat_top, lon_left)
        path = _write_synthetic_tile(Path(target_dir), fname)
        if created_files is not None:
            created_files.add(path)
        return path

    monkeypatch.setattr(cd_del, 'download_tile_from_gcs', _mock_dl)
    rc2 = main(
        [
            '--lat',
            '39.6828',
            '--lon',
            '-88.7729',
            '--gcs-uri',
            'gs://test-bucket/tiles',
            '--cache-dir',
            str(cache_dir),
            '--clean-cache',
            '-o',
            str(tmp_path / 'out_gcs.geojson'),
        ]
    )
    assert rc2 == 0
    assert preexisting.is_file()
    assert not (cache_dir / 'n40w090.npy').exists()


@pytest.mark.unit
def test_cli_preserve_caravan_dirs(tmp_path: Path) -> None:
    """Verify --preserve-caravan-dirs outputs standard shapefiles/<subdataset>."""
    tiles_dir = tmp_path / 'tiles'
    _write_synthetic_tile(tiles_dir)

    csv_file = tmp_path / 'caravan_multi.csv'
    pd.DataFrame(
        {
            'gauge_id': ['camels_01013500', 'grdc_6340110'],
            'gauge_lat': [39.6828, 38.3333],
            'gauge_lon': [-88.7729, -86.8858],
        }
    ).to_csv(csv_file, index=False)

    out_dir = tmp_path / 'caravan_root'
    rc = main(
        [
            '--csv',
            str(csv_file),
            '--tiles-dir',
            str(tiles_dir),
            '--output-dir',
            str(out_dir),
            '--preserve-caravan-dirs',
            '--format',
            'geoparquet',
        ]
    )
    assert rc == 0
    camels_pq = (
        out_dir / 'shapefiles' / 'camels' / 'camels_basin_shapes.geoparquet'
    )
    grdc_pq = out_dir / 'shapefiles' / 'grdc' / 'grdc_basin_shapes.geoparquet'
    assert camels_pq.is_file()
    assert grdc_pq.is_file()
    gdf = gpd.read_parquet(camels_pq)
    assert list(gdf['gauge_id']) == ['camels_01013500']


@pytest.mark.unit
def test_gcs_helpers() -> None:
    assert is_gcs_path('gs://bucket/path')
    assert is_gcs_path('gcs://bucket/path')
    assert not is_gcs_path('/local/path')
    assert normalize_gcs_path('gs:/bucket/path') == 'gs://bucket/path'
    with pytest.raises(ValueError, match='Cannot normalize'):
        normalize_gcs_path('')


@pytest.mark.unit
def test_expected_area_hint_snaps_past_small_tributary(
    tmp_path: Path,
) -> None:
    """When expected_area_km2 is supplied, snap skips a tiny local creek to find the matching river."""
    tile_arr = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    # Small 3-cell creek at (r=100, c=100)
    tile_arr[99, 100] = _SOUTH_D8
    tile_arr[98, 100] = _SOUTH_D8

    # Larger 30-cell river at (r=100, c=135) -- 35 cells east (outside default 12-cell snap window)
    for row in range(71, 101):
        tile_arr[row, 135] = _SOUTH_D8
    np.save(tmp_path / 'n40w090.npy', tile_arr)

    delin = DemDelineator(tiles_dir=tmp_path)
    lat = 40.0 - 100 * RES_DEG
    lon = -90.0 + 100 * RES_DEG

    no_hint = delin.delineate(lat=lat, lon=lon, snap_window_cells=12)
    assert no_hint['properties']['upstream_cells_count'] == 3

    # Expected area for 30 cells at ~40N (~0.0066 km2/cell -> ~0.20 km2)
    with_hint = delin.delineate(
        lat=lat,
        lon=lon,
        snap_window_cells=12,
        expected_area_km2=0.20,
        area_tolerance=0.40,
    )
    assert with_hint['properties']['upstream_cells_count'] == 30


@pytest.mark.unit
def test_expected_area_hint_failure_logs_loudly_and_produces_no_polygon(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """If expected_area_km2 cannot be matched, log loudly to stderr and produce no polygon."""
    from multimet.catchment_delineation import CatchmentAreaMismatchError

    _write_synthetic_tile(tmp_path)
    delin = DemDelineator(tiles_dir=tmp_path)
    with pytest.raises(CatchmentAreaMismatchError, match='AREA HINT FAILURE'):
        delin.delineate(
            lat=39.6828,
            lon=-88.7729,
            expected_area_km2=5000.0,
            catchment_id='bad_gauge_99',
        )
    captured = capsys.readouterr()
    assert '[AREA HINT FAILURE]' in captured.err
    assert 'bad_gauge_99' in captured.err

    out_file = tmp_path / 'should_not_exist.geojson'
    rc = main(
        [
            '--lat',
            '39.6828',
            '--lon',
            '-88.7729',
            '--tiles-dir',
            str(tmp_path),
            '--expected-area',
            '5000.0',
            '-o',
            str(out_file),
        ]
    )
    assert rc == 1
    assert not out_file.exists()


@pytest.mark.unit
def test_missing_local_tile_raises_file_not_found_in_delineate_point_and_area_hint(
    tmp_path: Path,
) -> None:
    """Missing local tile in tiles_dir raises FileNotFoundError in direct and area-hint paths."""
    assert not issubclass(CatchmentCoverageError, FileNotFoundError)

    empty_dir = tmp_path / 'empty_tiles'
    empty_dir.mkdir()
    delin_empty = DemDelineator(tiles_dir=empty_dir)
    with pytest.raises(
        FileNotFoundError, match=r'Missing flow direction tile.*n40w090\.npy'
    ):
        delin_empty.delineate_point(lat=39.6828, lon=-88.7729)

    tiles_dir = tmp_path / 'partial_tiles'
    tiles_dir.mkdir()
    south = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    col = 1200
    for row in (0, 1, 2):
        south[row, col] = _SOUTH_D8
    np.save(tiles_dir / 'n40w090.npy', south)

    lat = 40.0 - 3 * RES_DEG
    lon = -90.0 + col * RES_DEG
    delin = DemDelineator(tiles_dir=tiles_dir)

    with pytest.raises(
        FileNotFoundError, match=r'Missing flow direction tile.*n45w090\.npy'
    ):
        delin.delineate_point(lat=lat, lon=lon, snap_window_cells=0)

    with pytest.raises(
        FileNotFoundError, match=r'Missing flow direction tile.*n45w090\.npy'
    ):
        delin.delineate(
            lat=lat,
            lon=lon,
            snap_window_cells=0,
            expected_area_km2=0.50,
        )


@pytest.mark.unit
def test_area_hint_skips_candidates_exceeding_max_cells_without_try_except(
    tmp_path: Path,
) -> None:
    """Candidate cells exceeding max_cells are skipped cleanly via abort_on_limit=False."""
    tile_arr = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    # Oversized channel at c=100 starting at row=10 so every cell inside
    # the 80-cell window around r=180 has >90 upstream cells (exceeding max_cells=25).
    for row in range(10, 181):
        tile_arr[row, 100] = _SOUTH_D8
    # Matching 10-cell channel at (r=171..180, c=120)
    for row in range(171, 181):
        tile_arr[row, 120] = _SOUTH_D8
    np.save(tmp_path / 'n40w090.npy', tile_arr)

    delin = DemDelineator(tiles_dir=tmp_path)
    vt_over, cnt_over = delin._traverse_upstream_bfs(
        (40, -90), 180, 100, max_cells=25, abort_on_limit=False
    )
    assert vt_over == {}
    assert cnt_over == -1

    lat = 40.0 - 180 * RES_DEG
    lon = -90.0 + 100 * RES_DEG
    feat = delin.delineate(
        lat=lat,
        lon=lon,
        snap_window_cells=5,
        max_cells=25,
        expected_area_km2=0.064,
        area_tolerance=0.25,
    )
    assert feat['properties']['upstream_cells_count'] == 10


@pytest.mark.unit
def test_dem_datasets_and_merit_high_latitude_delineation(tmp_path: Path) -> None:
    """MERIT-Hydro supports 60°N–90°N while HydroSHEDS raises CatchmentCoverageError."""
    from multimet.catchment_delineation import (
        HYDROSHEDS_90M,
        MERIT_HYDRO_90M,
        is_coord_in_coverage,
        is_tile_in_coverage,
        resolve_dem_dataset,
    )

    assert resolve_dem_dataset('hydrosheds') == HYDROSHEDS_90M
    assert resolve_dem_dataset('merit') == MERIT_HYDRO_90M
    assert resolve_dem_dataset(MERIT_HYDRO_90M) == MERIT_HYDRO_90M
    with pytest.raises(ValueError, match='Unknown DEM dataset'):
        resolve_dem_dataset('nonexistent_dem')

    # 63.5°N is outside HydroSHEDS (-56..60) but inside MERIT-Hydro (-60..90)
    assert not is_coord_in_coverage(63.5, -145.2, dataset=HYDROSHEDS_90M)
    assert is_coord_in_coverage(63.5, -145.2, dataset=MERIT_HYDRO_90M)
    assert not is_tile_in_coverage(65, -150, dataset=HYDROSHEDS_90M)
    assert is_tile_in_coverage(65, -150, dataset=MERIT_HYDRO_90M)

    # Write a synthetic high-latitude tile n65w150.npy (covering 60..65°N, -150..-145°E)
    tile_arr = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    for r in range(200, 215):
        tile_arr[r, 300] = _SOUTH_D8
    np.save(tmp_path / 'n65w150.npy', tile_arr)

    hs_delin = DemDelineator(tiles_dir=tmp_path, dataset='hydrosheds_90m')
    with pytest.raises(
        CatchmentCoverageError,
        match=r'outside the global DEM coverage domain \(-56\.0° to 60\.0°',
    ):
        hs_delin.delineate_point(
            lat=65.0 - 210 * RES_DEG, lon=-150.0 + 300 * RES_DEG
        )

    merit_delin = DemDelineator(tiles_dir=tmp_path, dataset='merit_hydro_90m')
    feat = merit_delin.delineate_point(
        lat=65.0 - 210 * RES_DEG,
        lon=-150.0 + 300 * RES_DEG,
        snap_window_cells=5,
    )
    props = feat['properties']
    assert props['dem_id'] == 'merit_hydro_90m'
    assert props['dem_name'] == 'MERIT-Hydro 90m DEM (3 arc-sec)'
    assert props['grid_resolution'] == '90m (3 arc-second)'
    assert (
        props['delineation_method']
        == 'DEM Digital Elevation Flow-Routing (90m MERIT-Hydro Multi-Tile Seamless Grid)'
    )
    assert props['upstream_cells_count'] >= 10


@pytest.mark.unit
def test_elevation_tiles_and_global_grid(tmp_path: Path) -> None:
    """ElevationTiles and GlobalElevationGrid sample elevation and convert nodata to NaN."""
    from multimet.catchment_delineation import ElevationTiles, GlobalElevationGrid

    elv_dir = tmp_path / 'elv_tiles'
    elv_dir.mkdir()
    arr = np.full((TILE_CELLS, TILE_CELLS), 250, dtype=np.int16)
    arr[10, 20] = -9999
    np.save(elv_dir / 'n40w090.npy', arr)

    tiles = ElevationTiles(elv_dir)
    assert tiles.has_tile(40, -90)
    sampled = tiles.sample(np.array([39.5]), np.array([-89.5]))
    assert sampled[0] == pytest.approx(250.0)
    nodata_lat = 40.0 - 10 * RES_DEG
    nodata_lon = -90.0 + 20 * RES_DEG
    nodata_sampled = tiles.sample(
        np.array([nodata_lat]), np.array([nodata_lon])
    )
    assert np.isnan(nodata_sampled[0])

    global_npy = tmp_path / 'global_dem.npy'
    g_arr = np.full((140, 360), 120.0, dtype=np.float32)
    np.save(global_npy, g_arr)
    grid = GlobalElevationGrid(global_npy, res_deg=1.0)
    g_sampled = grid.sample(np.array([10.0]), np.array([20.0]))
    assert g_sampled[0] == pytest.approx(120.0)


@pytest.mark.unit
def test_backend_hydrography_vector_and_hybrid_delineation(tmp_path: Path) -> None:
    """RiverNetwork, UnitCatchmentDelineator, and delineate_hybrid operate with explicit paths."""
    from shapely.geometry import LineString, box
    from multimet.catchment_delineation import (
        HydroBasinsLayer,
        RiverNetwork,
        UnitCatchmentDelineator,
        delineate_hybrid,
    )

    rivers_dir = tmp_path / 'rivers'
    rivers_dir.mkdir()
    basins_dir = tmp_path / 'basins'
    basins_dir.mkdir()
    dem_dir = tmp_path / 'dem'
    dem_dir.mkdir()

    rivers_shp = rivers_dir / 'HydroRIVERS_v10_na.shp'
    rivers_gdf = gpd.GeoDataFrame(
        {
            'HYRIV_ID': [101, 102],
            'NEXT_DOWN': [102, 0],
            'MAIN_RIV': [102, 102],
            'LENGTH_KM': [2.5, 3.0],
            'DIST_DN_KM': [3.0, 0.0],
            'DIST_UP_KM': [2.5, 5.5],
            'CATCH_SKM': [1.2, 1.5],
            'UPLAND_SKM': [1.2, 2.7],
            'DIS_AV_CMS': [0.5, 1.2],
            'ORD_STRA': [2, 3],
            'ORD_CLAS': [1, 1],
            'ORD_FLOW': [6, 5],
            'HYBAS_L12': [1001, 1002],
        },
        geometry=[
            LineString([(-88.78, 39.70), (-88.78, 39.68)]),
            LineString([(-88.78, 39.68), (-88.78, 39.66)]),
        ],
        crs='EPSG:4326',
    )
    rivers_gdf.to_file(rivers_shp)

    basins_gdf = gpd.GeoDataFrame(
        {
            'HYBAS_ID': [1001, 1002],
            'NEXT_DOWN': [1002, 0],
            'NEXT_SINK': [1002, 1002],
            'MAIN_BAS': [1002, 1002],
            'DIST_SINK': [3.0, 0.0],
            'DIST_MAIN': [3.0, 0.0],
            'SUB_AREA': [1.2, 1.5],
            'UP_AREA': [1.2, 2.7],
            'PFAF_ID': [71201, 71202],
            'ENDO': [0, 0],
            'COAST': [0, 0],
            'ORDER': [2, 3],
            'SORT': [1, 2],
        },
        geometry=[
            box(-88.79, 39.68, -88.77, 39.70),
            box(-88.79, 39.66, -88.77, 39.68),
        ],
        crs='EPSG:4326',
    )
    basins_gdf.to_file(basins_dir / 'hybas_na_lev12_v1c.shp')
    _write_synthetic_tile(dem_dir)

    network = RiverNetwork.from_hydrorivers(rivers_shp)
    reaches = network.query_reaches(
        (-88.80, 39.65, -88.75, 39.71), min_stream_order=1
    )
    assert len(reaches) == 2

    snap = network.snap_to_reach(39.67, -88.78)
    assert snap.reach.reach_id == 102

    unit_layer = HydroBasinsLayer(basins_dir)
    vec_delin = UnitCatchmentDelineator(unit_layer)
    vec_res = vec_delin.delineate_exact_pour_point(
        1002, snap.lat, snap.lon, snap.reach.geometry
    )
    assert vec_res.outlet_unit_id == 1002
    assert len(vec_res.unit_ids) == 2
    assert vec_res.area_km2 > 0

    dem_delin = DemDelineator(tiles_dir=dem_dir)
    hyb_feat = delineate_hybrid(
        dem_delin,
        network,
        39.6828,
        -88.7729,
    )
    assert hyb_feat['type'] == 'Feature'
    assert hyb_feat['properties']['area_km2'] > 0


@pytest.mark.unit
def test_merit_d8_tile_download_with_custom_fetcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """download_merit_d8_tile assembles top and bottom halves atomically."""
    import multimet.catchment_delineation.merit as merit_mod
    from multimet.catchment_delineation import download_merit_d8_tile

    def fake_half_tile(
        lat_top: float,
        lon_left: float,
        *,
        ee_project: str,
        credentials: object = None,
        retries: int = 3,
    ) -> np.ndarray:
        assert ee_project == 'test-ee-proj'
        val = 1 if lat_top == 45.0 else 4
        return np.full((3000, 6000), val, dtype=np.uint8)

    monkeypatch.setattr(merit_mod, 'fetch_merit_d8_half_tile', fake_half_tile)

    out_path = download_merit_d8_tile(
        45, -90, tmp_path, ee_project='test-ee-proj'
    )
    assert out_path == tmp_path / 'n45w090.npy'
    loaded = np.load(out_path)
    assert loaded.shape == (6000, 6000)
    assert int(loaded[0, 0]) == 1
    assert int(loaded[3000, 0]) == 4

