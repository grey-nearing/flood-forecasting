# Copyright 2026 Google LLC
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

"""Spatial river-network indexing and upstream-area snapping for `maas`.

Supports four model river networks:
  1. HydroSHEDS HydroRIVERS (`floodhub`)
  2. LISFLOOD 0.05° river grid (`glofas`)
  3. TDX-Hydro streams (`geoglows`)
  4. CaMa-Flood 0.25° unit-catchment network (`todays_earth`)
"""

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyogrio
import shapely
import xarray as xr
from shapely.geometry import Point, mapping

from maas.config import (
    CAMA_GRID_RES_DEG,
    EARTH_RADIUS_KM,
    GEOGLOWS_MIN_AREA_KM2,
    GLOFAS_MIN_AREA_KM2,
    GLOFAS_NLAT,
    GLOFAS_NLON,
    GLOFAS_RES_DEG,
    NETWORK_LABELS,
    PROVIDERS,
    TE_BLOCK,
    TE_MIN_AREA_KM2,
    parse_finite_float,
    parse_int,
)

logger = logging.getLogger(__name__)

MODELS: tuple[str, ...] = PROVIDERS

GLOFAS_LOD: list[tuple[int, float, float]] = [
    (3, 25000, 0.08),
    (4, 15000, 0.04),
    (5, 10000, 0.02),
    (6, 5000, 0.01),
    (7, 2500, 0.005),
    (8, 1000, 0.0),
    (9, 500, 0.0),
    (10, 250, 0.0),
    (99, 100, 0.0),
]

TE_LOD: list[tuple[int, float, float]] = [
    (3, 25000, 0.08),
    (4, 15000, 0.04),
    (5, 10000, 0.02),
    (6, 5000, 0.0),
    (7, 2500, 0.0),
    (8, 1000, 0.0),
    (99, 500, 0.0),
]

GEOGLOWS_LOD: list[tuple[int, float, float | None]] = [
    (3, 25000, 0.08),
    (4, 15000, 0.04),
    (5, 10000, 0.02),
    (6, 5000, 0.01),
    (7, 2500, 0.005),
    (8, 1000, None),
    (9, 500, None),
    (10, 250, None),
    (99, 100, None),
]

FLOODHUB_LOD: list[tuple[int, int]] = [
    (4, 9),
    (5, 8),
    (6, 7),
    (7, 6),
    (8, 5),
    (9, 3),
    (10, 2),
    (99, 1),
]

CACHE_VERSION = 1

# PCRaster LDD keypad codes (7 8 9 / 4 5 6 / 1 2 3; 5 = pit) as row/col steps.
LDD_DROW = np.array([0, 1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int64)
LDD_DCOL = np.array([0, -1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int64)

DISTANCE_WEIGHT = 0.35
RANGE_SLACK = 1.15
MAX_CORRIDOR_SNAP_KM = 15.0


def lod_for_zoom(
    table: Sequence[tuple[Any, ...]],
    zoom: int,
) -> tuple[Any, ...]:
    """Return the first level-of-detail row matching `zoom <= row[0]`."""
    return next(row for row in table if zoom <= row[0])


def pyramid_signature(table: Sequence[tuple[Any, ...]]) -> str:
    """Return a versioned cache signature for a level-of-detail table."""
    return f'v{CACHE_VERSION}:' + ';'.join(
        f'{r[1]}/{r[2]}' for r in table if r[2] is not None
    )


def glofas_cell_center(
    lat: float,
    lon: float,
    res: float = GLOFAS_RES_DEG,
) -> tuple[float, float]:
    """Center of the GloFAS v4 0.05° grid cell containing `(lat, lon)`."""
    return (
        round(math.floor(lat / res) * res + res / 2.0, 3),
        round(math.floor(lon / res) * res + res / 2.0, 3),
    )


def glofas_cell_polygon(
    lat: float,
    lon: float,
    res: float = GLOFAS_RES_DEG,
) -> dict[str, Any]:
    """GeoJSON Feature for the GloFAS v4 0.05° LISFLOOD grid cell."""
    cell_lat, cell_lon = glofas_cell_center(lat, lon, res)
    half = res / 2.0
    box_coords = [
        [round(cell_lon - half, 5), round(cell_lat - half, 5)],
        [round(cell_lon + half, 5), round(cell_lat - half, 5)],
        [round(cell_lon + half, 5), round(cell_lat + half, 5)],
        [round(cell_lon - half, 5), round(cell_lat + half, 5)],
        [round(cell_lon - half, 5), round(cell_lat - half, 5)],
    ]
    area_km2 = round(
        (res * 111.0) * (res * 111.0 * math.cos(math.radians(cell_lat))), 1
    )
    return {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [box_coords],
        },
        'properties': {
            'fabric': 'glofas_cell',
            'fabric_name': 'Copernicus GloFAS 0.05° River Cell',
            'model': 'LISFLOOD Routing Grid',
            'cell_center_lat': round(cell_lat, 4),
            'cell_center_lon': round(cell_lon, 4),
            'area_km2': area_km2,
            'resolution': '0.05° (~5 km)',
            'bbox': {
                'min_lon': round(cell_lon - half, 5),
                'min_lat': round(cell_lat - half, 5),
                'max_lon': round(cell_lon + half, 5),
                'max_lat': round(cell_lat + half, 5),
            },
        },
    }


def snap_cama_cell(
    lat: float,
    lon: float,
    res: float = CAMA_GRID_RES_DEG,
) -> tuple[float, float]:
    """Center of the CaMa-Flood regular grid cell that contains `(lat, lon)`."""
    clat = min(max(float(lat), -89.9999), 89.9999)
    clon = ((float(lon) + 180.0) % 360.0) - 180.0
    return (
        round(math.floor(clat / res) * res + res / 2.0, 4),
        round(math.floor(clon / res) * res + res / 2.0, 4),
    )


def cama_cell_id(cell_lat: float, cell_lon: float) -> str:
    """Canonical CaMa-Flood 0.25° grid cell identifier."""
    return f'cama_025_{cell_lat:.3f}_{cell_lon:.3f}'


def cama_cell_polygon(
    cell_lat: float,
    cell_lon: float,
    res: float = CAMA_GRID_RES_DEG,
) -> tuple[list[list[float]], dict[str, float]]:
    """Closed GeoJSON ring and bbox of a `res x res` cell centered at `(cell_lat, cell_lon)`."""
    half = res / 2.0
    w, e = round(cell_lon - half, 5), round(cell_lon + half, 5)
    s, n = round(cell_lat - half, 5), round(cell_lat + half, 5)
    ring = [[w, s], [e, s], [e, n], [w, n], [w, s]]
    return ring, {'min_lon': w, 'min_lat': s, 'max_lon': e, 'max_lat': n}


def cama_cell_area_km2(
    cell_lat: float,
    res: float = CAMA_GRID_RES_DEG,
) -> float:
    """Exact spherical area (km²) of a `res x res` degree cell centered at `cell_lat`."""
    phi1 = math.radians(max(cell_lat - res / 2.0, -90.0))
    phi2 = math.radians(min(cell_lat + res / 2.0, 90.0))
    return round(
        EARTH_RADIUS_KM
        * EARTH_RADIUS_KM
        * math.radians(res)
        * abs(math.sin(phi2) - math.sin(phi1)),
        2,
    )


def is_geoglows_river_id(value: Any) -> bool:
    """Check whether `value` is a 9-digit GEOGLOWS v2 (TDX-Hydro) river ID."""
    v = parse_int(value)
    if v is None:
        return False
    return 100_000_000 <= v <= 999_999_999


def as_linkno(value: Any) -> int | None:
    """Parse a 9-digit GEOGLOWS `LINKNO`, or return `None`."""
    text = str(value or '').strip()
    return int(text) if text.isdigit() and len(text) == 9 else None


def _split_chains(
    down: np.ndarray,
    keep: np.ndarray,
    include_end: bool,  # noqa: FBT001
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the kept part of a river tree into chains running downstream."""
    d = np.where(keep, down, -1)
    linked = d >= 0
    d[linked] = np.where(keep[d[linked]], d[linked], -1)
    indeg = np.bincount(d[d >= 0], minlength=len(d))
    heads = np.flatnonzero(keep & (indeg != 1))
    dl, il = d.tolist(), indeg.tolist()
    nodes: list[int] = []
    own: list[bool] = []
    offsets = [0]
    min_len = 2 if include_end else 1
    for h in heads.tolist():
        seq = [h]
        c = h
        extra_end = False
        while len(seq) <= len(dl):
            nxt = dl[c]
            if nxt < 0:
                break
            if il[nxt] != 1:
                if include_end:
                    seq.append(nxt)
                    extra_end = True
                break
            seq.append(nxt)
            c = nxt
        if len(seq) >= min_len:
            nodes.extend(seq)
            own.extend([True] * (len(seq) - 1))
            own.append(not extra_end)
            offsets.append(len(nodes))
    return (
        np.asarray(nodes, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(own, dtype=bool),
    )


def _gather_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenated index ranges `[starts[i], starts[i] + lengths[i])`."""
    total = int(lengths.sum())
    if not total:
        return np.zeros(0, dtype=np.int64)
    shift = np.repeat(
        starts - np.concatenate([[0], np.cumsum(lengths)[:-1]]),
        lengths,
    )
    return shift + np.arange(total, dtype=np.int64)


def _pack_level(  # noqa: PLR0913, PLR0917
    xy: np.ndarray,
    line_of_vertex: np.ndarray,
    n_lines: int,
    amin: np.ndarray,
    amax: np.ndarray,
    tol: float,
) -> dict[str, np.ndarray]:
    """Build (optionally simplified) CSR polylines with bounding boxes and area ranges."""
    lines = shapely.linestrings(xy, indices=line_of_vertex)
    if tol and tol > 0:
        lines = shapely.simplify(lines, tol, preserve_topology=False)
    coords, idx = shapely.get_coordinates(lines, return_index=True)
    counts = np.bincount(idx, minlength=n_lines)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    starts = offsets[:-1]
    ok = counts > 0
    bbox = np.zeros((n_lines, 4), dtype=np.float32)
    if ok.any():
        s = starts[ok]
        bbox[ok, 0] = np.minimum.reduceat(coords[:, 0], s)
        bbox[ok, 1] = np.minimum.reduceat(coords[:, 1], s)
        bbox[ok, 2] = np.maximum.reduceat(coords[:, 0], s)
        bbox[ok, 3] = np.maximum.reduceat(coords[:, 1], s)
    bbox[~ok] = np.nan
    return {
        'coords': coords.astype(np.float32),
        'offsets': offsets,
        'bbox': bbox,
        'amin': amin.astype(np.float32),
        'amax': amax.astype(np.float32),
    }


def _point_levels(
    node_xy: np.ndarray,
    node_area: np.ndarray,
    down: np.ndarray,
    table: Sequence[tuple[Any, ...]],
) -> list[dict[str, np.ndarray]]:
    """Build zoom-stratified levels for point-based river networks."""
    levels = []
    for _, min_area, tol in table:
        nodes, offsets, own = _split_chains(
            down, node_area >= min_area, include_end=True
        )
        n = len(offsets) - 1
        counts = np.diff(offsets)
        a = np.where(own, node_area[nodes], np.nan)
        levels.append(
            _pack_level(
                node_xy[nodes],
                np.repeat(np.arange(n), counts),
                n,
                np.fmin.reduceat(a, offsets[:-1]) if n else a[:0],
                np.fmax.reduceat(a, offsets[:-1]) if n else a[:0],
                float(tol or 0.0),
            )
        )
    return levels


def save_network_pyramid(
    path: Path,
    levels: Sequence[Mapping[str, np.ndarray]],
    signature: str,
    **extra: np.ndarray,
) -> None:
    """Atomically save a multi-level river network pyramid to an `.npz` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f'L{i}_{k}': v for i, lvl in enumerate(levels) for k, v in lvl.items()
    }
    tmp = path.with_suffix('.tmp.npz')
    np.savez(
        tmp,
        signature=np.array(signature),
        n_levels=np.array(len(levels)),
        **arrays,
        **extra,
    )
    tmp.replace(path)


def load_network_pyramid(
    path: Path,
    signature: str,
) -> dict[str, Any] | None:
    """Load a cached river network pyramid if `path` exists and matches `signature`."""
    if not path.exists():
        return None
    with np.load(path) as z:
        if 'signature' not in z.files or str(z['signature']) != signature:
            return None
        n = int(z['n_levels'])
        keys = ('coords', 'offsets', 'bbox', 'amin', 'amax')
        out: dict[str, Any] = {
            'levels': [{k: z[f'L{i}_{k}'] for k in keys} for i in range(n)]
        }
        for k in z.files:
            if not k.startswith('L') and k not in ('signature', 'n_levels'):
                out[k] = z[k]
    return out


def _read_glofas_rasters(glofas_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read JRC GloFAS v4 `upArea_repaired.nc` and `ldd_repaired.nc` rasters."""
    up_path = glofas_dir / 'upArea_repaired.nc'
    ldd_path = glofas_dir / 'ldd_repaired.nc'
    if not up_path.exists():
        raise FileNotFoundError(
            f'Missing GloFAS upstream area raster: {up_path}'
        )
    if not ldd_path.exists():
        raise FileNotFoundError(f'Missing GloFAS LDD raster: {ldd_path}')
    with xr.open_dataset(up_path) as ds:
        if 'Band1' not in ds or 'lat' not in ds:
            raise KeyError(
                f"Expected 'Band1' and 'lat' in {up_path}, got {list(ds.variables)}"
            )
        up = ds['Band1'].to_numpy()
        lat0 = float(ds['lat'].to_numpy()[0])
    with xr.open_dataset(ldd_path) as ds:
        if 'Band1' not in ds:
            raise KeyError(
                f"Expected 'Band1' in {ldd_path}, got {list(ds.variables)}"
            )
        ldd = ds['Band1'].to_numpy()
    if (
        up.shape != (GLOFAS_NLAT, GLOFAS_NLON)
        or abs(lat0 - (90.0 - GLOFAS_RES_DEG / 2.0)) > 1e-6
    ):
        raise ValueError(
            f'Unexpected GloFAS grid {up.shape}, first latitude {lat0}'
        )
    up_km2 = np.where(np.isfinite(up), up / 1e6, 0.0).astype(np.float32)
    ldd_arr = np.where(np.isfinite(ldd), ldd, 0).astype(np.int64)
    return up_km2, ldd_arr


def _step(
    rows: np.ndarray,
    cols: np.ndarray,
    ldd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One step down the PCRaster drain directions; off-grid links are cut."""
    nlat, nlon = ldd.shape
    code = ldd[rows, cols]
    r2 = rows + LDD_DROW[code]
    c2 = cols + LDD_DCOL[code]
    ok = (
        (code >= 1)
        & (code != 5)
        & (r2 >= 0)
        & (r2 < nlat)
        & (c2 >= 0)
        & (c2 < nlon)
    )
    return np.where(ok, r2, 0), np.where(ok, c2, 0), ok


def _cell_xy(
    rows: np.ndarray,
    cols: np.ndarray,
    res: float = GLOFAS_RES_DEG,
) -> np.ndarray:
    """Convert `(rows, cols)` indices to `(lon, lat)` cell-center coordinates."""
    return np.stack(
        [-180.0 + (cols + 0.5) * res, 90.0 - (rows + 0.5) * res],
        axis=1,
    )


def _index_of(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Position of each key in `sorted_keys`, or `-1`."""
    if not len(sorted_keys):
        return np.full(len(keys), -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return np.where(sorted_keys[pos] == keys, pos, -1)


def build_glofas_and_te_pyramids(
    glofas_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build both the GloFAS 0.05° and Today's Earth 0.25° pyramids from rasters."""
    t0 = time.time()
    up, ldd = _read_glofas_rasters(glofas_dir)

    rows, cols = np.nonzero(up >= GLOFAS_MIN_AREA_KM2)
    lin = rows * GLOFAS_NLON + cols
    area = up[rows, cols]
    r2, c2, ok = _step(rows, cols, ldd)
    down = np.where(ok, _index_of(lin, r2 * GLOFAS_NLON + c2), -1)
    glofas = {
        'levels': _point_levels(_cell_xy(rows, cols), area, down, GLOFAS_LOD),
        'cell_lin': lin.astype(np.int64),
        'cell_area': area.astype(np.float32),
    }

    b = TE_BLOCK
    blocks = (
        up.reshape(GLOFAS_NLAT // b, b, GLOFAS_NLON // b, b)
        .transpose(0, 2, 1, 3)
        .reshape(GLOFAS_NLAT // b, GLOFAS_NLON // b, b * b)
    )
    k = blocks.argmax(axis=2)
    kmax = np.take_along_axis(blocks, k[..., None], axis=2)[..., 0]
    bi, bj = np.nonzero(kmax >= TE_MIN_AREA_KM2)
    kk = k[bi, bj]
    orow, ocol = bi * b + kk // b, bj * b + kk % b
    olin = orow * GLOFAS_NLON + ocol
    order = np.argsort(olin)
    olin_sorted = olin[order]
    te_down = np.full(len(olin), -1, dtype=np.int64)
    r, c = orow.copy(), ocol.copy()
    active = np.arange(len(olin))
    for _ in range(1000):
        if not active.size:
            break
        r2, c2, ok = _step(r[active], c[active], ldd)
        active, r2, c2 = active[ok], r2[ok], c2[ok]
        r[active], c[active] = r2, c2
        pos = _index_of(olin_sorted, r2 * GLOFAS_NLON + c2)
        hit = pos >= 0
        te_down[active[hit]] = order[pos[hit]]
        active = active[~hit]
    te = {
        'levels': _point_levels(
            _cell_xy(orow, ocol), kmax[bi, bj], te_down, TE_LOD
        )
    }
    logger.info(
        "Built GloFAS (%d cells) and Today's Earth (%d units) networks in %.1f s",
        len(lin),
        len(olin),
        time.time() - t0,
    )
    return glofas, te


def load_geoglows_lookup(
    attrs_path: Path,
    min_area_km2: float = GEOGLOWS_MIN_AREA_KM2,
) -> tuple[np.ndarray, np.ndarray]:
    """Load `(sorted LINKNO, DSContArea km²)` for reaches >= `min_area_km2`."""
    if not attrs_path.exists():
        raise FileNotFoundError(
            f'Missing GEOGLOWS attributes archive: {attrs_path}'
        )
    with np.load(attrs_path) as z:
        keep = z['area_km2'] >= min_area_km2
        return z['linkno'][keep], z['area_km2'][keep]


def build_geoglows_pyramid(geoglows_dir: Path) -> dict[str, Any]:
    """Build the zoom-stratified GEOGLOWS v2 TDX-Hydro pyramid."""
    gpkg_path = geoglows_dir / 'global_streams_simplified.gpkg'
    attrs_path = geoglows_dir / 'geoglows_attrs_v1.npz'
    if not gpkg_path.exists():
        raise FileNotFoundError(f'Missing GEOGLOWS GeoPackage: {gpkg_path}')
    if not attrs_path.exists():
        raise FileNotFoundError(f'Missing GEOGLOWS attributes: {attrs_path}')

    t0 = time.time()
    table = [row for row in GEOGLOWS_LOD if row[2] is not None]
    with np.load(attrs_path) as z:
        sel = z['area_km2'] >= min(row[1] for row in table)
        linkno = z['linkno'][sel]
        area = z['area_km2'][sel]
        ds_link = z['ds_linkno'][sel]

    ids = pyogrio.read_dataframe(
        gpkg_path,
        read_geometry=False,
        fid_as_index=True,
        columns=['LINKNO'],
    )
    gpkg_link = ids['LINKNO'].to_numpy()
    gpkg_fid = ids.index.to_numpy()
    o = np.argsort(gpkg_link)
    pos = _index_of(gpkg_link[o], linkno)
    found = pos >= 0
    linkno = linkno[found]
    area = area[found]
    ds_link = ds_link[found]
    fids = gpkg_fid[o][pos[found]]
    df = pyogrio.read_dataframe(gpkg_path, fids=fids)
    where = _index_of(linkno, df['LINKNO'].to_numpy())
    geoms = np.empty(len(linkno), dtype=object)
    geoms[where[where >= 0]] = df.geometry.to_numpy()[where >= 0]
    has_geom = np.array([g is not None for g in geoms])
    linkno = linkno[has_geom]
    area = area[has_geom]
    ds_link = ds_link[has_geom]
    geoms = geoms[has_geom]

    coords, vidx = shapely.get_coordinates(geoms, return_index=True)
    vcount = np.bincount(vidx, minlength=len(geoms))
    voff = np.concatenate([[0], np.cumsum(vcount)]).astype(np.int64)
    down = _index_of(linkno, ds_link)

    first = coords[voff[:-1]]
    last = coords[np.maximum(voff[1:] - 1, 0)]
    has_down = down >= 0
    ds = down[has_down]

    def _gap(p: np.ndarray) -> np.ndarray:
        return np.minimum(
            np.hypot(*(p - first[ds]).T),
            np.hypot(*(p - last[ds]).T),
        )

    reversed_ = np.zeros(len(geoms), dtype=bool)
    reversed_[has_down] = _gap(first[has_down]) + 1e-9 < _gap(last[has_down])
    for i in np.flatnonzero(reversed_).tolist():
        coords[voff[i] : voff[i + 1]] = coords[voff[i] : voff[i + 1]][::-1]

    levels = []
    for _, min_area, tol in table:
        nodes, offsets, _ = _split_chains(
            down, area >= min_area, include_end=False
        )
        n = len(offsets) - 1
        lengths = vcount[nodes]
        vertex = _gather_ranges(voff[nodes], lengths)
        line_of_vertex = np.repeat(
            np.repeat(np.arange(n), np.diff(offsets)),
            lengths,
        )
        a = area[nodes]
        levels.append(
            _pack_level(
                coords[vertex],
                line_of_vertex,
                n,
                np.minimum.reduceat(a, offsets[:-1]),
                np.maximum.reduceat(a, offsets[:-1]),
                float(tol or 0.0),
            )
        )
    logger.info(
        'Built GEOGLOWS pyramid (%d reaches, %d flipped) in %.1f s',
        len(linkno),
        int(reversed_.sum()),
        time.time() - t0,
    )
    return {'levels': levels}


def query_geoglows_reaches(  # noqa: PLR0913, PLR0917
    gpkg_path: Path,
    lookup: tuple[np.ndarray, np.ndarray],
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    min_area: float,
) -> list[dict[str, Any]]:
    """Query single TDX-Hydro reaches (with `LINKNO`) in a bounding box."""
    if not gpkg_path.exists():
        return []
    df = pyogrio.read_dataframe(
        gpkg_path,
        bbox=(min_lon, min_lat, max_lon, max_lat),
    )
    if df.empty:
        return []
    keys, areas = lookup
    links = df['LINKNO'].to_numpy()
    pos = _index_of(keys, links)
    a = np.where(pos >= 0, areas[np.maximum(pos, 0)], 0.0)
    feats: list[dict[str, Any]] = []
    for i in np.flatnonzero(a >= min_area):
        geom = df.geometry.to_numpy()[i]
        if geom is None or geom.is_empty:
            continue
        feats.append(
            {
                'type': 'Feature',
                'geometry': mapping(geom),
                'properties': {
                    'river_id': int(links[i]),
                    'upstream_area_km2': round(float(a[i]), 1),
                },
            }
        )
    return feats


def extract_level_features(
    level: Mapping[str, np.ndarray],
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> list[dict[str, Any]]:
    """Extract GeoJSON LineString features intersecting a bounding box from a pyramid level."""
    b = level['bbox']
    idx = np.flatnonzero(
        (b[:, 2] >= min_lon)
        & (b[:, 0] <= max_lon)
        & (b[:, 3] >= min_lat)
        & (b[:, 1] <= max_lat)
    )
    coords = level['coords']
    offsets = level['offsets']
    amin = level['amin']
    amax = level['amax']
    feats: list[dict[str, Any]] = []
    for i in idx.tolist():
        xy = coords[offsets[i] : offsets[i + 1]]
        if len(xy) < 2:
            continue
        feats.append(
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': np.round(xy.astype(np.float64), 4).tolist(),
                },
                'properties': {
                    'upstream_area_km2': round(float(amax[i]), 1),
                    'area_min_km2': round(float(amin[i]), 1),
                },
            }
        )
    return feats


def _choose(
    areas: np.ndarray,
    dist: np.ndarray,
    radius: float,
    target: float,
    area_range: tuple[float, float] | None,
) -> int:
    """Select the index best matching `target` upstream area and proximity."""
    if area_range:
        lo, hi = area_range
        inside = np.flatnonzero(
            (areas >= lo / RANGE_SLACK) & (areas <= hi * RANGE_SLACK)
        )
        if inside.size:
            return int(inside[np.lexsort((-areas[inside], dist[inside]))[0]])
        target = math.sqrt(lo * hi)
    return int(
        np.argmin(
            np.abs(np.log(areas / target))
            + DISTANCE_WEIGHT * dist / max(float(radius), 1e-6)
        )
    )


def snap_glofas_cell_from_network(  # noqa: PLR0913, PLR0917
    net: Mapping[str, Any] | None,
    lat: float,
    lon: float,
    target_area_km2: float,
    area_range: tuple[float, float] | None = None,
    radius_cells: int = 3,
    nlat: int = GLOFAS_NLAT,
    nlon: int = GLOFAS_NLON,
    res: float = GLOFAS_RES_DEG,
) -> dict[str, Any] | None:
    """Snap `(lat, lon)` to a GloFAS river cell matching `target_area_km2`."""
    if not target_area_km2 or target_area_km2 <= 0:
        return None
    if net is None or 'cell_lin' not in net or 'cell_area' not in net:
        return None
    fr = (90.0 - lat) / res - 0.5
    fc = (lon + 180.0) / res - 0.5
    dr, dc = np.mgrid[
        -radius_cells : radius_cells + 1,
        -radius_cells : radius_cells + 1,
    ]
    rr = int(round(fr)) + dr.ravel()
    cc = int(round(fc)) + dc.ravel()
    ok = (rr >= 0) & (rr < nlat) & (cc >= 0) & (cc < nlon)
    rr, cc = rr[ok], cc[ok]
    pos = _index_of(net['cell_lin'], rr * nlon + cc)
    hit = pos >= 0
    if not hit.any():
        return None
    rr, cc = rr[hit], cc[hit]
    areas = net['cell_area'][pos[hit]].astype(np.float64)
    dist = np.hypot(rr - fr, (cc - fc) * math.cos(math.radians(lat)))
    k = _choose(areas, dist, radius_cells, target_area_km2, area_range)
    cell_lat = round(90.0 - (int(rr[k]) + 0.5) * res, 3)
    cell_lon = round(-180.0 + (int(cc[k]) + 0.5) * res, 3)
    return {
        'lat': cell_lat,
        'lon': cell_lon,
        'query_lat': cell_lat,
        'query_lon': round(cell_lon - res, 3),
        'upstream_area_km2': round(float(areas[k]), 1),
        'offset_cells': round(float(dist[k]), 2),
    }


def snap_geoglows_reach_from_gpkg(  # noqa: PLR0913, PLR0917
    gpkg_path: Path,
    lookup: tuple[np.ndarray, np.ndarray],
    lat: float,
    lon: float,
    target_area_km2: float,
    area_range: tuple[float, float] | None = None,
    radius_deg: float = 0.15,
) -> dict[str, Any] | None:
    """Snap `(lat, lon)` to a GEOGLOWS TDX-Hydro reach matching `target_area_km2`."""
    if not target_area_km2 or target_area_km2 <= 0 or not gpkg_path.exists():
        return None
    df = pyogrio.read_dataframe(
        gpkg_path,
        bbox=(
            lon - radius_deg,
            lat - radius_deg,
            lon + radius_deg,
            lat + radius_deg,
        ),
    )
    if df.empty:
        return None
    keys, areas = lookup
    links = df['LINKNO'].to_numpy()
    pos = _index_of(keys, links)
    ok = pos >= 0
    if not ok.any():
        return None
    links = links[ok]
    a = areas[pos[ok]].astype(np.float64)
    geoms = df.geometry.to_numpy()[ok]
    kx = math.cos(math.radians(lat))
    scaled = shapely.transform(geoms, lambda xy: xy * np.array([kx, 1.0]))
    dist = shapely.distance(scaled, shapely.Point(lon * kx, lat))
    k = _choose(a, dist, radius_deg, target_area_km2, area_range)
    return {
        'river_id': int(links[k]),
        'upstream_area_km2': round(float(a[k]), 1),
        'offset_km': round(float(dist[k]) * 111.2, 2),
    }


def resolve_cross_network_click(  # noqa: PLR0913, PLR0917
    lat: float,
    lon: float,
    upstream_area_km2: Any,
    snap_glofas_fn: Callable[..., dict[str, Any] | None],
    snap_geoglows_fn: Callable[..., dict[str, Any] | None],
    area_min_km2: Any = None,
    network: str | None = None,
    river_id: Any = None,
) -> dict[str, Any] | None:
    """Map a click on one model's river line to every model's forecast element on the same river."""
    area = parse_finite_float(upstream_area_km2)
    if area is None or area <= 0.0:
        return None
    amin = (
        parse_finite_float(area_min_km2)
        if area_min_km2 not in (None, '')
        else None
    )
    rng = (amin, area) if amin is not None and 0.0 < amin <= area else None
    out: dict[str, Any] = {
        'network': network,
        'target_area_km2': area,
        'glofas': None,
        'geoglows': None,
    }
    linkno = as_linkno(river_id)
    if network == 'geoglows':
        if linkno is not None:
            out['geoglows'] = {
                'river_id': linkno,
                'upstream_area_km2': round(area, 1),
                'offset_km': 0.0,
            }
        else:
            out['geoglows'] = snap_geoglows_fn(lat, lon, area, area_range=rng)
        if out['geoglows']:
            out['target_area_km2'] = out['geoglows']['upstream_area_km2']
        out['glofas'] = snap_glofas_fn(
            lat, lon, out['target_area_km2'], radius_cells=4
        )
    elif network in ('glofas', 'todays_earth'):
        out['glofas'] = snap_glofas_fn(
            lat,
            lon,
            area,
            area_range=rng,
            radius_cells=5 if network == 'todays_earth' else 3,
        )
        gg_lat, gg_lon = lat, lon
        if out['glofas']:
            out['target_area_km2'] = out['glofas']['upstream_area_km2']
            gg_lat = float(out['glofas']['lat'])
            gg_lon = float(out['glofas']['lon'])
        out['geoglows'] = snap_geoglows_fn(
            gg_lat, gg_lon, out['target_area_km2'], radius_deg=0.20
        )
    else:
        out['glofas'] = snap_glofas_fn(lat, lon, area, radius_cells=4)
        out['geoglows'] = snap_geoglows_fn(lat, lon, area, radius_deg=0.20)
    return out


def query_hydrorivers_reaches(
    shp_path: Path | None,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> list[dict[str, Any]]:
    """Query HydroRIVERS reaches intersecting a bounding box."""
    if shp_path is None or not shp_path.exists():
        return []
    df = pyogrio.read_dataframe(
        str(shp_path),
        bbox=(min_lon, min_lat, max_lon, max_lat),
        columns=[
            'HYRIV_ID',
            'NEXT_DOWN',
            'MAIN_RIV',
            'UPLAND_SKM',
            'DIS_AV_CMS',
            'ORD_STRA',
        ],
    )
    reaches: list[dict[str, Any]] = []
    for rec in df.itertuples(index=False):
        geom = rec.geometry
        if geom is None or geom.is_empty:
            continue
        reaches.append(
            {
                'hyriv_id': int(rec.HYRIV_ID),
                'next_down': int(rec.NEXT_DOWN or 0),
                'main_riv': int(rec.MAIN_RIV or 0),
                'upstream_area_km2': float(rec.UPLAND_SKM or 0.0),
                'mean_discharge_m3s': float(rec.DIS_AV_CMS or 0.0),
                'stream_order': int(rec.ORD_STRA or 0),
                'geometry': geom,
            }
        )
    return reaches


def trace_main_stem_chain(
    reaches: Sequence[Mapping[str, Any]],
    lat: float,
    lon: float,
    reach_id: str | None = None,
    max_steps: int = 40,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float | None]:
    """Trace `(start reach, main-stem chain up/downstream, snap distance km)`."""
    if not reaches:
        return None, [], None
    by_id = {int(r['hyriv_id']): dict(r) for r in reaches}
    pt = Point(lon, lat)
    start: dict[str, Any] | None = None
    if reach_id:
        cleaned = str(reach_id).strip().upper().replace('HYRIV_', '')
        parsed_id = parse_int(cleaned)
        if parsed_id is not None:
            start = by_id.get(parsed_id)
    if start is None:
        scored = sorted(
            (r['geometry'].distance(pt), int(r['hyriv_id'])) for r in reaches
        )
        near = [by_id[h] for d, h in scored if d <= scored[0][0] + 0.0135]
        start = max(near, key=lambda r: float(r['upstream_area_km2']))
    snap_km = round(float(start['geometry'].distance(pt)) * 111.32, 2)
    if snap_km > MAX_CORRIDOR_SNAP_KM:
        return start, [], snap_km
    chain = [start]
    seen = {int(start['hyriv_id'])}
    cur = start
    for _ in range(max_steps):
        nxt = by_id.get(int(cur['next_down']))
        if nxt is None or int(nxt['hyriv_id']) in seen:
            break
        chain.append(nxt)
        seen.add(int(nxt['hyriv_id']))
        cur = nxt
    upstream_of: dict[int, list[dict[str, Any]]] = {}
    for r in reaches:
        upstream_of.setdefault(int(r['next_down']), []).append(dict(r))
    cur = start
    for _ in range(max_steps):
        ups = [
            u
            for u in upstream_of.get(int(cur['hyriv_id']), [])
            if int(u['hyriv_id']) not in seen
        ]
        if not ups:
            break
        cur = max(ups, key=lambda u: float(u['upstream_area_km2']))
        chain.append(cur)
        seen.add(int(cur['hyriv_id']))
    return start, chain, snap_km


__all__ = [
    'CACHE_VERSION',
    'FLOODHUB_LOD',
    'GEOGLOWS_LOD',
    'GLOFAS_LOD',
    'MODELS',
    'NETWORK_LABELS',
    'TE_LOD',
    'as_linkno',
    'build_geoglows_pyramid',
    'build_glofas_and_te_pyramids',
    'cama_cell_area_km2',
    'cama_cell_id',
    'cama_cell_polygon',
    'extract_level_features',
    'glofas_cell_center',
    'glofas_cell_polygon',
    'is_geoglows_river_id',
    'load_geoglows_lookup',
    'load_network_pyramid',
    'lod_for_zoom',
    'pyramid_signature',
    'query_geoglows_reaches',
    'query_hydrorivers_reaches',
    'resolve_cross_network_click',
    'save_network_pyramid',
    'snap_cama_cell',
    'snap_geoglows_reach_from_gpkg',
    'snap_glofas_cell_from_network',
    'trace_main_stem_chain',
]
