"""Per-model river networks for the MaaS map, plus upstream-area snapping.

Each forecast model runs on its own river network. The MaaS map draws the selected model's network, so a click
lands on one of that model's own forecast elements:

* FloodHub: HydroSHEDS HydroRIVERS reaches (FloodHub's virtual gauges are HydroBASINS units, e.g. `hybas_…`).
* GloFAS v4: the LISFLOOD 0.05° river grid, vectorised from the JRC static maps `ldd_repaired.nc` (PCRaster
  local drain directions) and `upArea_repaired.nc`. Vertices are GloFAS cell centres.
* GEOGLOWS v2: TDX-Hydro streams (`global_streams_simplified.gpkg`, attributes from `v2-model-table.parquet`).
  Zoomed-in views return single reaches with their LINKNO.
* Today's Earth: a CaMa-Flood-style 0.25° unit-catchment network upscaled here from the GloFAS grid (the outlet of
  each 0.25° block is its largest-upstream-area cell, linked to the next outlet down the drain directions). The
  official CaMa-Flood map is not bundled; the Today's Earth forecast itself is emulated from GloFAS (see
  `maas_engine.fetch_todays_earth_forecast`).

Zoomed-out views merge elements into continuous rivers between confluences and drop small rivers, so each request
returns a few thousand lines. `snap_glofas_cell` and `snap_geoglows_reach` pick, near a point, the element whose
upstream area best matches a target, so every model is read on the same river as the clicked one instead of the
nearest small tributary.

Data (downloaded once into ~/.cache/earthkit_hydro/data, or data/base_layers/river_networks/<dir>):
  glofas_v4/{ldd_repaired.nc, upArea_repaired.nc}
    https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/LISFLOOD_static_and_parameter_maps_for_GloFAS/
    v1.1.1_OS-LISFLOOD-v4.x/Catchments_morphology_and_river_network/
  geoglows_v2/global_streams_simplified.gpkg
    https://geoglows-v2.s3-us-west-2.amazonaws.com/hydrography-global/global_streams_simplified.gpkg
  geoglows_v2/geoglows_attrs_v1.npz: LINKNO, DSLINKNO, strmOrder and DSContArea (km²) for reaches >= 50 km²,
    converted from https://geoglows-v2.s3-us-west-2.amazonaws.com/tables/v2-model-table.parquet.
Derived caches (`*_network_v1.npz`) are built on first use (about a minute) or with
`python3 maas_networks.py --build`.
"""

import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from frontend.config import CACHE_DIR, RIVER_NETWORKS_DIR

_LOG = logging.getLogger(__name__)

MODELS = ("floodhub", "glofas", "geoglows", "todays_earth")
NETWORK_LABELS = {
    "floodhub": "HydroSHEDS HydroRIVERS reaches",
    "glofas": "GloFAS v4 LISFLOOD 0.05° river grid",
    "geoglows": "GEOGLOWS v2 TDX-Hydro streams",
    "todays_earth": "CaMa-Flood-style 0.25° unit catchments (upscaled from the GloFAS grid)",
}


def _data_dir(name: str) -> Path:
  path = RIVER_NETWORKS_DIR / name
  return path if path.exists() else CACHE_DIR / name


GLOFAS_DIR = _data_dir("glofas_v4")
GEOGLOWS_DIR = _data_dir("geoglows_v2")
GEOGLOWS_GPKG = GEOGLOWS_DIR / "global_streams_simplified.gpkg"
GEOGLOWS_ATTRS = GEOGLOWS_DIR / "geoglows_attrs_v1.npz"

GLOFAS_RES = 0.05
GLOFAS_NLAT, GLOFAS_NLON = 3600, 7200
GLOFAS_MIN_AREA_KM2 = 100.0
TE_BLOCK = 5  # 0.25° Today's Earth units = 5 x 5 GloFAS cells
TE_MIN_AREA_KM2 = 500.0
GEOGLOWS_MIN_AREA_KM2 = 100.0

# Level of detail per zoom: (max zoom, minimum upstream area km², simplification tolerance in degrees).
# Merged chains have one feature per river between confluences (~2,100 globally at >= 25,000 km²).
GLOFAS_LOD = [(3, 25000, 0.08), (4, 15000, 0.04), (5, 10000, 0.02), (6, 5000, 0.01), (7, 2500, 0.005),
              (8, 1000, 0.0), (9, 500, 0.0), (10, 250, 0.0), (99, 100, 0.0)]
TE_LOD = [(3, 25000, 0.08), (4, 15000, 0.04), (5, 10000, 0.02), (6, 5000, 0.0), (7, 2500, 0.0),
          (8, 1000, 0.0), (99, 500, 0.0)]
# Tolerance None: single reaches straight from the GeoPackage (each with its LINKNO).
GEOGLOWS_LOD = [(3, 25000, 0.08), (4, 15000, 0.04), (5, 10000, 0.02), (6, 5000, 0.01), (7, 2500, 0.005),
                (8, 1000, None), (9, 500, None), (10, 250, None), (99, 100, None)]
# HydroRIVERS minimum Strahler order per zoom (orders >= 7 come from the in-memory pyramid).
FLOODHUB_LOD = [(4, 9), (5, 8), (6, 7), (7, 6), (8, 5), (9, 3), (10, 2), (99, 1)]

_CACHE_VERSION = 1
# PCRaster LDD keypad codes (7 8 9 / 4 5 6 / 1 2 3; 5 = pit) as row/column steps, row 0 = north.
_LDD_DROW = np.array([0, 1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int64)
_LDD_DCOL = np.array([0, -1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int64)

_LOCK = threading.Lock()
_NETWORKS: Dict[str, Dict[str, Any]] = {}
_GEOGLOWS_LOOKUP: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (sorted LINKNO, area km²)


def _lod(table: List[Tuple[Any, ...]], zoom: int) -> Tuple[Any, ...]:
  return next(row for row in table if zoom <= row[0])


def _signature(table: List[Tuple[Any, ...]]) -> str:
  return f"v{_CACHE_VERSION}:" + ";".join(f"{r[1]}/{r[2]}" for r in table if r[2] is not None)


# ---------------------------------------------------------------------------------------------------------------
# Generic tree -> polyline helpers
# ---------------------------------------------------------------------------------------------------------------


def _split_chains(down: np.ndarray, keep: np.ndarray,
                  include_end: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Splits the kept part of a river tree into chains running from a source or confluence downstream.

  Args:
    down: Downstream node index per node (-1 at outlets).
    keep: Nodes in this level of detail. Links to dropped nodes are cut.
    include_end: Repeat the confluence node that ends a chain as its last vertex (point networks), so chains
      join up on the map.

  Returns:
    (nodes, offsets, own): chain i is nodes[offsets[i]:offsets[i + 1]], ordered downstream. `own` is False for
    a repeated confluence node, which belongs to the downstream chain.
  """
  d = np.where(keep, down, -1)
  linked = d >= 0
  d[linked] = np.where(keep[d[linked]], d[linked], -1)
  indeg = np.bincount(d[d >= 0], minlength=len(d))
  heads = np.flatnonzero(keep & (indeg != 1))
  dl, il = d.tolist(), indeg.tolist()
  nodes: List[int] = []
  own: List[bool] = []
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
  return (np.asarray(nodes, dtype=np.int64), np.asarray(offsets, dtype=np.int64),
          np.asarray(own, dtype=bool))


def _gather_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
  """Concatenated index ranges [starts[i], starts[i] + lengths[i])."""
  total = int(lengths.sum())
  if not total:
    return np.zeros(0, dtype=np.int64)
  shift = np.repeat(starts - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths)
  return shift + np.arange(total, dtype=np.int64)


def _pack_level(xy: np.ndarray, line_of_vertex: np.ndarray, n_lines: int, amin: np.ndarray, amax: np.ndarray,
                tol: float) -> Dict[str, np.ndarray]:
  """Builds (optionally simplified) CSR polylines with per-line bounding boxes and area ranges."""
  import shapely  # pylint: disable=g-import-not-at-top

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
  return {"coords": coords.astype(np.float32), "offsets": offsets, "bbox": bbox,
          "amin": amin.astype(np.float32), "amax": amax.astype(np.float32)}


def _point_levels(node_xy: np.ndarray, node_area: np.ndarray, down: np.ndarray,
                  table: List[Tuple[Any, ...]]) -> List[Dict[str, np.ndarray]]:
  levels = []
  for _, min_area, tol in table:
    nodes, offsets, own = _split_chains(down, node_area >= min_area, include_end=True)
    n = len(offsets) - 1
    counts = np.diff(offsets)
    a = np.where(own, node_area[nodes], np.nan)  # a chain's area range excludes its end confluence
    levels.append(_pack_level(node_xy[nodes], np.repeat(np.arange(n), counts), n,
                              np.fmin.reduceat(a, offsets[:-1]) if n else a[:0],
                              np.fmax.reduceat(a, offsets[:-1]) if n else a[:0], tol))
  return levels


def _save_levels(path: Path, levels: List[Dict[str, np.ndarray]], signature: str, **extra: np.ndarray) -> None:
  arrays = {f"L{i}_{k}": v for i, lvl in enumerate(levels) for k, v in lvl.items()}
  tmp = path.with_suffix(".tmp.npz")
  np.savez(tmp, signature=np.array(signature), n_levels=np.array(len(levels)), **arrays, **extra)
  tmp.replace(path)


def _load_levels(path: Path, signature: str) -> Optional[Dict[str, Any]]:
  if not path.exists():
    return None
  with np.load(path) as z:
    if str(z["signature"]) != signature:
      return None
    n = int(z["n_levels"])
    keys = ("coords", "offsets", "bbox", "amin", "amax")
    out: Dict[str, Any] = {"levels": [{k: z[f"L{i}_{k}"] for k in keys} for i in range(n)]}
    for k in z.files:
      if not k.startswith("L") and k not in ("signature", "n_levels"):
        out[k] = z[k]
  return out


# ---------------------------------------------------------------------------------------------------------------
# GloFAS (0.05° LISFLOOD grid) and Today's Earth (0.25° units upscaled from it)
# ---------------------------------------------------------------------------------------------------------------


def _read_glofas_rasters() -> Tuple[np.ndarray, np.ndarray]:
  import xarray as xr  # pylint: disable=g-import-not-at-top

  with xr.open_dataset(GLOFAS_DIR / "upArea_repaired.nc") as ds:
    up = ds["Band1"].values
    lat0 = float(ds["lat"].values[0])
  with xr.open_dataset(GLOFAS_DIR / "ldd_repaired.nc") as ds:
    ldd = ds["Band1"].values
  if up.shape != (GLOFAS_NLAT, GLOFAS_NLON) or abs(lat0 - (90 - GLOFAS_RES / 2)) > 1e-6:
    raise ValueError(f"Unexpected GloFAS grid {up.shape}, first latitude {lat0}")
  up_km2 = np.where(np.isfinite(up), up / 1e6, 0).astype(np.float32)
  ldd = np.where(np.isfinite(ldd), ldd, 0).astype(np.int64)
  return up_km2, ldd


def _step(rows: np.ndarray, cols: np.ndarray, ldd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """One step down the drain directions; links across the antimeridian or off the grid are cut."""
  code = ldd[rows, cols]
  r2, c2 = rows + _LDD_DROW[code], cols + _LDD_DCOL[code]
  ok = (code >= 1) & (code != 5) & (r2 >= 0) & (r2 < GLOFAS_NLAT) & (c2 >= 0) & (c2 < GLOFAS_NLON)
  return np.where(ok, r2, 0), np.where(ok, c2, 0), ok


def _cell_xy(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
  return np.stack([-180.0 + (cols + 0.5) * GLOFAS_RES, 90.0 - (rows + 0.5) * GLOFAS_RES], axis=1)


def _index_of(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
  """Position of each key in sorted_keys, or -1."""
  if not len(sorted_keys):
    return np.full(len(keys), -1, dtype=np.int64)
  pos = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
  return np.where(sorted_keys[pos] == keys, pos, -1)


def _build_glofas_and_te() -> Tuple[Dict[str, Any], Dict[str, Any]]:
  t0 = time.time()
  up, ldd = _read_glofas_rasters()

  # GloFAS river cells (>= GLOFAS_MIN_AREA_KM2) and their downstream cell.
  rows, cols = np.nonzero(up >= GLOFAS_MIN_AREA_KM2)
  lin = rows * GLOFAS_NLON + cols  # row-major, hence sorted
  area = up[rows, cols]
  r2, c2, ok = _step(rows, cols, ldd)
  down = np.where(ok, _index_of(lin, r2 * GLOFAS_NLON + c2), -1)
  glofas = {"levels": _point_levels(_cell_xy(rows, cols), area, down, GLOFAS_LOD),
            "cell_lin": lin.astype(np.int64), "cell_area": area.astype(np.float32)}

  # Today's Earth units: outlet = largest-upstream-area cell of each 0.25° block, linked to the next outlet
  # reached down the drain directions.
  b = TE_BLOCK
  blocks = up.reshape(GLOFAS_NLAT // b, b, GLOFAS_NLON // b, b).transpose(0, 2, 1, 3).reshape(
      GLOFAS_NLAT // b, GLOFAS_NLON // b, b * b)
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
  te = {"levels": _point_levels(_cell_xy(orow, ocol), kmax[bi, bj], te_down, TE_LOD)}
  _LOG.info("Built GloFAS (%d cells) and Today's Earth (%d units) networks in %.1f s", len(lin), len(olin),
            time.time() - t0)
  return glofas, te


# ---------------------------------------------------------------------------------------------------------------
# GEOGLOWS (TDX-Hydro)
# ---------------------------------------------------------------------------------------------------------------


def _geoglows_lookup() -> Tuple[np.ndarray, np.ndarray]:
  """(sorted LINKNO, DSContArea km²) for reaches >= GEOGLOWS_MIN_AREA_KM2."""
  global _GEOGLOWS_LOOKUP
  if _GEOGLOWS_LOOKUP is None:
    with np.load(GEOGLOWS_ATTRS) as z:
      keep = z["area_km2"] >= GEOGLOWS_MIN_AREA_KM2
      _GEOGLOWS_LOOKUP = (z["linkno"][keep], z["area_km2"][keep])
  return _GEOGLOWS_LOOKUP


def _build_geoglows() -> Dict[str, Any]:
  import pyogrio  # pylint: disable=g-import-not-at-top
  import shapely  # pylint: disable=g-import-not-at-top

  t0 = time.time()
  table = [row for row in GEOGLOWS_LOD if row[2] is not None]
  with np.load(GEOGLOWS_ATTRS) as z:
    sel = z["area_km2"] >= min(row[1] for row in table)
    linkno, area, ds_link = z["linkno"][sel], z["area_km2"][sel], z["ds_linkno"][sel]

  # Geometries of the selected reaches, fetched by feature ID.
  ids = pyogrio.read_dataframe(GEOGLOWS_GPKG, read_geometry=False, fid_as_index=True, columns=["LINKNO"])
  gpkg_link = ids["LINKNO"].to_numpy()
  gpkg_fid = ids.index.to_numpy()
  o = np.argsort(gpkg_link)
  pos = _index_of(gpkg_link[o], linkno)
  found = pos >= 0
  linkno, area, ds_link, fids = linkno[found], area[found], ds_link[found], gpkg_fid[o][pos[found]]
  df = pyogrio.read_dataframe(GEOGLOWS_GPKG, fids=fids)
  where = _index_of(linkno, df["LINKNO"].to_numpy())  # align rows with `linkno` order
  geoms = np.empty(len(linkno), dtype=object)
  geoms[where[where >= 0]] = df.geometry.values[where >= 0]
  has_geom = np.array([g is not None for g in geoms])
  linkno, area, ds_link, geoms = linkno[has_geom], area[has_geom], ds_link[has_geom], geoms[has_geom]

  coords, vidx = shapely.get_coordinates(geoms, return_index=True)
  vcount = np.bincount(vidx, minlength=len(geoms))
  voff = np.concatenate([[0], np.cumsum(vcount)]).astype(np.int64)
  down = _index_of(linkno, ds_link)

  # TDX-Hydro reaches are expected to run downstream. Flip any whose first vertex, not its last, touches the
  # downstream reach (either end of it, whatever that reach's own orientation).
  first, last = coords[voff[:-1]], coords[np.maximum(voff[1:] - 1, 0)]
  has_down = down >= 0
  ds = down[has_down]

  def _gap(p: np.ndarray) -> np.ndarray:
    return np.minimum(np.hypot(*(p - first[ds]).T), np.hypot(*(p - last[ds]).T))

  reversed_ = np.zeros(len(geoms), dtype=bool)
  reversed_[has_down] = _gap(first[has_down]) + 1e-9 < _gap(last[has_down])
  for i in np.flatnonzero(reversed_).tolist():
    coords[voff[i]:voff[i + 1]] = coords[voff[i]:voff[i + 1]][::-1]

  levels = []
  for _, min_area, tol in table:
    nodes, offsets, _ = _split_chains(down, area >= min_area, include_end=False)
    n = len(offsets) - 1
    lengths = vcount[nodes]
    vertex = _gather_ranges(voff[nodes], lengths)
    line_of_vertex = np.repeat(np.repeat(np.arange(n), np.diff(offsets)), lengths)
    a = area[nodes]
    levels.append(_pack_level(coords[vertex], line_of_vertex, n, np.minimum.reduceat(a, offsets[:-1]),
                              np.maximum.reduceat(a, offsets[:-1]), tol))
  _LOG.info("Built GEOGLOWS pyramid (%d reaches, %d flipped) in %.1f s", len(linkno), int(reversed_.sum()),
            time.time() - t0)
  return {"levels": levels}


def _geoglows_reaches(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                      min_area: float) -> List[Dict[str, Any]]:
  """Single TDX-Hydro reaches (with LINKNO) from the GeoPackage R-tree, for zoomed-in views."""
  import pyogrio  # pylint: disable=g-import-not-at-top
  from shapely.geometry import mapping  # pylint: disable=g-import-not-at-top

  df = pyogrio.read_dataframe(GEOGLOWS_GPKG, bbox=(min_lon, min_lat, max_lon, max_lat))
  if df.empty:
    return []
  keys, areas = _geoglows_lookup()
  links = df["LINKNO"].to_numpy()
  pos = _index_of(keys, links)
  a = np.where(pos >= 0, areas[np.maximum(pos, 0)], 0.0)
  feats = []
  for i in np.flatnonzero(a >= min_area):
    geom = df.geometry.values[i]
    if geom is None or geom.is_empty:
      continue
    feats.append({"type": "Feature", "geometry": mapping(geom),
                  "properties": {"river_id": int(links[i]), "upstream_area_km2": round(float(a[i]), 1)}})
  return feats


# ---------------------------------------------------------------------------------------------------------------
# Loading and serving
# ---------------------------------------------------------------------------------------------------------------


def _cache_path(model: str) -> Path:
  return (GEOGLOWS_DIR if model == "geoglows" else GLOFAS_DIR) / f"{model}_network_v{_CACHE_VERSION}.npz"


def _network(model: str) -> Optional[Dict[str, Any]]:
  """Loads (building and caching on first use) the gridded/merged network for glofas, todays_earth or geoglows."""
  if model in _NETWORKS:
    return _NETWORKS[model]
  table = {"glofas": GLOFAS_LOD, "todays_earth": TE_LOD, "geoglows": GEOGLOWS_LOD}[model]
  with _LOCK:
    if model in _NETWORKS:
      return _NETWORKS[model]
    net = _load_levels(_cache_path(model), _signature(table))
    if net is None:
      try:
        if model == "geoglows":
          net = _build_geoglows()
          _save_levels(_cache_path(model), net["levels"], _signature(table))
        else:
          glofas, te = _build_glofas_and_te()
          _save_levels(_cache_path("glofas"), glofas["levels"], _signature(GLOFAS_LOD),
                       cell_lin=glofas["cell_lin"], cell_area=glofas["cell_area"])
          _save_levels(_cache_path("todays_earth"), te["levels"], _signature(TE_LOD))
          _NETWORKS["glofas"], _NETWORKS["todays_earth"] = glofas, te
          net = _NETWORKS[model]
      except (OSError, ValueError, KeyError, ImportError) as e:
        _LOG.warning("River network for %s unavailable: %s", model, e)
        return None
    _NETWORKS[model] = net
    return net


def _level_features(level: Dict[str, np.ndarray], min_lon: float, min_lat: float, max_lon: float,
                    max_lat: float) -> List[Dict[str, Any]]:
  b = level["bbox"]
  idx = np.flatnonzero((b[:, 2] >= min_lon) & (b[:, 0] <= max_lon) & (b[:, 3] >= min_lat) & (b[:, 1] <= max_lat))
  coords, offsets, amin, amax = level["coords"], level["offsets"], level["amin"], level["amax"]
  feats = []
  for i in idx.tolist():
    xy = coords[offsets[i]:offsets[i + 1]]
    if len(xy) < 2:
      continue
    feats.append({
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": np.round(xy.astype(np.float64), 4).tolist()},
        "properties": {"upstream_area_km2": round(float(amax[i]), 1), "area_min_km2": round(float(amin[i]), 1)},
    })
  return feats


def _floodhub_features(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                       zoom: int) -> Tuple[List[Dict[str, Any]], int]:
  from frontend import river_indexer  # pylint: disable=g-import-not-at-top

  min_order = _lod(FLOODHUB_LOD, zoom)[1]
  gj = river_indexer.HydroRiverNetwork("hydroatlas").get_rivers_in_bbox(
      min_lon, min_lat, max_lon, max_lat, zoom=zoom, min_stream_order=min_order, prefer_cache=True)
  feats = []
  for f in gj.get("features") or []:
    p = f.get("properties") or {}
    reach_id = p.get("reach_id")
    # New dicts: the pyramid's cached features must not be mutated.
    feats.append({"type": "Feature", "geometry": f.get("geometry"),
                  "properties": {"river_id": reach_id, "upstream_area_km2": p.get("upstream_area_km2"),
                                 "stream_order": p.get("stream_order")}})
  return feats, min_order


def get_model_network(model: str, min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                      zoom: int) -> Dict[str, Any]:
  """GeoJSON FeatureCollection of `model`'s river network in a bounding box at a zoom-dependent level of detail.

  Line properties: `upstream_area_km2` (largest along the line), plus `area_min_km2` for merged rivers, or
  `river_id` for single elements (HydroRIVERS `HYRIV_…`, GEOGLOWS LINKNO).
  """
  model = model if model in MODELS else "floodhub"
  min_lon, max_lon = max(-180.0, min_lon), min(180.0, max_lon)
  min_lat, max_lat = max(-85.0, min_lat), min(85.0, max_lat)
  if min_lon >= max_lon:
    min_lon, max_lon = -180.0, 180.0
  if min_lat >= max_lat:
    min_lat, max_lat = -85.0, 85.0
  props: Dict[str, Any] = {"model": model, "network": NETWORK_LABELS[model], "zoom": zoom}
  feats: List[Dict[str, Any]] = []
  if model == "floodhub":
    feats, props["min_stream_order"] = _floodhub_features(min_lon, min_lat, max_lon, max_lat, zoom)
  else:
    table = {"glofas": GLOFAS_LOD, "todays_earth": TE_LOD, "geoglows": GEOGLOWS_LOD}[model]
    row = _lod(table, zoom)
    props["min_area_km2"] = row[1]
    try:
      if row[2] is None:  # GEOGLOWS single reaches
        feats = _geoglows_reaches(min_lon, min_lat, max_lon, max_lat, row[1])
      else:
        net = _network(model)
        if net is None:
          props["available"] = False
        else:
          level = net["levels"][[r for r in table if r[2] is not None].index(row)]
          feats = _level_features(level, min_lon, min_lat, max_lon, max_lat)
    except (OSError, ValueError, ImportError) as e:
      _LOG.warning("River network query failed for %s: %s", model, e)
      props["available"] = False
  props["feature_count"] = len(feats)
  return {"type": "FeatureCollection", "properties": props, "features": feats}


# ---------------------------------------------------------------------------------------------------------------
# Snapping by upstream area
# ---------------------------------------------------------------------------------------------------------------

# Without an area range: cost = |ln(area / target)| + _DISTANCE_WEIGHT * distance / radius.
_DISTANCE_WEIGHT = 0.35
# With an area range (a merged river line): the nearest element whose area is within the range, give or take this
# factor (the models' drainage areas differ by some percent).
_RANGE_SLACK = 1.15


def _choose(areas: np.ndarray, dist: np.ndarray, radius: float, target: float,
            area_range: Optional[Tuple[float, float]]) -> int:
  if area_range:
    lo, hi = area_range
    inside = np.flatnonzero((areas >= lo / _RANGE_SLACK) & (areas <= hi * _RANGE_SLACK))
    if inside.size:
      return int(inside[np.lexsort((-areas[inside], dist[inside]))[0]])  # nearest, then largest
    target = math.sqrt(lo * hi)
  return int(np.argmin(np.abs(np.log(areas / target)) + _DISTANCE_WEIGHT * dist / max(float(radius), 1e-6)))


def snap_glofas_cell(lat: float, lon: float, target_area_km2: float,
                     area_range: Optional[Tuple[float, float]] = None,
                     radius_cells: int = 3) -> Optional[Dict[str, Any]]:
  """GloFAS river cell near (lat, lon) matching an upstream area (radius 3 cells ≈ 16 km).

  Args:
    lat: Latitude of the clicked point.
    lon: Longitude of the clicked point.
    target_area_km2: Upstream area to match.
    area_range: (min, max) upstream area along the clicked river line, if known. Then the nearest cell in that
      range wins.
    radius_cells: Search radius in GloFAS cells.

  Returns:
    {lat, lon (cell centre), upstream_area_km2, offset_cells}, or None without GloFAS data or a nearby river cell.
  """
  if not target_area_km2 or target_area_km2 <= 0:
    return None
  net = _network("glofas")
  if net is None or "cell_lin" not in net:
    return None
  fr = (90.0 - lat) / GLOFAS_RES - 0.5  # fractional row/column of the point (cell centres are integers)
  fc = (lon + 180.0) / GLOFAS_RES - 0.5
  dr, dc = np.mgrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
  rr, cc = int(round(fr)) + dr.ravel(), int(round(fc)) + dc.ravel()
  ok = (rr >= 0) & (rr < GLOFAS_NLAT) & (cc >= 0) & (cc < GLOFAS_NLON)
  rr, cc = rr[ok], cc[ok]
  pos = _index_of(net["cell_lin"], rr * GLOFAS_NLON + cc)
  hit = pos >= 0
  if not hit.any():
    return None
  rr, cc = rr[hit], cc[hit]
  areas = net["cell_area"][pos[hit]].astype(np.float64)
  dist = np.hypot(rr - fr, (cc - fc) * math.cos(math.radians(lat)))  # in cell heights
  k = _choose(areas, dist, radius_cells, target_area_km2, area_range)
  cell_lat = round(90.0 - (int(rr[k]) + 0.5) * GLOFAS_RES, 3)
  cell_lon = round(-180.0 + (int(cc[k]) + 0.5) * GLOFAS_RES, 3)
  # Open-Meteo's /v1/flood grid is shifted 1 cell (-0.05°) in longitude relative to the JRC GloFAS v4 maps.
  return {"lat": cell_lat, "lon": cell_lon, "query_lat": cell_lat, "query_lon": round(cell_lon - GLOFAS_RES, 3),
          "upstream_area_km2": round(float(areas[k]), 1), "offset_cells": round(float(dist[k]), 2)}


def snap_geoglows_reach(lat: float, lon: float, target_area_km2: float,
                        area_range: Optional[Tuple[float, float]] = None,
                        radius_deg: float = 0.15) -> Optional[Dict[str, Any]]:
  """GEOGLOWS reach near (lat, lon) matching an upstream area; see `snap_glofas_cell` for the arguments.

  Returns:
    {river_id (LINKNO), upstream_area_km2, offset_km}, or None.
  """
  if not target_area_km2 or target_area_km2 <= 0 or not GEOGLOWS_GPKG.exists():
    return None
  import pyogrio  # pylint: disable=g-import-not-at-top
  import shapely  # pylint: disable=g-import-not-at-top

  df = pyogrio.read_dataframe(GEOGLOWS_GPKG, bbox=(lon - radius_deg, lat - radius_deg, lon + radius_deg,
                                                   lat + radius_deg))
  if df.empty:
    return None
  keys, areas = _geoglows_lookup()
  links = df["LINKNO"].to_numpy()
  pos = _index_of(keys, links)
  ok = pos >= 0
  if not ok.any():
    return None
  links, a, geoms = links[ok], areas[pos[ok]].astype(np.float64), df.geometry.values[ok]
  kx = math.cos(math.radians(lat))
  scaled = shapely.transform(geoms, lambda xy: xy * np.array([kx, 1.0]))
  dist = shapely.distance(scaled, shapely.Point(lon * kx, lat))
  k = _choose(a, dist, radius_deg, target_area_km2, area_range)
  return {"river_id": int(links[k]), "upstream_area_km2": round(float(a[k]), 1),
          "offset_km": round(float(dist[k]) * 111.2, 2)}


def _as_linkno(value: Any) -> Optional[int]:
  text = str(value or "").strip()
  return int(text) if text.isdigit() and len(text) == 9 else None


def resolve_click(lat: float, lon: float, upstream_area_km2: Any, area_min_km2: Any = None,
                  network: Optional[str] = None, river_id: Any = None) -> Optional[Dict[str, Any]]:
  """Maps a click on one model's river line to every model's forecast element on the same river.

  Args:
    lat: Latitude of the click (a point on the line).
    lon: Longitude of the click.
    upstream_area_km2: Upstream area of the clicked line (the largest along it, for merged rivers).
    area_min_km2: Smallest upstream area along a merged river line. With `upstream_area_km2` it bounds the
      clicked river, so snapping stays on it rather than jumping to a tributary.
    network: The model whose network was clicked (see `MODELS`).
    river_id: The clicked element's ID (GEOGLOWS LINKNO, HydroRIVERS `HYRIV_…`), if any.

  Returns:
    {network, target_area_km2, glofas: snap_glofas_cell(...) (also used for the Today's Earth forcing),
    geoglows: snap_geoglows_reach(...)}, or None without a valid upstream area.
  """
  try:
    area = float(upstream_area_km2)
  except (TypeError, ValueError):
    return None
  if not math.isfinite(area) or area <= 0:
    return None
  try:
    amin = float(area_min_km2) if area_min_km2 not in (None, "") else None
  except (TypeError, ValueError):
    amin = None
  rng = (amin, area) if amin and 0 < amin <= area else None
  out: Dict[str, Any] = {"network": network, "target_area_km2": area, "glofas": None, "geoglows": None}
  linkno = _as_linkno(river_id)
  if network == "geoglows":
    if linkno is not None:
      out["geoglows"] = {"river_id": linkno, "upstream_area_km2": round(area, 1), "offset_km": 0.0}
    else:
      out["geoglows"] = snap_geoglows_reach(lat, lon, area, area_range=rng)
    if out["geoglows"]:
      out["target_area_km2"] = out["geoglows"]["upstream_area_km2"]
    out["glofas"] = snap_glofas_cell(lat, lon, out["target_area_km2"], radius_cells=4)
  elif network in ("glofas", "todays_earth"):
    out["glofas"] = snap_glofas_cell(lat, lon, area, area_range=rng,
                                     radius_cells=5 if network == "todays_earth" else 3)
    gg_lat, gg_lon = lat, lon
    if out["glofas"]:
      out["target_area_km2"] = out["glofas"]["upstream_area_km2"]
      gg_lat, gg_lon = float(out["glofas"]["lat"]), float(out["glofas"]["lon"])
    out["geoglows"] = snap_geoglows_reach(gg_lat, gg_lon, out["target_area_km2"], radius_deg=0.20)
  else:  # a HydroRIVERS reach (FloodHub network)
    out["glofas"] = snap_glofas_cell(lat, lon, area, radius_cells=4)
    out["geoglows"] = snap_geoglows_reach(lat, lon, area, radius_deg=0.20)
  return out


def build_all() -> None:
  """Builds every derived network cache (run once after downloading the data)."""
  for model in ("glofas", "todays_earth", "geoglows"):
    t = time.time()
    net = _network(model)
    sizes = [len(lvl["offsets"]) - 1 for lvl in net["levels"]] if net else None
    print(f"{model}: lines per level {sizes} ({time.time() - t:.1f} s)", flush=True)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  if "--build" in sys.argv:
    build_all()
