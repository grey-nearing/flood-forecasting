# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static Catchment Attribute Extraction Engine for Caravan & HydroATLAS.

Interfaces with global BasinATLAS (HydroATLAS v1.0 Level 12) local geodatabase
or shapefile to compute exact area-weighted physiographic, hydro-environmental,
soil, land-cover, climatology, and anthropogenic attributes for user-supplied
watershed polygons following the Caravan aggregation methodology:

- Area-weighted majority voting for discrete categorical classes
- Downstream topological outlet tracing via NEXT_DOWN for pour-point metrics
- Area-weighted averaging for continuous physiographic & hydro-climatic properties
- 40-year ERA5 climate indices (1981-2020) resolved by default in the order:
  pre-extracted basin timeseries (if available) -> HydroATLAS Level 12 precomputed
  tables ('hybas') -> gridded ERA5-Land Zarr archive ('gridded'), with an override
  option (`prefer_gridded_over_hybas=True` / `--prefer-gridded-over-hybas`) to
  prioritize 'gridded' ahead of 'hybas'.
"""

from __future__ import annotations

from collections import defaultdict
import io
import logging
import math
from pathlib import Path
import re
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pyproj
import shapely.geometry
from tqdm.auto import tqdm
import xarray as xr

from multimet.static_extractor.climate import (
    ERA5ClimateLoader,
    ERA5GriddedExtractor,
    ERA5TimeseriesZarrLoader,
    compute_climate_metrics_from_dataframe,
)
from multimet.static_extractor.schema import CatchmentAttributes
from multimet.static_extractor.config import (
    ADDITIONAL_PROPERTIES,
    CONTINENT_BBOXES,
    IGNORE_PROPERTIES,
    MAJORITY_PROPERTIES,
    POUR_POINT_PROPERTIES,
    UPSTREAM_PROPERTIES,
)
from multimet.utils.gcs import (
    download_hydroatlas_from_gcs,
    is_gcs_path,
    normalize_gcs_path,
    strip_gcs_prefix,
)

shape = shapely.geometry.shape
Point = shapely.geometry.Point
Polygon = shapely.geometry.Polygon
MultiPolygon = shapely.geometry.MultiPolygon
box = shapely.geometry.box
_WGS84_GEOD = pyproj.Geod(ellps="WGS84")

warnings.filterwarnings("ignore", category=FutureWarning, module="google.auth.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.auth.*")

logger = logging.getLogger(__name__)

_VALID_SINGLE_ERA5_SOURCES = frozenset({"timeseries", "hybas", "gridded"})

_WORKER_EXTRACTOR: Optional[StaticAttributesExtractor] = None
_WORKER_EXTRACTOR_KEY: Optional[Tuple[Any, ...]] = None


class _WorkerExtractTask(NamedTuple):
  geom: Any
  gid: str
  min_overlap_threshold: float
  era5_source: str
  gdb_path: Optional[str]
  era5_cache_dir: Optional[str]
  gridded_era5_uri: Optional[str]
  skip_climate: bool
  gcs_gdb_uri: Optional[str] = None
  gcs_era5_climate_uri: Optional[str] = None
  no_download: bool = False
  era5_timeseries_zarr: Optional[Tuple[str, ...]] = None
  prefer_gridded_over_hybas: bool = False


def _normalize_hybas_id_series(s: pd.Series) -> pd.Series:
  """Normalizes HYBAS_ID values (e.g. 'hybas_7120000010' or int) to int64."""
  if s.dtype == object or pd.api.types.is_string_dtype(s):
    return s.astype(str).str.replace("hybas_", "", regex=False).astype("int64")
  return s.astype("int64")


def _bboxes_intersect(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> bool:
  """Returns True if two (minx, miny, maxx, maxy) bounding boxes intersect."""
  return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])


def _normalize_timeseries_zarr_arg(
    era5_timeseries_zarr: Optional[Union[str, Path, Sequence[Union[str, Path]]]]
) -> Optional[Tuple[str, ...]]:
  """Normalizes an optional timeseries Zarr path/URI or sequence into a tuple of strings."""
  if era5_timeseries_zarr is None:
    return None
  if isinstance(era5_timeseries_zarr, (str, Path)):
    parts = tuple(
        s.strip() for s in str(era5_timeseries_zarr).split(",") if s.strip()
    )
    return parts if parts else None
  parts = tuple(str(u).strip() for u in era5_timeseries_zarr if str(u).strip())
  return parts if parts else None


def _validate_era5_source_spec(era5_source: Optional[str]) -> Optional[str]:
  """Validates an era5_source string ('auto', 'timeseries', 'hybas', 'gridded', or comma-separated list)."""
  if era5_source is None:
    return None
  cleaned = era5_source.strip().lower()
  if not cleaned or cleaned == "auto":
    return "auto"
  tokens = [t.strip() for t in cleaned.split(",") if t.strip()]
  if not tokens or any(t not in _VALID_SINGLE_ERA5_SOURCES for t in tokens):
    raise ValueError(
        f"Invalid era5_source {era5_source!r}; must be 'auto', 'timeseries', "
        "'hybas', 'gridded', or a comma-separated priority list of those sources."
    )
  return ",".join(tokens)


def _get_worker_extractor(
    gdb_path: Optional[str],
    era5_cache_dir: Optional[str],
    gridded_era5_uri: Optional[str],
    gcs_gdb_uri: Optional[str] = None,
    gcs_era5_climate_uri: Optional[str] = None,
    no_download: bool = False,
    era5_timeseries_zarr: Optional[Tuple[str, ...]] = None,
    prefer_gridded_over_hybas: bool = False,
) -> StaticAttributesExtractor:
  global _WORKER_EXTRACTOR, _WORKER_EXTRACTOR_KEY
  key = (
      gdb_path,
      era5_cache_dir,
      gridded_era5_uri,
      gcs_gdb_uri,
      gcs_era5_climate_uri,
      no_download,
      era5_timeseries_zarr,
      prefer_gridded_over_hybas,
  )
  if _WORKER_EXTRACTOR is None or _WORKER_EXTRACTOR_KEY != key:
    _WORKER_EXTRACTOR = StaticAttributesExtractor(
        gdb_path=gdb_path,
        era5_cache_dir=era5_cache_dir,
        gridded_era5_uri=gridded_era5_uri,
        era5_timeseries_zarr=era5_timeseries_zarr,
        prefer_gridded_over_hybas=prefer_gridded_over_hybas,
        gcs_gdb_uri=gcs_gdb_uri,
        gcs_era5_climate_uri=gcs_era5_climate_uri,
        no_download=no_download,
    )
    _WORKER_EXTRACTOR_KEY = key
  return _WORKER_EXTRACTOR


def _worker_extract_polygon(args: tuple) -> CatchmentAttributes:
  task = args if isinstance(args, _WorkerExtractTask) else _WorkerExtractTask(*args)
  ext = _get_worker_extractor(
      task.gdb_path,
      task.era5_cache_dir,
      None if task.skip_climate else task.gridded_era5_uri,
      gcs_gdb_uri=task.gcs_gdb_uri,
      gcs_era5_climate_uri=task.gcs_era5_climate_uri,
      no_download=task.no_download,
      era5_timeseries_zarr=None if task.skip_climate else task.era5_timeseries_zarr,
      prefer_gridded_over_hybas=False if task.skip_climate else task.prefer_gridded_over_hybas,
  )
  return ext.extract_attributes_for_polygon(
      task.geom,
      catchment_id=task.gid,
      min_overlap_threshold=task.min_overlap_threshold,
      era5_source=task.era5_source,
      prefer_gridded_over_hybas=task.prefer_gridded_over_hybas,
      _batch_mode=True,
      _skip_climate=task.skip_climate,
  )


def compute_pour_point_properties(
    basin_data: Dict[str, List[Any]],
    min_overlap_threshold: float = 0.0,
    pour_point_properties: Optional[List[str]] = None,
) -> Dict[str, float]:
  """Computes Caravan pour-point metrics by following NEXT_DOWN to find the outlet sub-basins."""
  props_to_compute = pour_point_properties or POUR_POINT_PROPERTIES
  if not props_to_compute:
    return {}

  weights = np.array(basin_data.get("weights", []))
  sub_areas = np.array(basin_data.get("SUB_AREA", []))
  if len(weights) == 0 or len(sub_areas) == 0:
    return {p: np.nan for p in props_to_compute}

  percentage_overlap = np.where(sub_areas > 0, weights / sub_areas, 0.0)
  if len(percentage_overlap) == 0:
    return {p: np.nan for p in props_to_compute}

  current_basin_pos = int(np.argmax(percentage_overlap))
  next_down_id = basin_data["NEXT_DOWN"][current_basin_pos]

  # Traverse downstream until leaving the polygon or hitting ocean (0)
  while True:
    if next_down_id == 0:
      break
    if next_down_id not in basin_data["HYBAS_ID"]:
      break
    next_down_pos = basin_data["HYBAS_ID"].index(next_down_id)
    if percentage_overlap[next_down_pos] < 0.5:
      break
    next_down_id = basin_data["NEXT_DOWN"][next_down_pos]

  # Find all sub-basins draining into the terminal downstream outlet
  direct_upstream_polygons = []
  for i, next_down in enumerate(basin_data["NEXT_DOWN"]):
    if (next_down == next_down_id) and (
        (basin_data["weights"][i] > min_overlap_threshold)
        or (percentage_overlap[i] > 0.5)
    ):
      direct_upstream_polygons.append(i)

  if not direct_upstream_polygons:
    direct_upstream_polygons = [current_basin_pos]

  aggregated = {}
  for prop in props_to_compute:
    if prop in basin_data:
      aggregated[prop] = float(
          sum(basin_data[prop][i] for i in direct_upstream_polygons)
      )
    else:
      aggregated[prop] = np.nan
  return aggregated


class StaticAttributesExtractor:
  """Extracts and computes Caravan & HydroATLAS static catchment attributes."""

  def __init__(
      self,
      gdb_path: Optional[Union[str, Path]] = None,
      era5_source: Optional[str] = "auto",
      era5_cache_dir: Optional[Union[str, Path]] = None,
      gridded_era5_uri: Optional[Union[str, Path]] = None,
      era5_timeseries_zarr: Optional[
          Union[str, Path, Sequence[Union[str, Path]]]
      ] = None,
      prefer_gridded_over_hybas: bool = False,
      gcs_gdb_uri: Optional[str] = None,
      gcs_era5_climate_uri: Optional[str] = None,
      no_download: bool = False,
  ):
    """Initializes the StaticAttributesExtractor.

    Args:
      gdb_path: Path to local BasinATLAS_v10.gdb directory, shapefile, or
        GeoParquet file (or GCS URI when streaming in memory).
      era5_source: Sourcing mode for ERA5 climate attributes. Defaults to
        ``"auto"``, which checks configured sources in the order:
        ``timeseries -> hybas -> gridded`` (or ``timeseries -> gridded -> hybas``
        when ``prefer_gridded_over_hybas=True``). Can also be set to an explicit
        single source (``"timeseries"``, ``"hybas"``, or ``"gridded"``) or a
        comma-separated priority list (e.g. ``"timeseries,gridded,hybas"``).
      era5_cache_dir: Directory containing continental ERA5 climate index files
        for ``hybas`` mode.
      gridded_era5_uri: GCS URI or local path to gridded daily ERA5-Land Zarr
        store for ``gridded`` mode (and optional fallback for ``*_ERA5_LAND``
        columns if a legacy ``hybas`` table omits them).
      era5_timeseries_zarr: Optional path/URI (or sequence of paths/URIs) to
        pre-extracted basin-level ``ERA5_LAND/timeseries.zarr`` store(s).
      prefer_gridded_over_hybas: If True, prioritizes ``gridded`` ahead of
        ``hybas`` when ``era5_source="auto"``, changing the resolution order to
        ``timeseries -> gridded -> hybas``.
      gcs_gdb_uri: Optional GCS URI for HydroATLAS data. Downloaded to gdb_path
        when no_download=False, or streamed directly in memory when
        no_download=True.
      gcs_era5_climate_uri: Optional GCS URI for continental ERA5 climate tables.
        Downloaded to era5_cache_dir when no_download=False, or streamed
        directly in memory when no_download=True.
      no_download: If True, streams HydroATLAS and ERA5 data directly in memory
        from Google Cloud Storage without writing files to local disk.
    """
    self.era5_source = _validate_era5_source_spec(era5_source)
    self.prefer_gridded_over_hybas = bool(prefer_gridded_over_hybas)
    self.no_download = bool(no_download)
    self.gcs_gdb_uri = gcs_gdb_uri

    self._hydroatlas_mode: str = "pyogrio"
    self._available_levels: tuple[int, ...] = (12,)
    self._level_layers: dict[int, str] = {}
    self._in_memory_gdf: Optional[gpd.GeoDataFrame] = None
    self._subpolygon_files: Dict[str, str] = {}
    self._cloud_attr_pq_path: Optional[str] = None
    self._cloud_attr_parquet_bytes: Optional[bytes] = None
    self._cloud_continent_gdfs: Dict[str, gpd.GeoDataFrame] = {}
    self.is_shapefile: bool = False
    self.layer_name: Optional[str] = None

    if self.no_download:
      if gcs_gdb_uri:
        active_source = str(gcs_gdb_uri)
      elif gdb_path:
        active_source = str(gdb_path)
      else:
        raise ValueError(
            "Either gdb_path or gcs_gdb_uri must be explicitly provided."
        )
      self.gdb_path = (
          Path(gdb_path) if (gdb_path and not is_gcs_path(gdb_path)) else None
      )
    else:
      if not gdb_path:
        raise ValueError("gdb_path must be explicitly provided.")
      if is_gcs_path(gdb_path):
        active_source = str(gdb_path)
        self.gdb_path = None
        self.no_download = True
      else:
        self.gdb_path = Path(gdb_path)
        if (
            not self.gdb_path.exists()
            or (self.gdb_path.is_dir() and not any(self.gdb_path.iterdir()))
        ):
          if gcs_gdb_uri:
            logger.info(
                "BasinATLAS GDB not found at %s. Downloading from %s...",
                self.gdb_path,
                gcs_gdb_uri,
            )
            self.gdb_path = download_hydroatlas_from_gcs(
                target_dir=self.gdb_path, source_uri=gcs_gdb_uri
            )
          else:
            raise FileNotFoundError(
                f"BasinATLAS dataset not found at {self.gdb_path}."
            )
        active_source = str(self.gdb_path)

    self._init_hydroatlas_source(active_source)

    self.use_properties = [
        p
        for p in self.all_gdb_fields
        if p not in IGNORE_PROPERTIES + UPSTREAM_PROPERTIES
    ]
    self.caravan_feature_names = [
        p for p in self.use_properties if p not in ADDITIONAL_PROPERTIES
    ]

    self.era5_cache_dir = (
        Path(era5_cache_dir)
        if (era5_cache_dir and not is_gcs_path(era5_cache_dir))
        else None
    )
    self.gcs_era5_climate_uri = (
        str(era5_cache_dir)
        if (era5_cache_dir and is_gcs_path(era5_cache_dir) and not gcs_era5_climate_uri)
        else gcs_era5_climate_uri
    )

    has_era5_hybas_config = (
        self.era5_cache_dir is not None
        or self.gcs_era5_climate_uri is not None
    )
    self.era5_loader = (
        ERA5ClimateLoader(
            cache_dir=self.era5_cache_dir,
            gcs_source_uri=self.gcs_era5_climate_uri,
            no_download=self.no_download,
        )
        if has_era5_hybas_config
        else None
    )

    self.gridded_era5_uri = str(gridded_era5_uri) if gridded_era5_uri else None
    self.gridded_extractor = (
        ERA5GriddedExtractor(zarr_uri=self.gridded_era5_uri)
        if self.gridded_era5_uri is not None
        else None
    )

    self.era5_timeseries_zarr = _normalize_timeseries_zarr_arg(
        era5_timeseries_zarr
    )
    self.timeseries_loader = (
        ERA5TimeseriesZarrLoader(self.era5_timeseries_zarr)
        if self.era5_timeseries_zarr is not None
        else None
    )

    if self.era5_source == "hybas" and self.era5_loader is None:
      raise ValueError(
          "era5_cache_dir (or gcs_era5_climate_uri with no_download=True) must be provided when era5_source='hybas'."
      )
    if self.era5_source == "gridded" and self.gridded_extractor is None:
      raise ValueError("gridded_era5_uri must be provided when era5_source='gridded'.")
    if self.era5_source == "timeseries" and self.timeseries_loader is None:
      raise ValueError(
          "era5_timeseries_zarr must be provided when era5_source='timeseries'."
      )
    if self.prefer_gridded_over_hybas and self.gridded_extractor is None:
      raise ValueError(
          "gridded_era5_uri must be provided when prefer_gridded_over_hybas=True."
      )


  @property
  def available_levels(self) -> tuple[int, ...]:
    """Returns the HydroATLAS levels available in this extractor."""
    return tuple(self._available_levels)

  @staticmethod
  def _validate_subbasin_bbox(bbox: tuple[float, float, float, float]) -> None:
    """Validates a geographic bounding box `(minx, miny, maxx, maxy)`."""
    if len(bbox) != 4:
      raise ValueError("bbox must be a tuple of 4 floats")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) for v in bbox
    ):
      raise ValueError("bbox values must be numbers")
    if not all(math.isfinite(v) for v in bbox):
      raise ValueError("bbox values must be finite")
    minx, miny, maxx, maxy = bbox
    if minx >= maxx or miny >= maxy:
      raise ValueError(
          "bbox must be (minx, miny, maxx, maxy) and have non-zero area"
      )
    if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
      raise ValueError("bbox values out of range")

  @staticmethod
  def _subset_subbasin_columns(
      gdf: gpd.GeoDataFrame,
      columns: list[str] | None,
  ) -> gpd.GeoDataFrame:
    """Normalizes and subsets a sub-basin GeoDataFrame to requested columns."""
    if not columns:
      return gdf
    ordered_cols = list(columns)
    if "HYBAS_ID" not in ordered_cols:
      ordered_cols.insert(0, "HYBAS_ID")
    if "geometry" not in ordered_cols:
      ordered_cols.append("geometry")
    if gdf.empty:
      return gpd.GeoDataFrame(columns=ordered_cols, crs="EPSG:4326")
    missing_in_gdf = [c for c in ordered_cols if c not in gdf.columns]
    if missing_in_gdf:
      raise KeyError(
          f"Requested column {missing_in_gdf[0]!r} missing from sub-basin layer"
      )
    return gdf[ordered_cols]

  def read_subbasins(
      self,
      bbox: tuple[float, float, float, float],
      level: int = 12,
      columns: list[str] | None = None,
  ) -> gpd.GeoDataFrame:
    """Reads sub-basins within bounding box."""
    if (
        isinstance(level, bool)
        or not isinstance(level, int)
        or level not in self.available_levels
    ):
      raise UnsupportedHydroATLASLevelError(f"level {level} is not available")

    self._validate_subbasin_bbox(bbox)

    if columns:
      missing = [
          c
          for c in columns
          if c not in self.all_gdb_fields and c not in ("geometry", "HYBAS_ID")
      ]
      if missing:
        raise ValueError(
            f"Unknown HydroATLAS level {level} column: {missing[0]}"
        )

    if self._hydroatlas_mode == "pyogrio" and not self.is_shapefile:
      if self.gdb_path is None or not self.gdb_path.exists():
        raise FileNotFoundError(
            f"BasinATLAS dataset not found at {self.gdb_path}."
        )
      layer_name = self._level_layers.get(
          level, f"BasinATLAS_v10_lev{level:02d}"
      )
      read_kwargs: dict[str, Any] = {
          "bbox": bbox,
          "layer": layer_name,
      }
      if columns:
        cols_to_read = [c for c in columns if c != "geometry"]
        if "HYBAS_ID" not in cols_to_read:
          cols_to_read.insert(0, "HYBAS_ID")
        read_kwargs["columns"] = cols_to_read
      gdf = pyogrio.read_dataframe(self.gdb_path, **read_kwargs)
      return self._subset_subbasin_columns(gdf, columns)

    gdf = self._read_subbasins_in_bbox(bbox)
    return self._subset_subbasin_columns(gdf, columns)

  def read_subbasin_climate_indices(self, hybas_ids: List[int]) -> pd.DataFrame:
    """Reads climate indices for a list of HYBAS_IDs."""
    if self.era5_loader is None:
      raise ValueError(
          "StaticAttributesExtractor was not configured with era5_source='hybas' and an era5_cache_dir."
      )
    return self.era5_loader.get_indices_table(hybas_ids)


  def _init_hydroatlas_source(self, active_source: str) -> None:
    """Initializes HydroATLAS metadata from a local path or GCS URI."""
    if is_gcs_path(active_source):
      import gcsfs
      import pyarrow.parquet as pq

      fs = gcsfs.GCSFileSystem()
      normalized_src = normalize_gcs_path(active_source)
      clean_src = strip_gcs_prefix(normalized_src).rstrip("/")
      if clean_src.endswith((".parquet", ".geoparquet")):
        if not fs.exists(clean_src):
          raise FileNotFoundError(
              f"Cloud HydroATLAS parquet file not found at {active_source}."
          )
        raw_bytes = fs.cat_file(clean_src)
        gdf = gpd.read_parquet(io.BytesIO(raw_bytes))
        gdf["HYBAS_ID"] = _normalize_hybas_id_series(gdf["HYBAS_ID"])
        _ = gdf.sindex
        self._in_memory_gdf = gdf
        self._hydroatlas_mode = "in_memory_gdf"
        self.all_gdb_fields = [c for c in gdf.columns if c != "geometry"]
        return

      clean_base = (
          clean_src.rsplit("/", 1)[0]
          if clean_src.endswith(".gdb")
          else clean_src
      )
      attr_pq = f"{clean_base}/hydro_atlas_lev12.parquet"
      subpoly_dir = f"{clean_base}/subpolygons"
      if not fs.exists(attr_pq) or not fs.exists(subpoly_dir):
        raise FileNotFoundError(
            f"Cloud HydroATLAS parquet files not found under gs://{clean_base} "
            "(expected hydro_atlas_lev12.parquet and subpolygons/)."
        )
      for fpath in sorted(fs.ls(subpoly_dir)):
        fname = fpath.rsplit("/", 1)[-1]
        if fname.startswith("hybas_") and fname.endswith("_lev12_v1c.geoparquet"):
          cont = fname.split("_")[1]
          self._subpolygon_files[cont] = fpath
      if not self._subpolygon_files:
        raise FileNotFoundError(
            f"No hybas_*_lev12_v1c.geoparquet files found in gs://{subpoly_dir}."
        )
      with fs.open(attr_pq, "rb") as fp:
        schema = pq.read_schema(fp)
      self.all_gdb_fields = list(schema.names)
      self._cloud_attr_pq_path = attr_pq
      self._hydroatlas_mode = "gcs_partitioned"
      return

    local_src = Path(active_source)
    if not local_src.exists():
      raise FileNotFoundError(f"BasinATLAS dataset not found at {local_src}.")

    if str(local_src).endswith((".parquet", ".geoparquet")):
      gdf = gpd.read_parquet(local_src)
      gdf["HYBAS_ID"] = _normalize_hybas_id_series(gdf["HYBAS_ID"])
      _ = gdf.sindex
      self._in_memory_gdf = gdf
      self._hydroatlas_mode = "in_memory_gdf"
      self.all_gdb_fields = [c for c in gdf.columns if c != "geometry"]
      return

    if (
        local_src.is_dir()
        and (local_src / "hydro_atlas_lev12.parquet").exists()
        and (local_src / "subpolygons").is_dir()
    ):
      import pyarrow.parquet as pq

      attr_pq_local = local_src / "hydro_atlas_lev12.parquet"
      for fpath in sorted(
          (local_src / "subpolygons").glob("hybas_*_lev12_v1c.geoparquet")
      ):
        cont = fpath.name.split("_")[1]
        self._subpolygon_files[cont] = str(fpath)
      if not self._subpolygon_files:
        raise FileNotFoundError(
            f"No hybas_*_lev12_v1c.geoparquet files found in {local_src / 'subpolygons'}."
        )
      schema = pq.read_schema(attr_pq_local)
      self.all_gdb_fields = list(schema.names)
      self._cloud_attr_pq_path = str(attr_pq_local)
      self._hydroatlas_mode = "local_partitioned"
      return

    self._hydroatlas_mode = "pyogrio"
    self.is_shapefile = str(local_src).endswith(".shp")
    self.layer_name = None if self.is_shapefile else "BasinATLAS_v10_lev12"
    if self.is_shapefile:
      info = pyogrio.read_info(local_src)
    else:
      layers = pyogrio.list_layers(local_src)
      level_layers: dict[int, str] = {}
      for layer_info in layers:
        layer_name = str(layer_info[0])
        match = re.match(r"^BasinATLAS_v10_lev(\d{2})$", layer_name)
        if match:
          lvl = int(match.group(1))
          if 1 <= lvl <= 12:
            level_layers[lvl] = layer_name
      if 12 not in level_layers:
        raise ValueError(
            "Required HydroATLAS layer 'BasinATLAS_v10_lev12' not found in "
            f"{local_src}."
        )
      self._level_layers = level_layers
      self._available_levels = tuple(sorted(level_layers))
      self.layer_name = level_layers[12]
      info = pyogrio.read_info(local_src, layer=self.layer_name)
    self.all_gdb_fields = list(info["fields"])

  def _get_partitioned_continent_gdf(self, cont: str) -> gpd.GeoDataFrame:
    """Loads and merges a continental GeoParquet file with HydroATLAS attributes in memory."""
    if cont in self._cloud_continent_gdfs:
      return self._cloud_continent_gdfs[cont]

    import pyarrow.parquet as pq

    cols_to_read = ["HYBAS_ID"] + [
        c for c in self.use_properties if c != "HYBAS_ID"
    ]
    subpoly_path = self._subpolygon_files[cont]

    if self._hydroatlas_mode == "gcs_partitioned":
      import gcsfs

      fs = gcsfs.GCSFileSystem()
      if self._cloud_attr_parquet_bytes is None:
        logger.info(
            "Streaming HydroATLAS attributes in memory from gs://%s...",
            self._cloud_attr_pq_path,
        )
        self._cloud_attr_parquet_bytes = fs.cat_file(self._cloud_attr_pq_path)
      attr_df = pq.read_table(
          io.BytesIO(self._cloud_attr_parquet_bytes), columns=cols_to_read
      ).to_pandas()
      logger.info(
          "Streaming HydroATLAS '%s' sub-basin geometries in memory from gs://%s...",
          cont,
          subpoly_path,
      )
      raw_sub = fs.cat_file(subpoly_path)
      gdf_sub = gpd.read_parquet(io.BytesIO(raw_sub))
    else:
      attr_df = pq.read_table(
          self._cloud_attr_pq_path, columns=cols_to_read
      ).to_pandas()
      gdf_sub = gpd.read_parquet(subpoly_path)

    attr_df["HYBAS_ID"] = _normalize_hybas_id_series(attr_df["HYBAS_ID"])
    gdf_sub["HYBAS_ID"] = _normalize_hybas_id_series(gdf_sub["HYBAS_ID"])
    merged = gdf_sub[["HYBAS_ID", "geometry"]].merge(
        attr_df, on="HYBAS_ID", how="inner"
    )
    gdf_merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf_sub.crs)
    _ = gdf_merged.sindex
    self._cloud_continent_gdfs[cont] = gdf_merged
    return gdf_merged

  def _read_subbasins_in_bbox(
      self, bbox: Tuple[float, float, float, float]
  ) -> gpd.GeoDataFrame:
    """Reads Level 12 sub-basins within bounding box from local or in-memory cloud source."""
    if self._hydroatlas_mode == "in_memory_gdf":
      return self._in_memory_gdf.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]].copy()

    if self._hydroatlas_mode in ("gcs_partitioned", "local_partitioned"):
      matched_gdfs = []
      for cont in sorted(self._subpolygon_files.keys()):
        if cont in CONTINENT_BBOXES and not _bboxes_intersect(
            bbox, CONTINENT_BBOXES[cont]
        ):
          continue
        gdf_cont = self._get_partitioned_continent_gdf(cont)
        sub = gdf_cont.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]].copy()
        if len(sub) > 0:
          matched_gdfs.append(sub)
      if not matched_gdfs:
        return gpd.GeoDataFrame()
      if len(matched_gdfs) == 1:
        return matched_gdfs[0]
      return gpd.GeoDataFrame(
          pd.concat(matched_gdfs, ignore_index=True), crs=matched_gdfs[0].crs
      )

    if self.gdb_path is None or not self.gdb_path.exists():
      raise FileNotFoundError(
          f"BasinATLAS dataset not found at {self.gdb_path}."
      )

    read_kwargs = {"bbox": bbox}
    if not self.is_shapefile:
      read_kwargs["layer"] = self.layer_name

    return pyogrio.read_dataframe(self.gdb_path, **read_kwargs)

  def _is_source_configured(self, src: str) -> bool:
    if src == "timeseries":
      return self.timeseries_loader is not None
    if src == "hybas":
      return self.era5_loader is not None
    if src == "gridded":
      return self.gridded_extractor is not None
    return False

  def _resolve_era5_priority(
      self,
      era5_source: Optional[str] = None,
      prefer_gridded_over_hybas: Optional[bool] = None,
  ) -> Tuple[str, ...]:
    """Resolves the ordered tuple of ERA5 climate sources to consult."""
    spec = _validate_era5_source_spec(era5_source) or self.era5_source or "auto"
    prefer_gridded = (
        self.prefer_gridded_over_hybas
        if prefer_gridded_over_hybas is None
        else bool(prefer_gridded_over_hybas)
    )
    if prefer_gridded and self.gridded_extractor is None:
      raise ValueError(
          "gridded_era5_uri must be provided when prefer_gridded_over_hybas=True."
      )

    if spec == "auto":
      candidate_order = (
          ("timeseries", "gridded", "hybas")
          if prefer_gridded
          else ("timeseries", "hybas", "gridded")
      )
      active = tuple(s for s in candidate_order if self._is_source_configured(s))
      if not active:
        raise ValueError(
            "At least one ERA5 climate source must be configured "
            "(era5_timeseries_zarr, era5_cache_dir/gcs_era5_climate_uri, or gridded_era5_uri)."
        )
      return active

    candidate_order = tuple(t.strip() for t in spec.split(",") if t.strip())
    for s in candidate_order:
      if s == "hybas" and self.era5_loader is None:
        raise ValueError(
            "era5_cache_dir (or gcs_era5_climate_uri with no_download=True) "
            f"must be provided when era5_source={spec!r}."
        )
      if s == "gridded" and self.gridded_extractor is None:
        raise ValueError(
            f"gridded_era5_uri must be provided when era5_source={spec!r}."
        )
      if s == "timeseries" and self.timeseries_loader is None:
        raise ValueError(
            f"era5_timeseries_zarr must be provided when era5_source={spec!r}."
        )
    return candidate_order

  def _select_basin_era5_source(
      self, basin_id: str, priority: Tuple[str, ...]
  ) -> str:
    """Selects the first available ERA5 source in ``priority`` that can serve ``basin_id``."""
    for src in priority:
      if src == "timeseries":
        if (
            self.timeseries_loader is not None
            and self.timeseries_loader.has_basin(basin_id)
        ):
          return "timeseries"
      elif src == "hybas" and self.era5_loader is not None:
        return "hybas"
      elif src == "gridded" and self.gridded_extractor is not None:
        return "gridded"

    raise KeyError(
        f"Catchment {basin_id!r} was not found in era5_timeseries_zarr and no "
        f"fallback ERA5 source ('hybas' or 'gridded') is available in priority={priority}."
    )

  def extract_attributes_for_polygon(
      self,
      polygon_geojson: Union[Dict, Polygon, MultiPolygon, gpd.GeoSeries, gpd.GeoDataFrame],
      catchment_id: Optional[str] = None,
      min_overlap_threshold: float = 0.0,
      baseline_years: Tuple[int, int] = (1981, 2020),
      timeseries_df: Optional[pd.DataFrame] = None,
      era5_source: Optional[str] = None,
      prefer_gridded_over_hybas: Optional[bool] = None,
      _batch_mode: bool = False,
      _skip_climate: bool = False,
  ) -> CatchmentAttributes:
    """Calculates Caravan HydroATLAS static attributes for a watershed polygon.

    Args:
      polygon_geojson: GeoJSON Feature, Geometry dict, Shapely Polygon/MultiPolygon,
        or single-row GeoDataFrame.
      catchment_id: Catchment identifier string (required unless present as
        'gauge_id' or 'catchment_id' in the input Feature or GeoDataFrame).
      min_overlap_threshold: Minimum area threshold in km2 for filtering small overlap slivers.
      baseline_years: Tuple of start and end years for climate baseline (default 1981-2020).
      timeseries_df: Optional daily timeseries DataFrame containing columns
        (total_precipitation or prcp, temperature or 2m_temperature,
        potential_evaporation or pet) to compute climate indices directly.
      era5_source: Optional ERA5 sourcing mode override ('auto', 'timeseries',
        'hybas', 'gridded', or comma-separated priority list). Defaults to the
        extractor's configured ``era5_source`` ('auto': ``timeseries -> hybas -> gridded``).
      prefer_gridded_over_hybas: Optional override to prioritize ``gridded``
        ahead of ``hybas`` when ``era5_source="auto"`` (changing the priority
        order to ``timeseries -> gridded -> hybas``).

    Returns:
      CatchmentAttributes dataclass containing extracted attributes, summary,
      categories, and area metadata.
    """
    priority_sources: Tuple[str, ...] = ()
    actual_era5_source: Optional[str] = None
    if timeseries_df is not None and not timeseries_df.empty:
      actual_era5_source = "timeseries"
    elif _skip_climate and not era5_source:
      actual_era5_source = ""
    elif not _skip_climate:
      priority_sources = self._resolve_era5_priority(
          era5_source, prefer_gridded_over_hybas=prefer_gridded_over_hybas
      )
    elif era5_source:
      actual_era5_source = str(era5_source)

    # 1. Parse Input Geometry and Catchment ID
    if isinstance(polygon_geojson, (gpd.GeoDataFrame, gpd.GeoSeries)):
      if len(polygon_geojson) != 1:
        raise ValueError(
            f"Expected a single-row GeoDataFrame or GeoSeries, got {len(polygon_geojson)} rows."
        )
      geom = (
          polygon_geojson.geometry.iloc[0]
          if hasattr(polygon_geojson, "geometry")
          else polygon_geojson.iloc[0]
      )
      if catchment_id is None and hasattr(polygon_geojson, "columns"):
        for id_col in ("gauge_id", "catchment_id"):
          if id_col in polygon_geojson.columns:
            catchment_id = str(polygon_geojson[id_col].iloc[0])
            break
    elif isinstance(polygon_geojson, dict):
      if polygon_geojson.get("type") == "Feature":
        geom_dict = polygon_geojson["geometry"]
        props = polygon_geojson.get("properties") or {}
        if catchment_id is None:
          catchment_id = props.get("gauge_id") or props.get("catchment_id")
      else:
        geom_dict = polygon_geojson
      geom = shape(geom_dict)
    else:
      geom = polygon_geojson

    if not catchment_id:
      raise ValueError(
          "catchment_id must be provided explicitly or present as 'gauge_id' / 'catchment_id' in the input feature."
      )
    catchment_id = str(catchment_id)

    if geom is None or geom.is_empty or not geom.is_valid:
      raise ValueError(
          f"Target polygon geometry for catchment {catchment_id!r} is empty or topologically invalid."
      )

    if geom.area <= 0:
      raise ValueError("Target polygon area must be greater than 0.")

    chosen_era5_source: Optional[str] = None
    if (timeseries_df is None or timeseries_df.empty) and not _skip_climate:
      chosen_era5_source = self._select_basin_era5_source(
          catchment_id, priority_sources
      )
      actual_era5_source = chosen_era5_source

    minx, miny, maxx, maxy = geom.bounds

    # 2. Read BasinATLAS Level 12 Sub-basins within Bounding Box
    bbox = (minx - 0.02, miny - 0.02, maxx + 0.02, maxy + 0.02)
    gdf_subbasins = self._read_subbasins_in_bbox(bbox)

    # 3. Calculate exact geometric intersections and WGS84 geodesic area weights in km²
    if len(gdf_subbasins) > 0:
      intersections = gdf_subbasins.geometry.intersection(geom)
      valid_mask = ~intersections.is_empty
      gdf_matched = gdf_subbasins[valid_mask].copy()
      gdf_matched["intersect_geom"] = intersections[valid_mask]
      gdf_matched["intersect_area_km2"] = [
          float(abs(_WGS84_GEOD.geometry_area_perimeter(g)[0]) / 1e6)
          for g in gdf_matched["intersect_geom"]
      ]
    else:
      gdf_matched = gpd.GeoDataFrame()

    # 4. Collect Sub-basin Data with Caravan Overlap Rules
    basin_data = defaultdict(list)
    if len(gdf_matched) == 0:
      logger.warning(
          "No BasinATLAS Level 12 units intersect catchment '%s' (bounds=%s); "
          "setting HydroATLAS attributes to NaN.",
          catchment_id,
          bbox,
      )
    else:
      for _, row in gdf_matched.iterrows():
        int_area = float(row["intersect_area_km2"])
        if "SUB_AREA" not in row or pd.isna(row["SUB_AREA"]) or float(row["SUB_AREA"]) <= 0:
          raise ValueError(
              "HydroATLAS sub-basin is missing a valid positive 'SUB_AREA' attribute."
          )
        sub_area = float(row["SUB_AREA"])

        # Caravan filtering threshold: either > min_overlap_threshold or >50% of sub-basin
        if (int_area > min_overlap_threshold) or (int_area / sub_area > 0.5):
          for prop in self.use_properties:
            if prop in row:
              basin_data[prop].append(row[prop])
          basin_data["weights"].append(int_area)

        basin_data["area_fragments"].append(int_area)

      if not basin_data["weights"]:
        logger.warning(
            "All intersecting sub-basins for catchment '%s' fell below "
            "min_overlap_threshold=%.3f km²; setting HydroATLAS attributes to NaN.",
            catchment_id,
            min_overlap_threshold,
        )

    weights = np.array(basin_data["weights"], dtype=float)
    mask = weights > min_overlap_threshold
    masked_weights = weights[mask]

    # 5. Aggregate Caravan Properties
    caravan_attributes: Dict[str, Any] = {}
    skip_props = {
        "weights",
        "UP_AREA",
        "area_fragments",
        "HYBAS_ID",
        "NEXT_DOWN",
        "SUB_AREA",
        "geometry",
        "geom",
        "Shape",
    }

    for key in self.use_properties:
      if key in skip_props or key in POUR_POINT_PROPERTIES:
        continue
      if key not in basin_data or len(masked_weights) == 0:
        caravan_attributes[key] = np.nan
        continue

      val = np.array(basin_data[key], dtype=float)
      masked_val = val[mask]

      # Caravan rule for wetland classes: no wetland (-999 / -9999 / <0) is mapped to class 13
      if key == "wet_cl_smj":
        masked_val = np.where(
            (masked_val == -999) | (masked_val == -9999) | (masked_val < 0),
            13,
            masked_val,
        )

      valid_idx = (masked_val > -900) & (~np.isnan(masked_val))

      if not np.any(valid_idx):
        caravan_attributes[key] = np.nan
      else:
        if key in MAJORITY_PROPERTIES:
          # Area-Weighted Majority Vote
          valid_vals = masked_val[valid_idx].astype(int)
          valid_w = masked_weights[valid_idx]
          val_counts = np.bincount(valid_vals, weights=valid_w)
          caravan_attributes[key] = int(val_counts.argmax())
        else:
          # Area-Weighted Average
          caravan_attributes[key] = float(
              np.average(
                  masked_val[valid_idx], weights=masked_weights[valid_idx]
              )
          )

    # 5b. Downstream Outlet Pour-Point Properties
    pour_point_attrs = compute_pour_point_properties(
        basin_data,
        min_overlap_threshold=min_overlap_threshold,
        pour_point_properties=POUR_POINT_PROPERTIES,
    )
    for k, v in pour_point_attrs.items():
      caravan_attributes[k] = v

    # 6. Extract / Compute ERA5-Land Climate Attributes (1981-2020)
    era5_indices = {}
    if not _skip_climate:
      if timeseries_df is not None and not timeseries_df.empty:
        era5_indices = compute_climate_metrics_from_dataframe(
            timeseries_df,
            baseline_years=baseline_years,
            include_fao_pm=True,
        )
      elif chosen_era5_source == "timeseries":
        era5_indices = self.timeseries_loader.get_metrics_for_basin(
            catchment_id, baseline_years=baseline_years
        )
      elif chosen_era5_source == "gridded":
        era5_indices = self.gridded_extractor.extract_climate_metrics_for_polygon(
            geom, baseline_years=baseline_years
        )
      else:
        hybas_ids = (
            [int(hid) for hid in gdf_matched["HYBAS_ID"].values]
            if len(gdf_matched) > 0
            else []
        )
        intersect_weights = (
            [float(w) for w in gdf_matched["intersect_area_km2"].values]
            if len(gdf_matched) > 0
            else []
        )
        era5_indices = self.era5_loader.get_indices_for_subbasins(
            hybas_ids, intersect_weights
        )

      for k, v in era5_indices.items():
        caravan_attributes[k] = v

    # 7. Drainage Area & Aggregation Fraction
    total_frag_area = (
        float(sum(basin_data["area_fragments"]))
        if basin_data["area_fragments"]
        else float(abs(_WGS84_GEOD.geometry_area_perimeter(geom)[0]) / 1e6)
    )
    caravan_attributes["area"] = total_frag_area
    caravan_attributes["basin_area"] = total_frag_area
    caravan_attributes["area_fraction_used_for_aggregation"] = (
        float(sum(masked_weights) / total_frag_area)
        if total_frag_area > 0 and len(masked_weights) > 0
        else 0.0
    )

    res = CatchmentAttributes(
        catchment_id=catchment_id,
        attributes=caravan_attributes,
        area_km2=round(float(total_frag_area), 2),
        area_fraction_used_for_aggregation=float(sum(masked_weights) / total_frag_area) if total_frag_area > 0 and len(masked_weights) > 0 else 0.0,
        subbasin_ids=tuple([int(hid) for hid in gdf_matched["HYBAS_ID"].values] if len(gdf_matched) > 0 else []),
        subbasin_weights_km2=tuple([float(w) for w in gdf_matched["intersect_area_km2"].values] if len(gdf_matched) > 0 else []),
        min_overlap_threshold_km2=min_overlap_threshold,
        baseline_years=baseline_years,
        era5_source=actual_era5_source if actual_era5_source else "",
    )
    return res

  def _populate_gridded_climate_batch(
      self,
      results: List[CatchmentAttributes],
      poly_tasks: List[Tuple[Any, str]],
      baseline_years: Tuple[int, int] = (1981, 2020),
  ) -> None:
    """Computes gridded ERA5 climate metrics in a single pass over Zarr chunks for all polygons."""
    if not poly_tasks:
      return
    if self.gridded_extractor is None:
      raise ValueError("gridded_era5_uri must be provided when using 'gridded' ERA5 source.")
    scheduled_ids = {cid for _, cid in poly_tasks}
    climate_map = self.gridded_extractor.extract_climate_metrics_for_polygons_batch(
        poly_tasks, baseline_years=baseline_years
    )
    for i, r in enumerate(results):
      if r and r.catchment_id in scheduled_ids:
        results[i] = r.with_attributes(climate_map[r.catchment_id])

  def _populate_timeseries_climate_batch(
      self,
      results: List[CatchmentAttributes],
      basin_ids: List[str],
      baseline_years: Tuple[int, int] = (1981, 2020),
  ) -> None:
    """Computes ERA5 climate metrics from pre-extracted timeseries Zarr store(s) in bulk."""
    if not basin_ids:
      return
    if self.timeseries_loader is None:
      raise ValueError(
          "era5_timeseries_zarr must be provided when using 'timeseries' ERA5 source."
      )
    scheduled_ids = set(basin_ids)
    climate_map = self.timeseries_loader.get_metrics_for_basins_batch(
        basin_ids, baseline_years=baseline_years
    )
    for i, r in enumerate(results):
      if r and r.catchment_id in scheduled_ids:
        results[i] = r.with_attributes(climate_map[r.catchment_id])

  def extract_attributes_batch(
      self,
      features: List[Dict[str, Any]],
      min_overlap_threshold: float = 0.0,
      era5_source: Optional[str] = None,
      prefer_gridded_over_hybas: Optional[bool] = None,
  ) -> List[CatchmentAttributes]:
    """Extracts Caravan attributes for a list of GeoJSON Feature dicts."""
    priority = self._resolve_era5_priority(
        era5_source, prefer_gridded_over_hybas=prefer_gridded_over_hybas
    )

    results = []
    gridded_tasks: List[Tuple[Any, str]] = []
    ts_basin_ids: List[str] = []

    for feat in features:
      if not isinstance(feat, dict):
        raise ValueError(
            "Each item in features must be a GeoJSON Feature dictionary containing 'properties' with 'gauge_id' or 'catchment_id'."
        )
      props = feat.get("properties") or {}
      c_id = props.get("gauge_id") or props.get("catchment_id")
      if not c_id:
        raise ValueError(
            "Each feature in extract_attributes_batch must specify 'gauge_id' or 'catchment_id' in its 'properties'."
        )
      c_id = str(c_id)
      geom = shape(feat["geometry"] if feat.get("type") == "Feature" else feat)
      chosen_src = self._select_basin_era5_source(c_id, priority)
      skip_climate = chosen_src in ("timeseries", "gridded")
      if chosen_src == "gridded":
        gridded_tasks.append((geom, c_id))
      elif chosen_src == "timeseries":
        ts_basin_ids.append(c_id)

      res = self.extract_attributes_for_polygon(
          feat,
          catchment_id=c_id,
          min_overlap_threshold=min_overlap_threshold,
          era5_source=chosen_src,
          _skip_climate=skip_climate,
      )
      results.append(res)

    if ts_basin_ids:
      self._populate_timeseries_climate_batch(results, ts_basin_ids)
    if gridded_tasks:
      self._populate_gridded_climate_batch(results, gridded_tasks)

    return results

  def extract_attributes_from_file(
      self,
      input_path: Union[str, Path],
      output_csv_path: Optional[Union[str, Path]] = None,
      id_column: Optional[str] = None,
      min_overlap_threshold: float = 0.0,
      era5_source: Optional[str] = None,
      prefer_gridded_over_hybas: Optional[bool] = None,
      workers: int = 1,
      show_progress: bool = True,
      dataset_name: Optional[str] = None,
  ) -> pd.DataFrame:
    """Extracts Caravan attributes for all features in a vector file (Shapefile, GeoJSON, GeoPackage, Parquet).

    Args:
      input_path: Path to vector polygon file.
      output_csv_path: Optional path to save extracted attributes CSV.
      id_column: Name of column to use for basin / gauge ID (defaults to 'gauge_id').
      min_overlap_threshold: Minimum area threshold in km2.
      era5_source: Optional ERA5 sourcing mode override ('auto', 'timeseries',
        'hybas', 'gridded', or comma-separated priority list).
      prefer_gridded_over_hybas: Optional override to prioritize ``gridded``
        ahead of ``hybas`` when ``era5_source="auto"``.
      workers: Number of parallel processes to use (default 1).
      show_progress: Whether to show an interactive tqdm progress bar.
      dataset_name: Optional dataset label to display in the progress bar.

    Returns:
      Pandas DataFrame with extracted attributes, indexed by gauge_id.
    """
    priority = self._resolve_era5_priority(
        era5_source, prefer_gridded_over_hybas=prefer_gridded_over_hybas
    )
    prefer_gridded = (
        self.prefer_gridded_over_hybas
        if prefer_gridded_over_hybas is None
        else bool(prefer_gridded_over_hybas)
    )

    in_path = Path(input_path)
    if not in_path.exists():
      raise FileNotFoundError(f"Input vector file does not exist: {in_path}")

    if str(in_path).endswith((".parquet", ".geoparquet")):
      gdf = gpd.read_parquet(in_path)
    else:
      gdf = gpd.read_file(in_path)

    if gdf.crs is None:
      raise ValueError(
          f"Input file {in_path} has no coordinate reference system (CRS) defined."
      )
    if not gdf.crs.is_geographic:
      gdf = gdf.to_crs(epsg=4326)

    target_id_col = id_column if id_column is not None else "gauge_id"
    if target_id_col not in gdf.columns:
      raise ValueError(
          f"ID column {target_id_col!r} not found in {in_path} (available columns: {list(gdf.columns)}). "
          "Pass id_column explicitly."
      )

    if gdf[target_id_col].isna().any():
      raise ValueError(
          f"ID column {target_id_col!r} in {in_path} contains null values."
      )
    id_series = gdf[target_id_col].astype(str)
    if id_series.duplicated().any():
      dup_ids = id_series[id_series.duplicated()].unique().tolist()
      raise ValueError(
          f"ID column {target_id_col!r} in {in_path} contains duplicate IDs: {dup_ids[:5]}"
      )

    tasks: List[Tuple[Any, str, str, bool]] = []
    gridded_tasks: List[Tuple[Any, str]] = []
    ts_basin_ids: List[str] = []

    for _, row in gdf.iterrows():
      gid = str(row[target_id_col])
      geom = row.geometry
      chosen_src = self._select_basin_era5_source(gid, priority)
      skip_climate = chosen_src in ("timeseries", "gridded")
      if chosen_src == "gridded":
        gridded_tasks.append((geom, gid))
      elif chosen_src == "timeseries":
        ts_basin_ids.append(gid)
      tasks.append((geom, gid, chosen_src, skip_climate))

    ds_label = dataset_name or in_path.stem.replace("_basin_shapes", "").replace("_basins", "")

    if workers > 1 and len(tasks) > 1:
      import concurrent.futures
      import multiprocessing as mp

      worker_args = [
          (
              geom,
              gid,
              min_overlap_threshold,
              chosen_src,
              str(self.gdb_path) if self.gdb_path else None,
              str(self.era5_cache_dir) if self.era5_cache_dir else None,
              self.gridded_era5_uri,
              skip_climate,
              self.gcs_gdb_uri,
              self.gcs_era5_climate_uri,
              self.no_download,
              self.era5_timeseries_zarr,
              prefer_gridded,
          )
          for geom, gid, chosen_src, skip_climate in tasks
      ]
      logger.debug(
          "Processing %d catchments in parallel with %d workers...",
          len(tasks),
          workers,
      )
      ctx = mp.get_context("spawn")
      results = [None] * len(tasks)
      with concurrent.futures.ProcessPoolExecutor(
          max_workers=workers, mp_context=ctx
      ) as executor:
        future_to_idx = {
            executor.submit(_worker_extract_polygon, arg): i
            for i, arg in enumerate(worker_args)
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(tasks),
            desc=f"  ↳ {ds_label}",
            unit="basin",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        ):
          idx = future_to_idx[future]
          results[idx] = future.result()
    else:
      results = []
      for geom, gid, chosen_src, skip_climate in tqdm(
          tasks,
          desc=f"  ↳ {ds_label}",
          unit="basin",
          leave=False,
          dynamic_ncols=True,
          disable=not show_progress,
      ):
        res = self.extract_attributes_for_polygon(
            geom,
            catchment_id=gid,
            min_overlap_threshold=min_overlap_threshold,
            era5_source=chosen_src,
            prefer_gridded_over_hybas=prefer_gridded,
            _batch_mode=True,
            _skip_climate=skip_climate,
        )
        results.append(res)

    if ts_basin_ids:
      self._populate_timeseries_climate_batch(results, ts_basin_ids)
    if gridded_tasks:
      self._populate_gridded_climate_batch(results, gridded_tasks)

    df = self.export_caravan_csv(
        results, output_csv_path=output_csv_path if output_csv_path else None
    )
    return df

  def export_caravan_csv(
      self,
      results: Union[List[CatchmentAttributes], pd.DataFrame],
      output_csv_path: Optional[Union[str, Path]] = None,
  ) -> pd.DataFrame:
    """Formats and exports Caravan attributes to standard CSV."""
    if isinstance(results, pd.DataFrame):
      df = results
    else:
      rows = []
      gauge_ids = []
      for r in results:
        if hasattr(r, "catchment_id"):
          gid = r.catchment_id
          rows.append(r.attributes)
        else:
          gid = r["catchment_id"]
          rows.append(r["attributes"])
        gauge_ids.append(gid)
      df = pd.DataFrame(rows, index=gauge_ids)
      df.index.name = "gauge_id"

    # Sort columns alphabetically, ensuring basin_area is first
    sorted_cols = sorted(df.columns)
    if "basin_area" in sorted_cols:
      sorted_cols.remove("basin_area")
      sorted_cols = ["basin_area"] + sorted_cols
    df = df[sorted_cols].sort_index(axis=0)

    if output_csv_path:
      p = Path(output_csv_path)
      p.parent.mkdir(parents=True, exist_ok=True)
      df.to_csv(p)
      logger.debug("Saved Caravan static attributes to %s (shape: %s)", p, df.shape)

    return df

  def append_attributes_to_zarr(
      self,
      master_zarr_path: Union[str, Path],
      basin_id: str,
      attributes_dict: Optional[Union[Dict[str, Any], CatchmentAttributes]] = None,
      *,
      attributes: Optional[Union[Dict[str, Any], CatchmentAttributes]] = None,
  ) -> None:
    """Appends static Caravan & HydroATLAS attributes to a master Zarr store along the 'basin' dimension."""
    master_path = Path(master_zarr_path)
    if not master_path.exists():
      raise FileNotFoundError(f"Zarr store does not exist at {master_path}.")

    payload = attributes if attributes is not None else attributes_dict
    if not payload:
      raise ValueError("No attributes dictionary provided to append_attributes_to_zarr.")

    if isinstance(payload, CatchmentAttributes):
      caravan_attrs = payload.attributes
    else:
      caravan_attrs = (
          payload["attributes"]
          if "attributes" in payload
          else payload
      )
    if not caravan_attrs:
      raise ValueError("Provided attributes dictionary is empty.")

    ds = xr.open_zarr(str(master_path)).load()
    if "basin" not in ds.dims:
      raise KeyError(
          f"Dimension 'basin' not found in Zarr store {master_path} (dims: {list(ds.dims)})."
      )

    basin_list = [str(b) for b in ds["basin"].values]
    if str(basin_id) not in basin_list:
      raise KeyError(
          f"Basin ID {basin_id!r} not found in 'basin' coordinate of {master_path}."
      )

    basin_idx = basin_list.index(str(basin_id))

    for key, val in caravan_attrs.items():
      if isinstance(val, (int, float, np.integer, np.floating)):
        var_name = f"caravan_{key}"
        if var_name not in ds:
          arr = np.full((len(basin_list),), np.nan, dtype=np.float32)
          arr[basin_idx] = float(val)
          ds[var_name] = (["basin"], arr)
        else:
          ds[var_name].values[basin_idx] = float(val)

    ds.to_zarr(str(master_path), mode="w", consolidated=True)
    logger.info(
        "Appended Caravan static attributes for %s to %s", basin_id, master_path
    )

class UnsupportedHydroATLASLevelError(Exception):
    pass
