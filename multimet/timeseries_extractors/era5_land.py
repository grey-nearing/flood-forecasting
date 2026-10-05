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

"""Extractor for ECMWF ERA5-Land Daily Surface Reanalysis (0.1 deg, 17 bands).

By design, ``ERA5LandExtractor`` reads **exclusively** from a user-supplied
ERA5-Land gridded Zarr archive (e.g., ``gs://open-multimet/data/era5_land/daily_surface.zarr``).
Direct extraction from third-party/upstream sources (such as WeatherBench 2's
0.25 deg atmospheric ERA5 store) is intentionally disabled to prevent silent
model/resolution substitution.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from multimet.timeseries_extractors.base import BaseExtractor
from multimet.timeseries_extractors.config import Product
from multimet.timeseries_extractors.gridded_archive import extract_nowcast_from_archive
from multimet.utils.zonal import ZonalWeightCalculator, ZonalWeightMatrix

logger = logging.getLogger(__name__)

_ALLOWED_ERA5_SOURCES = frozenset(
    {"archive", "gridded_archive", "zarr", "zarr_archive", "auto", "default"}
)


class ERA5LandExtractor(BaseExtractor):
  """Extractor for ECMWF ERA5-Land Reanalysis (0.1 deg, 17 bands).

  Reads exclusively from a user-supplied ERA5-Land gridded Zarr archive.
  """

  def __init__(
      self,
      data_dir: Optional[str] = None,
      source: str = "archive",
  ):
    super().__init__(Product.ERA5_LAND, data_dir)
    source_lower = source.lower().strip()
    if source_lower not in _ALLOWED_ERA5_SOURCES:
      raise ValueError(
          f"ERA5LandExtractor does not support third-party source={source!r}. "
          "ERA5-Land can only be extracted from a user-supplied gridded "
          "archive (source='archive')."
      )
    self.source = "archive"
    self.data_dir = str(data_dir).strip() if data_dir is not None else ""

    # Standard ERA5-Land 0.1 deg grid: 1801 lats (90 -> -90) x 3600 lons (-180 -> 179.9)
    self.lats = np.linspace(90.0, -90.0, 1801, dtype=np.float64)
    lons_raw = np.linspace(0.0, 359.9, 3600, dtype=np.float64)
    lons_shifted = np.where(lons_raw >= 180.0, lons_raw - 360.0, lons_raw)
    self.sort_lon_idx = np.argsort(lons_shifted)
    self.lons = lons_shifted[self.sort_lon_idx]
    self.zonal_calc = ZonalWeightCalculator(
        self.lats, self.lons, cell_res_lat=0.1, cell_res_lon=0.1
    )

  def _require_archive_uri(self) -> str:
    if not self.data_dir:
      raise ValueError(
          "ERA5LandExtractor requires an explicit gridded archive Zarr URI "
          "or path via data_dir (or --archive-store ERA5_LAND=<URI>). "
          "Default or fallback paths are not permitted."
      )
    return self.data_dir

  def extract_day(
      self,
      dt: Union[str, pd.Timestamp],
      basins_gdf: gpd.GeoDataFrame,
      matrix: Optional[ZonalWeightMatrix] = None,
      weights_dict: Optional[
          Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
      ] = None,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of ERA5-Land across all basins from the gridded archive."""
    del weights_dict
    uri = self._require_archive_uri()
    dt_ts = pd.to_datetime(dt)
    ds_day = extract_nowcast_from_archive(
        Product.ERA5_LAND,
        uri,
        basins_gdf,
        start_date=dt_ts,
        end_date=dt_ts,
        weights_matrix=matrix,
        use_bounding_box=True,
    )
    return {
        var: ds_day[var].values[:, 0].astype(np.float32)
        for var in ds_day.data_vars
    }

  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
      **kwargs,
  ) -> xr.Dataset:
    """Extracts ERA5-Land daily surface variables from the gridded archive."""
    del kwargs
    uri = self._require_archive_uri()
    if start_date is None or end_date is None:
      raise ValueError(
          "ERA5LandExtractor.extract_for_basins requires both start_date and "
          "end_date to be explicitly provided."
      )
    return extract_nowcast_from_archive(
        Product.ERA5_LAND,
        uri,
        basins_gdf,
        start_date=start_date,
        end_date=end_date,
        weights_matrix=weights_matrix,
        use_bounding_box=use_bounding_box,
    )
