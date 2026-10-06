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

"""Unit tests for DynamicalDataLoader and dynamical.org extractors."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyproj
import pytest
import xarray as xr

from multimet.timeseries_extractors.config import Product
from multimet.timeseries_extractors.dynamical import (
    AIFSExtractor,
    DynamicalDataLoader,
    DynamicalDatasetInfo,
    DynamicalExtractor,
    DynamicalForecastExtractor,
    DynamicalIMERGExtractor,
    GEFSExtractor,
    GFSExtractor,
    IFSEnsExtractor,
    find_latest_dynamical_forecast_date,
)
from multimet.timeseries_extractors.zarr_writer import MultiMetZarrWriter
from multimet.utils.geometry import load_basin_geometries


pytestmark = pytest.mark.unit

_HRRR_LCC_PROJ4 = (
    "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 "
    "+x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
)
_HRRR_CRS_WKT = pyproj.CRS.from_proj4(_HRRR_LCC_PROJ4).to_wkt()


@pytest.fixture(scope="module")
def basins_gdf():
  path = (
      Path(__file__).parent
      / "test_data"
      / "shapefiles"
      / "us"
      / "us_basin_shapes.geojson"
  )
  return load_basin_geometries(path)


def _make_imerg_half_hourly_dataset(
    basins_gdf,
    dates=("2023-01-01",),
    rate_kg_m2_s: float = 10.0 / 86400.0,
    steps_on_last_day: int = 48,
) -> xr.Dataset:
  """Builds a realistic half-hourly IMERG-schema xarray.Dataset covering basins_gdf."""
  minx, miny, maxx, maxy = basins_gdf.total_bounds
  lats = np.linspace(maxy + 0.5, miny - 0.5, 12, dtype=np.float64)
  lons = np.linspace(minx - 0.5, maxx + 0.5, 12, dtype=np.float64)

  timestamps = []
  for idx, d_str in enumerate(dates):
    n_steps = steps_on_last_day if idx == len(dates) - 1 else 48
    day_ts = pd.date_range(
        f"{d_str}T00:00:00", periods=n_steps, freq="30min"
    )
    timestamps.extend(day_ts)
  time_idx = pd.DatetimeIndex(timestamps)

  shape = (len(time_idx), len(lats), len(lons))
  pr = np.full(shape, rate_kg_m2_s, dtype=np.float32)
  q_idx = np.ones(shape, dtype=np.float32)

  return xr.Dataset(
      data_vars={
          "precipitation_surface": (["time", "latitude", "longitude"], pr),
          "precipitation_quality_index_surface": (
              ["time", "latitude", "longitude"],
              q_idx,
          ),
      },
      coords={
          "time": time_idx,
          "latitude": lats,
          "longitude": lons,
      },
  )


def _make_projected_hrrr_dataset(basins_gdf) -> xr.Dataset:
  """Builds a projected 2D (Lambert Conformal Conic) xarray.Dataset covering basins_gdf."""
  minx, miny, maxx, maxy = basins_gdf.total_bounds
  transformer = pyproj.Transformer.from_crs(
      "EPSG:4326", _HRRR_CRS_WKT, always_xy=True
  )
  px, py = transformer.transform(
      [minx, maxx, maxx, minx], [miny, miny, maxy, maxy]
  )
  y_vals = np.linspace(min(py) - 15000.0, max(py) + 15000.0, 10, dtype=np.float64)
  x_vals = np.linspace(min(px) - 15000.0, max(px) + 15000.0, 10, dtype=np.float64)
  time_idx = pd.date_range("2023-01-01T00:00:00", periods=3, freq="1h")

  pr = np.full((len(time_idx), len(y_vals), len(x_vals)), 2.5, dtype=np.float32)
  spatial_ref = xr.DataArray(0, attrs={"crs_wkt": _HRRR_CRS_WKT})

  return xr.Dataset(
      data_vars={
          "precipitation_surface": (["time", "y", "x"], pr),
      },
      coords={
          "time": time_idx,
          "y": y_vals,
          "x": x_vals,
          "spatial_ref": spatial_ref,
      },
  )


def _make_forecast_dataset(
    basins_gdf,
    init_times,
    lead_hours,
    *,
    t2m_val: float = 15.0,
    precip_mm_per_day: float = 4.0,
    u10_val: float = 2.5,
    v10_val: float = -1.5,
    ensemble_offsets: tuple[float, ...] | None = None,
) -> xr.Dataset:
  """Builds a realistic 4D or 5D dynamical.org forecast xarray.Dataset."""
  minx, miny, maxx, maxy = basins_gdf.total_bounds
  lats = np.linspace(maxy + 0.5, miny - 0.5, 6, dtype=np.float64)
  lons = np.linspace(minx - 0.5, maxx + 0.5, 6, dtype=np.float64)
  init_idx = pd.to_datetime(init_times)
  lead_td = pd.to_timedelta(lead_hours, unit="h")

  pr_rate = float(precip_mm_per_day) / 86400.0

  if ensemble_offsets is None:
    shape = (len(init_idx), len(lead_td), len(lats), len(lons))
    t2m = np.full(shape, t2m_val, dtype=np.float32)
    pr = np.full(shape, pr_rate, dtype=np.float32)
    pr[:, 0, :, :] = np.nan  # 0h analysis step is NaN for flux variables
    u10 = np.full(shape, u10_val, dtype=np.float32)
    v10 = np.full(shape, v10_val, dtype=np.float32)
    dims = ["init_time", "lead_time", "latitude", "longitude"]
    coords = {
        "init_time": init_idx,
        "lead_time": lead_td,
        "latitude": lats,
        "longitude": lons,
    }
  else:
    n_ens = len(ensemble_offsets)
    shape = (len(init_idx), n_ens, len(lead_td), len(lats), len(lons))
    t2m = np.empty(shape, dtype=np.float32)
    pr = np.empty(shape, dtype=np.float32)
    u10 = np.empty(shape, dtype=np.float32)
    v10 = np.empty(shape, dtype=np.float32)
    for m_idx, offset in enumerate(ensemble_offsets):
      t2m[:, m_idx, :, :, :] = t2m_val + offset
      pr[:, m_idx, :, :, :] = pr_rate * (1.0 + 0.1 * offset)
      u10[:, m_idx, :, :, :] = u10_val + offset
      v10[:, m_idx, :, :, :] = v10_val - offset
    pr[:, :, 0, :, :] = np.nan
    dims = [
        "init_time",
        "ensemble_member",
        "lead_time",
        "latitude",
        "longitude",
    ]
    coords = {
        "init_time": init_idx,
        "ensemble_member": np.arange(n_ens, dtype=np.int32),
        "lead_time": lead_td,
        "latitude": lats,
        "longitude": lons,
    }

  return xr.Dataset(
      data_vars={
          "temperature_2m": (dims, t2m),
          "precipitation_surface": (dims, pr),
          "wind_u_10m": (dims, u10),
          "wind_v_10m": (dims, v10),
      },
      coords=coords,
  )


def test_loader_schema_analysis_geographic_1d(basins_gdf):
  """Tests schema and coordinate inspection on a 1D geographic dataset."""
  ds = _make_imerg_half_hourly_dataset(basins_gdf)
  loader = DynamicalDataLoader(
      "nasa-imerg-analysis-early", cache_dataset=False, ds=ds
  )
  assert loader.grid_type == "geographic_1d"
  assert loader.spatial_dims == ("latitude", "longitude")
  assert loader.lat_coord == "latitude"
  assert loader.lon_coord == "longitude"
  assert loader.lat_descending is True
  assert loader.lon_ascending is True
  assert loader.time_dim == "time"
  assert loader.has_lead_time is False

  info = loader.get_info()
  assert isinstance(info, DynamicalDatasetInfo)
  assert info.dataset_id == "nasa-imerg-analysis-early"
  assert "precipitation_surface" in info.variables


def test_spatial_slice_computation_geographic(basins_gdf):
  """Verifies spatial bounding slice calculation for 1D geographic datasets."""
  ds = _make_imerg_half_hourly_dataset(basins_gdf)
  loader = DynamicalDataLoader(
      "nasa-imerg-analysis-early", cache_dataset=False, ds=ds
  )

  slices = loader.compute_spatial_slices(basins_gdf, buffer=0.2)
  lat_slice = slices["latitude"]
  lon_slice = slices["longitude"]

  minx, miny, maxx, maxy = basins_gdf.total_bounds
  assert lat_slice.start > lat_slice.stop
  assert np.isclose(lat_slice.start, maxy + 0.2, atol=1e-3)
  assert np.isclose(lat_slice.stop, miny - 0.2, atol=1e-3)
  assert np.isclose(lon_slice.start, minx - 0.2, atol=1e-3)
  assert np.isclose(lon_slice.stop, maxx + 0.2, atol=1e-3)

  bbox = (-88.0, 39.0, -85.0, 41.0)
  slices_bbox = loader.compute_spatial_slices(bbox, buffer=0.1)
  assert slices_bbox["latitude"].start > slices_bbox["latitude"].stop
  assert np.isclose(slices_bbox["latitude"].start, 41.1)
  assert np.isclose(slices_bbox["latitude"].stop, 38.9)


def test_loader_schema_analysis_projected_2d(basins_gdf):
  """Tests schema inspection on a 2D projected dataset (HRRR Lambert Conformal)."""
  ds = _make_projected_hrrr_dataset(basins_gdf)
  loader = DynamicalDataLoader(
      "noaa-hrrr-analysis", cache_dataset=False, ds=ds
  )
  assert loader.grid_type == "projected_2d"
  assert loader.spatial_dims == ("y", "x")
  assert loader.y_coord == "y"
  assert loader.x_coord == "x"
  assert loader.crs_wkt is not None
  assert "Lambert_Conformal_Conic" in loader.crs_wkt or "PROJCRS" in loader.crs_wkt

  info = loader.get_info()
  assert info.grid_type == "projected_2d"
  assert "precipitation_surface" in info.variables


def test_spatial_slice_computation_projected(basins_gdf):
  """Verifies spatial bounding slice calculation with pyproj reprojection for HRRR."""
  ds = _make_projected_hrrr_dataset(basins_gdf)
  loader = DynamicalDataLoader(
      "noaa-hrrr-analysis", cache_dataset=False, ds=ds
  )
  slices = loader.compute_spatial_slices(basins_gdf)
  assert "y" in slices
  assert "x" in slices
  y_slice = slices["y"]
  x_slice = slices["x"]

  assert abs(y_slice.start) > 10000.0 or abs(y_slice.stop) > 10000.0
  assert abs(x_slice.start) > 10000.0 or abs(x_slice.stop) > 10000.0


def test_load_spatial_subset_and_missing_variable_error(basins_gdf):
  """Tests spatial/temporal subsetting and strict KeyError when any variable is missing."""
  ds = _make_imerg_half_hourly_dataset(basins_gdf)
  loader = DynamicalDataLoader(
      "nasa-imerg-analysis-early", cache_dataset=False, ds=ds
  )

  sub = loader.load_spatial_subset(
      watersheds=basins_gdf,
      variables=["precipitation_surface"],
      start_date="2023-01-01T00:00:00",
      end_date="2023-01-01T02:00:00",
      buffer=0.1,
      compute=True,
  )

  assert isinstance(sub, xr.Dataset)
  assert "precipitation_surface" in sub.data_vars
  assert "precipitation_quality_index_surface" not in sub.data_vars
  assert sub.sizes["time"] == 5

  # Requesting one valid and one non-existent variable must raise KeyError (never silently drop)
  with pytest.raises(KeyError, match="non_existent_band"):
    loader.load_spatial_subset(
        watersheds=basins_gdf,
        variables=["precipitation_surface", "non_existent_band"],
        start_date="2023-01-01T00:00:00",
        end_date="2023-01-01T01:00:00",
    )


def test_extract_basin_timeseries_geographic_and_projected(basins_gdf):
  """Tests exact catchment zonal averaging for both 1D geographic and 2D projected datasets."""
  ds_geo = _make_imerg_half_hourly_dataset(basins_gdf)
  loader_geo = DynamicalDataLoader(
      "nasa-imerg-analysis-early", cache_dataset=False, ds=ds_geo
  )
  ts_geo = loader_geo.extract_basin_timeseries(
      watersheds=basins_gdf,
      variables=["precipitation_surface"],
      start_date="2023-01-01T00:00:00",
      end_date="2023-01-01T01:00:00",
      buffer=0.1,
  )
  assert list(ts_geo.basin.values) == list(basins_gdf.index)
  assert ts_geo["precipitation_surface"].shape == (len(basins_gdf), 3)
  assert not np.isnan(ts_geo["precipitation_surface"].values).any()

  ds_proj = _make_projected_hrrr_dataset(basins_gdf)
  loader_proj = DynamicalDataLoader(
      "noaa-hrrr-analysis", cache_dataset=False, ds=ds_proj
  )
  ts_proj = loader_proj.extract_basin_timeseries(
      watersheds=basins_gdf,
      variables=["precipitation_surface"],
      start_date="2023-01-01T00:00:00",
      end_date="2023-01-01T01:00:00",
  )
  assert list(ts_proj.basin.values) == list(basins_gdf.index)
  assert ts_proj["precipitation_surface"].shape == (len(basins_gdf), 2)
  assert np.allclose(ts_proj["precipitation_surface"].values, 2.5)


def test_dynamical_extractor_adapter(basins_gdf):
  """Tests DynamicalExtractor adapter conforming to BaseExtractor."""
  ds = _make_imerg_half_hourly_dataset(basins_gdf, rate_kg_m2_s=0.002)
  loader = DynamicalDataLoader(
      "nasa-imerg-analysis-early", cache_dataset=False, ds=ds
  )
  extractor = DynamicalExtractor(
      dataset_id="nasa-imerg-analysis-early",
      variable_map={"precipitation_surface": "imerg_precipitation"},
      unit_conversions={"imerg_precipitation": lambda x: x * 1000.0},
      loader=loader,
  )

  res = extractor.extract_for_basins(
      basins_gdf=basins_gdf,
      start_date="2023-01-01T00:00:00",
      end_date="2023-01-01T01:00:00",
  )

  assert "imerg_precipitation" in res.data_vars
  assert np.allclose(res["imerg_precipitation"].values, 2.0)


def test_imerg_extractor_complete_and_incomplete_days(basins_gdf, tmp_path):
  """Tests DynamicalIMERGExtractor produces 48-step daily accumulations and NaN on partial days."""
  ds = _make_imerg_half_hourly_dataset(
      basins_gdf,
      dates=("2023-01-01", "2023-01-02"),
      rate_kg_m2_s=12.0 / 86400.0,
      steps_on_last_day=40,  # 2023-01-02 has only 40 of 48 half-hourly steps
  )
  loader = DynamicalDataLoader(
      "nasa-imerg-analysis-early", cache_dataset=False, ds=ds
  )
  extractor = DynamicalIMERGExtractor(loader=loader)

  out_ds = extractor.extract_for_basins(
      basins_gdf=basins_gdf,
      start_date="2023-01-01",
      end_date="2023-01-02",
  )

  assert out_ds.dims == {"basin": len(basins_gdf), "date": 2}
  # Day 1 (48 steps): 12.0 mm/day, missing_fraction == 0.0
  assert np.allclose(out_ds["imerg_precipitation"].values[:, 0], 12.0, atol=1e-4)
  assert np.allclose(out_ds["imerg_missing_fraction"].values[:, 0], 0.0)
  # Day 2 (40 steps < 48): strictly NaN and missing_fraction == 1.0 (never imputed!)
  assert np.all(np.isnan(out_ds["imerg_precipitation"].values[:, 1]))
  assert np.allclose(out_ds["imerg_missing_fraction"].values[:, 1], 1.0)

  day_res = extractor.extract_day(pd.Timestamp("2023-01-01"), basins_gdf)
  assert "imerg_precipitation" in day_res
  assert "imerg_missing_fraction" in day_res
  assert np.allclose(day_res["imerg_precipitation"], 12.0, atol=1e-4)
  assert np.allclose(day_res["imerg_missing_fraction"], 0.0)

  writer = MultiMetZarrWriter(tmp_path)
  writer.validate_dataset_schema(out_ds, Product.DYNAMICAL_IMERG)


def test_aifs_extractor(basins_gdf, tmp_path):
  """Tests AIFSExtractor (6-hourly steps) produces schema-compliant 10-day daily forecasts."""
  lead_hours = list(range(0, 241, 6))  # 41 steps (0h..240h)
  synth_ds = _make_forecast_dataset(
      basins_gdf,
      init_times=["2024-05-01T00:00:00"],
      lead_hours=lead_hours,
      t2m_val=18.5,
      precip_mm_per_day=5.0,
      u10_val=3.0,
      v10_val=-2.0,
  )
  loader = DynamicalDataLoader(
      "ecmwf-aifs-single-forecast", cache_dataset=False, ds=synth_ds
  )
  extractor = AIFSExtractor(loader=loader)
  ds = extractor.extract_for_basins(
      basins_gdf=basins_gdf,
      start_date="2024-05-01",
      end_date="2024-05-01",
  )

  assert ds.dims == {"basin": len(basins_gdf), "date": 1, "lead_time": 10}
  assert np.allclose(ds["aifs_temperature_2m"].values, 18.5)
  assert np.allclose(ds["aifs_total_precipitation"].values, 5.0, atol=1e-4)
  assert np.allclose(ds["aifs_u_component_of_wind_10m"].values, 3.0)
  assert np.allclose(ds["aifs_v_component_of_wind_10m"].values, -2.0)
  assert np.allclose(ds["aifs_missing_fraction"].values, 0.0)

  day_res = extractor.extract_day(pd.Timestamp("2024-05-01"), basins_gdf)
  assert day_res["aifs_temperature_2m"].shape == (len(basins_gdf), 10)
  assert "aifs_missing_fraction" in day_res

  writer = MultiMetZarrWriter(tmp_path)
  writer.validate_dataset_schema(ds, Product.AIFS)


def test_gfs_extractor(basins_gdf, tmp_path):
  """Tests GFSExtractor across mixed 1-hourly (days 1..5) and 3-hourly (days 6..10) steps."""
  lead_hours = list(range(0, 121, 1)) + list(range(123, 241, 3))
  synth_ds = _make_forecast_dataset(
      basins_gdf,
      init_times=["2023-01-01T00:00:00"],
      lead_hours=lead_hours,
      t2m_val=14.0,
      precip_mm_per_day=8.0,
      u10_val=1.5,
      v10_val=0.5,
  )
  loader = DynamicalDataLoader(
      "noaa-gfs-forecast", cache_dataset=False, ds=synth_ds
  )
  extractor = GFSExtractor(loader=loader)
  ds = extractor.extract_for_basins(
      basins_gdf=basins_gdf,
      start_date="2023-01-01",
      end_date="2023-01-01",
  )

  assert ds.dims == {"basin": len(basins_gdf), "date": 1, "lead_time": 10}
  assert np.allclose(ds["gfs_temperature_2m"].values, 14.0)
  assert np.allclose(ds["gfs_total_precipitation"].values, 8.0, atol=1e-4)
  assert np.allclose(ds["gfs_u_component_of_wind_10m"].values, 1.5)
  assert np.allclose(ds["gfs_v_component_of_wind_10m"].values, 0.5)
  assert np.allclose(ds["gfs_missing_fraction"].values, 0.0)

  writer = MultiMetZarrWriter(tmp_path)
  writer.validate_dataset_schema(ds, Product.GFS)


def test_gefs_and_ifs_ens_ensemble_member_and_mean(basins_gdf, tmp_path):
  """Tests GEFSExtractor and IFSEnsExtractor for control member, specific member, and mean."""
  # 1. GEFS (3-hourly steps 0..240h, 3 ensemble members with offsets 0.0, +2.0, +4.0 -> mean +2.0)
  gefs_hours = list(range(0, 241, 3))
  gefs_ds = _make_forecast_dataset(
      basins_gdf,
      init_times=["2026-04-01T00:00:00"],
      lead_hours=gefs_hours,
      t2m_val=10.0,
      precip_mm_per_day=6.0,
      u10_val=1.0,
      v10_val=-1.0,
      ensemble_offsets=(0.0, 2.0, 4.0),
  )
  gefs_loader = DynamicalDataLoader(
      "noaa-gefs-forecast-35-day", cache_dataset=False, ds=gefs_ds
  )

  ext_m0 = GEFSExtractor(loader=gefs_loader, ensemble_member=0)
  ds_m0 = ext_m0.extract_for_basins(
      basins_gdf, start_date="2026-04-01", end_date="2026-04-01"
  )
  assert np.allclose(ds_m0["gefs_temperature_2m"].values, 10.0)

  ext_m2 = GEFSExtractor(loader=gefs_loader, ensemble_member=2)
  ds_m2 = ext_m2.extract_for_basins(
      basins_gdf, start_date="2026-04-01", end_date="2026-04-01"
  )
  assert np.allclose(ds_m2["gefs_temperature_2m"].values, 14.0)

  ext_mean = GEFSExtractor(loader=gefs_loader, ensemble_member="mean")
  ds_mean = ext_mean.extract_for_basins(
      basins_gdf, start_date="2026-04-01", end_date="2026-04-01"
  )
  assert np.allclose(ds_mean["gefs_temperature_2m"].values, 12.0)
  assert np.allclose(ds_mean["gefs_missing_fraction"].values, 0.0)

  writer = MultiMetZarrWriter(tmp_path)
  writer.validate_dataset_schema(ds_mean, Product.GEFS)

  # 2. IFS_ENS (mixed 3-hourly days 1..6 [0..144h] and 6-hourly days 7..10 [150..240h])
  ifs_hours = list(range(0, 145, 3)) + list(range(150, 241, 6))
  ifs_ds = _make_forecast_dataset(
      basins_gdf,
      init_times=["2026-04-01T00:00:00"],
      lead_hours=ifs_hours,
      t2m_val=20.0,
      precip_mm_per_day=3.0,
      u10_val=4.0,
      v10_val=-3.0,
      ensemble_offsets=(0.0, 6.0),
  )
  ifs_loader = DynamicalDataLoader(
      "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
      cache_dataset=False,
      ds=ifs_ds,
  )
  ifs_mean = IFSEnsExtractor(loader=ifs_loader, ensemble_member="mean")
  ds_ifs = ifs_mean.extract_for_basins(
      basins_gdf, start_date="2026-04-01", end_date="2026-04-01"
  )
  assert np.allclose(ds_ifs["ifs_ens_temperature_2m"].values, 23.0)
  assert np.allclose(ds_ifs["ifs_ens_missing_fraction"].values, 0.0)
  writer.validate_dataset_schema(ds_ifs, Product.IFS_ENS)

  # 3. Invalid ensemble_member values must raise ValueError (never silently select member 0)
  for bad_member in ("median", "0", -1, True, False, None):
    with pytest.raises(ValueError, match="Invalid ensemble_member"):
      GEFSExtractor(loader=gefs_loader, ensemble_member=bad_member)  # type: ignore[arg-type]


def test_dynamical_forecast_strict_nan_and_spinup_1d(basins_gdf):
  """Verifies strict NaN propagation on partial 24h windows and spinup_only_before 1D optimization."""
  init_times = [
      "2026-04-01T00:00:00",
      "2026-04-01T06:00:00",  # non-00z should be ignored
      "2026-04-02T00:00:00",
      "2026-04-03T00:00:00",
  ]
  lead_hours = list(range(0, 121, 1)) + list(range(123, 241, 3))
  synth_ds = _make_forecast_dataset(
      basins_gdf,
      init_times=init_times,
      lead_hours=lead_hours,
      t2m_val=15.0,
      precip_mm_per_day=1.0,
      u10_val=2.5,
      v10_val=-1.5,
  )

  # Inject a single NaN step at lead_time=18h (index 18) on 2026-04-02 (init index 2)
  synth_ds["precipitation_surface"].values[2, 18, :, :] = np.nan
  # Inject NaN at terminal 240h step in wind_u_10m on 2026-04-03 (init index 3 -> lead day 10 incomplete)
  synth_ds["wind_u_10m"].values[3, -1, :, :] = np.nan

  loader = DynamicalDataLoader(
      "noaa-gfs-forecast", cache_dataset=False, ds=synth_ds
  )

  # 1. Auto-discovery with require_full_10d=True probes all 4 forecast variables
  # and skips 2026-04-03 because wind_u_10m at 240h is NaN!
  latest_full = find_latest_dynamical_forecast_date(
      dataset_id="noaa-gfs-forecast",
      reference_date="2026-04-03",
      require_full_10d=True,
      loader=loader,
  )
  assert latest_full == pd.Timestamp("2026-04-02")

  # 2. Extract with spinup_only_before="2026-04-02" using real DynamicalDataLoader
  extractor = GFSExtractor(loader=loader)
  ds_out = extractor.extract_for_basins(
      basins_gdf,
      start_date="2026-04-01",
      end_date="2026-04-03",
      spinup_only_before="2026-04-02",
  )

  # Spin-up date 2026-04-01 (index 0): only lead_time=1D (index 0) is populated; leads 2D..10D are NaN
  assert np.allclose(ds_out["gfs_temperature_2m"].values[:, 0, 0], 15.0)
  assert np.allclose(ds_out["gfs_total_precipitation"].values[:, 0, 0], 1.0, atol=1e-4)
  assert np.allclose(ds_out["gfs_missing_fraction"].values[:, 0, 0], 0.0)
  assert np.all(np.isnan(ds_out["gfs_temperature_2m"].values[:, 0, 1:]))
  assert np.allclose(ds_out["gfs_missing_fraction"].values[:, 0, 1:], 1.0)

  # Forecast date 2026-04-02 (index 1): lead day 1 had 1 NaN sub-step in pr -> pr is NaN and missing_fraction is 1.0!
  assert np.all(np.isnan(ds_out["gfs_total_precipitation"].values[:, 1, 0]))
  assert np.allclose(ds_out["gfs_missing_fraction"].values[:, 1, 0], 1.0)
  # Lead days 2..10 on 2026-04-02 are 100% valid
  assert np.allclose(ds_out["gfs_total_precipitation"].values[:, 1, 1:], 1.0, atol=1e-4)
  assert np.allclose(ds_out["gfs_temperature_2m"].values[:, 1, 1:], 15.0)
  assert np.allclose(ds_out["gfs_missing_fraction"].values[:, 1, 1:], 0.0)

  # Forecast date 2026-04-03 (index 2): lead day 10 (index 9) had NaN at 240h in wind_u_10m -> missing_fraction is 1.0!
  assert np.all(np.isnan(ds_out["gfs_u_component_of_wind_10m"].values[:, 2, 9]))
  assert np.allclose(ds_out["gfs_missing_fraction"].values[:, 2, 9], 1.0)
  assert np.allclose(ds_out["gfs_temperature_2m"].values[:, 2, :9], 15.0)


def test_find_latest_dynamical_forecast_date_rejects_short_horizon(basins_gdf):
  """Verifies find_latest_dynamical_forecast_date raises FileNotFoundError if lead_time < 10 days."""
  short_ds = _make_forecast_dataset(
      basins_gdf,
      init_times=["2026-04-01T00:00:00"],
      lead_hours=[0, 6, 12, 18, 24, 30, 36, 42, 48],  # Only 2 days (48h)
  )
  loader = DynamicalDataLoader(
      "ecmwf-aifs-single-forecast", cache_dataset=False, ds=short_ds
  )
  with pytest.raises(FileNotFoundError, match="Terminal 10-day lead step"):
    find_latest_dynamical_forecast_date(
        dataset_id="ecmwf-aifs-single-forecast",
        reference_date="2026-04-01",
        require_full_10d=True,
        lead_days=10,
        loader=loader,
    )


def test_dynamical_extractors_reject_archive_source(basins_gdf):
  """Verifies dynamical.org extractors explicitly reject source='archive'."""
  for cls in (
      AIFSExtractor,
      GFSExtractor,
      GEFSExtractor,
      IFSEnsExtractor,
      DynamicalIMERGExtractor,
      DynamicalForecastExtractor,
  ):
    with pytest.raises(ValueError, match="does not support source='archive'"):
      cls(source="archive", data_dir="gs://some-bucket/store.zarr")
