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

"""Point probe, catchment polygon meteogram extractor, and WeatherViewerEngine."""

from __future__ import annotations

import datetime
import math
import mmap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from multimet.weather_viewer.config import (
    NUM_STEPS,
    RUN_METADATA_FILE,
    STEP_HOURS,
    STREAM_SUFFIX,
    SUPPORTED_MODELS,
)
from multimet.weather_viewer.sync import (
    current_run_dir,
    read_sync_status,
    require_data_dir,
)
from multimet.weather_viewer.tiles import (
    compute_frame_index,
    file_step_for_lead,
    get_model_data_info_from_streams,
    grid_indices,
    rate_file_steps,
    render_colorbar_lut_png,
    render_raster_tile,
    render_weather_frame,
    scan_streams,
)
from multimet.weather_viewer.wind import extract_wind_vectors

_PROBE_STREAM_SUFFIX: Dict[str, str] = {
    **STREAM_SUFFIX,
    "wind_u": "u10",
    "wind_v": "v10",
}


def round_or_none(value: Optional[float], digits: int) -> Optional[float]:
  """Rounds finite floats to `digits` decimals, returning None for non-finite."""
  if value is None or not math.isfinite(value):
    return None
  return round(float(value), digits)


def extract_point_value(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    lead_h: float,
    lat: float,
    lon: float,
) -> Optional[float]:
  """Extracts a single scalar forecast value at `(lat, lon)` from synced streams.

  Raises:
    FileNotFoundError: If the required model stream is not present in `arrays`.
  """
  suffix = _PROBE_STREAM_SUFFIX.get(var_key)
  if suffix is None:
    raise ValueError(f"Unsupported probe variable '{var_key}'")
  stream_id = f"{model_key}_{suffix}"
  info = stream_info.get(stream_id)
  if not info or stream_id not in arrays:
    raise FileNotFoundError(
        f"Weather stream '{stream_id}' not found in synced data directory."
    )

  rows, cols = grid_indices(
      np.array([lat], dtype=np.float64), np.array([lon], dtype=np.float64)
  )
  if suffix == "precip":
    rate_steps = rate_file_steps(info, lead_h)
    if not rate_steps:
      return None
    if len(rate_steps) > 1:
      values = arrays[stream_id][rate_steps, rows[0], cols[0]].astype(
          np.float64
      )
      return float(np.nan_to_num(values, nan=0.0).mean())
    file_step = rate_steps[0]
  else:
    file_step = file_step_for_lead(info, lead_h, is_rate=False)
    if file_step is None:
      return None

  value = float(arrays[stream_id][file_step, rows[0], cols[0]]) + float(
      info.get("offset", 0.0)
  )
  return value if math.isfinite(value) else None


def extract_accumulation_series(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    lat: float,
    lon: float,
    lead_hours: Sequence[float],
) -> List[Optional[float]]:
  """Computes cumulative precipitation (mm) since forecast start at `(lat, lon)`."""
  stream_id = f"{model_key}_precip"
  info = stream_info.get(stream_id)
  if not info or stream_id not in arrays:
    raise FileNotFoundError(
        f"Precipitation stream '{stream_id}' not found in synced data directory."
    )
  rows, cols = grid_indices(
      np.array([lat], dtype=np.float64), np.array([lon], dtype=np.float64)
  )
  rates = np.nan_to_num(
      arrays[stream_id][:, rows[0], cols[0]].astype(np.float64), nan=0.0
  )
  leads = info["lead_hours"]
  totals: List[float] = [0.0]
  for i in range(1, len(leads)):
    totals.append(totals[-1] + float(rates[i]) * float(leads[i] - leads[i - 1]))
  out: List[Optional[float]] = []
  for lead_h in lead_hours:
    if lead_h > leads[-1]:
      out.append(None)
      continue
    idx = max(i for i, lead in enumerate(leads) if lead <= lead_h)
    out.append(totals[idx])
  return out


def extract_point_probe(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    lat: float,
    lon: float,
    models: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
  """Extracts comparative 10-day multi-model meteorological soundings at `(lat, lon)`.

  Raises:
    FileNotFoundError: If no requested models have synced data in `stream_info`.
  """
  lead_hours = [i * STEP_HOURS for i in range(NUM_STEPS)]
  now_utc = datetime.datetime.now(datetime.timezone.utc).replace(
      minute=0, second=0, microsecond=0
  )

  if models is not None:
    target_models = list(models)
  else:
    target_models = [
        m
        for m in SUPPORTED_MODELS
        if f"{m}_precip" in stream_info and f"{m}_temp" in stream_info
    ]

  if not target_models:
    raise FileNotFoundError(
        "No synced forecast models available in data directory for point probe."
    )

  results: Dict[str, Any] = {
      "latitude": round(float(lat), 4),
      "longitude": round(float(lon), 4),
      "query_time_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
      "lead_hours": lead_hours,
      "models": {},
  }

  for m_key in target_models:
    if m_key not in SUPPORTED_MODELS:
      raise ValueError(f"Unknown weather model '{m_key}'")
    m_info = SUPPORTED_MODELS[m_key]
    data_info = get_model_data_info_from_streams(stream_info, m_key)
    accum = extract_accumulation_series(
        arrays, stream_info, m_key, lat, lon, lead_hours
    )

    has_wind = (
        f"{m_key}_u10" in stream_info
        and f"{m_key}_v10" in stream_info
        and f"{m_key}_u10" in arrays
        and f"{m_key}_v10" in arrays
    )
    has_mslp = (
        f"{m_key}_mslp" in stream_info and f"{m_key}_mslp" in arrays
    )

    precip_curve: List[Optional[float]] = []
    temp_curve: List[Optional[float]] = []
    wind_spd_curve: List[Optional[float]] = []
    wind_dir_curve: List[Optional[int]] = []
    pressure_curve: List[Optional[float]] = []

    for lead_h in lead_hours:
      precip_curve.append(
          round_or_none(
              extract_point_value(
                  arrays, stream_info, m_key, "precipitation", lead_h, lat, lon
              ),
              2,
          )
      )
      temp_curve.append(
          round_or_none(
              extract_point_value(
                  arrays, stream_info, m_key, "temperature", lead_h, lat, lon
              ),
              1,
          )
      )
      if has_wind:
        u_w = extract_point_value(
            arrays, stream_info, m_key, "wind_u", lead_h, lat, lon
        )
        v_w = extract_point_value(
            arrays, stream_info, m_key, "wind_v", lead_h, lat, lon
        )
        if u_w is not None and v_w is not None:
          wind_spd_curve.append(round(math.sqrt(u_w**2 + v_w**2), 1))
          wind_dir_curve.append(
              int((math.degrees(math.atan2(-u_w, -v_w)) + 360) % 360)
          )
        else:
          wind_spd_curve.append(None)
          wind_dir_curve.append(None)
      else:
        wind_spd_curve.append(None)
        wind_dir_curve.append(None)

      if has_mslp:
        p_val = extract_point_value(
            arrays, stream_info, m_key, "pressure", lead_h, lat, lon
        )
        pressure_curve.append(round_or_none(p_val, 1))
      else:
        pressure_curve.append(None)

    results["models"][m_key] = {
        "name": m_info["name"],
        "badge": m_info["badge"],
        "precip_rate_mmh": precip_curve,
        "accum_precip_mm": [round_or_none(a, 1) for a in accum],
        "temp_c": temp_curve,
        "wind_speed_mps": wind_spd_curve,
        "wind_direction_deg": wind_dir_curve,
        "pressure_hpa": pressure_curve,
        **data_info,
    }

  return results


def geometry_points(geom: Mapping[str, Any]) -> Tuple[List[float], List[float]]:
  """Extracts exterior ring `(lats, lons)` lists from a GeoJSON geometry."""
  lats: List[float] = []
  lons: List[float] = []
  coords = geom.get("coordinates", [])
  rings = []
  if geom.get("type") == "Polygon" and coords:
    rings = [coords[0]]
  elif geom.get("type") == "MultiPolygon" and coords:
    rings = [poly[0] for poly in coords if poly]
  for ring in rings:
    for pt in ring:
      if len(pt) >= 2:
        lons.append(float(pt[0]))
        lats.append(float(pt[1]))
  return lats, lons


def extract_catchment_weather_summary(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    geojson_feature: Mapping[str, Any],
    step_idx: int = 0,
    model_key: str = "ecmwf_ifs",
) -> Dict[str, Any]:
  """Calculates basin-averaged precipitation and temperature for a catchment.

  Raises:
    FileNotFoundError: If synced streams for `model_key` are not present.
    ValueError: If `model_key` is unsupported.
  """
  if model_key not in SUPPORTED_MODELS:
    raise ValueError(f"Unknown weather model '{model_key}'")
  if (
      f"{model_key}_precip" not in stream_info
      or f"{model_key}_temp" not in stream_info
  ):
    raise FileNotFoundError(
        f"Synced precipitation/temperature streams for '{model_key}' not found."
    )

  step_idx_clamped = max(0, int(step_idx))
  lead_h = step_idx_clamped * STEP_HOURS

  props = geojson_feature.get("properties") or {}
  catchment_id = (
      props.get("catchment_id") or geojson_feature.get("id") or "basin"
  )
  area_km2 = float(props.get("area_km2", 1250.0))

  lats, lons = geometry_points(geojson_feature.get("geometry") or {})
  if lats and lons:
    c_lat = sum(lats) / len(lats)
    c_lon = sum(lons) / len(lons)
  else:
    c_lat = float(props.get("outlet_latitude", 40.0))
    c_lon = float(props.get("outlet_longitude", -86.0))

  sample_points = list(zip(lats, lons)) if lats and lons else [(c_lat, c_lon)]
  stride = max(1, len(sample_points) // 10)
  rates = [
      r
      for r in (
          extract_point_value(
              arrays, stream_info, model_key, "precipitation", lead_h, p_lat, p_lon
          )
          for p_lat, p_lon in sample_points[::stride]
      )
      if r is not None
  ]
  temp_c = extract_point_value(
      arrays, stream_info, model_key, "temperature", lead_h, c_lat, c_lon
  )

  data_info = get_model_data_info_from_streams(stream_info, model_key)
  window_h = int(data_info["max_lead_hours"] or 0)
  accum_list = extract_accumulation_series(
      arrays, stream_info, model_key, c_lat, c_lon, [window_h]
  )
  accum = accum_list[0] if accum_list else None

  init_time = data_info.get("init_time")
  if init_time:
    base_dt = datetime.datetime.fromisoformat(
        str(init_time).replace("Z", "+00:00")
    )
  else:
    base_dt = datetime.datetime.now(datetime.timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
  valid_dt = base_dt + datetime.timedelta(hours=lead_h)

  return {
      "catchment_id": catchment_id,
      "area_km2": round(area_km2, 1),
      "step_hours": lead_h,
      "valid_time_utc": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
      "basin_mean_precip_mmh": (
          round(sum(rates) / len(rates), 2) if rates else None
      ),
      "basin_max_precip_mmh": round(max(rates), 2) if rates else None,
      "basin_accumulated_10d_mm": round_or_none(accum, 1),
      "basin_mean_temp_c": round_or_none(temp_c, 1),
      "centroid": {"latitude": round(c_lat, 4), "longitude": round(c_lon, 4)},
      "model": model_key,
      "data_source": data_info["data_source"],
      "accumulation_hours": window_h,
  }


class WeatherViewerEngine:
  """Core gridded NWP weather visualization and query engine.

  Requires an explicit `data_dir` path and never fabricates synthetic weather
  grids. If a requested model run is missing from `data_dir`, methods raise
  `FileNotFoundError`.
  """

  def __init__(self, data_dir: Union[str, Path]):
    self.data_dir: Path = require_data_dir(data_dir)
    self.handles: Dict[str, Tuple[mmap.mmap, int, int, int, bool]] = {}
    self.stream_info: Dict[str, Dict[str, Any]] = {}
    self.arrays: Dict[str, np.ndarray] = {}
    self._loaded_signature: Optional[Tuple[str, int]] = None
    self._loaded_dir: Optional[Path] = None
    self._load_streams()

  def _resolve_active_dir(self) -> Path:
    active = current_run_dir(self.data_dir)
    return active if active is not None else self.data_dir

  def _compute_signature(
      self, target_dir: Path
  ) -> Optional[Tuple[str, int]]:
    if not target_dir.exists():
      return None
    resolved = target_dir.resolve()
    meta_file = resolved / RUN_METADATA_FILE
    mtime = meta_file.stat().st_mtime_ns if meta_file.exists() else 0
    return (str(resolved), mtime)

  def _load_streams(self) -> None:
    target_dir = self._resolve_active_dir()
    handles, infos, arrays = scan_streams(target_dir)
    self.handles = handles
    self.stream_info = infos
    self.arrays = arrays
    self._loaded_signature = self._compute_signature(target_dir)
    self._loaded_dir = target_dir.resolve() if target_dir.exists() else None

  def reload_if_changed(self) -> bool:
    """Reloads memory-mapped streams if a newer run was swapped into data_dir."""
    target_dir = self._resolve_active_dir()
    sig = self._compute_signature(target_dir)
    if sig == self._loaded_signature:
      return False
    self._load_streams()
    return True

  def get_model_info(self, model_key: str) -> Dict[str, Any]:
    """Returns metadata and availability status for `model_key`."""
    return get_model_data_info_from_streams(self.stream_info, model_key)

  def get_all_models_info(self) -> List[Dict[str, Any]]:
    """Returns metadata and availability status for all supported models."""
    return [
        {**info, **self.get_model_info(key)}
        for key, info in SUPPORTED_MODELS.items()
    ]

  def get_sync_status(self) -> Dict[str, Any]:
    """Returns synchronization status from `<data_dir>/sync_status.json`."""
    status = read_sync_status(self.data_dir)
    if not status:
      status = {
          "last_result": "never",
          "message": "No automatic update has run yet.",
      }
    return {
        "last_check_utc": status.get("last_check_utc"),
        "last_success_utc": status.get("last_success_utc"),
        "last_result": status.get("last_result"),
        "message": status.get("message"),
        "check_interval_minutes": status.get("check_interval_minutes", 60),
        "auto_update": self._loaded_dir is not None,
        "data_dir": str(self._loaded_dir) if self._loaded_dir else None,
    }

  def render_tile(
      self,
      model_key: str,
      var_key: str,
      step_idx: int,
      z: int,
      x: int,
      y: int,
      bilinear: bool = True,
  ) -> bytes:
    """Renders a 256x256 Web Mercator PNG tile for the specified layer."""
    return render_raster_tile(
        self.arrays,
        self.stream_info,
        model_key,
        var_key,
        step_idx,
        z,
        x,
        y,
        bilinear=bilinear,
    )

  def get_frame_index(self, model_key: str, var_key: str) -> Dict[str, Any]:
    """Returns animation frame index metadata for `model_key` and `var_key`."""
    return compute_frame_index(self.stream_info, model_key, var_key)

  def render_frame(
      self, model_key: str, var_key: str, step_idx: int
  ) -> bytes:
    """Renders a whole-world indexed PNG animation frame."""
    return render_weather_frame(
        self.arrays, self.stream_info, model_key, var_key, step_idx
    )

  def render_colorbar(
      self, var_key: str, width: int = 256, height: int = 16
  ) -> bytes:
    """Renders a horizontal colorbar LUT PNG for `var_key`."""
    return render_colorbar_lut_png(var_key, width=width, height=height)

  def get_wind_vectors(
      self,
      model_key: str = "ecmwf_ifs",
      step_idx: int = 0,
      subsample: int = 2,
      bbox: Optional[Tuple[float, float, float, float]] = None,
      bilinear: bool = False,
  ) -> Dict[str, Any]:
    """Extracts subsampled U/V wind vectors for streamline rendering."""
    return extract_wind_vectors(
        self.arrays,
        self.stream_info,
        model_key=model_key,
        step_idx=step_idx,
        subsample=subsample,
        bbox=bbox,
        bilinear=bilinear,
    )

  def probe_point(
      self,
      lat: float,
      lon: float,
      models: Optional[Sequence[str]] = None,
  ) -> Dict[str, Any]:
    """Extracts 10-day multi-model meteogram time series at `(lat, lon)`."""
    return extract_point_probe(
        self.arrays, self.stream_info, lat=lat, lon=lon, models=models
    )

  def summarize_catchment(
      self,
      geojson_feature: Mapping[str, Any],
      step_idx: int = 0,
      model_key: str = "ecmwf_ifs",
  ) -> Dict[str, Any]:
    """Computes catchment-averaged precipitation and temperature summary."""
    return extract_catchment_weather_summary(
        self.arrays,
        self.stream_info,
        geojson_feature=geojson_feature,
        step_idx=step_idx,
        model_key=model_key,
    )
