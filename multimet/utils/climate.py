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

"""Shared meteorological and hydrological climate utility functions.

Implements:
- FAO-56 Penman-Monteith potential evapotranspiration (Allen et al., 1998)
- Knoben et al. (2018) annual moisture index and seasonality index
- Addor et al. (2017) Caravan climate indices and precipitation extremes
- Meteorological depth and temperature unit normalizers
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_fao56_penman_monteith_pet(
    t2m_k: np.ndarray,
    d2m_k: np.ndarray,
    sp_pa: np.ndarray,
    ssr_jm2: np.ndarray,
    str_jm2: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
) -> np.ndarray:
  """Computes daily FAO-56 Penman-Monteith Reference Evapotranspiration (mm/day).

  Args:
    t2m_k: Daily mean 2m air temperature in Kelvin (K).
    d2m_k: Daily mean 2m dewpoint temperature in Kelvin (K).
    sp_pa: Daily mean surface pressure in Pascals (Pa).
    ssr_jm2: Daily accumulated surface net solar radiation in J/m^2.
    str_jm2: Daily accumulated surface net thermal radiation in J/m^2.
    u10_ms: Daily mean 10m eastward wind component in m/s.
    v10_ms: Daily mean 10m northward wind component in m/s.

  Returns:
    Array of daily potential evapotranspiration in mm/day (float32).
  """
  t_c = t2m_k - 273.15
  td_c = d2m_k - 273.15
  p_kpa = sp_pa / 1000.0
  rn_mj = (ssr_jm2 + str_jm2) / 1.0e6
  g_mj = 0.0

  u10 = np.sqrt(u10_ms**2 + v10_ms**2)
  u2 = u10 * (4.87 / np.log(67.8 * 10.0 - 5.42))

  es = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
  ea = 0.6108 * np.exp((17.27 * td_c) / (td_c + 237.3))
  vpd = np.maximum(0.0, es - ea)

  delta = (4098.0 * es) / ((t_c + 237.3) ** 2)
  gamma = 0.000665 * p_kpa

  numerator = (
      0.408 * delta * (rn_mj - g_mj)
      + gamma * (900.0 / (t_c + 273.0)) * u2 * vpd
  )
  denominator = delta + gamma * (1.0 + 0.34 * u2)

  pet = np.where(denominator > 0.0, numerator / denominator, np.nan)
  return np.maximum(0.0, pet).astype(np.float32)


def calculate_fao_pm_pet(
    surface_pressure_kpa: pd.Series,
    temperature_2m_c: pd.Series,
    dewpoint_temperature_2m_c: pd.Series,
    u_component_of_wind_10m: pd.Series,
    v_component_of_wind_10m: pd.Series,
    surface_net_solar_radiation_mean: pd.Series,
    surface_net_thermal_radiation_mean: pd.Series,
    radiation_units: str = "W/m^2",
) -> pd.Series:
  """Calculates daily potential evapotranspiration (PET) following Caravan / FAO-56 Penman-Monteith guidelines.

  Args:
    surface_pressure_kpa: Daily mean surface pressure in kPa.
    temperature_2m_c: Daily mean 2m air temperature in degrees Celsius.
    dewpoint_temperature_2m_c: Daily mean 2m dewpoint temperature in degrees Celsius.
    u_component_of_wind_10m: Daily mean 10m eastward wind component in m/s.
    v_component_of_wind_10m: Daily mean 10m northward wind component in m/s.
    surface_net_solar_radiation_mean: Daily mean surface net solar radiation (default W/m^2).
    surface_net_thermal_radiation_mean: Daily mean surface net thermal radiation (default W/m^2).
    radiation_units: Units of the input radiation series ('W/m^2' [default], 'J/m^2/day', or 'J/m^2/hr').
  """
  rad_u = (radiation_units or "W/m^2").strip().lower()
  if rad_u in {"w/m^2", "w m-2", "w m**-2", "w/m2", "w m^-2"}:
    rad_scale = 86400.0 / 1e6
  elif rad_u in {"j/m^2", "j m-2", "j m**-2", "j/m2", "j/m^2/day", "j/m2/day"}:
    rad_scale = 1.0 / 1e6
  elif rad_u in {"j/m^2/h", "j/m^2/hr", "j/m2/h", "j/m2/hr"}:
    rad_scale = 24.0 / 1e6
  else:
    raise ValueError(
        f"Unsupported radiation_units {radiation_units!r}; expected 'W/m^2', 'J/m^2/day', or 'J/m^2/hr'."
    )

  temp_windspeed10m_m_s = np.sqrt(
      u_component_of_wind_10m**2 + v_component_of_wind_10m**2
  )
  windspeed2m_m_s = (
      temp_windspeed10m_m_s * 4.87 / (np.log(67.8 * 10.0 - 5.42))
  )

  net_radiation_mj_m2 = (
      surface_net_solar_radiation_mean + surface_net_thermal_radiation_mean
  ) * rad_scale

  lmbda = 2.45
  cp = 1.013e-3
  eps = 0.622

  soil_heat_flux = np.zeros_like(surface_pressure_kpa)
  p_kpa = surface_pressure_kpa
  psychometric_kpa_c = cp * p_kpa / (eps * lmbda)
  svp_kpa = 0.6108 * np.exp(
      (17.27 * temperature_2m_c) / (temperature_2m_c + 237.3)
  )
  delta_kpa_c = 4098.0 * svp_kpa / (temperature_2m_c + 237.3) ** 2
  avp_kpa = 0.6108 * np.exp(
      (17.27 * dewpoint_temperature_2m_c)
      / (dewpoint_temperature_2m_c + 237.3)
  )
  svpdeficit_kpa = np.maximum(0.0, svp_kpa - avp_kpa)

  numerator = (
      0.408 * delta_kpa_c * (net_radiation_mj_m2 - soil_heat_flux)
      + psychometric_kpa_c
      * (900.0 / (temperature_2m_c + 273.15))
      * windspeed2m_m_s
      * svpdeficit_kpa
  )
  denominator = delta_kpa_c + psychometric_kpa_c * (
      1.0 + 0.34 * windspeed2m_m_s
  )

  et0_mm_day = numerator / denominator
  return pd.Series(
      np.clip(et0_mm_day, 0.0, None), index=temperature_2m_c.index
  )


def calculate_knoben_moisture_and_seasonality(
    precipitation: pd.Series,
    pet: pd.Series,
) -> Tuple[float, float]:
  """Computes Knoben et al. (2018) annual moisture index and seasonality index.

  Reference:
    Knoben, W. J., et al. (2018). Climate indices for catchment comparison:
    A global evaluation of the moisture and seasonality indices.
    Hydrological Processes, 32(23), 3502-3518.

  Args:
    precipitation: Daily precipitation series (mm/day).
    pet: Daily potential evapotranspiration series (mm/day).

  Returns:
    Tuple of (annual_moisture_index, seasonality_index).
  """
  mean_monthly_precip = precipitation.groupby(precipitation.index.month).mean()
  mean_monthly_pet = pet.groupby(pet.index.month).mean()

  # Monthly moisture index: piecewise formulation in Knoben et al. (2018) Eq. 1-2
  p_gt_et = (
      1.0
      - mean_monthly_pet.loc[mean_monthly_precip > mean_monthly_pet]
      / mean_monthly_precip.loc[mean_monthly_precip > mean_monthly_pet]
  )
  srs = pd.Series(
      np.zeros(len(mean_monthly_pet), dtype=np.float32),
      index=mean_monthly_pet.index,
      name="dummy",
  )
  p_eq_et = srs.loc[mean_monthly_precip == mean_monthly_pet]
  p_lt_et = (
      mean_monthly_precip.loc[mean_monthly_precip < mean_monthly_pet]
      / mean_monthly_pet.loc[mean_monthly_precip < mean_monthly_pet]
      - 1.0
  )
  non_empty = [s for s in (p_gt_et, p_eq_et, p_lt_et) if not s.empty]
  monthly_moisture_index = (
      pd.concat(non_empty) if non_empty else pd.Series(dtype=np.float64)
  )

  annual_moisture_index = float(monthly_moisture_index.mean())
  seasonality = float(
      monthly_moisture_index.max() - monthly_moisture_index.min()
  )
  return annual_moisture_index, seasonality


def _split_list(indices: np.ndarray) -> List[List[int]]:
  """Splits an array of consecutive integer indices into consecutive groups."""
  if len(indices) == 0:
    return []
  groups = []
  current_group = [int(indices[0])]
  for i in range(1, len(indices)):
    if indices[i] == indices[i - 1] + 1:
      current_group.append(int(indices[i]))
    else:
      groups.append(current_group)
      current_group = [int(indices[i])]
  if current_group:
    groups.append(current_group)
  return groups


def compute_caravan_climate_metrics(
    precipitation: pd.Series,
    temperature: pd.Series,
    pet_era5: Optional[pd.Series] = None,
    pet_fao: Optional[pd.Series] = None,
) -> Dict[str, float]:
  """Computes the Caravan climate indices.

  According to Addor et al. (2017) and Knoben et al. (2018).

  Args:
    precipitation: Daily basin-mean precipitation series (mm/day).
    temperature: Daily basin-mean 2m air temperature series (°C).
    pet_era5: Optional daily basin-mean ERA5-Land native potential evaporation (mm/day).
    pet_fao: Optional daily basin-mean FAO-56 Penman-Monteith PET (mm/day).

  Returns:
    Dictionary mapping all Caravan climate index names to their computed values.
  """
  p = np.asarray(precipitation.values, dtype=float)
  p_mean = float(np.nanmean(p)) if len(p) > 0 else np.nan

  # 1. ERA5-Land Native PEV Metrics
  if pet_era5 is not None:
    e_era5 = np.asarray(pet_era5.values, dtype=float)
    pet_mean_era5 = float(np.nanmean(e_era5)) if len(e_era5) > 0 else np.nan
    aridity_era5 = (
        float(pet_mean_era5 / p_mean)
        if (p_mean and not np.isnan(p_mean) and p_mean > 0)
        else np.nan
    )
    mi_era5, seas_era5 = calculate_knoben_moisture_and_seasonality(
        precipitation, pet_era5
    )
  else:
    logger.warning(
        "ERA5-Land potential evaporation series is missing; "
        "setting *_ERA5_LAND climate attributes to NaN."
    )
    pet_mean_era5 = np.nan
    aridity_era5 = np.nan
    mi_era5 = np.nan
    seas_era5 = np.nan

  # 2. FAO-56 Penman-Monteith Metrics
  if pet_fao is not None:
    e_fao = np.asarray(pet_fao.values, dtype=float)
    pet_mean_fao = float(np.nanmean(e_fao)) if len(e_fao) > 0 else np.nan
    aridity_fao = (
        float(pet_mean_fao / p_mean)
        if (p_mean and not np.isnan(p_mean) and p_mean > 0)
        else np.nan
    )
    mi_fao, seas_fao = calculate_knoben_moisture_and_seasonality(
        precipitation, pet_fao
    )
  else:
    logger.warning(
        "FAO-56 Penman-Monteith PET series is missing; "
        "setting unsuffixed and *_FAO_PM climate attributes to NaN."
    )
    pet_mean_fao = np.nan
    aridity_fao = np.nan
    mi_fao = np.nan
    seas_fao = np.nan

  # 3. Fraction of Snow (Knoben et al. 2018, Eq. 4)
  if np.isnan(p_mean) or temperature.dropna().empty:
    frac_snow = np.nan
  else:
    mean_monthly_precip = precipitation.groupby(precipitation.index.month).mean()
    mean_monthly_temp = temperature.groupby(temperature.index.month).mean()
    tot_monthly_p = mean_monthly_precip.sum()
    if tot_monthly_p > 0:
      frac_snow = float(
          mean_monthly_precip.loc[mean_monthly_temp < 0.0].sum() / tot_monthly_p
      )
    else:
      frac_snow = np.nan

  # 4. Extreme Precipitation Frequency & Duration (Addor et al. 2017)
  if p_mean and not np.isnan(p_mean) and p_mean > 0:
    high_p_thresh = 5.0 * p_mean
    high_prec_freq = float(
        len(precipitation.loc[precipitation >= high_p_thresh])
        / len(precipitation)
    )

    high_idx = np.where(p >= high_p_thresh)[0]
    if high_idx.size > 0:
      groups = _split_list(high_idx)
      high_prec_dur = (
          float(np.mean([len(g) for g in groups])) if groups else 0.0
      )
    else:
      high_prec_dur = 0.0

    low_p_thresh = 1.0
    low_prec_freq = float(
        len(precipitation.loc[precipitation < low_p_thresh]) / len(precipitation)
    )

    low_idx = np.where(p < low_p_thresh)[0]
    if low_idx.size > 0:
      groups = _split_list(low_idx)
      low_prec_dur = float(np.mean([len(g) for g in groups])) if groups else 0.0
    else:
      low_prec_dur = 0.0
  else:
    high_prec_freq = np.nan
    high_prec_dur = np.nan
    low_prec_freq = np.nan
    low_prec_dur = np.nan

  return {
      "p_mean": round(p_mean, 4),
      # FAO-56 Penman-Monteith (Standard Caravan)
      "pet_mean": round(pet_mean_fao, 4),
      "pet_mean_FAO_PM": round(pet_mean_fao, 4),
      "aridity": round(aridity_fao, 4),
      "aridity_FAO_PM": round(aridity_fao, 4),
      "moisture_index": round(mi_fao, 4),
      "moisture_index_FAO_PM": round(mi_fao, 4),
      "seasonality": round(seas_fao, 4),
      "seasonality_FAO_PM": round(seas_fao, 4),
      # ERA5-Land Native PEV
      "pet_mean_ERA5_LAND": round(pet_mean_era5, 4),
      "aridity_ERA5_LAND": round(aridity_era5, 4),
      "moisture_index_ERA5_LAND": round(mi_era5, 4),
      "seasonality_ERA5_LAND": round(seas_era5, 4),
      # Climate Extremes & Snow
      "frac_snow": round(frac_snow, 4),
      "high_prec_freq": round(high_prec_freq, 4),
      "high_prec_dur": round(high_prec_dur, 4),
      "low_prec_freq": round(low_prec_freq, 4),
      "low_prec_dur": round(low_prec_dur, 4),
  }


def depth_to_mm(
    series: np.ndarray, units: Optional[str], var_name: str
) -> np.ndarray:
  """Converts a precipitation or PET series to mm/day using declared array units."""
  if units is None:
    return series
  u = units.strip().lower()
  if u in {
      "m",
      "meter",
      "meters",
      "metre",
      "metres",
      "m of water equivalent",
      "m/day",
      "m d-1",
      "m d**-1",
  }:
    return series * 1000.0
  if u in {
      "mm",
      "mm/day",
      "mm d-1",
      "mm d**-1",
      "millimeter",
      "millimeters",
      "millimetre",
      "millimetres",
      "kg m-2",
      "kg m**-2",
      "kg/m2/day",
  }:
    return series
  raise ValueError(
      f"Unsupported units {units!r} for climate variable {var_name!r}; expected mm or m."
  )


def temp_to_celsius(
    series: np.ndarray, units: Optional[str], var_name: str
) -> np.ndarray:
  """Converts a temperature series to degrees Celsius using declared array units."""
  if units is None:
    return series
  u = units.strip().lower()
  if u in {"k", "kelvin", "kelvins", "degk", "degrees_kelvin"}:
    return series - 273.15
  if u in {"c", "degc", "°c", "celsius", "degrees_celsius", "degree_celsius"}:
    return series
  raise ValueError(
      f"Unsupported temperature units {units!r} for variable {var_name!r}; expected K or degC."
  )
