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

"""Attribute registry for Caravan / HydroATLAS static catchment attributes.

The registry is the single source of truth for attribute metadata (display
name, HydroATLAS category, units, unit scale factor, provenance and aggregation
rule). It is explicit and enumerable: every entry is listed (or expanded from a
documented family such as the twelve monthly columns of a variable). Unknown
keys are never guessed -- :func:`get_attribute_definition` returns ``None`` and
:func:`to_physical_units` passes the value through unchanged.

Two unit conventions are distinguished for every attribute:

* ``native_unit`` -- the unit of the value as produced by the extractor and
  written to the Caravan CSV. For HydroATLAS attributes this is the integer
  encoding documented in the HydroATLAS catalog (e.g. ``tmp_dc_*`` is stored in
  tenths of a degree Celsius, ``ari_ix_sav`` in hundredths). For the Caravan
  ERA5 climate indices it is the unit of the Caravan computation (e.g.
  ``frac_snow`` is a fraction, ``high_prec_freq`` a fraction of days).
* ``physical_unit`` -- the human-readable unit obtained by multiplying the
  native value by ``scale``.

Models trained on Caravan expect the native encodings, therefore extraction
results are always native and conversion is an explicit, opt-in operation.
"""

from __future__ import annotations

import dataclasses
import math
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import warnings

from multimet.static_extractor.config import (
    CARAVAN_CLIMATE_COLUMNS,
    MAJORITY_PROPERTIES,
    POUR_POINT_PROPERTIES,
)

SOURCE_HYDROATLAS = "hydroatlas"
SOURCE_ERA5_CLIMATE = "era5_climate"
SOURCE_DERIVED = "derived"

AGGREGATION_AREA_WEIGHTED_MEAN = "area_weighted_mean"
AGGREGATION_MAJORITY = "majority"
AGGREGATION_POUR_POINT_SUM = "pour_point_sum"
AGGREGATION_IDENTITY = "identity"

ATTRIBUTE_CATEGORIES: Tuple[str, ...] = (
    "Topography",
    "Climate",
    "Soils",
    "Land Cover",
    "Hydrology",
    "Anthropogenic",
)

_MONTH_NAMES: Tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


class UnknownAttributeWarning(UserWarning):
  """Emitted when a unit conversion encounters attribute keys absent from the registry."""


@dataclasses.dataclass(frozen=True)
class AttributeDefinition:
  """Metadata describing one Caravan / HydroATLAS static attribute.

  Attributes:
    key: Attribute column name (e.g. ``"tmp_dc_syr"``).
    name: Human-readable attribute name.
    category: HydroATLAS taxonomy group (one of :data:`ATTRIBUTE_CATEGORIES`).
    native_unit: Unit of the value as extracted and written to the Caravan CSV.
    physical_unit: Unit after multiplying the native value by ``scale``.
    scale: Multiplicative factor converting native to physical units. ``None``
      means the factor is unknown; such attributes are never rescaled.
    description: Short scientific description.
    source: Provenance (``"hydroatlas"``, ``"era5_climate"`` or ``"derived"``).
    aggregation: Catchment aggregation rule applied by the extractor
      (``"area_weighted_mean"``, ``"majority"``, ``"pour_point_sum"`` or
      ``"identity"``).
  """

  key: str
  name: str
  category: str
  native_unit: str
  physical_unit: str
  scale: Optional[float]
  description: str
  source: str
  aggregation: str

  def __post_init__(self) -> None:
    if self.category not in ATTRIBUTE_CATEGORIES:
      raise ValueError(
          f"Unknown attribute category {self.category!r} for {self.key!r}; "
          f"expected one of {ATTRIBUTE_CATEGORIES}."
      )
    if self.scale is not None and (
        not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0
    ):
      raise ValueError(f"scale for {self.key!r} must be a positive finite number or None.")

  @property
  def is_categorical(self) -> bool:
    """True for class-valued attributes aggregated by area-weighted majority vote."""
    return self.aggregation == AGGREGATION_MAJORITY

  def to_dict(self) -> Dict[str, Any]:
    """Returns the definition as a plain dictionary."""
    return dataclasses.asdict(self)


def _aggregation_for(key: str, source: str) -> str:
  if source == SOURCE_DERIVED:
    return AGGREGATION_IDENTITY
  if key in MAJORITY_PROPERTIES:
    return AGGREGATION_MAJORITY
  if key in POUR_POINT_PROPERTIES:
    return AGGREGATION_POUR_POINT_SUM
  return AGGREGATION_AREA_WEIGHTED_MEAN


def _d(
    key: str,
    name: str,
    category: str,
    native_unit: str,
    physical_unit: Optional[str],
    scale: Optional[float],
    description: str,
    source: str = SOURCE_HYDROATLAS,
) -> AttributeDefinition:
  return AttributeDefinition(
      key=key,
      name=name,
      category=category,
      native_unit=native_unit,
      physical_unit=physical_unit if physical_unit is not None else native_unit,
      scale=scale,
      description=description,
      source=source,
      aggregation=_aggregation_for(key, source),
  )


def _monthly(
    prefix: str,
    name: str,
    category: str,
    native_unit: str,
    physical_unit: Optional[str],
    scale: float,
    description: str,
) -> List[AttributeDefinition]:
  """Expands the twelve HydroATLAS monthly columns ``<prefix>s01..s12`` of one variable."""
  return [
      _d(
          f"{prefix}s{m:02d}",
          f"{_MONTH_NAMES[m - 1]} {name}",
          category,
          native_unit,
          physical_unit,
          scale,
          f"{description} ({_MONTH_NAMES[m - 1]})",
      )
      for m in range(1, 13)
  ]


def _class_extents(
    prefix: str,
    name: str,
    category: str,
    n_classes: int,
    description: str,
) -> List[AttributeDefinition]:
  """Expands HydroATLAS per-class spatial extent columns ``<prefix>s01..sNN`` (percent)."""
  return [
      _d(
          f"{prefix}s{c:02d}",
          f"{name} Class {c} Extent",
          category,
          "%",
          "%",
          1.0,
          f"{description} class {c} spatial extent",
      )
      for c in range(1, n_classes + 1)
  ]


def _era5(key: str, name: str, native_unit: str, physical_unit: Optional[str], scale: float, description: str) -> AttributeDefinition:
  return _d(key, name, "Climate", native_unit, physical_unit, scale, description, source=SOURCE_ERA5_CLIMATE)


_DEFINITIONS: List[AttributeDefinition] = [
    # --- Derived catchment geometry -------------------------------------------------------
    _d("basin_area", "Basin Area", "Topography", "km²", "km²", 1.0,
       "Total catchment drainage area (geodesic WGS84) in square kilometers", SOURCE_DERIVED),
    _d("area", "Catchment Area", "Topography", "km²", "km²", 1.0,
       "Geodesic WGS84 area of the catchment polygon (identical to basin_area)", SOURCE_DERIVED),
    _d("area_fraction_used_for_aggregation", "Aggregation Area Fraction", "Topography", "fraction", "fraction", 1.0,
       "Fraction of the catchment area covered by sub-basins retained for aggregation", SOURCE_DERIVED),

    # --- Topography & Physiography --------------------------------------------------------
    _d("ele_mt_sav", "Mean Elevation", "Topography", "m", "m", 1.0, "Spatial mean catchment elevation"),
    _d("ele_mt_smn", "Min Elevation", "Topography", "m", "m", 1.0, "Minimum catchment elevation"),
    _d("ele_mt_smx", "Max Elevation", "Topography", "m", "m", 1.0, "Maximum catchment elevation"),
    _d("slp_dg_sav", "Mean Slope", "Topography", "deg × 10", "deg", 0.1, "Spatial mean terrain slope in degrees"),
    _d("sgr_dk_sav", "Stream Gradient", "Topography", "dm/km", "dm/km", 1.0, "Stream gradient of river reaches"),

    # --- Climate (HydroATLAS) -------------------------------------------------------------
    _d("tmp_dc_syr", "Mean Annual Temperature", "Climate", "°C × 10", "°C", 0.1, "Spatial mean annual air temperature (HydroATLAS)"),
    _d("tmp_dc_smn", "Min Monthly Temperature", "Climate", "°C × 10", "°C", 0.1, "Spatial mean temperature of coldest month (HydroATLAS)"),
    _d("tmp_dc_smx", "Max Monthly Temperature", "Climate", "°C × 10", "°C", 0.1, "Spatial mean temperature of warmest month (HydroATLAS)"),
    _d("pre_mm_syr", "Mean Annual Precipitation", "Climate", "mm/yr", "mm/yr", 1.0, "Spatial mean annual precipitation total (HydroATLAS)"),
    _d("pet_mm_syr", "Potential Evapotranspiration", "Climate", "mm/yr", "mm/yr", 1.0, "Spatial mean annual potential evapotranspiration (HydroATLAS)"),
    _d("aet_mm_syr", "Actual Evapotranspiration", "Climate", "mm/yr", "mm/yr", 1.0, "Spatial mean annual actual evapotranspiration (AET)"),
    _d("ari_ix_sav", "Global Aridity Index", "Climate", "index × 100", "index", 0.01, "Ratio of mean annual precipitation to PET (HydroATLAS)"),
    _d("cmi_ix_syr", "Climate Moisture Index", "Climate", "index × 100", "index", 0.01, "Indicator of water availability vs evaporative demand (annual)"),
    _d("snw_pc_syr", "Snow Cover Extent", "Climate", "%", "%", 1.0, "Average annual snow cover extent percentage"),
    _d("snw_pc_smx", "Max Monthly Snow Cover Extent", "Climate", "%", "%", 1.0, "Maximum monthly snow cover extent percentage"),
    _d("run_mm_syr", "Annual Runoff", "Climate", "mm/yr", "mm/yr", 1.0, "Spatial mean annual natural surface runoff"),
    _d("clz_cl_smj", "Climate Zone Class", "Climate", "class (1-18)", "class (1-18)", 1.0, "Majority Global Environmental Stratification (GEnS) climate zone"),
    _d("cls_cl_smj", "Climate Strata Class", "Climate", "class (1-125)", "class (1-125)", 1.0, "Majority Global Environmental Stratification (GEnS) climate stratum"),

    # --- Caravan ERA5-Land climate indices (1981-2020) -----------------------------------
    _era5("p_mean", "Mean Daily Precip (ERA5)", "mm/day", "mm/day", 1.0, "Long-term daily mean precipitation from ERA5-Land (1981-2020)"),
    _era5("pet_mean", "Potential Evapotranspiration (FAO PM)", "mm/day", "mm/day", 1.0, "FAO-56 Penman-Monteith potential evapotranspiration (1981-2020)"),
    _era5("pet_mean_FAO_PM", "Potential Evapotranspiration (FAO PM)", "mm/day", "mm/day", 1.0, "FAO-56 Penman-Monteith potential evapotranspiration (1981-2020)"),
    _era5("pet_mean_ERA5_LAND", "Potential Evaporation (ERA5-Land Native)", "mm/day", "mm/day", 1.0, "Long-term daily mean potential evaporation from ERA5-Land (1981-2020)"),
    _era5("aridity", "Aridity Index (FAO PM)", "ratio", "ratio", 1.0, "Ratio of FAO Penman-Monteith PET to precipitation (1981-2020)"),
    _era5("aridity_FAO_PM", "Aridity Index (FAO PM)", "ratio", "ratio", 1.0, "Ratio of FAO Penman-Monteith PET to precipitation (1981-2020)"),
    _era5("aridity_ERA5_LAND", "Aridity Index (ERA5-Land Native)", "ratio", "ratio", 1.0, "Ratio of ERA5-Land native potential evaporation to precipitation (1981-2020)"),
    _era5("frac_snow", "Snow Fraction (ERA5)", "fraction", "%", 100.0, "Fraction of precipitation falling when mean temperature < 0°C (1981-2020)"),
    _era5("moisture_index", "Moisture Index (FAO PM)", "index", "index", 1.0, "Annual moisture index following Knoben et al. 2018 with FAO PM PET (1981-2020)"),
    _era5("moisture_index_FAO_PM", "Moisture Index (FAO PM)", "index", "index", 1.0, "Annual moisture index following Knoben et al. 2018 with FAO PM PET (1981-2020)"),
    _era5("moisture_index_ERA5_LAND", "Moisture Index (ERA5-Land Native)", "index", "index", 1.0, "Annual moisture index following Knoben et al. 2018 with ERA5-Land native PEV (1981-2020)"),
    _era5("seasonality", "Seasonality Index (FAO PM)", "index", "index", 1.0, "Precipitation and FAO PM PET seasonality following Knoben et al. 2018 (1981-2020)"),
    _era5("seasonality_FAO_PM", "Seasonality Index (FAO PM)", "index", "index", 1.0, "Precipitation and FAO PM PET seasonality following Knoben et al. 2018 (1981-2020)"),
    _era5("seasonality_ERA5_LAND", "Seasonality Index (ERA5-Land Native)", "index", "index", 1.0, "Precipitation and ERA5-Land native PEV seasonality following Knoben et al. 2018 (1981-2020)"),
    _era5("high_prec_freq", "High Precip Frequency", "fraction of days", "days/yr", 365.25, "Frequency of extreme precipitation days (>= 5x mean daily precip) (1981-2020)"),
    _era5("high_prec_dur", "High Precip Duration", "days", "days", 1.0, "Mean duration of consecutive extreme precipitation days (1981-2020)"),
    _era5("low_prec_freq", "Low Precip Frequency", "fraction of days", "days/yr", 365.25, "Frequency of dry days (< 1 mm/day) (1981-2020)"),
    _era5("low_prec_dur", "Low Precip Duration", "days", "days", 1.0, "Mean duration of consecutive dry spells (< 1 mm/day) (1981-2020)"),

    # --- Soils & Geology ------------------------------------------------------------------
    _d("cly_pc_sav", "Clay Content", "Soils", "%", "%", 1.0, "Volumetric fraction of clay in topsoil"),
    _d("slt_pc_sav", "Silt Content", "Soils", "%", "%", 1.0, "Volumetric fraction of silt in topsoil"),
    _d("snd_pc_sav", "Sand Content", "Soils", "%", "%", 1.0, "Volumetric fraction of sand in topsoil"),
    _d("soc_th_sav", "Soil Organic Carbon", "Soils", "t/ha", "t/ha", 1.0, "Mass density of organic carbon in soil column"),
    _d("swc_pc_syr", "Soil Water Content", "Soils", "%", "%", 1.0, "Annual average volumetric soil water content"),
    _d("gwt_cm_sav", "Water Table Depth", "Soils", "cm", "cm", 1.0, "Mean depth from land surface to groundwater table"),
    _d("kar_pc_sse", "Karst Area Fraction", "Soils", "%", "%", 1.0, "Percentage of catchment underlain by karst formations"),
    _d("ero_kh_sav", "Soil Erodibility", "Soils", "K-factor × 100", "K-factor", 0.01, "USLE soil erodibility factor"),
    _d("lit_cl_smj", "Dominant Lithology Class", "Soils", "class (1-16)", "class (1-16)", 1.0, "Majority Global Lithological Map (GLiM) rock type class"),

    # --- Land Cover & Vegetation ----------------------------------------------------------
    _d("for_pc_sse", "Forest Fraction", "Land Cover", "%", "%", 1.0, "Total forest canopy cover percentage"),
    _d("crp_pc_sse", "Cropland Fraction", "Land Cover", "%", "%", 1.0, "Agricultural cropland percentage"),
    _d("pst_pc_sse", "Pasture Fraction", "Land Cover", "%", "%", 1.0, "Managed grazing / pasture land percentage"),
    _d("ire_pc_sse", "Irrigated Land", "Land Cover", "%", "%", 1.0, "Equipped for irrigation agricultural land"),
    _d("urb_pc_sse", "Urban Extent", "Land Cover", "%", "%", 1.0, "Artificial impervious urban land percentage"),
    _d("gla_pc_sse", "Glacier Fraction", "Land Cover", "%", "%", 1.0, "Perennial glacier and ice sheet coverage"),
    _d("prm_pc_sse", "Permafrost Extent", "Land Cover", "%", "%", 1.0, "Catchment area under continuous or discontinuous permafrost"),
    _d("pac_pc_sse", "Protected Natural Areas", "Land Cover", "%", "%", 1.0, "Catchment area inside designated nature reserves"),
    _d("glc_cl_smj", "Dominant Land Cover Class", "Land Cover", "class (1-22)", "class (1-22)", 1.0, "Majority Global Land Cover 2000 (GLC2000) class"),
    _d("pnv_cl_smj", "Potential Natural Vegetation Class", "Land Cover", "class (1-15)", "class (1-15)", 1.0, "Majority Potential Natural Vegetation (PNV) class"),
    _d("tbi_cl_smj", "Terrestrial Biome Class", "Land Cover", "class (1-14)", "class (1-14)", 1.0, "Majority terrestrial biome class (WWF TEOW)"),
    _d("tec_cl_smj", "Terrestrial Ecoregion Class", "Land Cover", "class (1-846)", "class (1-846)", 1.0, "Majority terrestrial ecoregion class (WWF TEOW)"),

    # --- Hydrology, Discharge & Inundation ------------------------------------------------
    _d("dis_m3_pyr", "Natural Annual Discharge", "Hydrology", "m³/s", "m³/s", 1.0, "Natural annual average river discharge at basin pour point"),
    _d("dis_m3_pmn", "Min Monthly Discharge", "Hydrology", "m³/s", "m³/s", 1.0, "Minimum monthly natural river discharge at basin pour point"),
    _d("dis_m3_pmx", "Max Monthly Discharge", "Hydrology", "m³/s", "m³/s", 1.0, "Maximum monthly natural river discharge at basin pour point"),
    _d("lka_pc_sse", "Lake Area Fraction", "Hydrology", "% × 10", "%", 0.1, "Catchment surface area covered by natural lakes"),
    _d("lkv_mc_usu", "Upstream Lake Volume", "Hydrology", "M m³", "M m³", 1.0, "Total upstream lake storage volume at pour point"),
    _d("rev_mc_usu", "Upstream Reservoir Volume", "Hydrology", "M m³", "M m³", 1.0, "Total upstream reservoir storage capacity at pour point"),
    _d("ria_ha_usu", "Upstream River Area", "Hydrology", "ha", "ha", 1.0, "Total upstream river surface area at pour point"),
    _d("riv_tc_usu", "Upstream River Volume", "Hydrology", "thousand m³", "thousand m³", 1.0, "Total upstream river channel volume at pour point"),
    _d("dor_pc_pva", "Degree of Regulation", "Hydrology", "% × 10", "%", 0.1, "Upstream degree of river regulation by reservoirs"),
    _d("inu_pc_smn", "Min Inundation Extent", "Hydrology", "%", "%", 1.0, "Minimum monthly surface water inundation extent"),
    _d("inu_pc_smx", "Max Inundation Extent", "Hydrology", "%", "%", 1.0, "Maximum monthly surface water inundation extent"),
    _d("inu_pc_slt", "Long-Term Inundation", "Hydrology", "%", "%", 1.0, "Long-term maximum surface water inundation extent"),
    _d("wet_cl_smj", "Dominant Wetland Class", "Hydrology", "class (1-13)", "class (1-13)", 1.0, "Majority Global Lakes and Wetlands Database (GLWD) class"),
    _d("wet_pc_sg1", "Wetland Group 1 Extent", "Hydrology", "%", "%", 1.0, "GLWD wetland group 1 (lakes, reservoirs, rivers) spatial extent"),
    _d("wet_pc_sg2", "Wetland Group 2 Extent", "Hydrology", "%", "%", 1.0, "GLWD wetland group 2 (marshes, floodplains, swamps) spatial extent"),
    _d("fmh_cl_smj", "Freshwater Major Habitat Type Class", "Hydrology", "class (1-13)", "class (1-13)", 1.0, "Majority freshwater major habitat type class (FEOW)"),
    _d("fec_cl_smj", "Freshwater Ecoregion Class", "Hydrology", "class (1-426)", "class (1-426)", 1.0, "Majority freshwater ecoregion class (FEOW)"),

    # --- Anthropogenic & Human Modification -----------------------------------------------
    _d("ppd_pk_sav", "Population Density", "Anthropogenic", "people/km²", "people/km²", 1.0, "Spatial mean human population density"),
    _d("pop_ct_usu", "Upstream Population Count", "Anthropogenic", "count", "count", 1.0, "Total upstream population count at pour point (HydroATLAS pop_ct)"),
    _d("nli_ix_sav", "Nighttime Lights Index", "Anthropogenic", "index × 10", "index", 0.1, "Satellite-derived nighttime luminosity"),
    _d("rdd_mk_sav", "Road Network Density", "Anthropogenic", "m/km²", "m/km²", 1.0, "Total road length per unit catchment area"),
    _d("rdl_km_sav", "Road Network Density", "Anthropogenic", "m/km²", "m/km²", 1.0, "Total road length per unit catchment area"),
    _d("hft_ix_s09", "Human Footprint Index (2009)", "Anthropogenic", "index × 10", "index (0-100)", 0.1, "Cumulative terrestrial human footprint score (2009)"),
    _d("hft_ix_s93", "Human Footprint Index (1993)", "Anthropogenic", "index × 10", "index (0-100)", 0.1, "Cumulative terrestrial human footprint score (1993)"),
    _d("gdp_ud_sav", "Gross Domestic Product (GDP)", "Anthropogenic", "USD/capita", "USD/capita", 1.0, "Economic output per capita in the basin"),
    _d("gdp_ud_ssu", "Sub-basin GDP Sum", "Anthropogenic", "USD", "USD", 1.0, "Total economic output within the basin"),
    _d("hdi_ix_sav", "Human Development Index (HDI)", "Anthropogenic", "index × 1000", "index (0-1)", 0.001, "Socioeconomic human development index"),
]

# Documented HydroATLAS column families expanded explicitly (monthly and per-class columns).
_DEFINITIONS += _monthly("tmp_dc_", "Air Temperature", "Climate", "°C × 10", "°C", 0.1, "Spatial mean monthly air temperature (HydroATLAS)")
_DEFINITIONS += _monthly("pre_mm_", "Precipitation", "Climate", "mm", "mm", 1.0, "Spatial mean monthly precipitation (HydroATLAS)")
_DEFINITIONS += _monthly("pet_mm_", "Potential Evapotranspiration", "Climate", "mm", "mm", 1.0, "Spatial mean monthly potential evapotranspiration (HydroATLAS)")
_DEFINITIONS += _monthly("aet_mm_", "Actual Evapotranspiration", "Climate", "mm", "mm", 1.0, "Spatial mean monthly actual evapotranspiration (HydroATLAS)")
_DEFINITIONS += _monthly("cmi_ix_", "Climate Moisture Index", "Climate", "index × 100", "index", 0.01, "Monthly climate moisture index (HydroATLAS)")
_DEFINITIONS += _monthly("snw_pc_", "Snow Cover Extent", "Climate", "%", "%", 1.0, "Monthly snow cover extent percentage (HydroATLAS)")
_DEFINITIONS += _monthly("swc_pc_", "Soil Water Content", "Soils", "%", "%", 1.0, "Monthly volumetric soil water content (HydroATLAS)")
_DEFINITIONS += _class_extents("glc_pc_", "Land Cover", "Land Cover", 22, "Global Land Cover 2000 (GLC2000)")
_DEFINITIONS += _class_extents("pnv_pc_", "Potential Natural Vegetation", "Land Cover", 15, "Potential Natural Vegetation (PNV)")
_DEFINITIONS += _class_extents("wet_pc_", "Wetland", "Hydrology", 9, "Global Lakes and Wetlands Database (GLWD)")


def _build_registry(definitions: Iterable[AttributeDefinition]) -> Mapping[str, AttributeDefinition]:
  registry: Dict[str, AttributeDefinition] = {}
  for defn in definitions:
    if defn.key in registry:
      raise ValueError(f"Duplicate attribute definition for {defn.key!r}.")
    registry[defn.key] = defn
  return MappingProxyType(registry)


ATTRIBUTE_REGISTRY: Mapping[str, AttributeDefinition] = _build_registry(_DEFINITIONS)

for _climate_key in CARAVAN_CLIMATE_COLUMNS:
  if _climate_key not in ATTRIBUTE_REGISTRY:
    raise RuntimeError(f"Caravan climate column {_climate_key!r} is missing from ATTRIBUTE_REGISTRY.")


def _legacy_view(defn: AttributeDefinition) -> Dict[str, Any]:
  view = defn.to_dict()
  view["unit"] = defn.physical_unit
  view["desc"] = defn.description
  return view


# Read-only dictionary view kept for callers that predate the typed registry.
# ``unit`` mirrors ``physical_unit`` and ``desc`` mirrors ``description``.
ATTRIBUTE_DEFINITIONS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    key: MappingProxyType(_legacy_view(defn)) for key, defn in ATTRIBUTE_REGISTRY.items()
})


def get_attribute_definition(key: str) -> Optional[AttributeDefinition]:
  """Returns the registry entry for ``key`` or ``None`` when the attribute is unknown."""
  return ATTRIBUTE_REGISTRY.get(str(key))


def unknown_attribute_keys(attrs: Mapping[str, Any]) -> Tuple[str, ...]:
  """Returns the keys of ``attrs`` that have no registry entry or no known unit scale."""
  return tuple(
      k for k in attrs if k not in ATTRIBUTE_REGISTRY or ATTRIBUTE_REGISTRY[k].scale is None
  )


def _is_missing(value: Any) -> bool:
  if value is None:
    return True
  import contextlib
  with contextlib.suppress(TypeError, ValueError):
    return bool(math.isnan(float(value)))
  return False


def to_physical_units(
    attrs: Mapping[str, Any],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
  """Converts Caravan-native attribute values to physical units using the registry.

  Only the multiplicative ``scale`` of each registry entry is applied; values are
  never rounded. Values with ``scale == 1.0`` (including categorical class codes)
  are returned unchanged, missing values (``None``/NaN) are returned as NaN and
  non-numeric values are passed through untouched.

  Args:
    attrs: Mapping of attribute key to native value (e.g.
      :attr:`CatchmentAttributes.attributes`).
    strict: If True, raise ``KeyError`` when ``attrs`` contains a key that is
      absent from the registry or whose scale is unknown. If False (default),
      such values are passed through unscaled and a single
      :class:`UnknownAttributeWarning` listing them is emitted.

  Returns:
    New dictionary with the same keys as ``attrs`` in the same order.
  """
  unknown = unknown_attribute_keys(attrs)
  if unknown:
    if strict:
      raise KeyError(
          f"Attributes without a registered unit scale: {list(unknown)}. "
          "Add them to multimet.static_extractor.schema.ATTRIBUTE_REGISTRY or call with strict=False."
      )
    warnings.warn(
        f"{len(unknown)} attribute(s) have no registered unit scale and were passed through "
        f"unscaled: {list(unknown)}",
        UnknownAttributeWarning,
        stacklevel=2,
    )

  out: Dict[str, Any] = {}
  import contextlib
  for key, value in attrs.items():
    defn = ATTRIBUTE_REGISTRY.get(key)
    if defn is None or defn.scale is None:
      out[key] = value
      continue
    if _is_missing(value):
      out[key] = math.nan
      continue
    if float(defn.scale) == 1.0:
      out[key] = value
      continue
    out[key] = value
    with contextlib.suppress(TypeError, ValueError):
      out[key] = float(value) * float(defn.scale)
  return out


__all__ = [
    "AGGREGATION_AREA_WEIGHTED_MEAN",
    "AGGREGATION_IDENTITY",
    "AGGREGATION_MAJORITY",
    "AGGREGATION_POUR_POINT_SUM",
    "ATTRIBUTE_CATEGORIES",
    "ATTRIBUTE_DEFINITIONS",
    "ATTRIBUTE_REGISTRY",
    "AttributeDefinition",
    "SOURCE_DERIVED",
    "SOURCE_ERA5_CLIMATE",
    "SOURCE_HYDROATLAS",
    "UnknownAttributeWarning",
    "get_attribute_definition",
    "to_physical_units",
    "unknown_attribute_keys",
]

@dataclasses.dataclass(frozen=True)
class CatchmentAttributes:
    """Core data structure representing extracted catchment attributes."""
    catchment_id: str
    attributes: Dict[str, Any]
    area_km2: float
    area_fraction_used_for_aggregation: float
    subbasin_ids: Tuple[int, ...] = dataclasses.field(default_factory=tuple)
    subbasin_weights_km2: Tuple[float, ...] = dataclasses.field(default_factory=tuple)
    min_overlap_threshold_km2: float = 0.0
    era5_source: str = ""
    baseline_years: Tuple[int, int] = (1981, 2020)

    def __post_init__(self):
        object.__setattr__(self, "catchment_id", str(self.catchment_id))
        object.__setattr__(self, "subbasin_ids", tuple(self.subbasin_ids))
        object.__setattr__(self, "subbasin_weights_km2", tuple(self.subbasin_weights_km2))
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
        object.__setattr__(self, "baseline_years", tuple(self.baseline_years))
        if len(self.subbasin_ids) != len(self.subbasin_weights_km2):
            raise ValueError("subbasin_ids and subbasin_weights_km2 must have the same length")

    def with_attributes(self, updates: Mapping[str, Any]) -> "CatchmentAttributes":
        new_attrs = dict(self.attributes)
        new_attrs.update(updates)
        return dataclasses.replace(self, attributes=new_attrs)

    @property
    def n_subbasins(self) -> int:
        return len(self.subbasin_ids)

    def physical_units(self, strict: bool = False) -> Dict[str, Any]:
        return to_physical_units(self.attributes, strict=strict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "catchment_id": self.catchment_id,
            "attributes": dict(self.attributes),
            "area_km2": self.area_km2,
            "area_fraction_used_for_aggregation": self.area_fraction_used_for_aggregation,
            "subbasin_ids": list(self.subbasin_ids),
            "subbasin_weights_km2": list(self.subbasin_weights_km2),
            "min_overlap_threshold_km2": self.min_overlap_threshold_km2,
            "era5_source": self.era5_source,
            "baseline_years": list(self.baseline_years),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CatchmentAttributes":
        if "attributes" in d:
            for k, v in d["attributes"].items():
                if v is None:
                    d["attributes"][k] = float('nan')
        try_cls = True
        missing = [f.name for f in dataclasses.fields(cls) if f.name not in d and f.default == dataclasses.MISSING and f.default_factory == dataclasses.MISSING]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return cls(**d)

    def to_series(self) -> pd.Series:
        import pandas as pd
        s = pd.Series(self.attributes)
        s.name = self.catchment_id
        return s


