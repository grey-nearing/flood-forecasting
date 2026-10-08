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

"""Property classifications and data-source constants for the Caravan static attributes extractor.

This module lists which HydroATLAS fields are aggregated by majority vote, which
are pour-point (outlet) sums, which are ignored, the HydroBASINS continent
prefixes and bounding boxes, the Caravan ERA5 climate column names, and the
canonical GCS locations of the library's input data.

Per-attribute metadata (names, units, scale factors, categories) lives in
:mod:`multimet.static_extractor.schema`. Local file-system paths are never
assumed here: callers (CLI, applications) pass them explicitly.
"""

from __future__ import annotations

from typing import List, Mapping, Tuple

# -------------------------------------------------------------------------
# Caravan HydroATLAS Property Definitions & Classifications
# Reference: https://github.com/kratzert/Caravan/blob/main/code/Caravan_part1_Earth_Engine.ipynb
# -------------------------------------------------------------------------

# 1. Discrete Categorical Properties (Aggregated by Area-Weighted Majority Vote)
MAJORITY_PROPERTIES: List[str] = [
    "clz_cl_smj",  # Climate zones (18 classes)
    "cls_cl_smj",  # Climate strata (125 classes)
    "glc_cl_smj",  # Land cover (22 classes)
    "pnv_cl_smj",  # Potential natural vegetation (15 classes)
    "wet_cl_smj",  # Wetland (12 classes)
    "tbi_cl_smj",  # Terrestrial biomes (14 classes)
    "tec_cl_smj",  # Terrestrial Ecoregions (846 classes)
    "fmh_cl_smj",  # Freshwater Major Habitat Types (13 classes)
    "fec_cl_smj",  # Freshwater Ecoregions (426 classes)
    "lit_cl_smj",  # Lithological classes (16 classes)
]

# 2. Pour-Point / Upstream Terminal Properties
POUR_POINT_PROPERTIES: List[str] = [
    "dis_m3_pmn",
    "dis_m3_pmx",
    "dis_m3_pyr",
    "lkv_mc_usu",
    "rev_mc_usu",
    "ria_ha_usu",
    "riv_tc_usu",
    "pop_ct_usu",
    "dor_pc_pva",
]

# 3. HydroSHEDS System / Topological Properties (Ignored or processed separately)
IGNORE_PROPERTIES: List[str] = [
    "system:index",
    "COAST",
    "DIST_MAIN",
    "DIST_SINK",
    "ENDO",
    "MAIN_BAS",
    "NEXT_SINK",
    "ORDER_",
    "PFAF_ID",
    "SORT",
    "Shape_Length",
    "Shape_Area",
    "Shape_Leng",
]

# 4. Topological Navigation Attributes (Used for subbasin accounting)
ADDITIONAL_PROPERTIES: List[str] = ["HYBAS_ID", "NEXT_DOWN", "SUB_AREA", "UP_AREA"]

# 5. Upstream Attributes Ignored (Per-subbasin '_s' areal properties used instead)
UPSTREAM_PROPERTIES: List[str] = [
    "aet_mm_uyr", "ari_ix_uav", "cly_pc_uav", "cmi_ix_uyr", "crp_pc_use", "ele_mt_uav", "ero_kh_uav",
    "for_pc_use", "gdp_ud_usu", "gla_pc_use", "glc_pc_u01", "glc_pc_u02", "glc_pc_u03", "glc_pc_u04",
    "glc_pc_u05", "glc_pc_u06", "glc_pc_u07", "glc_pc_u08", "glc_pc_u09", "glc_pc_u10", "glc_pc_u11",
    "glc_pc_u12", "glc_pc_u13", "glc_pc_u14", "glc_pc_u15", "glc_pc_u16", "glc_pc_u17", "glc_pc_u18",
    "glc_pc_u19", "glc_pc_u20", "glc_pc_u21", "glc_pc_u22", "hft_ix_u09", "hft_ix_u93", "inu_pc_ult",
    "inu_pc_umn", "inu_pc_umx", "ire_pc_use", "kar_pc_use", "lka_pc_use", "nli_ix_uav", "pac_pc_use",
    "pet_mm_uyr", "pnv_pc_u01", "pnv_pc_u02", "pnv_pc_u03", "pnv_pc_u04", "pnv_pc_u05", "pnv_pc_u06",
    "pnv_pc_u07", "pnv_pc_u08", "pnv_pc_u09", "pnv_pc_u10", "pnv_pc_u11", "pnv_pc_u12", "pnv_pc_u13",
    "pnv_pc_u14", "pnv_pc_u15", "pop_ct_ssu", "ppd_pk_uav", "pre_mm_uyr", "prm_pc_use", "pst_pc_use",
    "ria_ha_ssu", "riv_tc_ssu", "rdd_mk_uav", "slp_dg_uav", "slt_pc_uav", "snd_pc_uav", "snw_pc_uyr",
    "soc_th_uav", "swc_pc_uyr", "tmp_dc_uyr", "urb_pc_use", "wet_pc_u01", "wet_pc_u02", "wet_pc_u03",
    "wet_pc_u04", "wet_pc_u05", "wet_pc_u06", "wet_pc_u07", "wet_pc_u08", "wet_pc_u09", "wet_pc_ug1",
    "wet_pc_ug2", "gad_id_smj",
]

# 6. HydroBASINS Continent ID prefix mapping (First digit of HYBAS_ID)
CONTINENT_MAP: Mapping[int, str] = {
    1: "af",  # Africa
    2: "eu",  # Europe
    3: "si",  # Siberia
    4: "as",  # Central / South / East Asia
    5: "au",  # Australia / Oceania
    6: "sa",  # South America
    7: "na",  # North America
    8: "ar",  # Arctic
    9: "gr",  # Greenland
}

# Bounding boxes (minx, miny, maxx, maxy) of each continental Level 12 GeoParquet file
CONTINENT_BBOXES: Mapping[str, Tuple[float, float, float, float]] = {
    "af": (-18.2, -34.9, 54.6, 37.6),
    "ar": (-180.0, 51.2, -61.0, 83.3),
    "as": (57.6, 1.1, 151.0, 56.0),
    "au": (94.9, -55.2, 180.1, 24.4),
    "eu": (-24.6, 12.5, 69.6, 81.9),
    "gr": (-73.1, 59.7, -11.3, 83.7),
    "na": (-138.0, 5.4, -52.6, 62.8),
    "sa": (-92.1, -56.0, -32.3, 14.9),
    "si": (58.9, 45.5, 180.0, 81.3),
}

CARAVAN_CLIMATE_COLUMNS: List[str] = [
    "p_mean",
    "pet_mean",
    "pet_mean_FAO_PM",
    "pet_mean_ERA5_LAND",
    "aridity",
    "aridity_FAO_PM",
    "aridity_ERA5_LAND",
    "frac_snow",
    "moisture_index",
    "moisture_index_FAO_PM",
    "moisture_index_ERA5_LAND",
    "seasonality",
    "seasonality_FAO_PM",
    "seasonality_ERA5_LAND",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
]
CARAVAN_CLIMATE_INDICES = set(CARAVAN_CLIMATE_COLUMNS)

DEFAULT_GCS_HYDROATLAS_URI = "gs://open-multimet/ancillary-data/hydroatlas"
DEFAULT_GCS_ERA5_CLIMATE_URI = (
    "gs://open-multimet/ancillary-data/hydroatlas/era5_climate"
)
DEFAULT_GCS_GRIDDED_ERA5_URI = (
    "gs://open-multimet/data/era5_land/daily_surface.zarr"
)
