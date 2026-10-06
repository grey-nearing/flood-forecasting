"""Unit tests for hydrography snapping and catchment delineation."""

import unittest
from frontend.delineator import HydroDelineator


class HydroDelineatorTest(unittest.TestCase):

  def test_merit_hydro_delineation(self):
    delineator = HydroDelineator(dataset_id="merit-hydro")
    # Test point in Wabash River Basin (Terre Haute, IN)
    result = delineator.delineate_catchment(lat=39.46, lon=-87.41)

    self.assertEqual(result["type"], "Feature")
    self.assertIn("properties", result)
    props = result["properties"]
    self.assertEqual(props["dataset"], "merit-hydro")
    self.assertGreater(props["area_km2"], 0)
    self.assertIn("outlet", props)
    self.assertIn("latitude", props["outlet"])
    self.assertIn("longitude", props["outlet"])
    self.assertIn("geometry", result)
    self.assertIn(result["geometry"]["type"], ("Polygon", "MultiPolygon"))

  def test_hydroatlas_delineation(self):
    delineator = HydroDelineator(dataset_id="hydroatlas")
    # Test point in White River (Indianapolis / Wabash tributary)
    result = delineator.delineate_catchment(lat=38.60, lon=-86.80)

    props = result["properties"]
    self.assertEqual(props["dataset"], "hydroatlas")
    self.assertGreater(props["area_km2"], 0)
    self.assertIn("reach_id", props["outlet"])

  def test_exact_pour_point_delineation(self):
    """Tests that exact_pour_point mode clips the local sub-basin at the clicked pour point."""
    delineator = HydroDelineator(dataset_id="hydroatlas")
    # Dalton City, IL point from user screenshot
    res_official = delineator.delineate_catchment(
        lat=39.6828, lon=-88.7729, mode="official_ridgeline"
    )
    res_exact = delineator.delineate_catchment(
        lat=39.6828, lon=-88.7729, mode="exact_pour_point"
    )

    self.assertEqual(res_exact["type"], "Feature")
    self.assertEqual(
        res_exact["properties"]["delineation_mode"], "exact_pour_point"
    )
    self.assertIn(
        "Exact Pour-Point", res_exact["properties"]["delineation_method"]
    )

    # The exact pour point boundary should terminate at the snapped point (min latitude matches snapped lat)
    snapped_lat = res_exact["properties"]["outlet"]["latitude"]
    exact_min_lat = res_exact["properties"]["bbox"]["min_lat"]
    official_min_lat = res_official["properties"]["bbox"]["min_lat"]

    # Exact mode trims the downstream section so exact_min_lat is higher (further north) than official
    self.assertAlmostEqual(exact_min_lat, snapped_lat, places=3)
    self.assertGreater(exact_min_lat, official_min_lat)

  def test_dem_flow_direction_delineation(self):
    """Tests pure DEM flow direction D8 reverse routing on 90m terrain grid."""
    delineator = HydroDelineator(dataset_id="hydroatlas")
    res_dem = delineator.delineate_catchment(
        lat=39.6828, lon=-88.7729, mode="dem_flow_direction"
    )

    self.assertEqual(res_dem["type"], "Feature")
    self.assertEqual(
        res_dem["properties"]["delineation_mode"], "dem_flow_direction"
    )
    self.assertIn(
        "DEM Digital Elevation", res_dem["properties"]["delineation_method"]
    )
    self.assertGreater(res_dem["properties"]["area_km2"], 10.0)
    self.assertGreater(res_dem["properties"]["upstream_cells_count"], 1000)
    self.assertIn(res_dem["geometry"]["type"], ("Polygon", "MultiPolygon"))

  def test_custom_multi_watershed_upload(self):
    """Tests multi-watershed GeoJSON normalization."""
    from shapely.geometry import shape, mapping
    import numpy as np

    sample_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "gauge_id": "USGS_03335500",
                    "station_name": "Wabash River at Lafayette, IN",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-87.0, 40.3],
                        [-86.8, 40.3],
                        [-86.8, 40.5],
                        [-87.0, 40.5],
                        [-87.0, 40.3],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "gauge_id": "USGS_03335700",
                    "station_name": "Wildcat Creek near Lafayette, IN",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-86.9, 40.4],
                        [-86.6, 40.4],
                        [-86.6, 40.6],
                        [-86.9, 40.6],
                        [-86.9, 40.4],
                    ]],
                },
            },
        ],
    }

    features_in = sample_geojson["features"]
    normalized = []
    for idx, feat in enumerate(features_in):
      geom = shape(feat["geometry"])
      cent = geom.centroid
      cid = feat["properties"].get("gauge_id") or f"basin_{idx+1}"
      area_km2 = round(
          float(geom.area * 111.0 * 111.0 * np.cos(np.radians(cent.y))), 2
      )
      normalized.append({
          "type": "Feature",
          "geometry": mapping(geom),
          "properties": {
              **feat["properties"],
              "catchment_id": cid,
              "area_km2": area_km2,
          },
      })

    self.assertEqual(len(normalized), 2)
    self.assertEqual(
        normalized[0]["properties"]["catchment_id"], "USGS_03335500"
    )
    self.assertGreater(normalized[0]["properties"]["area_km2"], 0)
    self.assertEqual(
        normalized[1]["properties"]["catchment_id"], "USGS_03335700"
    )

  def test_one_to_one_dem_river_network_mapping(self):
    """Tests 1:1 association between DEM ID and River Network in config and delineator."""
    from frontend.config import (
        HYDRO_DATASETS,
        resolve_hydro_dataset_id,
    )

    self.assertEqual(resolve_hydro_dataset_id("hydrosheds_90m"), "hydroatlas")
    self.assertEqual(resolve_hydro_dataset_id("merit_hydro_90m"), "merit-hydro")
    self.assertEqual(
        HYDRO_DATASETS["hydroatlas"]["dem_id"], "hydrosheds_90m"
    )
    self.assertEqual(
        HYDRO_DATASETS["merit-hydro"]["dem_id"], "merit_hydro_90m"
    )

    delin_hs = HydroDelineator(dataset_id="hydrosheds_90m")
    self.assertEqual(delin_hs.dataset_id, "hydroatlas")
    res_hs = delin_hs.delineate_catchment(lat=38.60, lon=-86.80)
    self.assertEqual(res_hs["properties"]["dem_id"], "hydrosheds_90m")
    self.assertIn("HydroRIVERS", res_hs["properties"]["river_network"])

    delin_mh = HydroDelineator(dataset_id="merit_hydro_90m")
    self.assertEqual(delin_mh.dataset_id, "merit-hydro")
    res_mh = delin_mh.delineate_catchment(lat=39.46, lon=-87.41)
    self.assertEqual(res_mh["properties"]["dem_id"], "merit_hydro_90m")
    self.assertIn("MERIT-Basins", res_mh["properties"]["river_network"])

  def test_delineator_and_login_tab_ui_structure(self):
    """Verifies top header removal, dedicated Login tab, and simplified Delineation UI."""
    from pathlib import Path

    html_path = (
        Path(__file__).resolve().parent.parent / "static" / "index.html"
    )
    html = html_path.read_text(encoding="utf-8")

    # 1. Top header above #mainTabBar is hidden
    self.assertIn('<header class="hidden" aria-hidden="true">', html)
    # 2. Login lives in the single Account tab (no separate Login tab)
    self.assertNotIn('data-tab="tab-login"', html)
    self.assertNotIn('id="loginTabBtn"', html)
    self.assertNotIn('<section id="tab-login"', html)
    self.assertIn('data-tab="tab-account"', html)
    self.assertIn('<section id="tab-account"', html)
    self.assertIn('id="loginTabUsernameInput"', html)
    self.assertIn('id="loginTabSubmitBtn"', html)
    # 3. Single 1:1 DEM & River Network selector in #tab-delineation
    self.assertIn("1. DEM &amp; River Network", html)
    self.assertIn('data-alias="demSelect"', html)
    self.assertIn('data-dem-id="hydrosheds_90m"', html)
    self.assertIn('data-dem-id="merit_hydro_90m"', html)


if __name__ == "__main__":
  unittest.main()

