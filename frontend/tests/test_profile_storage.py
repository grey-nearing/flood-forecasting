"""Unit tests for the 13-Category googlehydrology Storage Architecture in config.py & profile_manager.py."""

import importlib
import json
from pathlib import Path
import sys
import tempfile

try:
  from absl.testing import absltest
except ImportError:
  import unittest as absltest

from frontend import config
from frontend.profile_manager import ProfileManager


class ProfileStorageArchitectureTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.profiles_root = Path(self.temp_dir.name) / "users"
    self.pm = ProfileManager(profiles_dir=self.profiles_root)

  def tearDown(self):
    self.temp_dir.cleanup()
    super().tearDown()

  def test_config_shared_and_legacy_constants_preserved(self):
    # Legacy constants
    for attr in (
        "BASE_DIR",
        "DATA_DIR",
        "POLYGONS_DIR",
        "ATTRIBUTES_DIR",
        "HISTORICAL_DIR",
        "FORECAST_DIR",
        "REALTIME_DIR",
        "ARCHIVES_DIR",
        "BASE_LAYERS_DIR",
        "DEM_DIR",
        "RIVER_NETWORKS_DIR",
        "HYDRO_BASINS_DIR",
        "CLIMATOLOGY_DIR",
        "STATIC_DIR",
        "HYDRO_DATASETS",
        "WEATHER_CONFIG",
    ):
      self.assertTrue(hasattr(config, attr), f"Missing legacy config constant: {attr}")

    # New Phase 1 storage constants
    self.assertIsInstance(config.FLOOD_FORECASTING_REPO_DIR, Path)
    self.assertEqual(config.SHARED_DIR, config.DATA_DIR / "shared")
    self.assertEqual(config.SHARED_DEMS_DIR, config.SHARED_DIR / "dems")
    self.assertEqual(config.SHARED_HYDROATLAS_DIR, config.SHARED_DIR / "hydroatlas")
    self.assertEqual(config.SHARED_HYDRORIVERS_DIR, config.SHARED_DIR / "hydrorivers")
    self.assertEqual(config.SHARED_HYDROFABRICS_DIR, config.SHARED_DIR / "hydrofabrics")
    self.assertEqual(
        config.SHARED_GRIDDED_ARCHIVES_DIR, config.SHARED_DIR / "gridded_archives"
    )
    self.assertEqual(
        config.SHARED_WEATHER_TILES_CACHE_DIR,
        config.SHARED_DIR / "weather_tiles_cache",
    )
    self.assertEqual(config.SHARED_MODELS_DIR, config.SHARED_DIR / "models")
    self.assertEqual(config.SHARED_MAAS_CACHE_DIR, config.SHARED_DIR / "maas_cache")
    self.assertEqual(config.USERS_DIR, config.DATA_DIR / "users")

  def test_ensure_flood_forecasting_on_sys_path(self):
    fake_repo = Path(self.temp_dir.name) / "fake_flood_forecasting"
    fake_src = fake_repo / "src"
    fake_src.mkdir(parents=True, exist_ok=True)

    result = config.ensure_flood_forecasting_on_sys_path(repo_dir=fake_repo)
    self.assertTrue(result)
    self.assertIn(str(fake_repo), sys.path)
    self.assertIn(str(fake_src), sys.path)

    # Clean up sys.path entries
    sys.path.remove(str(fake_repo))
    sys.path.remove(str(fake_src))

    missing_repo = Path(self.temp_dir.name) / "nonexistent_repo"
    self.assertFalse(config.ensure_flood_forecasting_on_sys_path(repo_dir=missing_repo))

  def test_extend_multimet_package_path_spans_sibling_checkouts(self):
    # Two sibling checkouts, each with its own `multimet` package, like
    # flood-forecasting-multimet (multimet.realtime, ...) and
    # flood-forecasting-static-extractor (multimet.static_extractor).
    projects = Path(self.temp_dir.name) / "projects"
    for repo, submodule in (
        ("flood-forecasting-multimet", "fake_realtime"),
        ("flood-forecasting-static-extractor", "fake_static"),
    ):
      package = projects / repo / "multimet"
      package.mkdir(parents=True)
      (package / "__init__.py").write_text("")
      (package / f"{submodule}.py").write_text(f"NAME = {submodule!r}\n")

    def is_multimet(name):
      return name == "multimet" or name.startswith("multimet.")

    saved_modules = {k: v for k, v in sys.modules.items() if is_multimet(k)}
    saved_path = list(sys.path)
    try:
      for name in saved_modules:
        del sys.modules[name]
      sys.path.insert(0, str(projects / "flood-forecasting-multimet"))

      self.assertTrue(config.extend_multimet_package_path(
          repo_dir=projects / "flood-forecasting"))
      importlib.invalidate_caches()
      self.assertEqual(importlib.import_module("multimet.fake_realtime").NAME, "fake_realtime")
      self.assertEqual(importlib.import_module("multimet.fake_static").NAME, "fake_static")
      # Calling it again doesn't add duplicates.
      self.assertTrue(config.extend_multimet_package_path(
          repo_dir=projects / "flood-forecasting"))
      self.assertEqual(len(sys.modules["multimet"].__path__), 2)
    finally:
      for name in [k for k in sys.modules if is_multimet(k)]:
        del sys.modules[name]
      sys.modules.update(saved_modules)
      sys.path[:] = saved_path

    # No sibling checkouts: nothing to do.
    self.assertFalse(config.extend_multimet_package_path(
        repo_dir=Path(self.temp_dir.name) / "lonely" / "flood-forecasting"))

  def test_13_category_directory_provisioning_and_accessors(self):
    self.pm.login_profile("ministry_kenya")
    p_dir = self.pm.get_profile_dir("ministry_kenya")

    # Legacy directories still provisioned
    for legacy_sub in ("polygons", "attributes", "historical", "forecast", "archives"):
      self.assertTrue((p_dir / legacy_sub).is_dir())

    # 13-category directory structure provisioned
    expected_subdirs = [
        p_dir / "catchments" / "shapes",
        p_dir / "catchments" / "zonal_weights",
        p_dir / "catchments" / "basin_lists",
        p_dir / "statics" / "caravan_csv",
        p_dir / "dynamics" / "ERA5_LAND",
        p_dir / "dynamics" / "CPC",
        p_dir / "dynamics" / "IMERG",
        p_dir / "dynamics" / "HRES",
        p_dir / "dynamics" / "GRAPHCAST",
        p_dir / "realtime" / "dynamics" / "ERA5_LAND",
        p_dir / "realtime" / "dynamics" / "CPC",
        p_dir / "realtime" / "dynamics" / "IMERG",
        p_dir / "realtime" / "dynamics" / "HRES",
        p_dir / "realtime" / "dynamics" / "GRAPHCAST",
        p_dir / "realtime" / "scenarios",
        p_dir / "targets" / "uploads",
        p_dir / "assimilation" / "uploads",
        p_dir / "assimilation" / "da_states",
        p_dir / "models" / "hot_start",
        p_dir / "models" / "runs",
        p_dir / "forecasts",
        p_dir / "maas" / "snapshots",
        p_dir / "jobs",
    ]
    for d in expected_subdirs:
      self.assertTrue(d.is_dir(), f"Expected directory not provisioned: {d}")

    # Test all 21 accessor methods
    self.assertEqual(self.pm.get_catchments_dir("ministry_kenya"), p_dir / "catchments")
    self.assertEqual(
        self.pm.get_catchment_shapes_dir("ministry_kenya"),
        p_dir / "catchments" / "shapes",
    )
    self.assertEqual(
        self.pm.get_zonal_weights_dir("ministry_kenya"),
        p_dir / "catchments" / "zonal_weights",
    )
    self.assertEqual(
        self.pm.get_basin_lists_dir("ministry_kenya"),
        p_dir / "catchments" / "basin_lists",
    )
    self.assertEqual(self.pm.get_statics_dir("ministry_kenya"), p_dir / "statics")
    self.assertEqual(
        self.pm.get_statics_zarr_path("ministry_kenya"),
        p_dir / "statics" / "attributes.zarr",
    )
    self.assertEqual(self.pm.get_dynamics_dir("ministry_kenya"), p_dir / "dynamics")
    self.assertEqual(
        self.pm.get_dynamics_dir("ministry_kenya", "cpc"),
        p_dir / "dynamics" / "CPC",
    )
    self.assertEqual(
        self.pm.get_dynamics_zarr_path("ministry_kenya", "era5_land"),
        p_dir / "dynamics" / "ERA5_LAND" / "timeseries.zarr",
    )
    self.assertEqual(
        self.pm.get_realtime_dir_v2("ministry_kenya"), p_dir / "realtime"
    )
    self.assertEqual(
        self.pm.get_realtime_dynamics_zarr_path("ministry_kenya", "graphcast"),
        p_dir / "realtime" / "dynamics" / "GRAPHCAST" / "timeseries.zarr",
    )
    self.assertEqual(self.pm.get_targets_dir("ministry_kenya"), p_dir / "targets")
    self.assertEqual(
        self.pm.get_targets_zarr_path("ministry_kenya"),
        p_dir / "targets" / "streamflow.zarr",
    )
    self.assertEqual(
        self.pm.get_return_periods_path("ministry_kenya"),
        p_dir / "targets" / "return_periods.json",
    )
    self.assertEqual(
        self.pm.get_assimilation_dir("ministry_kenya"), p_dir / "assimilation"
    )
    self.assertEqual(
        self.pm.get_assimilation_zarr_path("ministry_kenya"),
        p_dir / "assimilation" / "streamflow_realtime.zarr",
    )
    self.assertEqual(
        self.pm.get_da_states_dir("ministry_kenya"),
        p_dir / "assimilation" / "da_states",
    )
    self.assertEqual(self.pm.get_models_dir("ministry_kenya"), p_dir / "models")
    self.assertEqual(
        self.pm.get_hot_start_dir("ministry_kenya"), p_dir / "models" / "hot_start"
    )
    self.assertEqual(
        self.pm.get_model_runs_dir("ministry_kenya"), p_dir / "models" / "runs"
    )
    self.assertEqual(self.pm.get_forecasts_dir("ministry_kenya"), p_dir / "forecasts")
    self.assertEqual(
        self.pm.get_forecasts_dir("ministry_kenya", basin_id="tana_01"),
        p_dir / "forecasts" / "tana_01",
    )
    self.assertTrue((p_dir / "forecasts" / "tana_01" / "history").is_dir())
    self.assertEqual(self.pm.get_jobs_dir("ministry_kenya"), p_dir / "jobs")

  def test_save_watersheds_syncs_per_basin_geojson_and_basin_lists(self):
    self.pm.login_profile("hydro_agency")
    features = [
        {
            "type": "Feature",
            "id": "basin_alpha",
            "properties": {"name": "Alpha River", "area_km2": 320.5},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[36.0, -1.0], [36.5, -1.0], [36.5, -0.5], [36.0, -1.0]]],
            },
        },
        {
            "type": "Feature",
            "properties": {"basin_id": "basin_beta", "SUB_AREA": 510.0},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[37.0, -1.0], [37.5, -1.0], [37.5, -0.5], [37.0, -1.0]]],
            },
        },
        {
            "type": "Feature",
            "properties": {"HYBAS_ID": 1120034560, "UP_AREA": 1240.0},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[38.0, -1.0], [38.5, -1.0], [38.5, -0.5], [38.0, -1.0]]],
            },
        },
    ]

    self.assertTrue(self.pm.save_watersheds(features, username="hydro_agency"))

    # 1. Verify polygons/watersheds.json and catchments/watersheds.geojson
    poly_ws = self.pm.get_polygons_dir("hydro_agency") / "watersheds.json"
    catch_ws = self.pm.get_catchments_dir("hydro_agency") / "watersheds.geojson"
    self.assertTrue(poly_ws.exists())
    self.assertTrue(catch_ws.exists())

    catch_fc = json.loads(catch_ws.read_text(encoding="utf-8"))
    self.assertEqual(catch_fc["type"], "FeatureCollection")
    self.assertEqual(len(catch_fc["features"]), 3)

    # 2. Verify individual per-basin GeoJSON files in catchments/shapes/
    shapes_dir = self.pm.get_catchment_shapes_dir("hydro_agency")
    expected_ids = ["basin_alpha", "basin_beta", "1120034560"]
    for bid in expected_ids:
      sfile = shapes_dir / f"{bid}.geojson"
      self.assertTrue(sfile.exists(), f"Missing per-basin shape file: {sfile}")
      sdata = json.loads(sfile.read_text(encoding="utf-8"))
      self.assertEqual(sdata["type"], "Feature")
      self.assertEqual(sdata["id"], bid)
      self.assertEqual(sdata["properties"]["basin_id"], bid)

    # 3. Verify basin_lists/*.txt
    lists_dir = self.pm.get_basin_lists_dir("hydro_agency")
    for list_file in (
        "all_basins.txt",
        "train_basins.txt",
        "val_basins.txt",
        "test_basins.txt",
    ):
      fpath = lists_dir / list_file
      self.assertTrue(fpath.exists())
      lines = [line.strip() for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()]
      self.assertEqual(lines, expected_ids)

    # Verify stale per-basin shape file is removed when watershed list is updated
    self.pm.save_watersheds(features[:1], username="hydro_agency")
    self.assertTrue((shapes_dir / "basin_alpha.geojson").exists())
    self.assertFalse((shapes_dir / "basin_beta.geojson").exists())
    self.assertFalse((shapes_dir / "1120034560.geojson").exists())
    self.assertEqual(
        (lists_dir / "all_basins.txt").read_text(encoding="utf-8").strip().splitlines(),
        ["basin_alpha"],
    )

  def test_guest_cleanup_wipes_13_category_artifacts(self):
    # Write artifacts into guest's 13-category directories
    self.pm.save_watersheds(
        [
            {
                "type": "Feature",
                "id": "guest_basin_99",
                "properties": {"area_km2": 100.0},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
                },
            }
        ],
        username="guest",
    )
    dummy_target = self.pm.get_targets_dir("guest") / "uploads" / "guest_basin_99_historical.csv"
    dummy_target.write_text("date,discharge_cms\n2020-01-01,10.0\n", encoding="utf-8")
    dummy_job = self.pm.get_jobs_dir("guest") / "job_123.json"
    dummy_job.write_text("{}", encoding="utf-8")

    self.assertTrue(dummy_target.exists())
    self.assertTrue(dummy_job.exists())
    self.assertTrue(
        (self.pm.get_catchment_shapes_dir("guest") / "guest_basin_99.geojson").exists()
    )

    # Clear guest data
    self.pm.clear_guest_data()
    self.assertFalse(dummy_target.exists())
    self.assertFalse(dummy_job.exists())
    self.assertFalse(
        (self.pm.get_catchment_shapes_dir("guest") / "guest_basin_99.geojson").exists()
    )
    self.assertEqual(
        (self.pm.get_basin_lists_dir("guest") / "all_basins.txt").read_text(encoding="utf-8"),
        "",
    )
    self.assertEqual(self.pm.load_watersheds("guest"), [])

  def test_get_account_summary_inventory(self):
    import numpy as np
    import pandas as pd
    import xarray as xr

    self.pm.login_profile("acct_test_user")
    features = [
        {
            "type": "Feature",
            "id": "basin_001",
            "properties": {
                "id": "basin_001",
                "name": "Test Basin 1",
                "area_km2": 450.25,
                "source": "delineated",
                "dataset": "merit_hydro",
                "outlet_lat": 40.12,
                "outlet_lon": -86.45,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-86.5, 40.0], [-86.0, 40.0], [-86.0, 40.5], [-86.5, 40.0]]],
            },
        },
        {
            "type": "Feature",
            "id": "basin_002",
            "properties": {
                "id": "basin_002",
                "name": "Test Basin 2",
                "area_km2": 820.75,
                "source": "uploaded",
                "dataset": "custom_upload",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-85.5, 40.0], [-85.0, 40.0], [-85.0, 40.5], [-85.5, 40.0]]],
            },
        },
    ]
    self.pm.save_watersheds(features, username="acct_test_user")

    # Save Caravan attributes CSV
    attr_csv = self.pm.get_attributes_dir("acct_test_user") / "caravan_hydroatlas_attributes.csv"
    df_attr = pd.DataFrame(
        {
            "ele_mt_sav": [210.5, 340.0],
            "slp_dg_sav": [1.8, 3.2],
            "pre_mm_syr": [980.0, 1120.0],
            "area_km2": [450.25, 820.75],
        },
        index=pd.Index(["basin_001", "basin_002"], name="basin_id"),
    )
    df_attr.to_csv(attr_csv)

    # Save historical training Zarr
    hist_zarr = self.pm.get_historical_dir("acct_test_user") / "historical_training_master.zarr"
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    ds_hist = xr.Dataset(
        {
            "total_precipitation_sum": (
                ("basin", "date"),
                np.ones((2, 10), dtype=np.float32),
            ),
            "temperature_2m_mean": (
                ("basin", "date"),
                np.full((2, 10), 15.0, dtype=np.float32),
            ),
            "area": (("basin",), np.array([450.25, 820.75], dtype=np.float32)),
        },
        coords={"basin": ["basin_001", "basin_002"], "date": dates},
        attrs={"source_dataset": "ERA5_LAND", "temporal_resolution": "1D"},
    )
    ds_hist.to_zarr(hist_zarr, mode="w")

    # Save return periods JSON
    rp_path = self.pm.get_return_periods_path("acct_test_user")
    rp_path.write_text(
        json.dumps({"basin_001": {"return_periods_cms": {"Q2": 50.0, "Q10": 120.0}}}),
        encoding="utf-8",
    )

    summary = self.pm.get_account_summary("acct_test_user")
    self.assertEqual(summary["username"], "acct_test_user")
    self.assertFalse(summary["is_guest"])

    # Verify totals
    totals = summary["totals"]
    self.assertEqual(totals["polygon_count"], 2)
    self.assertAlmostEqual(totals["total_area_km2"], 1271.0, places=2)
    self.assertEqual(totals["attribute_basin_count"], 2)
    self.assertEqual(totals["attribute_variable_count"], 4)
    self.assertEqual(totals["historical_store_count"], 1)

    # Verify basin coverage matrix
    coverage = {row["basin_id"]: row for row in summary["basin_coverage"]}
    self.assertIn("basin_001", coverage)
    self.assertIn("basin_002", coverage)
    self.assertTrue(coverage["basin_001"]["has_polygon"])
    self.assertTrue(coverage["basin_001"]["has_attributes"])
    self.assertTrue(coverage["basin_001"]["has_historical"])
    self.assertTrue(coverage["basin_001"]["has_return_periods"])
    self.assertFalse(coverage["basin_002"]["has_return_periods"])

    # Verify historical training metadata
    hist_stores = summary["historical_training"]["stores"]
    self.assertEqual(len(hist_stores), 1)
    self.assertEqual(hist_stores[0]["source_dataset"], "ERA5_LAND")
    self.assertEqual(hist_stores[0]["n_timesteps"], 10)
    self.assertEqual(hist_stores[0]["start_date"], "2022-01-01")
    self.assertEqual(hist_stores[0]["end_date"], "2022-01-10")
    self.assertIn("total_precipitation_sum", hist_stores[0]["dynamic_variables"])

  def test_account_tab_dom_elements_in_index_html(self):
    index_path = config.STATIC_DIR / "index.html"
    html = index_path.read_text(encoding="utf-8")
    required_snippets = [
        'data-tab="tab-account"',
        'id="accountTabBtn"',
        'id="navAccountBadge"',
        'id="tab-account"',
        'id="accountTabProfileSelect"',
        'id="accountBasinCoverageTableBody"',
        'id="accountPolygonsContainer"',
        'id="accountAttributesContainer"',
        'id="accountHistoricalContainer"',
        'id="accountRealtimeAndModelsContainer"',
        "async function loadAccountSummary(",
        "function renderAccountSummaryUI(",
    ]
    for snippet in required_snippets:
      self.assertIn(snippet, html, f"Missing Account Tab element or function in index.html: {snippet}")


if __name__ == "__main__":
  absltest.main()

