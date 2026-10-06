"""Unit tests for Streamflow CSV Ingestion, Unit Normalization, Multi-Basin Zarr Persistence, and Return Periods."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import xarray as xr

try:
  from absl.testing import absltest
except ImportError:
  import unittest as absltest

from frontend.profile_manager import ProfileManager
from frontend.streamflow_csv_ingestor import (
    CMS_TO_MM_DAY_FACTOR,
    compute_return_periods,
    ingest_historical_streamflow_csv,
    ingest_realtime_streamflow_csv,
    read_basin_streamflow_series,
)


class StreamflowCsvIngestorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.profiles_root = Path(self.temp_dir.name) / "users"
    self.pm = ProfileManager(profiles_dir=self.profiles_root)
    self.pm.login_profile("hydro_tester")

  def tearDown(self):
    self.temp_dir.cleanup()
    super().tearDown()

  def test_unit_conversion_cms_and_mm_day_exact(self):
    """Verifies exact physical unit normalization: streamflow_mm = Q_cms * 86.4 / Area_km2."""
    area_km2 = 432.0  # 86.4 / 432.0 = 0.2 -> 10.0 m3/s == 2.0 mm/day
    csv_cms = (
        "date,discharge_cms\n"
        "2021-05-01,10.0\n"
        "2021-05-02,25.0\n"
        "2021-05-03,50.0\n"
    )
    res_cms = ingest_historical_streamflow_csv(
        csv_data=csv_cms,
        basin_id="basin_cms_test",
        username="hydro_tester",
        area_km2=area_km2,
        input_units="auto",
        profile_manager=self.pm,
    )
    self.assertEqual(res_cms["status"], "success")
    self.assertEqual(res_cms["detected_input_units"], "m3/s")

    ds = xr.open_zarr(res_cms["zarr_path"], consolidated=False)
    self.assertEqual(ds.sizes["basin"], 1)
    self.assertEqual(ds.sizes["date"], 3)
    self.assertEqual(ds["streamflow"].dtype, np.float32)
    self.assertEqual(ds["discharge_cms"].dtype, np.float32)
    self.assertEqual(ds["streamflow"].attrs.get("units"), "mm/day")
    self.assertEqual(ds["discharge_cms"].attrs.get("units"), "m3/s")

    np.testing.assert_allclose(
        ds["discharge_cms"].sel(basin="basin_cms_test").values,
        np.array([10.0, 25.0, 50.0], dtype=np.float32),
        rtol=1e-5,
    )
    expected_mm = np.array([10.0, 25.0, 50.0], dtype=np.float32) * (
        CMS_TO_MM_DAY_FACTOR / area_km2
    )
    np.testing.assert_allclose(
        ds["streamflow"].sel(basin="basin_cms_test").values,
        expected_mm,
        rtol=1e-5,
    )

    # Now ingest mm/day directly and verify inverse conversion to m3/s
    csv_mm = (
        "timestamp,streamflow_mm\n"
        "2021-05-01,2.0\n"
        "2021-05-02,5.0\n"
        "2021-05-03,10.0\n"
    )
    res_mm = ingest_historical_streamflow_csv(
        csv_data=csv_mm,
        basin_id="basin_mm_test",
        username="hydro_tester",
        area_km2=area_km2,
        input_units="auto",
        profile_manager=self.pm,
    )
    self.assertEqual(res_mm["detected_input_units"], "mm/day")
    ds2 = xr.open_zarr(res_mm["zarr_path"], consolidated=False)
    np.testing.assert_allclose(
        ds2["streamflow"].sel(basin="basin_mm_test").values,
        np.array([2.0, 5.0, 10.0], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        ds2["discharge_cms"].sel(basin="basin_mm_test").values,
        np.array([10.0, 25.0, 50.0], dtype=np.float32),
        rtol=1e-5,
    )

  def test_area_lookup_from_saved_watersheds(self):
    """Verifies automatic lookup of area_km2 / SUB_AREA / UP_AREA from saved ProfileManager watersheds."""
    self.pm.save_watersheds(
        [
            {
                "type": "Feature",
                "id": "saved_basin_01",
                "properties": {"SUB_AREA": 864.0},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[36.0, -1.0], [36.5, -1.0], [36.5, -0.5], [36.0, -1.0]]],
                },
            }
        ],
        username="hydro_tester",
    )
    csv_data = "date,q_cms\n2022-01-01,100.0\n2022-01-02,200.0\n"
    res = ingest_historical_streamflow_csv(
        csv_data=csv_data,
        basin_id="saved_basin_01",
        username="hydro_tester",
        area_km2=None,
        input_units="cms",
        profile_manager=self.pm,
    )
    self.assertAlmostEqual(res["area_km2"], 864.0)
    series = read_basin_streamflow_series(
        basin_id="saved_basin_01",
        username="hydro_tester",
        mode="historical",
        profile_manager=self.pm,
    )
    # 100.0 * 86.4 / 864.0 = 10.0 mm/day
    np.testing.assert_allclose(series["streamflow_mm_day"], [10.0, 20.0], rtol=1e-5)
    np.testing.assert_allclose(series["discharge_cms"], [100.0, 200.0], rtol=1e-5)

  def test_multi_basin_zarr_merging_historical_and_realtime(self):
    """Verifies multi-basin Zarr merging in targets/streamflow.zarr and assimilation/streamflow_realtime.zarr."""
    csv_b1 = "date,discharge_cms\n2023-01-01,12.0\n2023-01-02,15.0\n"
    csv_b2 = "date,discharge_cms\n2023-01-02,30.0\n2023-01-03,45.0\n"

    ingest_historical_streamflow_csv(
        csv_data=csv_b1,
        basin_id="catchment_a",
        username="hydro_tester",
        area_km2=100.0,
        profile_manager=self.pm,
    )
    res_hist = ingest_historical_streamflow_csv(
        csv_data=csv_b2,
        basin_id="catchment_longer_name_b",
        username="hydro_tester",
        area_km2=200.0,
        profile_manager=self.pm,
    )

    self.assertEqual(
        res_hist["all_basins"], ["catchment_a", "catchment_longer_name_b"]
    )
    hist_ds = xr.open_zarr(res_hist["zarr_path"], consolidated=False)
    self.assertEqual(tuple(hist_ds["streamflow"].dims), ("basin", "date"))
    self.assertEqual(hist_ds.sizes["basin"], 2)
    self.assertEqual(hist_ds.sizes["date"], 3)
    self.assertTrue(
        np.isnan(
            hist_ds["discharge_cms"]
            .sel(basin="catchment_a", date="2023-01-03")
            .values.item()
        )
    )
    self.assertAlmostEqual(
        float(
            hist_ds["discharge_cms"]
            .sel(basin="catchment_longer_name_b", date="2023-01-03")
            .values.item()
        ),
        45.0,
        places=4,
    )

    # Also test real-time assimilation multi-basin Zarr merging + sub-daily aggregation
    rt_b1 = (
        "datetime,discharge\n"
        "2026-09-20T06:00:00Z,10.0\n"
        "2026-09-20T18:00:00Z,20.0\n"
        "2026-09-21T12:00:00Z,30.0\n"
    )
    rt_b2 = (
        "datetime,discharge\n"
        "2026-09-21T00:00:00Z,50.0\n"
        "2026-09-22T00:00:00Z,60.0\n"
    )
    res_rt1 = ingest_realtime_streamflow_csv(
        csv_data=rt_b1.encode("utf-8"),
        basin_id="rt_basin_1",
        username="hydro_tester",
        area_km2=86.4,
        profile_manager=self.pm,
    )
    self.assertTrue(res_rt1["is_subdaily_aggregated"])
    self.assertEqual(len(res_rt1["raw_timestamps"]), 3)

    res_rt2 = ingest_realtime_streamflow_csv(
        csv_data=rt_b2,
        basin_id="rt_basin_2",
        username="hydro_tester",
        area_km2=86.4,
        profile_manager=self.pm,
    )
    rt_ds = xr.open_zarr(res_rt2["zarr_path"], consolidated=False)
    self.assertEqual(tuple(rt_ds["streamflow"].dims), ("basin", "date"))
    self.assertEqual(
        [str(b) for b in rt_ds.coords["basin"].values],
        ["rt_basin_1", "rt_basin_2"],
    )
    # On 2026-09-20, mean of 10.0 and 20.0 is 15.0 m3/s (and since area_km2=86.4, 15.0 mm/day)
    self.assertAlmostEqual(
        float(
            rt_ds["discharge_cms"]
            .sel(basin="rt_basin_1", date="2026-09-20")
            .values.item()
        ),
        15.0,
        places=4,
    )
    self.assertTrue(Path(res_rt1["raw_csv_path"]).exists())
    self.assertTrue(Path(res_rt2["raw_csv_path"]).exists())

  def test_sentinel_and_missing_value_handling(self):
    """Verifies sentinel values (-999, -9999, negative) and blanks become NaN without corrupting daily means."""
    csv_with_sentinels = (
        "date,discharge_cms\n"
        "2022-03-01,20.0\n"
        "2022-03-02,-9999\n"
        "2022-03-03,-999.0\n"
        "2022-03-04,\n"
        "2022-03-05,NaN\n"
        "2022-03-06,-5.0\n"
        "2022-03-07,40.0\n"
    )
    res = ingest_historical_streamflow_csv(
        csv_data=csv_with_sentinels,
        basin_id="sentinel_basin",
        username="hydro_tester",
        area_km2=86.4,
        profile_manager=self.pm,
    )
    self.assertEqual(res["n_observations"], 7)
    self.assertEqual(res["n_valid"], 2)
    self.assertEqual(res["n_missing"], 5)

    ds = xr.open_zarr(res["zarr_path"], consolidated=False)
    vals = ds["discharge_cms"].sel(basin="sentinel_basin").values
    self.assertAlmostEqual(float(vals[0]), 20.0, places=4)
    for idx in (1, 2, 3, 4, 5):
      self.assertTrue(np.isnan(vals[idx]), f"Expected NaN at index {idx}, got {vals[idx]}")
    self.assertAlmostEqual(float(vals[6]), 40.0, places=4)

    series = read_basin_streamflow_series(
        basin_id="sentinel_basin",
        username="hydro_tester",
        mode="historical",
        profile_manager=self.pm,
    )
    self.assertEqual(series["discharge_cms"], [20.0, None, None, None, None, None, 40.0])

  def test_return_period_calculation_gumbel_and_empirical_persistence(self):
    """Verifies Q2 < Q5 < Q10 < Q20 < Q50 < Q100 in both Gumbel (>=3 yrs) and empirical (<3 yrs) modes."""
    rng = np.random.default_rng(42)
    dates_5yr = pd.date_range("2018-01-01", "2022-12-31", freq="D")
    # Synthetic seasonal + extreme discharge record over 5 years
    day_of_year = dates_5yr.dayofyear.to_numpy()
    base_flow = 50.0 + 30.0 * np.sin(2.0 * np.pi * day_of_year / 365.25)
    peaks = rng.gumbel(loc=40.0, scale=25.0, size=len(dates_5yr))
    q_5yr = np.clip(base_flow + peaks, 5.0, None)

    lines = ["date,discharge_cms"] + [
        f"{d.strftime('%Y-%m-%d')},{v:.3f}" for d, v in zip(dates_5yr, q_5yr)
    ]
    csv_5yr = "\n".join(lines)

    res = ingest_historical_streamflow_csv(
        csv_data=csv_5yr,
        basin_id="gumbel_basin",
        username="hydro_tester",
        area_km2=500.0,
        profile_manager=self.pm,
    )
    rp = res["return_periods"]
    self.assertEqual(rp["method"], "gumbel")
    self.assertEqual(rp["n_years"], 5)

    # Verify strict monotonicity Q2 < Q5 < Q10 < Q20 < Q50 < Q100 in both m3/s and mm/day
    for unit_key in ("cms", "mm_day"):
      u_dict = rp[unit_key]
      self.assertLess(u_dict["Q2"], u_dict["Q5"])
      self.assertLess(u_dict["Q5"], u_dict["Q10"])
      self.assertLess(u_dict["Q10"], u_dict["Q20"])
      self.assertLess(u_dict["Q20"], u_dict["Q50"])
      self.assertLess(u_dict["Q50"], u_dict["Q100"])

    # Verify return_periods.json persistence on disk
    rp_path = self.pm.get_return_periods_path("hydro_tester")
    self.assertTrue(rp_path.exists())
    persisted = json.loads(rp_path.read_text(encoding="utf-8"))
    self.assertIn("gumbel_basin", persisted)
    self.assertAlmostEqual(
        persisted["gumbel_basin"]["Q100_mm_day"],
        persisted["gumbel_basin"]["Q100_cms"] * (CMS_TO_MM_DAY_FACTOR / 500.0),
        places=2,
    )

    # Also verify empirical fallback (< 3 years)
    short_dates = pd.date_range("2023-01-01", "2023-06-30", freq="D")
    short_q = np.linspace(10.0, 200.0, len(short_dates))
    rp_emp = compute_return_periods(
        short_q, dates=short_dates, area_km2=100.0, input_units="cms", basin_id="emp_basin"
    )
    self.assertEqual(rp_emp["method"], "empirical")
    self.assertLess(rp_emp["Q2"], rp_emp["Q5"])
    self.assertLess(rp_emp["Q5"], rp_emp["Q10"])
    self.assertLess(rp_emp["Q10"], rp_emp["Q20"])
    self.assertLess(rp_emp["Q20"], rp_emp["Q50"])
    self.assertLess(rp_emp["Q50"], rp_emp["Q100"])


if __name__ == "__main__":
  absltest.main()
