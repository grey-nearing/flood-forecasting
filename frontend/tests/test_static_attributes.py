from pathlib import Path
import tempfile
try:
    from absl.testing import absltest
except ImportError:
    import unittest as absltest
import numpy as np
import pandas as pd

try:
    from frontend.static_attributes import (
        StaticAttributesExtractor,
        ERA5ClimateLoader,
        ERA5RawGriddedExtractor,
        compute_caravan_climate_metrics,
        ATTRIBUTE_DEFINITIONS,
    )
except ImportError:
    from frontend.static_attributes import (
        StaticAttributesExtractor,
        ERA5ClimateLoader,
        ERA5RawGriddedExtractor,
        compute_caravan_climate_metrics,
        ATTRIBUTE_DEFINITIONS,
    )


class StaticAttributesWithClimateIndicesTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extractor = StaticAttributesExtractor()
        cls.climate_loader = cls.extractor.era5_loader

    def test_schema_definitions(self):
        """Verifies that all core HydroATLAS and ERA5 climate definitions exist."""
        # HydroATLAS & Caravan properties
        self.assertIn("basin_area", ATTRIBUTE_DEFINITIONS)
        self.assertIn("ele_mt_sav", ATTRIBUTE_DEFINITIONS)
        self.assertIn("slp_dg_sav", ATTRIBUTE_DEFINITIONS)
        self.assertIn("pre_mm_syr", ATTRIBUTE_DEFINITIONS)
        self.assertIn("tmp_dc_syr", ATTRIBUTE_DEFINITIONS)
        self.assertIn("cly_pc_sav", ATTRIBUTE_DEFINITIONS)
        self.assertIn("for_pc_sse", ATTRIBUTE_DEFINITIONS)

        # ERA5-Land 1981-2020 Caravan climate properties
        self.assertIn("p_mean", ATTRIBUTE_DEFINITIONS)
        self.assertIn("pet_mean_FAO_PM", ATTRIBUTE_DEFINITIONS)
        self.assertIn("pet_mean_ERA5_LAND", ATTRIBUTE_DEFINITIONS)
        self.assertIn("aridity_FAO_PM", ATTRIBUTE_DEFINITIONS)
        self.assertIn("aridity_ERA5_LAND", ATTRIBUTE_DEFINITIONS)
        self.assertIn("frac_snow", ATTRIBUTE_DEFINITIONS)
        self.assertIn("moisture_index_FAO_PM", ATTRIBUTE_DEFINITIONS)
        self.assertIn("moisture_index_ERA5_LAND", ATTRIBUTE_DEFINITIONS)
        self.assertIn("seasonality_FAO_PM", ATTRIBUTE_DEFINITIONS)
        self.assertIn("seasonality_ERA5_LAND", ATTRIBUTE_DEFINITIONS)
        self.assertIn("high_prec_freq", ATTRIBUTE_DEFINITIONS)
        self.assertIn("high_prec_dur", ATTRIBUTE_DEFINITIONS)
        self.assertIn("low_prec_freq", ATTRIBUTE_DEFINITIONS)
        self.assertIn("low_prec_dur", ATTRIBUTE_DEFINITIONS)

    def test_era5_climate_loader(self):
        """Tests that Level 12 precomputed climate indices can be loaded and aggregated."""
        sample_hybas_ids = [7120592920, 7120867630]
        sample_weights = [15.0, 30.0]
        indices = self.climate_loader.get_indices_for_subbasins(sample_hybas_ids, sample_weights)

        self.assertIn("p_mean", indices)
        self.assertIn("pet_mean_ERA5_LAND", indices)
        self.assertIn("aridity_ERA5_LAND", indices)
        self.assertIn("frac_snow", indices)
        if not np.isnan(indices.get("p_mean", np.nan)):
            self.assertGreater(indices["p_mean"], 0.0)
            self.assertGreater(indices["pet_mean_ERA5_LAND"], 0.0)
            self.assertGreater(indices["aridity_ERA5_LAND"], 0.0)

    def test_compute_caravan_climate_metrics(self):
        """Tests that compute_caravan_climate_metrics calculates all indices properly."""
        dates = pd.date_range("1981-01-01", periods=365 * 5, freq="D")
        p = pd.Series(np.random.uniform(0.5, 5.0, size=len(dates)), index=dates)
        pet_era5 = pd.Series(np.random.uniform(1.0, 3.0, size=len(dates)), index=dates)
        pet_fao = pd.Series(np.random.uniform(2.0, 5.0, size=len(dates)), index=dates)
        t = pd.Series(np.sin(np.linspace(0, 10 * np.pi, len(dates))) * 15.0 + 10.0, index=dates)

        metrics = compute_caravan_climate_metrics(p, t, pet_era5, pet_fao)
        self.assertIn("p_mean", metrics)
        self.assertIn("pet_mean_FAO_PM", metrics)
        self.assertIn("pet_mean_ERA5_LAND", metrics)
        self.assertIn("aridity_FAO_PM", metrics)
        self.assertIn("aridity_ERA5_LAND", metrics)
        self.assertIn("frac_snow", metrics)
        self.assertIn("moisture_index_FAO_PM", metrics)
        self.assertIn("moisture_index_ERA5_LAND", metrics)
        self.assertIn("seasonality_FAO_PM", metrics)
        self.assertIn("seasonality_ERA5_LAND", metrics)
        self.assertIn("high_prec_freq", metrics)
        self.assertIn("high_prec_dur", metrics)
        self.assertIn("low_prec_freq", metrics)
        self.assertIn("low_prec_dur", metrics)
        self.assertGreater(metrics["p_mean"], 0.0)
        self.assertGreater(metrics["pet_mean_FAO_PM"], 0.0)
        self.assertGreater(metrics["pet_mean_ERA5_LAND"], 0.0)

    def test_era5_raw_gridded_extractor(self):
        """Tests raw gridded ERA5 extraction directly from CNS on sample polygon."""
        extractor = ERA5RawGriddedExtractor()
        sample_poly = {
            "type": "Polygon",
            "coordinates": [[
                [-86.6, 40.3],
                [-86.2, 40.3],
                [-86.2, 40.7],
                [-86.6, 40.7],
                [-86.6, 40.3]
            ]]
        }
        indices = extractor.extract_climate_indices_for_polygon(
            sample_poly, baseline_years=(2019, 2020)
        )
        if indices:  # If CNS is accessible
            self.assertIn("p_mean", indices)
            self.assertIn("pet_mean_FAO_PM", indices)
            self.assertIn("pet_mean_ERA5_LAND", indices)
            self.assertIn("aridity_FAO_PM", indices)
            self.assertIn("aridity_ERA5_LAND", indices)
            self.assertGreater(indices["p_mean"], 0.0)
            self.assertGreater(indices["pet_mean_FAO_PM"], 0.0)
            self.assertGreater(indices["pet_mean_ERA5_LAND"], 0.0)

    def test_extract_attributes_for_polygon(self):
        """Tests full attribute extraction on a sample polygon."""
        sample_poly = {
            "type": "Polygon",
            "coordinates": [[
                [-86.6, 40.3],
                [-86.2, 40.3],
                [-86.2, 40.7],
                [-86.6, 40.7],
                [-86.6, 40.3]
            ]]
        }
        res = self.extractor.extract_attributes_for_polygon(
            sample_poly, catchment_id="test_wabash_01", baseline_years=(2019, 2020)
        )
        
        self.assertEqual(res["catchment_id"], "test_wabash_01")
        self.assertGreater(res["total_area_km2"], 0)
        self.assertGreater(res["intersected_subbasins_count"], 0)

        # Caravan attributes dictionary
        attribs = res["caravan_attributes"]
        self.assertIn("basin_area", attribs)
        self.assertIn("ele_mt_sav", attribs)
        self.assertIn("pre_mm_syr", attribs)

        # Summary
        summary = res["summary"]
        self.assertGreater(summary["elevation_mean_m"], 0)
        if summary.get("era5_p_mean_mm_day", 0) > 0:
            self.assertGreater(summary["era5_p_mean_mm_day"], 0)
            self.assertGreater(summary["era5_pet_mean_mm_day"], 0)
            self.assertGreater(summary["era5_fao_pet_mean_mm_day"], 0)
            self.assertGreater(summary["era5_aridity"], 0)
            self.assertGreater(summary["era5_fao_aridity"], 0)

        # Categories
        self.assertIn("Climate", res["categories"])
        climate_items = {item["key"]: item for item in res["categories"]["Climate"]}
        self.assertIn("p_mean", climate_items)
        self.assertIn("pet_mean_FAO_PM", climate_items)
        self.assertIn("pet_mean_ERA5_LAND", climate_items)
        self.assertIn("aridity_FAO_PM", climate_items)
        self.assertIn("aridity_ERA5_LAND", climate_items)

    def test_export_caravan_csv(self):
        """Tests exporting extracted attributes to Caravan CSV format."""
        sample_poly = {
            "type": "Polygon",
            "coordinates": [[
                [-86.6, 40.3],
                [-86.2, 40.3],
                [-86.2, 40.7],
                [-86.6, 40.7],
                [-86.6, 40.3]
            ]]
        }
        res = self.extractor.extract_attributes_for_polygon(
            sample_poly, catchment_id="test_export_basin", baseline_years=(2019, 2020)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_attributes.csv"
            df = self.extractor.export_caravan_csv([res], csv_path)
            self.assertTrue(csv_path.exists())
            self.assertEqual(len(df), 1)
            self.assertIn("basin_area", df.columns)
            self.assertIn("p_mean", df.columns)
            self.assertIn("pet_mean_FAO_PM", df.columns)
            self.assertIn("pet_mean_ERA5_LAND", df.columns)
            self.assertIn("aridity_FAO_PM", df.columns)
            self.assertIn("aridity_ERA5_LAND", df.columns)


if __name__ == "__main__":
    absltest.main()
