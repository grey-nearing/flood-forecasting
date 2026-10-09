"""CLI utility to build simplified z2/z4 river network pyramids for the web UI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from shapely.geometry import mapping

from frontend.config import ensure_flood_forecasting_on_sys_path
from frontend.river_indexer import _reach_to_ui_feature

ensure_flood_forecasting_on_sys_path()

from multimet.catchment_delineation.hydrography import RiverNetwork  # pylint: disable=g-import-not-at-top


def build_pyramid_geojson(
    network: RiverNetwork,
    *,
    min_stream_order: int,
    simplify_tolerance_deg: float,
) -> dict[str, object]:
  """Extracts reaches with `stream_order >= min_stream_order` and simplifies geometries."""
  reaches = network.query_reaches(
      bbox=(-180.0, -85.0, 180.0, 85.0),
      min_stream_order=min_stream_order,
  )
  features = []
  for r in reaches:
    feat = _reach_to_ui_feature(r)
    if simplify_tolerance_deg > 0 and r.geometry is not None:
      simp = r.geometry.simplify(simplify_tolerance_deg, preserve_topology=True)
      if not simp.is_empty:
        feat["geometry"] = mapping(simp)
    features.append(feat)
  return {"type": "FeatureCollection", "features": features}


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="Build z2 and z4 simplified river pyramids for OpenHydroNet UI."
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      required=True,
      help="Explicit output directory for pyramid GeoJSON files.",
  )
  parser.add_argument(
      "--hydrorivers-shp",
      type=Path,
      default=None,
      help="Optional path to HydroRIVERS_v10.shp.",
  )
  parser.add_argument(
      "--merit-rivers-dir",
      type=Path,
      default=None,
      help="Optional path to MERIT-Basins river shapefiles directory.",
  )
  args = parser.parse_args(argv)
  args.output_dir.mkdir(parents=True, exist_ok=True)

  if args.hydrorivers_shp is not None:
    net = RiverNetwork.from_hydrorivers(args.hydrorivers_shp)
    z2 = build_pyramid_geojson(
        net, min_stream_order=8, simplify_tolerance_deg=0.02
    )
    z4 = build_pyramid_geojson(
        net, min_stream_order=7, simplify_tolerance_deg=0.008
    )
    (args.output_dir / "global_rivers_fast_z2.geojson").write_text(
        json.dumps(z2), encoding="utf-8"
    )
    (args.output_dir / "global_rivers_fast_z4.geojson").write_text(
        json.dumps(z4), encoding="utf-8"
    )

  if args.merit_rivers_dir is not None:
    net = RiverNetwork.from_merit_basins(args.merit_rivers_dir)
    z2 = build_pyramid_geojson(
        net, min_stream_order=7, simplify_tolerance_deg=0.02
    )
    z4 = build_pyramid_geojson(
        net, min_stream_order=6, simplify_tolerance_deg=0.008
    )
    (args.output_dir / "merit_rivers_fast_z2.geojson").write_text(
        json.dumps(z2), encoding="utf-8"
    )
    (args.output_dir / "merit_rivers_fast_z4.geojson").write_text(
        json.dumps(z4), encoding="utf-8"
    )

  return 0


if __name__ == "__main__":
  sys.exit(main())
