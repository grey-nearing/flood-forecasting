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

"""Command-line interface for synchronizing gridded NWP weather runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from multimet.weather_fetcher.config import DYNAMICAL_MODELS, SOURCE_LABELS
from multimet.weather_fetcher.sync import read_sync_status, sync_all_models

# Exit code 0 only when the data directory is fully consistent afterwards.
_SUCCESS_RESULTS = ("updated", "up_to_date", "busy")


def build_parser() -> argparse.ArgumentParser:
  """Builds the argument parser for `sync-weather-forecasts`."""
  sources = ", ".join(dict.fromkeys(SOURCE_LABELS.values()))
  parser = argparse.ArgumentParser(
      prog="sync-weather-forecasts",
      description=(
          "Synchronize operational gridded weather runs from "
          f"{sources} into float16 binary streams on the 0.25 degree global "
          "grid with an atomic run-directory swap."
      ),
  )
  parser.add_argument(
      "--data-dir",
      "--data-root",
      dest="data_dir",
      type=str,
      required=True,
      help=(
          "Directory for downloaded forecast runs (required; runs are stored"
          " under <data-dir>/runs and exposed through <data-dir>/current)."
      ),
  )
  parser.add_argument(
      "--cpc-cache-dir",
      dest="cpc_cache_dir",
      type=str,
      default=None,
      help=(
          "Directory caching NOAA PSL CPC annual NetCDF files (defaults to"
          " <data-dir>/cpc_cache)."
      ),
  )
  parser.add_argument(
      "--models",
      type=str,
      default="",
      help=(
          "Comma-separated subset of models to synchronize: "
          + ", ".join(DYNAMICAL_MODELS.keys())
      ),
  )
  parser.add_argument(
      "--force",
      action="store_true",
      help=(
          "Force re-download even when the latest init_time is already"
          " synced."
      ),
  )
  parser.add_argument(
      "--status",
      action="store_true",
      help="Print the current synchronization status JSON and exit.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry point for `sync-weather-forecasts`.

  Returns:
    ``0`` when the run directory is consistent afterwards (``updated``,
    ``up_to_date`` or ``busy``), ``1`` when any selected model failed
    (``partial`` or ``error``). Unsupported model keys raise ``ValueError``.
  """
  parser = build_parser()
  args = parser.parse_args(argv)
  data_dir = Path(args.data_dir).expanduser().resolve()

  if args.status:
    print(json.dumps(read_sync_status(data_dir), indent=2))
    return 0

  models: list[str] | None = (
      [m.strip() for m in args.models.split(",") if m.strip()]
      if args.models
      else None
  )
  result = sync_all_models(
      data_dir=data_dir,
      models=models,
      force=args.force,
      log=lambda msg: print(msg, flush=True),
      cpc_cache_dir=args.cpc_cache_dir,
  )
  return 0 if result.get("last_result") in _SUCCESS_RESULTS else 1


if __name__ == "__main__":
  sys.exit(main())
