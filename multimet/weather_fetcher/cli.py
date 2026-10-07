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
import os
from pathlib import Path
import sys
from typing import List, Optional, Sequence

from multimet.weather_fetcher.config import DYNAMICAL_MODELS
from multimet.weather_fetcher.sync import read_sync_status, sync_all_models


def resolve_default_weather_data_dir() -> Path:
  """Resolves the default CLI weather cache directory."""
  env = os.environ.get("EARTHKIT_WEATHER_DATA_DIR")
  if env and env.strip():
    return Path(env).expanduser().resolve()
  return Path.home() / ".cache" / "earthkit_hydro_web" / "weather"


def build_parser() -> argparse.ArgumentParser:
  """Builds the argument parser for `sync-weather-forecasts`."""
  parser = argparse.ArgumentParser(
      prog="sync-weather-forecasts",
      description=(
          "Synchronize operational gridded NWP weather forecasts from "
          "dynamical.org into float16 binary streams with atomic directory swap."
      ),
  )
  parser.add_argument(
      "--data-dir",
      "--data-root",
      dest="data_dir",
      type=str,
      default=None,
      help=(
          "Directory for downloaded forecast runs "
          "(defaults to $EARTHKIT_WEATHER_DATA_DIR or "
          "~/.cache/earthkit_hydro_web/weather)."
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
      help="Force re-download even when the latest init_time is already synced.",
  )
  parser.add_argument(
      "--status",
      action="store_true",
      help="Print the current synchronization status JSON and exit.",
  )
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  """CLI entry point for `sync-weather-forecasts`."""
  parser = build_parser()
  args = parser.parse_args(argv)

  data_dir = (
      Path(args.data_dir).expanduser().resolve()
      if args.data_dir
      else resolve_default_weather_data_dir()
  )

  if args.status:
    status = read_sync_status(data_dir)
    print(json.dumps(status, indent=2))
    return 0

  models: Optional[List[str]] = (
      [m.strip() for m in args.models.split(",") if m.strip()]
      if args.models
      else None
  )

  result = sync_all_models(
      data_dir=data_dir,
      models=models,
      force=args.force,
      log=lambda msg: print(msg, flush=True),
  )
  ok = result.get("last_result") in ("updated", "up_to_date", "busy")
  if argv is None:
    sys.exit(0 if ok else 1)
  return 0 if ok else 1


if __name__ == "__main__":
  main()
