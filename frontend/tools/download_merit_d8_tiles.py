"""Downloads global 5x5 degree MERIT-Hydro 90m (3 arc-second) D8 flow-direction tiles.

Fetches the 'dir' band from the official MERIT/Hydro/v1_0_1 asset on Google
Earth Engine's high-volume pixel API and writes (6000, 6000) uint8 ESRI D8
flow-direction .npy tiles to ~/.cache/openhydronet/data/merit_dem/tiles_5deg.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Optional, Tuple
import urllib.request

import numpy as np

EE_MERIT_GET_PIXELS_URL = (
    "https://earthengine-highvolume.googleapis.com/v1/"
    "projects/earthengine-public/assets/MERIT/Hydro/v1_0_1:getPixels"
)
DEFAULT_EE_USER_PROJECT = os.environ.get("EE_USER_PROJECT", "ee-hydro_user")
DEFAULT_MERIT_TILES_DIR = (
    Path.home()
    / ".cache"
    / "openhydronet"
    / "data"
    / "merit_dem"
    / "tiles_5deg"
)
DEFAULT_HYDROSHEDS_TILES_DIR = (
    Path.home()
    / ".cache"
    / "openhydronet"
    / "data"
    / "hydrosheds_dem"
    / "tiles_5deg"
)

RES_DEG = 1.0 / 1200.0
HALF_ROWS = 3000
FULL_COLS = 6000
VALID_D8_CODES = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int16)

_TOKEN_LOCK = threading.Lock()
_CACHED_TOKEN: Optional[str] = None
_TOKEN_EXPIRY: float = 0.0


def get_adc_access_token(force_refresh: bool = False) -> str:
  """Returns a valid Google Cloud ADC OAuth2 access token."""
  global _CACHED_TOKEN, _TOKEN_EXPIRY
  now = time.time()
  with _TOKEN_LOCK:
    if not force_refresh and _CACHED_TOKEN and now < _TOKEN_EXPIRY:
      return _CACHED_TOKEN
    token = subprocess.check_output(
        ["gcloud", "auth", "application-default", "print-access-token"],
        text=True,
        timeout=15,
    ).strip()
    if not token:
      raise RuntimeError("Failed to obtain gcloud application-default token.")
    _CACHED_TOKEN = token
    _TOKEN_EXPIRY = now + 1800.0  # Refresh every 30 minutes
    return token


def parse_tile_stem(stem: str) -> Tuple[int, int]:
  """Parses a tile stem like 'n40w090' into (lat_top, lon_left)."""
  s = stem.strip().lower()
  if len(s) != 7 or s[0] not in ("n", "s") or s[3] not in ("e", "w"):
    raise ValueError(f"Invalid 5x5 tile stem: {stem!r}")
  lat_mag = int(s[1:3])
  lon_mag = int(s[4:7])
  lat_top = lat_mag if s[0] == "n" else -lat_mag
  lon_left = lon_mag if s[3] == "e" else -lon_mag
  return lat_top, lon_left


def format_tile_name(lat_top: int, lon_left: int) -> str:
  """Formats (lat_top, lon_left) into 'n40w090.npy'."""
  lat_str = f"n{lat_top:02d}" if lat_top >= 0 else f"s{abs(lat_top):02d}"
  lon_str = f"w{abs(lon_left):03d}" if lon_left < 0 else f"e{lon_left:03d}"
  return f"{lat_str}{lon_str}.npy"


def _fetch_half_tile(
    y_top: float,
    x_left: float,
    user_project: str = DEFAULT_EE_USER_PROJECT,
    retries: int = 3,
) -> np.ndarray:
  """Fetches a (3000, 6000) half-tile of MERIT/Hydro/v1_0_1 'dir' band."""
  payload = json.dumps({
      "fileFormat": "NPY",
      "bandIds": ["dir"],
      "grid": {
          "dimensions": {"width": FULL_COLS, "height": HALF_ROWS},
          "affineTransform": {
              "scaleX": RES_DEG,
              "shearX": 0.0,
              "translateX": float(x_left),
              "shearY": 0.0,
              "scaleY": -RES_DEG,
              "translateY": float(y_top),
          },
          "crsCode": "EPSG:4326",
      },
  }).encode("utf-8")

  last_err: Optional[Exception] = None
  for attempt in range(retries):
    try:
      token = get_adc_access_token(force_refresh=(attempt > 0))
      req = urllib.request.Request(
          EE_MERIT_GET_PIXELS_URL,
          data=payload,
          headers={
              "Authorization": f"Bearer {token}",
              "x-goog-user-project": user_project,
              "Content-Type": "application/json",
          },
      )
      with urllib.request.urlopen(req, timeout=60) as resp:
        raw = np.load(io.BytesIO(resp.read()))
      return raw["dir"]
    except Exception as err:  # pylint: disable=broad-except
      last_err = err
      time.sleep(0.5 * (attempt + 1))
  raise RuntimeError(
      f"Failed to fetch MERIT-Hydro dir half-tile ({y_top}, {x_left}): {last_err}"
  ) from last_err


def download_merit_d8_tile(
    lat_top: int,
    lon_left: int,
    target_dir: Path = DEFAULT_MERIT_TILES_DIR,
    user_project: str = DEFAULT_EE_USER_PROJECT,
) -> Path:
  """Downloads a single 5x5 degree (6000, 6000) uint8 MERIT-Hydro D8 tile atomically."""
  target_dir = Path(target_dir)
  target_dir.mkdir(parents=True, exist_ok=True)
  tile_name = format_tile_name(int(lat_top), int(lon_left))
  out_path = target_dir / tile_name
  if out_path.is_file() and out_path.stat().st_size >= 36_000_000:
    return out_path

  top_arr = _fetch_half_tile(float(lat_top), float(lon_left), user_project=user_project)
  bot_arr = _fetch_half_tile(
      float(lat_top) - 2.5, float(lon_left), user_project=user_project
  )
  full_raw = np.vstack([top_arr, bot_arr])
  valid_mask = np.isin(full_raw, VALID_D8_CODES)
  d8_uint8 = np.where(valid_mask, full_raw, 0).astype(np.uint8)

  tmp_path = target_dir / f".{tile_name}.tmp.{os.getpid()}.{threading.get_ident()}.npy"
  try:
    np.save(tmp_path, d8_uint8)
    os.replace(tmp_path, out_path)
  finally:
    if tmp_path.exists():
      try:
        tmp_path.unlink()
      except OSError:
        pass
  return out_path


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Download global 5x5 degree MERIT-Hydro 90m D8 tiles."
  )
  parser.add_argument(
      "--target-dir",
      type=Path,
      default=DEFAULT_MERIT_TILES_DIR,
      help="Destination directory for .npy tiles.",
  )
  parser.add_argument(
      "--reference-tiles-dir",
      type=Path,
      default=DEFAULT_HYDROSHEDS_TILES_DIR,
      help="Directory of land tile filenames to mirror.",
  )
  parser.add_argument(
      "--workers",
      type=int,
      default=12,
      help="Number of concurrent download threads.",
  )
  args = parser.parse_args()

  args.target_dir.mkdir(parents=True, exist_ok=True)
  ref_files = sorted(args.reference_tiles_dir.glob("*.npy"))
  tile_keys = [parse_tile_stem(p.stem) for p in ref_files]
  missing = [
      (lat, lon)
      for (lat, lon) in tile_keys
      if not (args.target_dir / format_tile_name(lat, lon)).is_file()
  ]
  print(
      f"Total land 5x5 tiles: {len(tile_keys)} | "
      f"Already downloaded: {len(tile_keys) - len(missing)} | "
      f"Remaining: {len(missing)}"
  )
  if not missing:
    return

  t0 = time.time()
  completed = 0
  failed = 0
  with ThreadPoolExecutor(max_workers=args.workers) as pool:
    fut_map = {
        pool.submit(download_merit_d8_tile, lat, lon, args.target_dir): (lat, lon)
        for (lat, lon) in missing
    }
    for fut in as_completed(fut_map):
      lat, lon = fut_map[fut]
      try:
        fut.result()
        completed += 1
      except Exception as exc:  # pylint: disable=broad-except
        failed += 1
        print(f"FAILED {format_tile_name(lat, lon)}: {exc}", flush=True)
      if (completed + failed) % 25 == 0 or (completed + failed) == len(missing):
        elapsed = time.time() - t0
        rate = (completed + failed) / max(elapsed, 1e-3)
        print(
            f"[{completed + failed}/{len(missing)}] "
            f"ok={completed} failed={failed} "
            f"({rate:.2f} tiles/s, elapsed={elapsed:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
  main()
