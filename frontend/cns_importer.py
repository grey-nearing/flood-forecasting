"""CNS Zarr Store Importer for Earthkit Hydro Web.

Allows developers and users to point to pre-extracted historical Zarr stores on CNS
and download them directly to local disk, skipping on-demand raw gridded extraction.
"""

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

try:
  from frontend.config import (
      ARCHIVES_DIR,
      HISTORICAL_DIR,
  )
except ImportError:
  try:
    from frontend.config import (
        ARCHIVES_DIR,
        HISTORICAL_DIR,
    )
  except ImportError:
    from config import (
        ARCHIVES_DIR,
        HISTORICAL_DIR,
    )

logger = logging.getLogger(__name__)


class CNSZarrImporter:
  """Imports and validates pre-extracted Zarr archives from CNS into local storage."""

  def __init__(self, output_dir: Optional[Path] = None):
    self.output_dir = Path(output_dir or HISTORICAL_DIR)
    self.output_dir.mkdir(parents=True, exist_ok=True)

  def import_zarr_from_cns(
      self,
      cns_path: str,
      dest_filename: Optional[str] = None,
      overwrite: bool = True,
  ) -> Dict[str, Any]:
    """Downloads a Zarr store directory from CNS to the local archives directory.

    Args:
        cns_path: Absolute path to the Zarr store on CNS (e.g. /cns/...).
        dest_filename: Name of the local destination directory/store. If None or
          'master', defaults to 'historical_training_master.zarr'.
        overwrite: Whether to overwrite if the local directory already exists.

    Returns:
        A dictionary containing metadata, dimension extents, variables, and
        chart preview.
    """
    cns_path = (cns_path or "").strip()
    if not cns_path:
      raise ValueError("CNS path cannot be empty.")

    if not cns_path.startswith("/cns/"):
      raise ValueError(f"Invalid CNS path '{cns_path}'. Must start with /cns/.")

    # Determine destination folder name
    if (
        not dest_filename
        or dest_filename.lower() in ("master", "master.zarr", "default")
        or "master" in dest_filename.lower()
    ):
      target_name = "historical_training_master.zarr"
    else:
      target_name = dest_filename if dest_filename.endswith(".zarr") else f"{dest_filename}.zarr"

    local_dest = self.output_dir / target_name

    logger.info("Starting CNS Zarr import from %s to %s", cns_path, local_dest)

    # 1. Check if source exists on CNS using fileutil
    check_cmd = ["fileutil", "ls", "-d", cns_path]
    check_res = subprocess.run(
        check_cmd, capture_output=True, text=True, timeout=30, check=False
    )
    if check_res.returncode != 0:
      raise FileNotFoundError(
          f"Source path '{cns_path}' does not exist on CNS or is inaccessible."
      )

    # 2. Prepare destination
    self.output_dir.mkdir(parents=True, exist_ok=True)
    if local_dest.exists() and overwrite:
      shutil.rmtree(str(local_dest), ignore_errors=True)

    # 3. Copy files recursively using fileutil cp
    # Strategy A: Copy source directory directly into local_dest path
    cp_cmd = ["fileutil", "cp", "-R", cns_path.rstrip("/"), str(local_dest)]
    logger.info("Running CNS copy: %s", " ".join(cp_cmd))

    cp_proc = subprocess.run(
        cp_cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 min timeout for large stores
        check=False,
    )

    if cp_proc.returncode != 0 and not local_dest.exists():
      # Strategy B: Pre-create destination and copy wildcard contents
      local_dest.mkdir(parents=True, exist_ok=True)
      cns_src = cns_path.rstrip("/") + "/*"
      cp_cmd_wildcard = ["fileutil", "cp", "-R", cns_src, str(local_dest)]
      logger.info("Direct copy failed, trying wildcard copy: %s", " ".join(cp_cmd_wildcard))
      cp_proc = subprocess.run(
          cp_cmd_wildcard,
          capture_output=True,
          text=True,
          timeout=300,
          check=False,
      )

    if not (local_dest.exists() and any(local_dest.iterdir())):
      raise RuntimeError(
          f"Failed to copy from CNS '{cns_path}' to '{local_dest}': {cp_proc.stderr}"
      )

    # 4. Compute total size on disk
    size_bytes = sum(f.stat().st_size for f in local_dest.rglob("*") if f.is_file())
    size_mb = round(size_bytes / (1024 * 1024), 2)

    # 5. Open and inspect with xarray
    zgroup_file = local_dest / ".zgroup"
    zarr_json = local_dest / "zarr.json"
    zarray_file = local_dest / ".zarray"
    if not zgroup_file.exists() and not zarr_json.exists() and not zarray_file.exists():
      try:
        with open(str(zgroup_file), "w") as f:
          f.write('{"zarr_format": 2}\n')
      except Exception:
        pass

    try:
      ds = xr.open_zarr(str(local_dest), consolidated=False)
    except Exception:
      try:
        ds = xr.open_zarr(str(local_dest))
      except Exception as e:
        raise ValueError(f"Downloaded directory is not a valid Zarr store: {e}")

    # Extract dimensions and coordinates
    basin_list = []
    if "basin" in ds.coords or "basin" in ds.dims:
      basin_list = [str(b) for b in ds.basin.values]
    elif "catchment_id" in ds.coords or "catchment_id" in ds.dims:
      basin_list = [str(b) for b in ds.catchment_id.values]

    # Extract time coordinates
    if "date" in ds.coords:
      time_coord = ds.coords["date"]
    elif "time" in ds.coords:
      time_coord = ds.coords["time"]
    else:
      time_coord = None

    timestamps = []
    if time_coord is not None:
      time_vals = pd.to_datetime(time_coord.values)
      timestamps = [t.strftime("%Y-%m-%d") for t in time_vals]
      start_date = timestamps[0] if timestamps else "N/A"
      end_date = timestamps[-1] if timestamps else "N/A"
      n_timesteps = len(timestamps)
    else:
      start_date = "N/A"
      end_date = "N/A"
      n_timesteps = 0

    var_names = list(ds.data_vars.keys())

    # 6. Build preview timeseries for the first basin
    preview_data = {}
    if timestamps and var_names and basin_list:
      primary_var = next(
          (v for v in var_names if "precip" in v.lower() or "tp" in v.lower()),
          var_names[0],
      )
      secondary_var = next(
          (v for v in var_names if "temp" in v.lower() or "t2m" in v.lower() or "pet" in v.lower()),
          None,
      )

      # Sample the first basin
      b0 = ds.sel(basin=basin_list[0]) if "basin" in ds.dims else ds
      if "lead_time" in b0.dims:
        b0 = b0.isel(lead_time=0)

      p_vals = [float(v) if not np.isnan(v) else None for v in b0[primary_var].values]
      s_vals = []
      if secondary_var and secondary_var in b0:
        s_vals = [float(v) if not np.isnan(v) else None for v in b0[secondary_var].values]

      preview_data = {
          "timestamps": timestamps[-365:] if len(timestamps) > 365 else timestamps,
          "primary_var_name": primary_var,
          "primary_var_values": p_vals[-365:] if len(p_vals) > 365 else p_vals,
          "secondary_var_name": secondary_var or "",
          "secondary_var_values": s_vals[-365:] if len(s_vals) > 365 else s_vals,
          "sample_basin_id": basin_list[0],
      }

    logger.info(
        "Successfully imported CNS Zarr store '%s' (%d basins, %d timesteps, %s MB)",
        target_name,
        len(basin_list),
        n_timesteps,
        size_mb,
    )

    return {
        "status": "completed",
        "cns_source_path": cns_path,
        "local_path": str(local_dest),
        "filename": target_name,
        "dest_filename": target_name,
        "size_mb": size_mb,
        "master_size_mb": size_mb,
        "total_basins": len(basin_list),
        "total_basins_in_master": len(basin_list),
        "basins_extracted_count": len(basin_list),
        "basin_id": basin_list[0] if basin_list else "Imported Store",
        "basins": basin_list,
        "variables": var_names,
        "start_date": start_date,
        "end_date": end_date,
        "n_timesteps": n_timesteps,
        "weather_source": "cns_import",
        "weather_source_name": f"CNS Import ({Path(cns_path).name})",
        "master_zarr_path": str(local_dest),
        "master_zarr_rel_path": f"data/archives/{target_name}",
        "preview": preview_data,
        "message": (
            f"Successfully downloaded and imported Zarr store from CNS: {target_name} "
            f"({len(basin_list)} basins, {n_timesteps} days, {size_mb} MB)."
        ),
    }
