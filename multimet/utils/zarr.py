# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared Zarr target classification, existence inspection, and CF time decoding."""

from __future__ import annotations

import os
from typing import Any

import gcsfs
from multimet.utils.storage import (
    decode_zarr_time_index,
    get_zarr_mapper,
    inspect_zarr_store,
    is_remote_target,
    managed_cache_dir,
    parse_cf_time_coordinate,
    plan_archive_resume,
    resolve_zarr_target,
    write_dataset_batch_in_place,
    write_dataset_batch_to_zarr,
)


def check_zarr_store_exists(
    uri: str,
    fs: Any = None,
    project: str | None = None,
) -> tuple[bool, bool, Any]:
  """Checks whether a local or GCS Zarr v2/v3 store exists without swallowing errors.

  Returns:
    Tuple of ``(exists, has_consolidated_metadata, resolved_store_or_mapper)``.
  """
  if uri.startswith(("gs://", "gcs://")):
    clean = uri.removeprefix("gs://").removeprefix("gcs://").rstrip("/")
    active_fs = fs
    if active_fs is None:
      if gcsfs is None:
        raise ImportError(f"gcsfs is required to access remote Zarr URI {uri}")
      fs_kwargs: dict[str, Any] = {"token": "anon"}
      if project:
        fs_kwargs = {"project": project}
      active_fs = gcsfs.GCSFileSystem(**fs_kwargs)
    has_zmeta = bool(active_fs.exists(f"{clean}/.zmetadata"))
    has_zgroup = bool(active_fs.exists(f"{clean}/.zgroup"))
    has_v3 = bool(active_fs.exists(f"{clean}/zarr.json"))
    exists = has_zmeta or has_zgroup or has_v3
    return exists, has_zmeta, active_fs.get_mapper(clean)

  local_path = uri[len("file://") :] if uri.startswith("file://") else uri
  if not os.path.exists(local_path):
    return False, False, local_path
  has_zmeta = os.path.exists(os.path.join(local_path, ".zmetadata"))
  has_zgroup = os.path.exists(os.path.join(local_path, ".zgroup"))
  has_v3 = os.path.exists(os.path.join(local_path, "zarr.json"))
  exists = has_zmeta or has_zgroup or has_v3
  return exists, has_zmeta, local_path


__all__ = [
    "check_zarr_store_exists",
    "decode_zarr_time_index",
    "get_zarr_mapper",
    "inspect_zarr_store",
    "is_remote_target",
    "managed_cache_dir",
    "parse_cf_time_coordinate",
    "plan_archive_resume",
    "resolve_zarr_target",
    "write_dataset_batch_in_place",
    "write_dataset_batch_to_zarr",
]
