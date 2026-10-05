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

"""Shared utilities for MultiMet data workflows."""

from multimet.utils.gcs import (
    download_directory_from_gcs,
    download_file_from_gcs,
    download_hydroatlas_from_gcs,
    gcs_path_exists,
    is_gcs_path,
    normalize_gcs_path,
    read_bytes_from_gcs,
    strip_gcs_prefix,
    sync_gcs_directory,
    upload_file_to_gcs,
    upload_to_gcs,
)
from multimet.utils.http import (
    DEFAULT_CMR_GRANULES_URL,
    EarthdataSession,
    check_http_url_exists,
    download_http_file,
    get_earthdata_credentials_from_netrc,
    query_cmr_granules,
)
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

__all__ = [
    "DEFAULT_CMR_GRANULES_URL",
    "EarthdataSession",
    "check_http_url_exists",
    "decode_zarr_time_index",
    "download_directory_from_gcs",
    "download_file_from_gcs",
    "download_http_file",
    "download_hydroatlas_from_gcs",
    "gcs_path_exists",
    "get_earthdata_credentials_from_netrc",
    "get_zarr_mapper",
    "inspect_zarr_store",
    "is_gcs_path",
    "is_remote_target",
    "managed_cache_dir",
    "normalize_gcs_path",
    "parse_cf_time_coordinate",
    "plan_archive_resume",
    "query_cmr_granules",
    "read_bytes_from_gcs",
    "resolve_zarr_target",
    "strip_gcs_prefix",
    "sync_gcs_directory",
    "upload_file_to_gcs",
    "upload_to_gcs",
    "write_dataset_batch_in_place",
    "write_dataset_batch_to_zarr",
]

