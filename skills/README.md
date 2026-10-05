# Project Agent Skills (`skills/`)

This directory contains project-level skill guides for AI coding assistants working on `flood-forecasting` (OpenHydroNet / Open-MultiMet):

| Skill | File | Purpose |
| :--- | :--- | :--- |
| **`repo-organization`** | [`skills/repo-organization.md`](./repo-organization.md) | Top-level package layout (`googlehydrology/`, `multimet/`, `catchment_delineation/`, `return_periods/`), `multimet/` subpackage boundaries, `multimet/utils/` shared helpers (no duplication or re-export shims), and packaging/CI synchronization. |
| **`algorithm-rules-and-norms`** | [`skills/algorithm-rules-and-norms.md`](./algorithm-rules-and-norms.md) | Non-negotiable algorithmic rules and data-integrity norms: missing data in $\implies$ `NaN` out, zero `try`/`except` blocks that mask errors, explicit paths only, 48-file IMERG daily accumulation, 80% valid zonal coverage, and 7-day tail publication lag / `--extend_archive` cache safety. |
| **`testing`** | [`skills/testing.md`](./testing.md) | Co-located `<package>/tests/` layout, native-resolution synthetic testing (never monkeypatch grid sizes to toy grids), strict mocking only at external network/cloud boundaries, and required data-integrity regression coverage. |
| **`documentation`** | [`skills/documentation.md`](./documentation.md) | Three-tier documentation strategy: concise root `README.md`, self-contained package/subpackage `README.md` guides (with *"Do you need these tools?"* callouts), and Sphinx ReadTheDocs (`docs/source/usage/` + `docs/source/api/`). |
| **`gcs-bucket-organization`** | [`skills/gcs-bucket-organization.md`](./gcs-bucket-organization.md) | Directory hierarchy, dataset schemas, and Python/CLI access patterns for `gs://open-multimet/` (`caravan-new/`, `caravan-old/`, `caravan-multimet/`, `gridded-data-archives/`, `ancillary-data/`, and `data/era5_land/`). |
