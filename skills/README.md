# Project Agent Skills (`skills/`)

This directory contains project-level `SKILL.md` guides for AI coding assistants working on `flood-forecasting` (OpenHydroNet / Open-MultiMet):

| Skill | Path | Purpose |
| :--- | :--- | :--- |
| **`repo-organization`** | [`skills/repo-organization/SKILL.md`](./repo-organization/SKILL.md) | Repository package/subpackage architecture, `<package>/tests/` co-location, `multimet/utils/` shared & domain helpers, strict data-integrity rules (no `try`/`except`, no default output/archive paths, no silent gap-filling, $\ge 80\%$ valid basin coverage, native-grid testing), and our 3-tier documentation strategy (`README.md` + Sphinx `docs/`). |
| **`gcs-bucket-organization`** | [`skills/gcs-bucket-organization/SKILL.md`](./gcs-bucket-organization/SKILL.md) | Directory hierarchy, dataset schemas, and Python/CLI access patterns for `gs://open-multimet/` (`caravan-new/`, `caravan-old/`, `caravan-multimet/`, `gridded-data-archives/`, `ancillary-data/`, and `data/era5_land/`). |
