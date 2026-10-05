---
name: documentation
description: >-
  Three-tier documentation strategy (root README.md, self-contained package and
  subpackage README.md guides, and Sphinx ReadTheDocs usage and API pages) and
  writing style conventions for flood-forecasting. Use whenever adding,
  updating, or reviewing README files, docstrings, benchmark reports, or Sphinx
  documentation in docs/.
---

# Documentation Strategy & Standards (`flood-forecasting`)

We maintain a three-tier documentation architecture across the `flood-forecasting` repository so that both end users (who just want to run models on pre-built datasets) and advanced pipeline operators (who delineate new gauges or build raw weather archives) can find clear instructions immediately.

---

## 1. Three-Tier Documentation Architecture

### Tier 1 — Concise Top-Level [`README.md`](../README.md)
- Keep the root `README.md` concise, scannable, and focused on:
  - High-level overview of the repository and production FloodHub models (`Mean-Embedding-Forecast-LSTM`, `Handoff-Forecast-LSTM`).
  - Conda environment installation (`environments/conda.yml` and `pip install -e .`).
  - Interactive Colab tutorial link and ReadTheDocs link (`https://openhydronet.readthedocs.io/`).
  - Quick pointers to pre-built Caravan/MultiMet datasets (`gs://caravan-multimet/v1.1`) and direct links to each package/subpackage `README.md` for advanced workflows.
- Do not crowd the root `README.md` with long tables of subpackage CLI flags—put those in Tier 2 and Tier 3.

### Tier 2 — Self-Contained Package & Subpackage `README.md`s
Every top-level package (`multimet/README.md`, `catchment_delineation/README.md`, `return_periods/README.md`) and major workflow subpackage (`multimet/gridded_archive_builders/README.md`, `multimet/timeseries_extractors/README.md`) must have its own self-contained `README.md` containing:

1. **"Do you need these tools?" Callout at the Top:**
   - Immediately explain when users do **not** need to run the raw pipeline because pre-built datasets already exist in `gs://open-multimet/` or `gs://caravan-multimet/v1.1`.
2. **Overview & Summary Table:**
   - Summarize the CLI entry points, Python classes, spatial resolutions, coordinate systems, and temporal coverage.
3. **Prerequisites & Credentials:**
   - Show Conda environment activation (`conda activate googlehydrology`), editable install (`pip install -e .`), and any external authentication setup (e.g., NASA Earthdata Login `.netrc` or bearer token, GCP application default credentials).
4. **Copy-Pasteable Quick Start Examples:**
   - Provide realistic, copy-pasteable CLI and Python examples covering initial builds, incremental archive extension (`--extend_archive`), cloud streaming (`gs://`), and cache cleanup (`--cleanup_cache` / `--clean-cache`).
5. **What to Watch Out For (Common Pitfalls):**
   - Document failure modes clearly (e.g., no hidden path defaults, area-hint mismatch errors, latitude coverage limits, 48-file daily IMERG requirement, 7-day tail publication lag).
6. **Complete Command-Line Arguments Reference:**
   - Document every CLI flag, its type, default value, and behavior.

### Tier 3 — Sphinx ReadTheDocs Documentation (`docs/source/`)
Every user-facing package or subpackage must be documented in Sphinx (`docs/`):

1. **Usage Guide (`docs/source/usage/<feature>.rst`):**
   - Step-by-step user guide in reStructuredText mirroring the package `README.md`.
   - Must be added to the `.. toctree::` in `docs/source/index.rst`.
2. **API Reference (`docs/source/api/<module>.rst`):**
   - Module reference using `.. automodule::` (`:members:`, `:undoc-members:`, `:show-inheritance:`).
   - Must be registered in `docs/source/api/modules.rst` and `docs/Makefile`.
3. **Build Verification:**
   - Always verify that `make -C docs html` builds cleanly with zero Sphinx warnings or broken cross-references.

---

## 2. Writing Style & Formatting Norms

- **Clear, Accessible Plain English:**
  - Write for both hydrologists and software engineers. Explain physical meaning and units explicitly (e.g., `mm/day`, $\text{km}^2$, `EPSG:4326` decimal degrees, UTC calendar days).
- **Use Standard Domain Terminology:**
  - Use accurate hydrological terminology (e.g., in `return_periods`, use official USGS Bulletin 17C terms: PILFs, MGBT, EMA, LP-III).
- **Clean Formatting:**
  - Avoid unnecessary inline bolding clutter in prose and benchmark reports (`REPORT.md`). Use Markdown tables, code blocks, and section headings for structure.
