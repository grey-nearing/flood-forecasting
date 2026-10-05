---
name: documentation
description: >-
  Core documentation rules (focus strictly on what exists and how to use it;
  write concisely for a high-school reading level, non-experts, and ESL
  readers) and three-tier documentation architecture (root README.md,
  package/subpackage README.md guides, and Sphinx ReadTheDocs) for
  flood-forecasting. Use whenever adding, updating, or reviewing README files,
  docstrings, benchmark reports, or Sphinx documentation in docs/.
---

# Documentation Strategy & Standards (`flood-forecasting`)

## 1. Core High-Level Rules (Mandatory)

### Rule 1 — Document Only What Exists and How to Use It
- **Focus strictly on the present system and how to use it:** Every README, usage guide, docstring, and report must describe **only what currently exists in the repository and how a user runs or interacts with it**.
- **Never expose development history or design decisions:** Do **not** explain the decisions made to arrive at the current design, the path or iterations taken to get here, what approaches were previously tried or replaced, or why alternative designs were rejected.
- **Never expose internal coding rules to users:** Do **not** clutter user-facing documentation with internal engineering or agent rules (for example, no user cares that we enforce a rule against `try`/`except` blocks—that belongs in [`skills/algorithm-rules-and-norms.md`](./algorithm-rules-and-norms.md), not in user documentation). Users only need to know what the software does, what inputs it requires, and what outputs or errors it produces.

### Rule 2 — Target Human Readability Over Technical Proficiency
- **Target reader profile:** Write all documentation for a reader at a **high school reading level** who is a **non-expert** in computer science, machine learning, mathematics, and hydrology.
- **Write for English as a Second Language (ESL) readers:** Many or most readers of this repository have **English as a second language**. Use short, direct sentences, active voice, and plain, everyday words. Avoid idioms, colloquialisms, complex nested clauses, or dense academic/software jargon. When a technical or hydrological term is required, define it simply in plain language at point of use.
- **Be strictly concise (never verbose):** Keep explanations as short and direct as possible. Cut filler paragraphs, repetitive explanations, and unnecessary theoretical background. Prefer brief bullet points, tables, and copy-pasteable commands over long blocks of text.

---

## 2. Three-Tier Documentation Architecture

We maintain a three-tier documentation structure so both everyday users (who want to run models on pre-built datasets) and pipeline operators (who delineate new gauges or build weather archives) can find clear instructions right away.

### Tier 1 — Concise Top-Level [`README.md`](../README.md)
- Keep the root `README.md` short, scannable, and focused on:
  - Brief overview of what is in the repository and how to run the models (`Mean-Embedding-Forecast-LSTM`, `Handoff-Forecast-LSTM`).
  - Conda environment installation (`environments/conda.yml` and `pip install -e .`).
  - Interactive Colab tutorial link and ReadTheDocs link (`https://openhydronet.readthedocs.io/`).
  - Quick links to pre-built datasets (`gs://caravan-multimet/v1.1`) and direct links to each package/subpackage `README.md`.
- Do not crowd the root `README.md` with long tables of subpackage CLI flags—put those in Tier 2 and Tier 3.

### Tier 2 — Self-Contained Package & Subpackage `README.md`s
Every top-level package (`multimet/README.md`, `return_periods/README.md`) and major workflow subpackage (`multimet/catchment_delineation/README.md`, `multimet/gridded_archive_builders/README.md`, `multimet/timeseries_extractors/README.md`) must have its own self-contained `README.md` containing:

1. **"Do you need these tools?" Callout at the Top:**
   - Immediately tell users when they do **not** need to run the raw pipeline because pre-built datasets already exist in `gs://open-multimet/` or `gs://caravan-multimet/v1.1`.
2. **Overview & Summary Table:**
   - Summarize the CLI commands, Python classes, spatial resolutions, coordinate systems, and date ranges.
3. **Prerequisites & Credentials:**
   - Show Conda environment activation (`conda activate openhydronet`), installation (`pip install -e .`), and any required login steps (such as NASA Earthdata Login `.netrc` or Google Cloud credentials).
4. **Copy-Pasteable Quick Start Examples:**
   - Provide short, copy-pasteable CLI and Python examples for initial runs, adding new dates (`--extend_archive`), cloud paths (`gs://`), and cache cleanup (`--cleanup_cache` / `--clean-cache`).
5. **What to Watch Out For (Common Pitfalls):**
   - Clearly list practical usage limits and requirements (such as required path flags, latitude limits, required input files, or data publication delays) without lecturing on internal code implementation.
6. **Complete Command-Line Arguments Reference:**
   - Document every CLI flag, its type, default value, and what it does in plain language.

### Tier 3 — Sphinx ReadTheDocs Documentation (`docs/source/`)
Every user-facing package or subpackage must be documented in Sphinx (`docs/`):

1. **Usage Guide (`docs/source/usage/<feature>.rst`):**
   - Concise step-by-step user guide in reStructuredText matching the package `README.md`.
   - Must be added to the `.. toctree::` in `docs/source/index.rst`.
2. **API Reference (`docs/source/api/<module>.rst`):**
   - Module reference using `.. automodule::` (`:members:`, `:undoc-members:`, `:show-inheritance:`).
   - Must be registered in `docs/source/api/modules.rst` and `docs/Makefile`.
3. **Build Verification:**
   - Always verify that `make -C docs html` builds cleanly with zero Sphinx warnings or broken links.

---

## 3. Formatting & Terminology Norms

- **State Units and Formats Explicitly:**
  - Always state physical units and formats clearly (for example, `mm/day`, $\text{km}^2$, `EPSG:4326` latitude/longitude degrees, UTC dates in `YYYY-MM-DD`).
- **Use Standard Domain Terms With Plain Explanations:**
  - Use standard hydrological terms (for example, official USGS Bulletin 17C terms in `return_periods`), and briefly state what they mean in plain words so non-experts can follow.
- **Clean Visual Formatting:**
  - Avoid excessive inline bolding in paragraphs and reports (`REPORT.md`). Use Markdown tables, code blocks, and short headings to organize information cleanly.
