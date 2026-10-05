---
name: pr-review
description: >-
  Comprehensive Pull Request (PR) code review checklist, standards, and workflow
  for flood-forecasting (OpenHydroNet / Open-MultiMet). Covers zero-tolerance
  checks for phantom/fake/masked/imputed data, removal of try/except error
  suppression, mathematical and autograd rigor, pretrained model weight and
  evaluation hygiene, non-trivial test and mock audits, object-oriented design
  and redundancy checks, Google Python readability, documentation rules, and
  author-facing review tone. Use whenever reviewing a PR or auditing code
  changes before merge.
---

# Pull Request Review Standards & Checklist (`flood-forecasting`)

Whenever asked to perform a thorough PR review in this repository, load this skill alongside [`skills/algorithm-rules-and-norms.md`](./algorithm-rules-and-norms.md), [`skills/testing.md`](./testing.md), [`skills/repo-organization.md`](./repo-organization.md), [`skills/documentation.md`](./documentation.md), and the Google Python `readability` skill, and evaluate every change against the following nine mandatory pillars.

---

## 1. Zero Tolerance for Phantom, Fake, Masked, or Imputed Data

Audit every line of the diff for any sliver of potential to introduce, fabricate, impute, or mask bad or missing data (see [`skills/algorithm-rules-and-norms.md`](./algorithm-rules-and-norms.md)):

1. **Missing Data In $\implies$ `NaN` Out (or Loud Exception):**
   - **Never** silently drop, zero-fill (`0.0`), forward-fill (`ffill`), backward-fill (`bfill`), linearly interpolate, or replace missing dates, sub-daily granules, grid cells, basins, or forecast lead times with climatology or constants.
   - **Never** substitute backup or fallback data products or sources when a requested source is unavailable, delayed, or fails to download (no `"auto"` source fallbacks, no silent fallback between NASA GES DISC and other buckets, no mixing of incompatible dataset versions such as IMERG V06 `precipitationCal` with V07 `precipitation`, and no mixing of prior and current forecast runs via `preserve_existing_valid` flags).
   - **Never** substitute a fallback statistical estimator, distribution, or fabricated quantile when an algorithm fails to converge or lacks sufficient valid observations.
   - **Never** broadcast a single analysis or forecast step across missing forecast lead times.
2. **Strict Sub-Daily & Spatial Coverage Thresholds:**
   - Verify sub-daily accumulations enforce complete slot coverage (e.g., all 48 unique 30-minute start tokens `-S000000` through `-S233000` for NASA IMERG daily UTC totals, requiring `valid_counts == 48` at every `(lat, lon)` cell; any cell with `< 48` valid observations must be `NaN`).
   - Verify spatial/zonal aggregation uses exact area-weighted polygon-grid / polygon-subbasin intersections and enforces the **$\ge 80\%$ valid (non-`NaN`) area coverage** threshold (`< 80%` $\implies$ `NaN`).
   - Reject any centroid or nearest-neighbor snapping for out-of-domain or non-intersecting polygons.
   - In catchment delineation, reject silent truncation at `max_cells` (`CatchmentCoverageError`), missing neighbor tiles (`FileNotFoundError` / `RuntimeError`), or `60°N` boundaries, and ensure unmatched `--expected-area` hints log `[AREA HINT FAILURE]` to `stderr` and raise `CatchmentAreaMismatchError` without writing an output polygon.
3. **Temporal Continuity, Publication Lag, Sentinel Values & Cache Safety:**
   - Gridded Zarr archives must maintain strictly contiguous daily time axes (`1D` frequency) with zero interior date gaps and zero interior all-`NaN` days.
   - Never allow trimming trailing `NaN` slices on historical years; only allow up to `MAX_PUBLICATION_LAG_DAYS = 7` days of upstream publication lag at the tail of the current year when `--end_date` is omitted (when `--end_date` is explicitly passed, require 100% finite coverage through `end_date`).
   - Never trust cached files by filename alone (`precip.{year}.nc`); verify internal coordinate timestamps **and** finite data values cover the required dates, and ensure `--extend_archive` bypasses pre-cached files to force fresh upstream downloads.
   - Never allow `--in_place` or incremental updates to overwrite existing valid archive slices with `NaN`s.
   - Verify that format decoders (GRIB2, NetCDF, HDF5, Zarr) explicitly convert bitmap/sentinel missing values (such as `9999.0` or `_FillValue`) to `np.nan` and validate coordinate axes, bounds, and latitude orientation (`-90..90` ascending) before use.

---

## 2. Eliminate All `try`/`except` Blocks and Error-Swallowing Logic

Flag and require removing **every** `try`/`except` block and equivalent error-masking construct:

1. **No `try`/`except` Control Flow or Suppression:**
   - Flag all `try`/`except` blocks, `contextlib.suppress(...)`, `shutil.rmtree(..., ignore_errors=True)`, and `warnings.catch_warnings` blocks that silence numerical or runtime failures.
   - Flag equivalent silent-fallback patterns: `.get(key, default)` on required dictionary/config fields, `getattr(obj, attr, default)` that hides missing class attributes or mismatches between private and public attributes, runtime `inspect.signature(...)` duck-typing shims, or early `return None` / `return np.nan` / `continue` branches on corrupt or missing inputs.
2. **Explicit Precondition Validation:**
   - Require explicit precondition checks (`os.path.exists` / `Path.exists()`, `in` checks on dictionary keys, DataFrame columns, and NetCDF/HDF5/Zarr variables/dimensions, and explicit HTTP status checks such as `response.status_code == 404` before `response.raise_for_status()`) that immediately raise descriptive `ValueError`, `KeyError`, `FileNotFoundError`, or `RuntimeError`.
3. **IEEE 754 `NaN` / `Inf` & Boolean Trap in Config and Input Validators:**
   - In Python/IEEE 754 floating-point, `float('nan') <= 0` and `float('nan') < minimum` both evaluate to `False`, and `isinstance(True, int)` is `True`.
   - Verify that every numeric config/parameter validator explicitly checks `not math.isfinite(value)` and `not isinstance(value, bool)` (and sets `allow_inf_nan=False` / `strict=True` on Pydantic models) so `NaN`, `Inf`, and booleans never pass validation silently.

---

## 3. Mathematical, Statistical, & Conceptual Rigor

Do not merely skim mathematical or domain code—verify formulas from first principles and write a small standalone Python reproduction script in `scratch/` to test edge cases against the PR branch:

1. **Forward-Pass, Loss, & Autograd Gradient Integrity Under `NaN`s:**
   - Trace how `NaN` values in dynamic inputs, static attributes, targets, or baseline embeddings propagate through **both** the forward pass and the **backward pass (`autograd` gradients)**.
   - Watch for `0.0 * NaN = NaN` in PyTorch autograd and **backward-in-time `NaN` gradient corruption through recurrent layers (`CudaLSTM` / `LSTM`)**: even when a loss is evaluated purely on finite historical steps (`t < seq_length`), an all-`NaN` timestep at a future forecast step (`t >= seq_length`) in a tensor passed through an LSTM will corrupt gradients backward in time into the historical window unless masked out before the recurrent layer.
   - In masked loss functions (`MaskedMSELoss`, `MaskedCMALLoss`, `MaskedNSELoss`) and regularizers (`BackgroundEmbeddingRegularization`):
     - Verify masking operates per-timestep/per-element rather than dropping entire sequences on a single `NaN`.
     - Check normalization denominators (`N_valid` vs. `B * W * E`), ensure an all-`NaN` sequence/batch returns a clean zero loss with finite zero gradients (no divide-by-zero or `NaN` norm that poisons `clip_gradient_norm_`), and verify changes do not silently rescale loss magnitudes (which alters the effective learning rate).
     - Check relative scale and dimensionality between loss terms and regularization terms (e.g., whether background embedding regularization needs scale normalization relative to the observation loss, and whether `_EPS` floors are non-vanishing across dtypes).
     - Test numerical stability across `float16`, `bfloat16`, `float32`, and `float64` (e.g., clamping in log/exp or Laplace CDF calculations where values inside clamp bounds can still overflow in `float16` and produce `-inf + inf = nan` gradients).
2. **Temporal Indexing, Sequence Slicing, & Future-Data Leakage:**
   - Check every sequence slice (`seq_length`, `lead_time`, `_min_lead_time`, `predict_last_n`, `assimilation_window`) for off-by-one shifts and future-data leakage.
   - Verify `sample['date']`, `sample['y']`, and `prediction['y_hat']` align to the exact same timestamps across both forecast (`lead_time > 0`) and hindcast-only configurations, including when `predict_last_n < lead_time`.
   - In Data Assimilation (DA) and hindcast/forecast models (`MeanEmbeddingForecastLSTM`, `HandoffForecastLSTM`):
     - Verify embedding overrides and observation losses are strictly bounded to the historical window (`0 <= start < end <= seq_length - lead_time`), cannot bypass forecast-horizon guards via full-length tensor overrides, and never slice into the future forecast horizon (`[:, -lead_time:]`) via suffix slicing (`predict_last_n`).
     - Check whether historical assimilation updates must apply consistently to both `hindcast_embedding` and the historical portion of `forecast_embedding` / `shared_embeddings`.
   - In Hot-Start (`save_state` / `load_state`): verify states are only saved/loaded during inference/evaluation (never during training or fine-tuning), include both hidden (`h`) and cell (`c`) states keyed by basin ID and timestamp, and use the proper config object (`self.cfg` vs. `self.tester_cfg`).
3. **Probabilistic Post-Processing, Geospatial, & Hydrological Algorithms:**
   - Distinguish between clipping, truncating, and resampling predictive distributions (e.g., stochastic vs. deterministic CMAL sampling in `cmal_deterministic._search_quantile` and root-bracketing math).
   - Check geospatial algorithms for spherical/equal-area cell-latitude math (never using tile-center latitude for every cell), D8 `int8` bitmask sign handling (`128` overflow in signed `int8`), half-cell grid registration, antimeridian/pole edge cases, and CRS consistency.
   - Check statistical hydrology algorithms (e.g., USGS Bulletin 17C EMA moments, MGBT thresholds, regional skew weighting) against authoritative references (`peakfq` Fortran / R `MGBT`) and published benchmark datasets.

---

## 4. Pretrained Model Weights, Numerical Equivalence, & Evaluation Integrity

Whenever a PR modifies model math, feature ordering, temporal alignment, scalers, or input/output data formats:

1. **Backwards Compatibility & Numerical Equivalence:**
   - Check whether existing `pretrained-models/`, `tutorial/model-runs/5-basin-example`, and saved scalers (`scaler.nc` / `scaler.zarr`) still load and evaluate cleanly, or whether the PR fixes a training/alignment bug that requires retraining the shipped weights.
   - For format or dataloader refactors, verify claimed numerical equivalence empirically (checking `float32` vs. `float64` promotion, chunking order, and coordinate decoding).
2. **Pretrained Weight & Checkpoint Hygiene:**
   - When updating pretrained model weights in the repository:
     - Commit **only** the single target epoch weight file (`model_epoch*.pt`) and a single set of test metrics per model.
     - **Never commit optimizer state (`optimizer_epoch*.pt`)** or intermediate epoch checkpoints, and check `git status` / `git diff --stat` for accidentally committed large binary blobs, scratch files, or credentials.
     - Select the **largest/final completed training epoch** rather than cherry-picking an intermediate epoch based on test-set metrics (to avoid test-set overfitting).
     - When comparing old vs. new model evaluation metrics, verify that both models were evaluated on the **exact same basin cohort and time period**.

---

## 5. Test Suite Rigor, Sincerity, & Mocking Audit

Inspect every test file in the PR with the same scrutiny as production code (see [`skills/testing.md`](./testing.md)):

1. **Reject Trivial or Shape-Only Tests:**
   - Flag tests that only check `.shape`, `isinstance`, or exit codes without asserting on exact numerical outputs, coordinate alignment, mathematical invariants, or persisted Zarr/Parquet contents.
2. **Verify Regression Tests Fail on Unpatched `main`:**
   - For bug-fix PRs, verify that the test calls the real public method where the bug occurred (not a private helper or a copy-pasted formula inside the test) and **actually fails when run against `main` without the fix**.
3. **Audit Every Mock, Monkeypatch, and Stub (`unittest.mock`, `monkeypatch`, `SimpleNamespace`):**
   - **Never allow mocks that mask core components under test:** Reject tests that mock out the mathematical function, parser, dataset loader, or model component being tested (e.g., mocking `_search_quantile` while testing deterministic clipping, mocking `ZonalEngine` or NetCDF/Zarr readers with pre-baked DataFrames, or passing `SimpleNamespace` stubs that omit real `Config`/`Dataset` attributes and hide `AttributeError` bugs).
   - Only external network or cloud boundaries (`requests`, Earthdata/CMR HTTP endpoints, `gcsfs`) may be mocked in hermetic unit tests.
4. **Enforce Native Spatial & Temporal Resolution in Synthetic Fixtures:**
   - Reject tests that monkeypatch grid dimensions (`LAT_COUNT`, `LON_COUNT`, `IMERG_LATS`, `TILE_CELLS`) to tiny toy grids (`2 × 4`, `3 × 3`), hardcode fake constant timestamps (`2024-01-01` across multi-day files), or use 3-day files as fake "full years" without passing explicit `start_date`/`end_date`.
5. **Check Completeness, Conciseness, & CI Wiring:**
   - Require end-to-end integration tests for new features (e.g., verifying partial-basin loading produces bit-identical metrics to full-basin loading, or running an actual training + evaluation loop).
   - Require explicit negative tests proving missing/corrupt/out-of-domain inputs fail loudly with the expected exception.
   - Flag redundant or bloated tests, verify Pytest markers (`@pytest.mark.unit`, `integration`, `slow`, `gpu`, `canary`), and confirm `pyproject.toml` (`testpaths`) and `.github/workflows/pytest-ci.yml` (`--cov=<package>`) include the package's test suite.

---

## 6. Architecture, Object-Oriented Structure, Redundancy, & Scope Discipline

Evaluate how the PR fits into the broader codebase (see [`skills/repo-organization.md`](./repo-organization.md)):

1. **Object-Oriented Design & Inheritance vs. Duplication:**
   - Check whether new classes (especially in student/intern or AI-generated PRs) replace or copy-paste whole components that should instead subclass and extend existing abstractions (for example, `AssimilationConfig` should inherit from `Config` rather than duplicating a parallel 370-line config parser; shared logic between `HandoffForecastLSTM` and `MeanEmbeddingForecastLSTM` should live on a shared base class or helper).
   - Ensure new hooks or features do not clutter the core forward pass or make future extensions harder.
2. **Code Redundancy & Shared Utilities (`multimet/utils/`):**
   - Search across packages and subpackages for duplicated utilities (GCS/Zarr storage, CF time decoding, HTTP/Earthdata downloads, FAO-56 Penman-Monteith PET, Caravan climate indices, polygon/zonal geometry).
   - Consolidate shared domain utilities into `multimet/utils/` (even if a specific function in that utility module is currently called by only one subpackage) and remove thin re-export shims.
3. **Package Layout, Explicit CLI Paths, & Dead-Code Removal:**
   - Verify files live in their canonical package/subpackage locations (`googlehydrology/` / `model/`, `multimet/<subpackage>/`, `return_periods/`), tests live in `<package>/tests/`, and auxiliary scripts live in `<package>/tools/` (never in root `scripts/` or `tools/` folders, and never with subpackage `.github/workflows/`).
   - Ensure all input and output paths are required explicitly via caller arguments or CLI flags—flag and remove any hardcoded `gs://open-multimet/data`, `/tmp/`, local user paths, or implicit default output paths.
   - Remove dead code, unused archival paths, vestigial multi-frequency code, out-of-scope product code (e.g., unfinished HRES/ERA5 code in a CPC/IMERG PR), and temporary working-note `.md` plan files in the repository root.
4. **PR Scope Discipline:**
   - Keep repository-organization and renaming PRs strictly separate from functional or algorithmic code fixes. If a review of a structural PR uncovers pre-existing code bugs, open tracked GitHub issues for those bugs rather than bundling functional changes into the reorganization PR.

---

## 7. Python Readability & Google Style Compliance

Apply the Google Python Style Guide (`readability` skill) and verify `ruff check` / `ruff format --check` pass with zero errors:

1. **Variable & Parameter Naming Clarity:**
   - Flag confusingly similar identifiers in the same scope (e.g., plural `hindcast_embeddings` vs. singular `hindcast_embedding` differing by a single letter `s` while holding different data structures).
   - Flag misleading parameter names when types are widened (e.g., keeping a parameter named `model` when it accepts `torch.nn.Module | Iterable[torch.Tensor] | Iterable[dict]`; rename to `model_or_params` or keep a narrow contract).
   - Flag single-letter variable names (`k`, `l`, `O`) outside brief comprehensions.
2. **Docstring Contracts & Type Annotations:**
   - Require complete Python 3.10+ type annotations and Google-style docstrings (`Args:`, `Returns:`, `Raises:`) on public classes and functions.
   - Verify that docstring promises match the actual code behavior (e.g., whether `predict_last_n` slices the tail of the passed tensor vs. the historical window, or whether a method mutates inputs).
3. **PyTorch & NumPy Idioms:**
   - Prefer clean tensor idioms (`tensor.new_zeros(...)`, `torch.zeros_like(...)`) over manual device plumbing, and vectorize array checks instead of writing Python loops over timesteps or grid cells where vectorization is natural.

---

## 8. Documentation Standards

Verify that documentation is updated in lockstep with code changes and obeys [`skills/documentation.md`](./documentation.md):

1. **Rule 1 — Document Only What Exists and How to Use It:**
   - Documentation must describe **only what currently exists in the repository and how a user runs it**.
   - Reject any documentation that narrates development history, design decisions, what was removed or replaced, or internal engineering/agent rules (such as our ban on `try`/`except` blocks).
2. **Rule 2 — Target Human Readability Over Technical Proficiency:**
   - Ensure all READMEs, usage guides, and reports are **concise** (never verbose or written in dense "Claude-speak") and written for a **high-school reading level**, **non-experts** in CS/ML/math/hydrology, and **English-as-a-Second-Language (ESL)** readers.
   - Keep root `README.md` descriptions of features extremely short, with detailed CLI tables and copy-pasteable quick-start examples in the package/subpackage `README.md` and Sphinx docs (`docs/source/usage/` and `docs/source/api/`).
   - Verify `make -C docs html` builds cleanly with zero warnings, and check that the PR title and PR description match the final state of the branch.

---

## 9. Review Tone, Register, & GitHub Mechanics

1. **Write for the PR Author, Not as a Reviewer Lab Notebook:**
   - Address comments directly to the author: state **what** is wrong, **why** it matters (citing concrete reproduction numbers, shapes, or counterexamples as facts), and **what** change is needed.
   - **Never** narrate your own investigation process ("I chased this down", "in my test harness", "I checked two checkouts") or deliberate over rejected options the author did not propose.
   - Preserve genuine caveats honestly (never overstate an unverified claim), and keep GitHub ```` ```suggestion ```` blocks byte-for-byte accurate so they can be applied with one click.
2. **Cite Exact File Paths and Line Numbers:**
   - Every finding in a review report or inline comment must include the exact file path and line numbers (`path/to/file.py:L123-L145`).
3. **Never Publish or Submit Reviews Without Explicit Approval:**
   - Do not run mutating `gh pr review` / `gh api -X POST` calls to publish a review unless explicitly instructed.
   - When asked to stage comments on GitHub so the user can review/edit them in the web UI first, post them as a **PENDING (unsubmitted) review** (omitting the `event` field in the GitHub Reviews API) and wait for explicit confirmation before submitting—and only submit for the exact PR number specified.
4. **Equal, Non-Deferential Scrutiny for All Authors:**
   - Apply the exact same rigor whether the PR is authored by an external contributor, a student intern, an AI coding agent, or the repository maintainer (`grey-nearing`). For external fork PRs, also check workflow/CI security and branch write-access risks.
