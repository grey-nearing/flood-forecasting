# Return Period Calculator (`return_periods`)

`return_periods` is a standalone flood frequency analysis package implementing the official USGS Bulletin 17C guidelines (*Guidelines for Determining Flood Flow Frequency — Bulletin 17C*, England et al., 2019; https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf).

## Key Features

* **Expected Moments Algorithm (`EMA`):** Iterative Log-Pearson Type III (`LP-III`) parameter estimation supporting systematic records, zero flood years, and interval-censored observations (`GEMAFitter`).
* **Multiple Grubbs-Beck Test (`MGBT`):** Full implementation of the Cohn et al. (2013) orthogonal-$t$ Gaussian quadrature low-outlier test (`MultipleGrubbsBeckTester`) with two-stage outward ($\alpha_{\text{out}} = 0.005$) and inward ($\alpha_{\text{in}} = 0.10$) significance sweeps for Potentially Influential Low Flood (`PILF`) screening.
* **Legacy Bulletin 17B Support:** Optional single-outlier Grubbs-Beck critical-value table test (`GrubbsBeckTester`, `use_multiple_grubbs_beck=False`).
* **Regional Skew Weighting:** Supports combining at-site station skew with regional skew and regional skew MSE per Bulletin 17C Equation 7-10 (`regional_skew`, `regional_skew_mse`).
* **Multiple Distribution Fitters:** Supports explicit fitter selection via `fitter='gema'` (`GEMAFitter`), `fitter='simple_lp3'` (`SimpleLogPearson3Fitter`), `fitter='log_linear'` (`LogLogTrendFitter`), or a custom `BaseFitter` subclass.
* **Hirsch-Stedinger Threshold Plotting Positions:** Empirical plotting positions (`simple_empirical_plotting_position` and `threshold_exceedance_empirical_plotting_position`) following Bulletin 17C Appendix 5.
* **Peak Extraction & Visualization:** Extract Annual Maximum Series (calendar or water year) or Peaks-Over-Threshold (POT) from daily hydrographs and plot fitted distributions, return period curves, and annotated hydrographs.

---

## Quick Start

### 1. From an Annual Peak Series

```python
import pandas as pd
from return_periods import ReturnPeriodCalculator

peaks = pd.Series(
    [2080, 1670, 1480, 2940, 1560, 2380, 2720, 2860, 2620, 1710],
    index=range(1947, 1957),
)

rpc = ReturnPeriodCalculator(peaks_series=peaks, fitter='gema')

# Compute discharge quantiles for 2-, 5-, 10-, 25-, 50-, and 100-year floods
return_periods = [2, 5, 10, 25, 50, 100]
flows = rpc.flow_values_from_return_periods(return_periods)

# Invert discharge values back to return periods (years)
estimated_rps = rpc.return_periods_from_flow_values(flows)
```

### 2. From a Daily Streamflow Hydrograph

```python
from return_periods import ReturnPeriodCalculator

# Extracts annual peaks (use water_year_start_month=10 for Oct-Sep water years)
rpc = ReturnPeriodCalculator(
    hydrograph_series=daily_streamflow_series,
    fitter='gema',
    max_missing_days_in_year=30,
    water_year_start_month=10,
)

fig, ax = rpc.plot_hydrograph(plot_return_periods=[2, 5, 10, 25, 50, 100])
```

### 3. Direct MGBT Low-Outlier Screening

```python
from return_periods import MultipleGrubbsBeckTester

tester = MultipleGrubbsBeckTester(
    data=annual_peaks,
    is_log_transformed=False,
)
print('Number of PILFs (klow):', tester.klow)
print('PILF threshold (cfs):', 10.0**tester.threshold)
```

---

## Executive Summary: Python vs. USGS Fortran and USGS R Bulletin 17C Flood Calculators

We benchmarked three versions of the US government's official flood-frequency calculator (USGS Bulletin 17C) across 11,213 global river basins (342,950 flood years):

1. **The Python Version (`return_periods`):** Our Python library.
2. **The USGS Fortran Version (`peakfqr` v8.0):** The official USGS Fortran program, compiled from unmodified Fortran source code (`gfortran`).
3. **The USGS R Version (`MGBT` v1.1.6):** The official USGS R package from CRAN, run in R 4.6.0.

Bulletin 17C has two steps:

* **Step 1 — Multiple Grubbs-Beck Test (`MGBT`):** Identifies Potentially Influential Low Floods (`PILFs`)—small drought-year peaks (including zeros) that can distort the lower tail of the flood distribution. Run by Python, USGS Fortran, and USGS R.
* **Step 2 — Expected Moments Algorithm (`EMA`):** Fits a Log-Pearson Type III (`LP-III`) distribution to estimate flood quantiles like the 10-year ($Q_{10}$) and 100-year ($Q_{100}$) floods. Run by Python and USGS Fortran (the USGS R package only implements Step 1).

### Key Quantitative Results

* **Step 1 — MGBT PILF Detection (Python vs. USGS Fortran):**
  * **100.00% Match (`11,171 / 11,171` completed basins):** Python and USGS Fortran flagged the exact same number of PILFs in both $\text{ft}^3/\text{s}$ and $\text{mm/day}$.
  * **100.00% Completion in Python (`11,213 / 11,213` basins):** Python completed all 11,213 basins in 18.2 seconds, whereas USGS Fortran exited early or did not converge on 42 basins (0.37%).
* **Step 1 — MGBT PILF Detection (Python & USGS Fortran vs. USGS R):**
  * **98.80% Match (`11,078 / 11,213` basins):** Default USGS R (`MGBT17c`) matched Python and USGS Fortran on 98.80% of basins on positive flows (98.27% in $\text{ft}^3/\text{s}$, 97.81% in $\text{mm/day}$ when including zero-flow rivers).
  * **Bulletin 17C Appendix 10 Benchmark (Orestimba Creek, CA):** Python and USGS Fortran match the published Bulletin 17C textbook result (30 PILFs, $T_{\text{PILF}} = 782\text{ ft}^3/\text{s}$, $Q_{100} = 13,820\text{ ft}^3/\text{s}$), whereas USGS R `MGBT17c` flags 38 PILFs due to differences in its non-central $t$ evaluation and zero-flow handling.
* **Step 2 — EMA LP-III Flood Quantiles (Python vs. USGS Fortran):**
  * **10-Digit Agreement on 97.35% of Basins (`10,875 / 11,171`):** Median $Q_{100}$ difference is 0.0000000071%, with 99.99% of basins within 0.01% and 100.00% within 0.1%.
  * **Unit-Conversion Consistency on 296 Basins (2.64%):** USGS Fortran uses a fixed $10^{-6}$ lower bound for censored PILFs (`gbtmin = -6.0`), which causes Fortran's $Q_{100}$ flood quantiles to shift by up to 58.6% when converting between $\text{ft}^3/\text{s}$ and $\text{mm/day}$. Python uses 0 as the lower bound and produces 100.00% identical flood quantiles across unit systems.
* **Turning Off the Iterative EMA Loop (Non-Iterative vs. Iterative Python):**
  * **72.71% of Basins Have Zero PILFs (`8,153 / 11,213`):** Non-iterative LP-III vs. full Bulletin 17C gives 0.00% difference.
  * **27.29% of Basins With PILFs (`3,060 / 11,213`):** Skipping both MGBT and EMA has 1.51% median error (vs. full Bulletin 17C) on $Q_{10}$ and 12.55% on $Q_{100}$; keeping Step 1 MGBT while turning off Step 2 EMA cuts that $Q_{100}$ error to 3.29%.

---

## How to Run the Benchmarks

### 1. Commands to Recreate the Benchmark

1. **Install system compilers (`gfortran` and `R`):**

```bash
sudo apt-get update && sudo apt-get install -y gfortran liblapack-dev libblas-dev r-base r-base-dev
```

2. **Clone and compile the official USGS Fortran package (`peakfqr` v8.0):**

```bash
mkdir -p /tmp/usgs_src
git clone https://code.usgs.gov/water/peakfqr.git /tmp/usgs_src/peakfqr
gfortran -shared -fPIC -O2 \
    /tmp/usgs_src/peakfqr/src/emafit.f \
    /tmp/usgs_src/peakfqr/src/probfun.f \
    /tmp/usgs_src/peakfqr/src/imslfake.f \
    -llapack -lblas \
    -o /tmp/usgs_src/peakfqr/src/peakfq.so
```

3. **Clone the official USGS R package (`cran/MGBT` v1.1.6):**

```bash
git clone https://github.com/cran/MGBT.git /tmp/usgs_src/MGBT
```

4. **Run the benchmark across all 11,213 Caravan basins:** Point `benchmarks.return_periods` (or the installed `benchmark-return-periods` CLI) at your local Caravan directory and the two cloned USGS repositories (this writes `caravan_benchmark_results.csv` and the comparison figures into `./benchmark_output`):

```bash
python -m benchmarks.return_periods \
    --caravan-dir /path/to/caravan \
    --peakfqr-repo /tmp/usgs_src/peakfqr \
    --mgbt-repo /tmp/usgs_src/MGBT \
    --output-dir ./benchmark_output \
    --workers 16
```

### 2. How the Executive Summary Results Were Calculated

For each of the 11,213 Caravan river basins with at least 10 complete October–September Water Years ($\ge 330$ daily observations per year), the benchmark script extracts the annual maximum flood series in native specific discharge ($\text{mm/day}$) and converts it to volumetric discharge ($\text{ft}^3/\text{s}$) using the basin's drainage area. It then runs all three implementations on both unit systems: our Python `MultipleGrubbsBeckTester` and `GEMAFitter`, the compiled USGS Fortran `emafit()` routine (`LOthresh = 0`, `rG = 0`, `rGmse = -1e99`), and the USGS R `MGBT17c()` function. The Step 1 MGBT agreement percentages (`100.00%` between Python and USGS Fortran on the 11,171 basins where Fortran completed; `98.80%` between Python/Fortran and USGS R on positive flows) count the fraction of basins where the two implementations flag the exact same number of Potentially Influential Low Floods (PILFs).

For Step 2 (EMA LP-III flood quantiles), we compare the 100-year flood ($Q_{100}$, $\text{AEP} = 0.01$) returned by Python `GEMAFitter` and USGS Fortran `emafit()` using the relative percentage difference $|Q_{100,\text{Python}} - Q_{100,\text{Fortran}}| / Q_{100,\text{Fortran}} \times 100\%$ across all 11,171 completed basins (`0.0000000071%` median difference on the 10,875 basins unaffected by Fortran's $10^{-6}$ PILF lower bound). Finally, for the non-iterative EMA comparisons, we run Python without MGBT and without the iterative EMA loop (`SimpleLogPearson3Fitter`) and Python with Step 1 MGBT PILF filtering but 0 EMA iterations (sample moments of the MGBT-retained sample), and compute the median absolute percentage difference $|Q_{T,\text{simplified}} - Q_{T,\text{full}}| / Q_{T,\text{full}} \times 100\%$ against the full iterative Bulletin 17C Python fit on the 8,153 basins with zero PILFs (`0.00%` difference) and on the 3,060 basins with one or more PILFs (`1.51%` on $Q_{10}$ and `12.55%` on $Q_{100}$ without MGBT, dropping to `3.29%` on $Q_{100}$ when keeping Step 1 MGBT).

---

## References

* England, J.F., Jr., Cohn, T.A., Faber, B.A., Stedinger, J.R., Thomas, W.O., Jr., Veilleux, A.G., Kiang, J.E., and Mason, R.R., Jr. (2019). *Guidelines for determining flood flow frequency — Bulletin 17C*. U.S. Geological Survey Techniques and Methods, book 4, chap. B5, 148 p. https://doi.org/10.3133/tm4B5
* Cohn, T.A., England, J.F., Jr., Berenbrock, C.E., Mason, R.R., Stedinger, J.R., and Lamontagne, J.R. (2013). *A generalized Grubbs-Beck test statistic for detecting multiple potentially influential low outliers in flood series*. Water Resources Research, 49(8), 5047–5058. https://doi.org/10.1002/wrcr.20392
* Cohn, T.A., Lane, W.L., and Baier, W.G. (1997). *An algorithm for computing moments-based flood quantile estimates when historical flood information is available*. Water Resources Research, 33(9), 2089–2096.
* Hirsch, R.M., and Stedinger, J.R. (1987). *Plotting positions for historical floods and their precision*. Water Resources Research, 23(4), 715–727.
