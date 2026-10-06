Return Period Calculator (USGS Bulletin 17C)
============================================

The ``return_periods`` package provides flood frequency analysis and return
period estimation following **USGS Bulletin 17C** (*Guidelines for Determining
Flood Flow Frequency — Bulletin 17C*, England et al., 2019).

Overview
--------

Estimating the relationship between flood discharge magnitude and annual
exceedance probability (or return period :math:`T = 1 / p`) is essential for
flood hazard assessment and operational warning thresholds. The
``return_periods`` package implements:

1. **Expected Moments Algorithm (EMA)**: Iterative fitting of the Log-Pearson
   Type III (LP3) distribution (:class:`~return_periods.GEMAFitter`) supporting
   systematic records, zero-flow years, interval-censored low outliers, and
   generalized regional skew weighting.
2. **Multiple Grubbs-Beck Test (MGBT)**: Full orthogonal-:math:`t` Gaussian
   quadrature low-outlier screening
   (:class:`~return_periods.MultipleGrubbsBeckTester`, Cohn et al., 2013) with
   the official Bulletin 17C two-stage outward (:math:`\alpha_{\text{out}} =
   0.005`) and inward (:math:`\alpha_{\text{in}} = 0.10`) significance sweeps
   to identify Potentially Influential Low Floods (PILFs).
3. **Multiple Distribution Fitters**: :class:`~return_periods.ReturnPeriodCalculator`
   supports explicit selection of ``GEMAFitter`` (``fitter='gema'``),
   :class:`~return_periods.SimpleLogPearson3Fitter` (``fitter='simple_lp3'``),
   :class:`~return_periods.LogLogTrendFitter` (``fitter='log_linear'``), or a
   custom :class:`~return_periods.BaseFitter` subclass.
4. **Hirsch-Stedinger Threshold Plotting Positions**: Empirical plotting
   positions (:func:`~return_periods.simple_empirical_plotting_position` and
   :func:`~return_periods.threshold_exceedance_empirical_plotting_position`)
   from Bulletin 17C Appendix 5.

Quickstart Examples
-------------------

Calculating Return Periods from Annual Peak Flows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from return_periods import ReturnPeriodCalculator

   peaks = pd.Series(
       [2080, 1670, 1480, 2940, 1560, 2380, 2720, 2860, 2620, 1710],
       index=range(1947, 1957),
   )

   rpc = ReturnPeriodCalculator(peaks_series=peaks, fitter='gema')

   # Calculate discharge quantiles for standard return periods (in years)
   flows = rpc.flow_values_from_return_periods([2, 5, 10, 25, 50, 100])

   # Convert arbitrary discharge values into return periods (in years)
   return_periods = rpc.return_periods_from_flow_values(flows)

Extracting Peaks and Return Periods from a Daily Hydrograph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from return_periods import ReturnPeriodCalculator

   rpc = ReturnPeriodCalculator(
       hydrograph_series=daily_flow_series,
       fitter='gema',
       max_missing_days_in_year=30,
       water_year_start_month=10,  # October 1 - September 30 water year
   )

   fig, ax = rpc.plot_hydrograph(plot_return_periods=[2, 5, 10, 25, 50, 100])

Using Regional Skew Weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per USGS Bulletin 17C Equation 7-10, at-site station skew can be weighted with
a generalized regional skew and its mean square error:

.. code-block:: python

   rpc = ReturnPeriodCalculator(
       peaks_series=peaks,
       fitter='gema',
       regional_skew=0.44,
       regional_skew_mse=0.078,
   )

Direct Low-Outlier Screening with MGBT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from return_periods import MultipleGrubbsBeckTester

   tester = MultipleGrubbsBeckTester(
       data=annual_peaks,
       is_log_transformed=False,
   )
   print("PILFs identified:", tester.klow)
   print("PILF threshold:", 10.0 ** tester.threshold)
