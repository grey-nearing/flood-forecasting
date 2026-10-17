============================================
Return Period Calculator (USGS Bulletin 17C)
============================================

The ``return_periods`` package calculates flood return periods and flood frequency curves from historical river discharge records following **USGS Bulletin 17C** (*Guidelines for Determining Flood Flow Frequency — Bulletin 17C*, England et al., 2019).

.. contents:: On this page
   :local:
   :depth: 2

What Is a Flood Return Period?
------------------------------

A **return period** :math:`T` (measured in years) describes how rare a flood of a given size is:

* A **2-year flood** (:math:`T = 2`) has a :math:`1/2 = 50\%` chance of being equaled or exceeded in any single year. This represents a typical annual high-water flow.
* A **10-year flood** (:math:`T = 10`) has a :math:`1/10 = 10\%` chance of happening in any given year.
* A **100-year flood** (:math:`T = 100`) has a :math:`1/100 = 1\%` chance of happening in any given year.

In general, a flood with return period :math:`T` years has an **annual exceedance probability** of :math:`p = 1 / T`. Flood forecasting systems use return period thresholds (such as the 2-, 5-, 20-, and 100-year flows in ``m³/s``) to translate predicted river discharge into clear warning levels.

How the USGS Bulletin 17C Algorithm Works
-----------------------------------------

To estimate flood return periods at a river gauge, ``return_periods`` performs three steps:

1. **Extracts Annual Peak Flows:** From a daily river discharge time series (in ``m³/s`` or ``ft³/s``), the calculator groups the record into **water years** (for example, October 1 through September 30) and finds the highest single-day flow in each year. Years with too many missing days (controlled by ``max_missing_days_in_year``, default ``30``) are skipped so incomplete years are not mistaken for dry years.
2. **Screens Out Dry-Year Low Outliers (`MGBT`):** In arid or seasonal rivers, some dry years have zero flow or a very small peak flow. These **Potentially Influential Low Floods (PILFs)** do not follow the same physical process as large floods and can bend the fitted curve, causing inaccurate 50-year or 100-year flood estimates. The **Multiple Grubbs-Beck Test (`MGBT`)** (:py:class:`~return_periods.MultipleGrubbsBeckTester`, Cohn et al., 2013) automatically detects unusually low annual peaks using the official Bulletin 17C two-stage outward (:math:`\alpha_{\text{out}} = 0.005`) and inward (:math:`\alpha_{\text{in}} = 0.10`) significance sweeps.
3. **Fits the Log-Pearson Type III Distribution (`EMA`):** The **Expected Moments Algorithm (`EMA`)** (:py:class:`~return_periods.GEMAFitter`) fits a Log-Pearson Type III (`LP-III`) probability distribution to the base-10 logarithms (:math:`\log_{10} Q`) of the annual peak flows. Any dry-year low outliers flagged by `MGBT` (as well as zero-flow years) are treated as **interval-censored observations** (known to be smaller than the `MGBT` threshold) rather than discarded, preserving the true length of the historical record.

Available Distribution Fitters
------------------------------

:py:class:`~return_periods.ReturnPeriodCalculator` supports three built-in fitting methods (or any custom :py:class:`~return_periods.BaseFitter` subclass):

* ``fitter='gema'`` (recommended default): Full USGS Bulletin 17C Expected Moments Algorithm (:py:class:`~return_periods.GEMAFitter`) with `MGBT` low-outlier censoring and optional regional skew weighting.
* ``fitter='simple_lp3'``: Standard Method-of-Moments Log-Pearson Type III fit (:py:class:`~return_periods.SimpleLogPearson3Fitter`) without `EMA` interval censoring.
* ``fitter='log_linear'``: Log-linear regression on empirical plotting positions (:py:class:`~return_periods.LogLogTrendFitter`).

It also provides the empirical plotting-position functions from Bulletin 17C Appendix 5 (:py:func:`~return_periods.simple_empirical_plotting_position` and :py:func:`~return_periods.threshold_exceedance_empirical_plotting_position`).

Quick Start Examples
--------------------

1. Calculate Return Period Flows from Annual Peaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have a ``pandas.Series`` of annual maximum flows (in ``m³/s``), pass ``peaks_series`` to :py:class:`~return_periods.ReturnPeriodCalculator`:

.. code-block:: python

   import pandas as pd
   from return_periods import ReturnPeriodCalculator

   # Annual peak discharge (m³/s) indexed by water year
   peaks = pd.Series(
       [2080, 1670, 1480, 2940, 1560, 2380, 2720, 2860, 2620, 1710],
       index=range(1947, 1957),
   )

   rpc = ReturnPeriodCalculator(peaks_series=peaks, fitter='gema')

   # 1. Compute discharge thresholds (m³/s) for standard return periods (years)
   flows = rpc.flow_values_from_return_periods([2, 5, 10, 25, 50, 100])

   # 2. Convert any observed or forecast flow (m³/s) into its return period (years)
   return_periods = rpc.return_periods_from_flow_values(flows)

2. Extract Annual Peaks from a Daily Streamflow Series
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have a daily streamflow time series (indexed by ``pd.DatetimeIndex``), pass ``hydrograph_series``:

.. code-block:: python

   from return_periods import ReturnPeriodCalculator

   rpc = ReturnPeriodCalculator(
       hydrograph_series=daily_flow_series,
       fitter='gema',
       max_missing_days_in_year=30,
       water_year_start_month=10,  # Water year: October 1 to September 30
   )

   # Plot the daily hydrograph with horizontal return-period threshold lines
   fig, ax = rpc.plot_hydrograph(plot_return_periods=[2, 5, 10, 25, 50, 100])

3. Combine Station Skew with a Regional Skew Prior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Short gauge records (for example, 10 to 30 years) can have noisy sample skewness. Following USGS Bulletin 17C Equation 7-10, you can weight the station's sample skew with a generalized regional skew ``regional_skew`` and its mean square error ``regional_skew_mse``:

.. code-block:: python

   rpc = ReturnPeriodCalculator(
       peaks_series=peaks,
       fitter='gema',
       regional_skew=0.44,
       regional_skew_mse=0.078,
   )

4. Run MGBT Low-Outlier Screening Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To inspect how many dry-year low outliers (`PILFs`) are flagged in an annual peak series and view the outlier threshold (in flow units):

.. code-block:: python

   from return_periods import MultipleGrubbsBeckTester

   tester = MultipleGrubbsBeckTester(
       data=annual_peaks,
       is_log_transformed=False,
   )
   print("Number of low outliers (PILFs) flagged:", tester.klow)
   print("Low-outlier flow threshold (m³/s):", 10.0 ** tester.threshold)

API reference: :doc:`../api/return_periods`.
