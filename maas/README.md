# `maas` — Models-as-a-Service Multi-Model Operational Flood Forecasting

`maas` is a standalone Python package for querying, harmonizing, and evaluating operational flood forecasts across four global and continental flood forecasting systems:

1. **Google FloodHub** (`maas.floodhub.FloodHubClient`) — Gauge- and site-level AI streamflow forecasts, warning severity levels (`NORMAL`, `WARNING`, `DANGER`, `EXTREME`), and KML inundation polygons via `https://apis.google.com/floodforecasting/v1`.
2. **Copernicus GloFAS v4** (`maas.glofas.GloFASClient`) — 51-member ECMWF-forced LISFLOOD global $0.05^\circ$ river discharge ensembles (`p10`, `p25`, `median`, `p75`, `p90`) via the Open-Meteo Flood API (`https://flood-api.open-meteo.com/v1/flood`) and ECMWF CEMS Zarr stores (`s3://geoglows-v2/glofas-v4/`).
3. **GEOGLOWS ECMWF v2** (`maas.geoglows.GeoGLOWSClient`) — Reach-level vector streamflow forecasts on ~6.8M global TDX-Hydro / HydroRIVERS river reaches (`LINKNO`) via the GEOGLOWS REST API (`https://geoglows.ecmwf.int/api/v2`) and AWS S3 retrospective Zarr archives (`s3://geoglows-v2/retrospective.zarr`).
4. **JAXA Today's Earth (`CaMa-Flood`)** (`maas.todays_earth.TodaysEarthClient`) — Global/regional hydrodynamic floodplain routing on $0.25^\circ$ (global) and $0.05^\circ$ (Japan) grids (`rivout` river discharge, `flddph` floodplain water depth, `fldfrc` flooded area fraction, and `fldare` inundated area) via JAXA Earth API STAC (`https://data.earth.jaxa.jp/stac/cog/v1/collections`), with physics-based floodplain routing emulation when live STAC tiles are unreachable.

---

## Architecture & Module Layout

```
maas/
├── __init__.py              # Public API exports (MaaSEngine, MaaSConfig, clients, network & threshold utilities)
├── config.py                # MaaSConfig dataclass, provider schemas, unit conversions, finite float parsers
├── thresholds.py            # Return-period threshold calculators (delegating to return_periods) & risk classification
├── networks.py              # Spatial reach snapping, upstream-area cross-network matching, LOD binary pyramids
├── floodhub.py              # Google FloodHub REST v1 client + KML inundation polygon parser
├── glofas.py                # Copernicus GloFAS v4 Open-Meteo REST + S3 Zarr client
├── geoglows.py              # GEOGLOWS ECMWF v2 REST + S3 retrospective Zarr client
├── todays_earth.py          # JAXA Today's Earth STAC client + CaMa-Flood 1D diffusive-wave floodplain router
├── engine.py                # MaaSEngine orchestrator, SQLiteCache, consensus & aligned timeline builders
├── cli.py                   # CLI entry point (fetch-maas-forecast / python -m maas.cli)
├── tools/
│   └── build_network_pyramids.py  # Offline spatial index builder for LOD binary pyramids (.npz)
└── tests/
    ├── test_networks.py     # Unit tests for spatial grid math, reach snapping, and cross-network matching
    ├── test_providers.py    # Offline unit tests for all 4 provider parsers, clients, and MaaSEngine aggregation
    ├── test_thresholds.py   # Unit tests for return-period fitting (Bulletin 17C EMA, Weibull, Gumbel) & exceedance
    └── test_canary_live.py  # Opt-in live network canary tests (MAAS_LIVE_CANARY=1)
```

---

## Core Mathematical & Hydrological Formulations

### 1. Return Period Thresholds (`maas/thresholds.py`)
All return-period exceedance calculations map discharge $Q$ ($\text{m}^3/\text{s}$) against canonical recurrence intervals $T \in \{2, 5, 10, 20, 50, 100\}$ years (`q_2yr`, `q_5yr`, `q_10yr`, `q_20yr`, `q_50yr`, `q_100yr`):

- **Parametric USGS Bulletin 17C EMA + MGBT (`compute_return_periods`)**: Delegates directly to `return_periods.ReturnPeriodCalculator(return_periods=[2, 5, 10, 20, 50, 100], fitter='gema')` to fit a Log-Pearson Type III distribution with Expected Moments Algorithm and Multiple Grubbs-Beck low-outlier screening on annual maximum series (requiring $\ge 10$ valid water years).
- **Empirical Weibull Log-Linear (`compute_empirical_weibull_return_periods`)**: Computes empirical plotting positions $P_i = \frac{\text{rank}_i}{N + 1}$ via `return_periods.plotting_positions.simple_empirical_plotting_position(..., plotting_position_type='weibull')` and interpolates quantiles via `ReturnPeriodCalculator(fitter='log_linear')`.
- **Method-of-Moments Gumbel EV1 (`compute_gumbel_return_periods`)**: Uses the theoretical Euler-Mascheroni reduced variate constants ($\gamma = 0.5772156649$, $\sigma_y = \pi / \sqrt{6} \approx 1.28255$):
  $$K_T = -\frac{\sqrt{6}}{\pi}\left(\gamma + \ln\left(-\ln\left(1 - \frac{1}{T}\right)\right)\right), \qquad Q_T = \mu_{\text{AMS}} + K_T \, \sigma_{\text{AMS}}$$

### 2. Cross-Network Upstream-Area Reach Snapping (`maas/networks.py`)
Because vector reaches (`GEOGLOWS LINKNO`, `HydroRIVERS HYRIV_ID`) and Eulerian raster cells (`GloFAS` $0.05^\circ$, `CaMa-Flood` $0.25^\circ$) frequently diverge by $1\text{--}5\text{ km}$ near river confluences, nearest-distance snapping alone can jump from a main stem ($A \sim 10^5\text{ km}^2$) onto a minor tributary ($A \sim 10^1\text{ km}^2$). `resolve_cross_network_click` minimizes a joint distance-and-scale objective:
$$J(i) = \frac{d_i}{R_{\max}} + \lambda \left|\log_{10}\!\left(\frac{\max(A_i, 1)}{\max(A_{\text{ref}}, 1)}\right)\right|$$
so that all four providers lock onto the same hydrological river order across networks.

---

## Programmatic Usage

```python
from pathlib import Path
from maas import MaaSConfig, MaaSEngine

config = MaaSConfig(
    cache_dir=Path('/tmp/maas_cache'),
    river_networks_dir=Path('data/river_networks'),
    floodhub_api_key=None,  # Optional: pass Google FloodHub API key
)
engine = MaaSEngine(config)

# Fetch unified multi-model forecast bundle (timeline, consensus, return periods, inundation)
bundle = engine.fetch_multi_model_forecast(
    lat=38.6270,
    lon=-90.1994,
    horizon_days=10,
    requested_models=['floodhub', 'glofas', 'geoglows', 'todays_earth'],
)
print(bundle['consensus']['overall']['severity'])
```

---

## CLI Usage

```bash
# Query all 4 flood models at a coordinate and emit formatted JSON
fetch-maas-forecast --lat 38.6270 --lon -90.1994 --horizon-days 10 --pretty

# Query specific providers with explicit cache and river network paths
python -m maas.cli \
  --lat 38.6270 \
  --lon -90.1994 \
  --models glofas,geoglows,todays_earth \
  --cache-dir ~/.cache/flood_forecasting/maas \
  --river-networks-dir data/river_networks \
  --output forecast_bundle.json
```

---

## Testing

```bash
# Run offline unit tests (100% hermetic, < 30s)
pytest maas/tests -v

# Opt-in live network canary tests against external provider APIs
MAAS_LIVE_CANARY=1 pytest maas/tests/test_canary_live.py -v
```
