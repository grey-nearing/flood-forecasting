# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frontend multi-model consensus badges, flood summary cards, and chart timeline formatting."""

import math
from typing import Any

import numpy as np

from maas.config import MAAS_MODEL_NAMES, parse_finite_float
from maas.fetcher import align_daily_series, daily_series, window_peak
from maas.thresholds import (
    EXCEEDANCE_CLASSES,
    RISK_RANK,
    UNASSESSED_COLOR,
    UNASSESSED_LABEL,
    classify_exceedance,
    estimate_return_period_years,
    gumbel_quantile_from_return_periods,
)
from maas.todays_earth import CAMA_FLDOUT_SHARE, CAMA_FLOODPLAIN_K


def route_floodplain_excess(
    series: list[float],
    q_bankfull: float,
    k: float = CAMA_FLOODPLAIN_K,
) -> list[float]:
    """Linear-reservoir routing of above-bankfull flow (daily explicit scheme)."""
    routed: list[float] = []
    state: float | None = None
    for q in series:
        excess = max(q - q_bankfull, 0.0)
        state = excess if state is None else state + k * (excess - state)
        routed.append(state)
    return routed


def emulate_camaflood_physics(
    glofas_records: list[dict[str, Any]],
    rps: dict[str, Any],
    elev: float = 80.0,
    elev_source: str = 'Open-Meteo DEM',
) -> dict[str, Any]:
    """Deterministic CaMa-Flood-style streamflow routing emulation from GloFAS v4."""
    records = (glofas_records or [])[:6]

    def _col(name: str, fallback: str = 'discharge_mean') -> list[float]:
        out: list[float] = []
        for r in records:
            v = parse_finite_float(r.get(name))
            if v is None:
                v = parse_finite_float(r.get(fallback))
            out.append(max(v or 0.0, 0.0))
        return out

    central = _col('discharge_median')
    med_val = float(np.median(central)) if central else 1.0
    q_clim = parse_finite_float((rps or {}).get('mean_flow')) or med_val or 1.0
    q_clim = max(q_clim, 0.05)
    width = max(0.40 * q_clim**0.75, 10.0)
    depth = max(0.10 * q_clim**0.5, 1.0)
    q_bf = max(
        gumbel_quantile_from_return_periods(rps or {}, 1.5) or 0.0,
        1.2 * q_clim,
        0.5,
    )
    elev_c = min(max(float(elev), 0.0), 1500.0)
    depth_scale = 1.0 + elev_c / 150.0
    f_max = (0.02 + 0.08 * math.log10(1.0 + q_clim / 10.0)) * (
        1.0 + 1.5 * math.exp(-elev_c / 30.0)
    )
    f_max = min(max(f_max, 0.02), 0.6)

    def _cama(
        series: list[float],
    ) -> tuple[list[float], list[float], list[float]]:
        routed = route_floodplain_excess(series, q_bf)
        total = [min(q, q_bf) + r for q, r in zip(series, routed)]
        fld = [CAMA_FLDOUT_SHARE * r for r in routed]
        return total, [t - f for t, f in zip(total, fld)], fld

    total, rivout, fldout = _cama(central)
    stage = [depth * (max(r, 0.0) / q_bf) ** 0.6 for r in rivout]
    flddph = [max(h - depth, 0.0) for h in stage]
    fldfrc = [100.0 * f_max * (1.0 - math.exp(-d / depth_scale)) for d in flddph]
    sfcelv = [max(max(float(elev), 0.0) - depth + h, 0.0) for h in stage]
    r2 = lambda xs: [round(x, 2) for x in xs]
    return {
        'series': {
            'timestamps': [
                f"{str(r.get('time'))[:10]}T00:00:00Z" for r in records
            ],
            'mean': r2(total),
            'rivout': r2(rivout),
            'fldout': r2(fldout),
            'p25': r2(_cama(_col('discharge_p25'))[0]),
            'p75': r2(_cama(_col('discharge_p75'))[0]),
            'max': r2(_cama(_col('discharge_max'))[0]),
            'min': r2(_cama(_col('discharge_min'))[0]),
            'flddph_m': [round(d, 3) for d in flddph],
            'fldfrc_pct': r2(fldfrc),
            'sfcelv_m': r2(sfcelv),
        },
        'channel_params': {
            'mean_flow_m3s': round(q_clim, 3),
            'bankfull_discharge_m3s': round(q_bf, 2),
            'channel_width_m': round(width, 1),
            'channel_depth_m': round(depth, 2),
            'ground_elevation_m': float(elev),
            'elevation_source': elev_source,
            'max_flooded_fraction_ceiling_pct': round(100.0 * f_max, 1),
        },
        'forcing_status': 'live',
        'return_period_status': (rps or {}).get('status'),
    }


def spread_confidence(
    central: float | None,
    p25: float | None,
    p75: float | None,
    live: bool = True,
    fallback_thresholds: bool = False,
) -> str:
    """High / Medium / Low from the relative inter-quartile spread at the peak."""
    if not live:
        return 'Low'
    level = 1
    if central is not None and p25 is not None and p75 is not None and central > 0:
        rel = max(p75 - p25, 0.0) / central
        level = 2 if rel < 0.25 else (1 if rel < 0.6 else 0)
    if fallback_thresholds:
        level = max(level - 1, 0)
    return ('Low', 'Medium', 'High')[level]


def build_consensus_row(
    model: str,
    available: bool,
    status: str | None,
    peak: float | None,
    peak_time: str | None,
    rps: dict[str, Any] | None,
    thresholds_source: str | None,
    confidence: str,
    unit: str = 'm³/s',
    independent: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """Build a single provider row for the Multi-Model Consensus Matrix."""
    synthetic = available and status == 'fallback'
    cls = classify_exceedance(peak, rps) if (available and not synthetic) else None
    row: dict[str, Any] = {
        'model': model,
        'name': MAAS_MODEL_NAMES.get(model, model),
        'available': available,
        'status': status,
        'unit': unit,
        'peak_flow': round(peak, 2) if peak is not None else None,
        'peak_time': peak_time,
        'return_period': cls['return_period'] if cls else None,
        'return_period_yrs': (
            estimate_return_period_years(peak, rps or {})
            if (cls and unit == 'm³/s')
            else None
        ),
        'exceedance_rank': cls['rank'] if cls else None,
        'exceedance_label': (
            cls['label'] if cls else (UNASSESSED_LABEL if synthetic else None)
        ),
        'risk_level': cls['risk_level'] if cls else 'UNKNOWN',
        'confidence': confidence if available else 'N/A',
        'thresholds_source': thresholds_source,
        'independent': independent,
    }
    row.update(extra)
    return row


def reach_exceedance_summary(
    glofas_fc: dict[str, Any] | None,
    glofas_rp: dict[str, Any] | None,
    geoglows_fc: dict[str, Any] | None,
    geoglows_rp: dict[str, Any] | None,
) -> dict[str, Any]:
    """Forecast-peak return-period exceedance of the reach (worst of live GloFAS / GEOGLOWS)."""
    per_model: dict[str, Any] = {}
    excluded: list[str] = []
    for key, fc, rps, value_keys in (
        ('glofas', glofas_fc, glofas_rp, ('discharge_median', 'discharge_mean')),
        ('geoglows', geoglows_fc, geoglows_rp, ('flow_med',)),
    ):
        if not fc:
            continue
        if fc.get('status') == 'fallback':
            excluded.append(key)
            continue
        peak, when = window_peak(daily_series(fc.get('data'), *value_keys))
        cls = classify_exceedance(peak, rps)
        per_model[key] = {
            'peak_flow': peak,
            'peak_time': when,
            'rank': cls['rank'],
            'label': cls['label'],
            'return_period_yrs': estimate_return_period_years(peak, rps or {}),
            'thresholds_status': (rps or {}).get('status'),
        }
    if not per_model:
        out = dict(EXCEEDANCE_CLASSES[0])
        out.update(
            {
                'label': (
                    UNASSESSED_LABEL if excluded else 'Not assessed (no forecast)'
                ),
                'risk_level': 'UNKNOWN',
                'return_period': None,
                'color': UNASSESSED_COLOR,
                'governing_model': None,
                'per_model': {},
                'excluded_models': excluded,
                'unit': 'm³/s',
            }
        )
        return out
    gov = max(per_model, key=lambda k: per_model[k]['rank'])
    out = dict(EXCEEDANCE_CLASSES[per_model[gov]['rank']])
    out.update(
        {
            'governing_model': gov,
            'per_model': per_model,
            'excluded_models': excluded,
            'unit': 'm³/s',
        }
    )
    return out


def build_flood_summary(
    consensus: list[dict[str, Any]],
    te: dict[str, Any] | None,
    fh_fc: dict[str, Any] | None,
    fh_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synthesize overall multi-model flood warning status across all independent models."""
    independent = [
        r
        for r in consensus
        if r.get('available')
        and r.get('independent')
        and r.get('status') in ('live', 'severity_only')
    ]
    excluded = [
        r['name']
        for r in consensus
        if r.get('available')
        and r.get('independent')
        and r.get('status') not in ('live', 'severity_only')
    ]
    n_exceed = sum(1 for r in independent if (r.get('exceedance_rank') or 0) >= 1)
    worst = (
        max(independent, key=lambda r: r.get('exceedance_rank') or 0)
        if independent
        else None
    )
    risks = [
        r['risk_level'] for r in independent if r.get('risk_level') in RISK_RANK
    ]
    overall = max(risks, key=lambda k: RISK_RANK[k]) if risks else 'UNKNOWN'
    agreement = (
        f'{n_exceed} of {len(independent)} independent models forecast ≥ 2-yr exceedance'
    )
    if te and te.get('emulated'):
        agreement += " (Today's Earth is emulated from GloFAS and excluded)"
    if excluded:
        agreement += f"; offline fallback excluded: {', '.join(excluded)}"
    fh_meta = fh_status or fh_fc or {}
    return {
        'floodhub_severity': (
            fh_meta.get('severity') if fh_fc else 'UNAVAILABLE'
        ),
        'floodhub_trend': fh_meta.get('trend') if fh_fc else None,
        'floodhub_severity_source': fh_meta.get('severity_source'),
        'return_period_exceedance': (
            worst['exceedance_label'] if worst else 'Unknown'
        ),
        'return_period_exceedance_model': worst['name'] if worst else None,
        'max_exceedance_rank': (
            (worst.get('exceedance_rank') or 0) if worst else None
        ),
        'models_exceeding_2yr': n_exceed,
        'independent_models_evaluated': len(independent),
        'overall_risk_level': overall,
        'agreement': agreement,
    }


def build_aligned_timeline(
    models: list[str],
    fh_fc: dict[str, Any] | None,
    fh_is_q: bool,
    gl_fc: dict[str, Any] | None,
    gg_fc: dict[str, Any] | None,
    te_fc: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a date-aligned daily timeline for multi-model hydrograph charting."""
    fh_daily = (
        daily_series((fh_fc or {}).get('data'), 'discharge')
        if (fh_fc and fh_fc.get('data'))
        else {}
    )
    gl_daily = (
        daily_series((gl_fc or {}).get('data'), 'discharge_median', 'discharge_mean')
        if gl_fc
        else {}
    )
    gl_p25_d = (
        daily_series((gl_fc or {}).get('data'), 'discharge_p25') if gl_fc else {}
    )
    gl_p75_d = (
        daily_series((gl_fc or {}).get('data'), 'discharge_p75') if gl_fc else {}
    )
    gl_max_d = (
        daily_series((gl_fc or {}).get('data'), 'discharge_max') if gl_fc else {}
    )
    gl_min_d = (
        daily_series((gl_fc or {}).get('data'), 'discharge_min') if gl_fc else {}
    )
    gg_daily = (
        daily_series((gg_fc or {}).get('data'), 'flow_med', 'flow_avg')
        if gg_fc
        else {}
    )
    gg_p25_d = (
        daily_series((gg_fc or {}).get('data'), 'flow_25p') if gg_fc else {}
    )
    gg_p75_d = (
        daily_series((gg_fc or {}).get('data'), 'flow_75p') if gg_fc else {}
    )
    gg_max_d = (
        daily_series((gg_fc or {}).get('data'), 'flow_max') if gg_fc else {}
    )
    gg_min_d = (
        daily_series((gg_fc or {}).get('data'), 'flow_min') if gg_fc else {}
    )
    te_daily = (
        daily_series((te_fc or {}).get('data'), 'discharge_mean') if te_fc else {}
    )
    te_p25_d = (
        daily_series((te_fc or {}).get('data'), 'discharge_p25') if te_fc else {}
    )
    te_p75_d = (
        daily_series((te_fc or {}).get('data'), 'discharge_p75') if te_fc else {}
    )
    te_riv_d = daily_series((te_fc or {}).get('data'), 'rivout') if te_fc else {}
    te_fld_d = daily_series((te_fc or {}).get('data'), 'fldout') if te_fc else {}
    te_dph_d = (
        daily_series((te_fc or {}).get('data'), 'flddph_m') if te_fc else {}
    )
    te_frc_d = (
        daily_series((te_fc or {}).get('data'), 'fldfrc_pct') if te_fc else {}
    )
    dates = sorted(
        set(fh_daily) | set(gl_daily) | set(gg_daily) | set(te_daily)
    )
    timeline_series: dict[str, Any] = {}
    if 'floodhub' in models and fh_daily:
        fh_aligned = align_daily_series(fh_daily, dates)
        timeline_series['floodhub'] = {
            'central': fh_aligned,
            'discharge': fh_aligned,
            'unit': 'm³/s' if fh_is_q else 'm',
            'axis': 'discharge' if fh_is_q else 'stage',
        }
    if 'glofas' in models and gl_daily:
        gl_aligned = align_daily_series(gl_daily, dates)
        timeline_series['glofas'] = {
            'central': gl_aligned,
            'median': gl_aligned,
            'p25': align_daily_series(gl_p25_d, dates),
            'p75': align_daily_series(gl_p75_d, dates),
            'max': align_daily_series(gl_max_d, dates),
            'min': align_daily_series(gl_min_d, dates),
        }
    if 'geoglows' in models and gg_daily:
        gg_aligned = align_daily_series(gg_daily, dates)
        timeline_series['geoglows'] = {
            'central': gg_aligned,
            'median': gg_aligned,
            'p25': align_daily_series(gg_p25_d, dates),
            'p75': align_daily_series(gg_p75_d, dates),
            'max': align_daily_series(gg_max_d, dates),
            'min': align_daily_series(gg_min_d, dates),
        }
    if 'todays_earth' in models and te_daily:
        te_aligned = align_daily_series(te_daily, dates)
        timeline_series['todays_earth'] = {
            'central': te_aligned,
            'mean': te_aligned,
            'p25': align_daily_series(te_p25_d, dates),
            'p75': align_daily_series(te_p75_d, dates),
            'rivout': align_daily_series(te_riv_d, dates),
            'fldout': align_daily_series(te_fld_d, dates),
            'flddph_m': align_daily_series(te_dph_d, dates, 3),
            'fldfrc_pct': align_daily_series(te_frc_d, dates),
        }
    return {
        'dates': dates,
        'unit': 'm³/s',
        'series': timeline_series,
        'status': {
            'floodhub': (fh_fc or {}).get('status') if fh_fc else None,
            'glofas': (gl_fc or {}).get('status') if gl_fc else None,
            'geoglows': (gg_fc or {}).get('status') if gg_fc else None,
            'todays_earth': (
                ('emulated' if te_fc.get('emulated') else te_fc.get('status'))
                if te_fc
                else None
            ),
        },
    }
