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

from typing import Any

from maas.config import MAAS_MODEL_NAMES, TODAYS_EARTH_SOURCE
from maas.fetcher import align_daily_series, daily_series, window_peak
from maas.thresholds import (
    EXCEEDANCE_CLASSES,
    RISK_RANK,
    UNASSESSED_COLOR,
    UNASSESSED_LABEL,
    classify_exceedance,
    estimate_return_period_years,
)


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
    fh_inund: dict[str, Any] | None,
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
    te_ff = (te or {}).get('flood_forecast') or {}
    fh_inund = fh_inund or {}
    return {
        'max_inundation_depth_m': te_ff.get('max_flood_depth_m'),
        'max_flooded_fraction_pct': te_ff.get('max_flooded_fraction_pct'),
        'peak_sfcelv_m': te_ff.get('max_sfcelv_m'),
        'peak_depth_time': te_ff.get('peak_depth_time'),
        'inundation_source': (
            (
                TODAYS_EARTH_SOURCE
                + (' — emulated' if te.get('emulated') else '')
            )
            if te
            else None
        ),
        'floodhub_severity': (
            fh_inund.get('severity') if fh_fc else 'UNAVAILABLE'
        ),
        'floodhub_trend': fh_inund.get('trend') if fh_fc else None,
        'floodhub_severity_source': fh_inund.get('severity_source'),
        'floodhub_inundation_maps_available': bool(
            fh_inund.get('inundation_maps_available')
        ),
        'floodhub_inundation_map_levels': (
            fh_inund.get('inundation_map_levels') or []
        ),
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
        if (fh_fc and fh_fc.get('data') and fh_is_q)
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
    gg_daily = (
        daily_series((gg_fc or {}).get('data'), 'flow_med') if gg_fc else {}
    )
    gg_p25_d = (
        daily_series((gg_fc or {}).get('data'), 'flow_25p') if gg_fc else {}
    )
    gg_p75_d = (
        daily_series((gg_fc or {}).get('data'), 'flow_75p') if gg_fc else {}
    )
    te_daily = (
        daily_series((te_fc or {}).get('data'), 'discharge_mean') if te_fc else {}
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
        timeline_series['floodhub'] = {
            'discharge': align_daily_series(fh_daily, dates)
        }
    if 'glofas' in models and gl_daily:
        timeline_series['glofas'] = {
            'median': align_daily_series(gl_daily, dates),
            'p25': align_daily_series(gl_p25_d, dates),
            'p75': align_daily_series(gl_p75_d, dates),
        }
    if 'geoglows' in models and gg_daily:
        timeline_series['geoglows'] = {
            'median': align_daily_series(gg_daily, dates),
            'p25': align_daily_series(gg_p25_d, dates),
            'p75': align_daily_series(gg_p75_d, dates),
        }
    if 'todays_earth' in models and te_daily:
        timeline_series['todays_earth'] = {
            'mean': align_daily_series(te_daily, dates),
            'rivout': align_daily_series(te_riv_d, dates),
            'fldout': align_daily_series(te_fld_d, dates),
            'flddph_m': align_daily_series(te_dph_d, dates, 3),
            'fldfrc_pct': align_daily_series(te_frc_d, dates),
        }
    return {
        'dates': dates,
        'unit': 'm³/s',
        'series': timeline_series,
    }
