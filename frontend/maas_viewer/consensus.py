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

"""Frontend multi-model consensus rows, flood summaries, and chart timelines."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from maas.config import MAAS_MODEL_NAMES, parse_finite_float
from maas.fetcher import align_daily_series, daily_series, window_peak
from maas.thresholds import (
    EXCEEDANCE_CLASSES,
    RISK_RANK,
    UNASSESSED_COLOR,
    classify_exceedance,
    estimate_return_period_years,
)

_HIGH_CONFIDENCE_REL_SPREAD: float = 0.25
_MEDIUM_CONFIDENCE_REL_SPREAD: float = 0.60


def spread_confidence(
    central: float | None,
    p25: float | None,
    p75: float | None,
    *,
    live: bool = True,
) -> str:
    """Compute High/Medium/Low confidence from peak interquartile spread."""
    if not live:
        return 'Low'
    level = 1
    if (
        central is not None
        and p25 is not None
        and p75 is not None
        and central > 0
    ):
        rel = max(p75 - p25, 0.0) / central
        if rel < _HIGH_CONFIDENCE_REL_SPREAD:
            level = 2
        elif rel < _MEDIUM_CONFIDENCE_REL_SPREAD:
            level = 1
        else:
            level = 0
    return ('Low', 'Medium', 'High')[level]


def build_consensus_row(  # noqa: PLR0913, PLR0917
    model: str,
    status: str | None,
    peak: float | None,
    peak_time: str | None,
    rps: dict[str, Any] | None,
    thresholds_source: str | None,
    confidence: str,
    unit: str = 'm³/s',
    *,
    available: bool,
    independent: bool = True,
    **extra: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Build a single provider row for the Multi-Model Consensus Matrix."""
    cls = classify_exceedance(peak, rps) if available else None
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
        'exceedance_label': cls['label'] if cls else None,
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
    """Compute forecast-peak return-period exceedance of the reach."""
    per_model: dict[str, Any] = {}
    for key, fc, rps, value_keys in (
        (
            'glofas',
            glofas_fc,
            glofas_rp,
            ('discharge_median', 'discharge_mean'),
        ),
        ('geoglows', geoglows_fc, geoglows_rp, ('flow_med',)),
    ):
        if not fc or not fc.get('data'):
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
                'label': 'Not assessed (no forecast)',
                'risk_level': 'UNKNOWN',
                'return_period': None,
                'color': UNASSESSED_COLOR,
                'governing_model': None,
                'per_model': {},
                'excluded_models': [],
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
            'excluded_models': [],
            'unit': 'm³/s',
        }
    )
    return out


def build_flood_summary(
    consensus: list[dict[str, Any]],
    fh_fc: dict[str, Any] | None,
    fh_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synthesize overall multi-model flood warning status across models."""
    independent = [
        r
        for r in consensus
        if r.get('available')
        and r.get('independent')
        and r.get('status') in ('live', 'severity_only')
    ]
    n_exceed = sum(
        1 for r in independent if (r.get('exceedance_rank') or 0) >= 1
    )
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
        f'{n_exceed} of {len(independent)} independent models forecast '
        '≥ 2-yr exceedance'
    )
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


_ISO_DATE_LEN: int = 10
_ISO_MINUTE_LEN: int = 16


def _normalize_timestamp(raw_time: object) -> str:
    """Validate and normalize a UTC timestamp to `'YYYY-MM-DDTHH:MM:SSZ'`."""
    if not isinstance(raw_time, str):
        raise TypeError(f'Expected ISO-8601 timestamp string, got {raw_time!r}')
    cleaned = raw_time.strip()
    if (
        len(cleaned) < _ISO_DATE_LEN
        or cleaned[4] != '-'
        or cleaned[7] != '-'
        or (len(cleaned) != _ISO_DATE_LEN and len(cleaned) < _ISO_MINUTE_LEN)
        or (len(cleaned) > _ISO_DATE_LEN and ':' not in cleaned[_ISO_DATE_LEN:])
    ):
        raise ValueError(f'Invalid ISO-8601 timestamp {raw_time!r}')
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is not None and parsed.utcoffset() != timedelta(0):
        raise ValueError(
            f'Non-UTC timestamp offset is not supported: {raw_time!r}'
        )
    return parsed.strftime('%Y-%m-%dT%H:%M:%SZ')


def _infer_cadence_label(timestamps: Sequence[str]) -> str:
    """Infer a cadence label (`'daily'`, `'3-hourly'`, etc.) from timestamps."""
    sorted_ts = sorted(set(timestamps))
    if len(sorted_ts) >= 2:  # noqa: PLR2004
        epochs = [
            datetime.fromisoformat(ts).replace(tzinfo=UTC).timestamp()
            for ts in sorted_ts
        ]
        diffs_hours = [
            (epochs[i + 1] - epochs[i]) / 3600.0
            for i in range(len(epochs) - 1)
            if epochs[i + 1] > epochs[i]
        ]
        if diffs_hours:
            median_h = statistics.median(diffs_hours)
            if abs(median_h - 24.0) < 0.5:  # noqa: PLR2004
                return 'daily'
            if abs(median_h - 3.0) < 0.25:  # noqa: PLR2004
                return '3-hourly'
            if abs(median_h - 1.0) < 0.1:  # noqa: PLR2004
                return 'hourly'
            return f'{median_h:g}-hourly' if median_h < 24.0 else 'daily'  # noqa: PLR2004
    if any(not ts.endswith('T00:00:00Z') for ts in sorted_ts):
        return '3-hourly'
    return 'daily'


def _timestamp_series(
    records: Sequence[Mapping[str, Any]] | None,
    key: str,
) -> dict[str, float]:
    """Extract `{YYYY-MM-DDTHH:MM:SSZ: value}` at native model cadence."""
    out: dict[str, float] = {}
    for record in records or ():
        if not isinstance(record, Mapping):
            raise TypeError(
                f'Expected mapping record in forecast data, got {record!r}'
            )
        timestamp = _normalize_timestamp(record.get('time'))
        val = parse_finite_float(record.get(key))
        if val is not None:
            out[timestamp] = round(val, 2)
    return out


def _align_glofas_members(
    gl_fc: Mapping[str, Any] | None,
    dates: Sequence[str],
) -> dict[str, list[float | None]]:
    """Align GloFAS ensemble members (`member01`..`member50`) onto `dates`."""
    if not isinstance(gl_fc, Mapping):
        return {}
    raw_members = gl_fc.get('ensemble_members')
    if not isinstance(raw_members, Mapping) or not raw_members:
        return {}
    raw_records = gl_fc.get('data') or ()
    raw_times: list[str] = []
    for record in raw_records:
        if not isinstance(record, Mapping):
            raise TypeError(
                f'Expected mapping record in GloFAS data, got {record!r}'
            )
        raw_times.append(_normalize_timestamp(record.get('time')))
    aligned_members: dict[str, list[float | None]] = {}
    for member_key, member_vals in sorted(raw_members.items()):
        if not isinstance(member_vals, Sequence) or isinstance(
            member_vals, (str, bytes)
        ):
            raise TypeError(
                f'Expected sequence for ensemble member {member_key!r}'
            )
        member_series: dict[str, float] = {}
        for timestamp, raw_val in zip(raw_times, member_vals, strict=True):
            finite_val = parse_finite_float(raw_val)
            if finite_val is not None:
                member_series[timestamp] = finite_val
        aligned_members[str(member_key)] = align_daily_series(
            member_series, dates
        )
    return aligned_members


def build_aligned_timeline(
    models: list[str],
    fh_fc: dict[str, Any] | None,
    fh_is_q: bool,  # noqa: FBT001
    gl_fc: dict[str, Any] | None,
    gg_fc: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a timestamp-aligned timeline at each model's native cadence."""
    fh_daily = (
        _timestamp_series((fh_fc or {}).get('data'), 'discharge')
        if (fh_fc and fh_fc.get('data'))
        else {}
    )
    gl_det_d = (
        _timestamp_series(
            (gl_fc or {}).get('data'),
            'discharge_deterministic',
        )
        if gl_fc
        else {}
    )
    gl_daily = (
        _timestamp_series((gl_fc or {}).get('data'), 'discharge_median')
        if gl_fc
        else {}
    )
    gl_mean_d = (
        _timestamp_series((gl_fc or {}).get('data'), 'discharge_mean')
        if gl_fc
        else {}
    )
    gl_p25_d = (
        _timestamp_series((gl_fc or {}).get('data'), 'discharge_p25')
        if gl_fc
        else {}
    )
    gl_p75_d = (
        _timestamp_series((gl_fc or {}).get('data'), 'discharge_p75')
        if gl_fc
        else {}
    )
    gl_max_d = (
        _timestamp_series((gl_fc or {}).get('data'), 'discharge_max')
        if gl_fc
        else {}
    )
    gl_min_d = (
        _timestamp_series((gl_fc or {}).get('data'), 'discharge_min')
        if gl_fc
        else {}
    )
    gg_daily = (
        _timestamp_series((gg_fc or {}).get('data'), 'flow_med')
        if gg_fc
        else {}
    )
    gg_p25_d = (
        _timestamp_series((gg_fc or {}).get('data'), 'flow_25p')
        if gg_fc
        else {}
    )
    gg_p75_d = (
        _timestamp_series((gg_fc or {}).get('data'), 'flow_75p')
        if gg_fc
        else {}
    )
    gg_max_d = (
        _timestamp_series((gg_fc or {}).get('data'), 'flow_max')
        if gg_fc
        else {}
    )
    gg_min_d = (
        _timestamp_series((gg_fc or {}).get('data'), 'flow_min')
        if gg_fc
        else {}
    )
    gl_ts = (
        set(gl_det_d)
        | set(gl_daily)
        | set(gl_mean_d)
        | set(gl_p25_d)
        | set(gl_p75_d)
        | set(gl_max_d)
        | set(gl_min_d)
    )
    dates = sorted(set(fh_daily) | gl_ts | set(gg_daily))
    timeline_series: dict[str, Any] = {}
    if 'floodhub' in models and fh_daily:
        fh_aligned = align_daily_series(fh_daily, dates)
        timeline_series['floodhub'] = {
            'central': fh_aligned,
            'discharge': fh_aligned,
            'unit': 'm³/s' if fh_is_q else 'm',
            'axis': 'discharge' if fh_is_q else 'stage',
            'cadence': _infer_cadence_label(tuple(fh_daily)),
        }
    if 'glofas' in models and gl_ts:
        gl_det_aligned = align_daily_series(gl_det_d, dates)
        gl_med_aligned = align_daily_series(gl_daily, dates)
        timeline_series['glofas'] = {
            'central': gl_det_aligned,
            'deterministic': gl_det_aligned,
            'median': gl_med_aligned,
            'mean': align_daily_series(gl_mean_d, dates),
            'p25': align_daily_series(gl_p25_d, dates),
            'p75': align_daily_series(gl_p75_d, dates),
            'max': align_daily_series(gl_max_d, dates),
            'min': align_daily_series(gl_min_d, dates),
            'members': _align_glofas_members(gl_fc, dates),
            'cadence': _infer_cadence_label(tuple(gl_ts)),
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
            'cadence': _infer_cadence_label(tuple(gg_daily)),
        }
    return {
        'dates': dates,
        'unit': 'm³/s',
        'series': timeline_series,
        'status': {
            'floodhub': (fh_fc or {}).get('status') if fh_fc else None,
            'glofas': (gl_fc or {}).get('status') if gl_fc else None,
            'geoglows': (gg_fc or {}).get('status') if gg_fc else None,
        },
    }
