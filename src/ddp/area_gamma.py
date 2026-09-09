"""Validated, portable area coefficients for citywide PB-RBAT.

JSON tables contain their complete historical manifest and coefficient vector.
A table may be evaluated on its target day or on a declared historical day.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

GRID = tuple(i / 10 for i in range(11))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_identity(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity():
    root = Path(__file__).resolve().parent
    return identity({str(p.relative_to(root)): file_identity(p) for p in sorted(root.rglob('*.py'))})


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def coefficient(value):
    if isinstance(value, bool):
        raise ValueError('gamma must be numeric, not boolean')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("gamma must be numeric") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError('area gamma must be finite and in [0, 1]')
    return result


def validate_table(table, *, d, dataset_id, dispatch="rbatch", tau_s=30):
    if not isinstance(table, dict):
        raise ValueError('Coefficient table must be a JSON object')
    required = {'schema_version', 'grouping', 'matching_scope', 'shadow', 'dispatch',
                'tau', 'window_seconds', 'target_day', 'historical_days', 'dataset_days', 'dataset_id',
                'training_manifest_id', 'coefficients', 'fallback_gamma', 'missing_area_policy'}
    if not required.issubset(table):
        raise ValueError(f'Missing coefficient table metadata: {sorted(required - table.keys())}')
    for key, expected in {'schema_version': 1, 'grouping': 'da_id', 'matching_scope': 'citywide',
                          'shadow': 'pb', 'dispatch': dispatch, 'tau': 0}.items():
        if table[key] != expected:
            raise ValueError(f'Coefficient table {key} mismatch')
    if not np.isscalar(d) or not math.isfinite(float(d)) or float(d) <= 0 or table['window_seconds'] != float(d):
        raise ValueError('Coefficient table window mismatch')
    if dispatch == 'rbatch2':
        if not math.isfinite(float(tau_s)) or not 0 < float(tau_s) <= float(d):
            raise ValueError('Periodic interval must be finite and in (0, window]')
        if table.get('tau_s') != float(tau_s):
            raise ValueError('Coefficient table periodic interval mismatch')
    if not dataset_id or table['dataset_id'] != dataset_id:
        raise ValueError('Coefficient table dataset identity mismatch')
    history = table['historical_days']
    if (not isinstance(table['target_day'], str) or not isinstance(history, list)
            or not history or any(not isinstance(day, str) for day in history)
            or len(set(history)) != len(history) or table['target_day'] in history):
        raise ValueError('Historical days must be unique and exclude target day')
    days = table['dataset_days']
    if (not isinstance(days, list) or any(not isinstance(day, str) for day in days)
            or len(set(days)) != len(days) or table['target_day'] not in days
            or set(history) != set(days) - {table['target_day']}):
        raise ValueError('History must contain exactly the other observed dataset days')
    if not isinstance(table['training_manifest_id'], str) or not table['training_manifest_id']:
        raise ValueError('Missing training manifest identity')
    if table['missing_area_policy'] not in {'error', 'historical_global'}:
        raise ValueError('Declare missing_area_policy: error or historical_global')
    coefficient(table['fallback_gamma'])
    if not isinstance(table['coefficients'], dict) or any(not isinstance(k, str) or not k for k in table['coefficients']):
        raise ValueError('Coefficients must map nonempty string area IDs to gammas')
    for value in table['coefficients'].values():
        coefficient(value)
    for area, gamma in table['coefficients'].items():
        if area in table.get('effective_coefficients', {}) and not math.isclose(
                table['effective_coefficients'][area], coefficient(gamma) / 2, abs_tol=1e-15):
            raise ValueError('Effective coefficient must equal gamma / 2')
        support = table.get('support', {}).get(area)
        if support and support.get('historical_arrivals') == 0 and coefficient(gamma) != coefficient(table['fallback_gamma']):
            raise ValueError('An area without history must use the historical global fallback')
    return table


def resolve_area_gammas(jobs, table, *, d, gamma=None, shadows=('pb',),
                        dispatches=('rbatch',), tau=0, reward_type='pooling', tau_s=30):
    if table is None:
        return None, {}
    if gamma is not None:
        raise ValueError('Scalar gamma and gamma_table are mutually exclusive')
    if tuple(shadows) != ('pb',) or tuple(dispatches) not in {('rbatch',), ('rbatch2',)} or tau != 0 or reward_type != 'pooling':
        raise ValueError('Area coefficients require PB, dispatch=rbatch or rbatch2, tau=0, pooling reward')
    if isinstance(table, (str, Path)):
        # Reject duplicate JSON keys instead of silently overwriting coefficients.
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f'Duplicate table key: {key}')
                result[key] = value
            return result
        table = json.loads(Path(table).read_text(), object_pairs_hook=unique)
    datasets = {job.dataset_id for job in jobs}
    days = {job.day for job in jobs}
    ids = [job.job_id for job in jobs]
    if len(datasets) != 1 or len(days) != 1 or None in days or None in ids or len(set(ids)) != len(ids):
        raise ValueError('Area coefficients require unique job identities and one labeled dataset/day')
    validate_table(table, d=d, dataset_id=next(iter(datasets)), dispatch=tuple(dispatches)[0], tau_s=tau_s)
    if next(iter(days)) not in [table['target_day'], *table['historical_days']]:
        raise ValueError('Job day is outside the table target/history manifest')
    values, fallback_jobs, missing = [], 0, set()
    for job in jobs:
        if job.da_id is None:
            raise ValueError('Missing job da_id')
        if job.da_id in table['coefficients']:
            values.append(coefficient(table['coefficients'][job.da_id]))
        elif table['missing_area_policy'] == 'historical_global':
            values.append(coefficient(table['fallback_gamma']))
            fallback_jobs += 1
            missing.add(job.da_id)
        else:
            raise ValueError(f'Missing area coefficient: {job.da_id}')
    provenance = {'gamma_mode': 'area_table', 'gamma_table_id': identity(table),
                  'coefficient_vector': json.dumps(table['coefficients'], sort_keys=True),
                  'target_day': table['target_day'], 'historical_days': json.dumps(table['historical_days']),
                  'training_manifest_id': table['training_manifest_id'], 'dataset_id': table['dataset_id'],
                  'fallback_jobs': fallback_jobs, 'fallback_areas': json.dumps(sorted(missing)),
                  'fallback_gamma': table['fallback_gamma'], 'matching_scope': 'citywide',
                  'job_day': next(iter(days)), 'code_id': source_identity(),
                  'job_alignment_id': identity([[j.job_id, j.da_id, j.origin, j.dest, j.timestamp] for j in jobs]),
                  'effective_coefficient_vector': json.dumps({k: coefficient(v) / 2 for k, v in table['coefficients'].items()}, sort_keys=True)}
    return np.asarray(values), provenance


def select_gamma(scores, current, *, tolerance=1e-12, grid=GRID):
    """Exact-grid conditional tie rule; tolerance is absolute savings units."""
    if set(scores) != set(grid) or any(not math.isfinite(v) for v in scores.values()):
        raise ValueError(f'Selection requires all {len(grid)} finite candidate totals')
    if current not in grid:
        raise ValueError('Current coefficient must be in the candidate grid')
    best = max(scores.values())
    maximizers = [g for g, score in scores.items() if best - score <= tolerance]
    return min(maximizers, key=lambda g: (g != current, round(abs(g - current), 12), g))
