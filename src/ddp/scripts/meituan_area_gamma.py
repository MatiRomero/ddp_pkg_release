"""Resumable citywide scalar baselines and conditional area-coordinate sweeps.

Configs live directly in configs/. Run staged arrays with run_from_config,
or fit-fold to complete dependent coordinates within one allocated batch job.
No scheduler is invoked by this module.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import tempfile
import time

import networkx as nx
import numpy as np

from ddp.area_gamma import GRID, atomic_json, file_identity, identity, select_gamma, source_identity, coefficient, resolve_area_gammas
from ddp.pb_experiment import PreparedPB
from ddp.scripts.csv_loader import load_jobs_from_csv


def code_identity():
    return source_identity()


def atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_experiment(path, *, check_code=True):
    path = Path(path).resolve()
    root = path.parent.parent
    manifest = read_json(path)
    if identity({k: v for k, v in manifest.items() if k != 'manifest_id'}) != manifest['manifest_id']:
        raise ValueError('Experiment manifest was changed')
    if check_code and code_identity() != manifest['code_id']:
        raise ValueError('Source code differs from experiment manifest; use the frozen source or a new experiment ID')
    validate_spec(manifest)
    dataset_path = root / manifest['dataset_manifest']
    if file_identity(dataset_path) != manifest['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest identity mismatch')
    data = read_json(dataset_path)
    for name, digest in data['diagnostic_files'].items():
        if file_identity(dataset_path.parent / name) != digest:
            raise ValueError(f'Diagnostic identity mismatch: {name}')
    return root, manifest, data, dataset_path.parent


def history_days(data, fold):
    if fold not in data['days']:
        raise ValueError('Unknown target fold')
    return [day for day in data['days'] if day != fold]


def gamma_grid(manifest):
    return tuple(manifest.get('gamma_grid', GRID))


def fixed_gamma(manifest):
    return manifest.get('fixed_gamma', .5)


def uses_fixed_reference(manifest):
    return manifest.get('initialization', 'historical_global') == 'fixed_reference'


def baseline_candidates(manifest):
    grid = gamma_grid(manifest)
    return ([grid.index(fixed_gamma(manifest))] if uses_fixed_reference(manifest)
            else list(range(len(grid))))


def evaluation_specs(manifest, data, vector, global_gamma):
    fixed = (f'fixed_global_{fixed_gamma(manifest):g}', dict.fromkeys(data['areas'], fixed_gamma(manifest)))
    area = ('historical_area', vector)
    if uses_fixed_reference(manifest):
        return [fixed, area]
    return [('plain_' + manifest['dispatch'], dict.fromkeys(data['areas'], 0.)), fixed,
            ('historical_global', dict.fromkeys(data['areas'], global_gamma)), area]


def validate_spec(manifest):
    if manifest.get('initialization', 'historical_global') not in {'historical_global', 'fixed_reference'}:
        raise ValueError('Unknown coefficient initialization')
    grid = gamma_grid(manifest)
    if not grid or len(set(grid)) != len(grid) or list(grid) != sorted(grid):
        raise ValueError('Gamma grid must be nonempty, unique and increasing')
    for value in grid:
        coefficient(value)
    if grid[0] != 0 or grid[-1] != 1:
        raise ValueError('Gamma grid must include zero and one')
    if fixed_gamma(manifest) not in grid or manifest.get('global_tie_reference', .5) not in grid:
        raise ValueError('Fixed gamma and global tie reference must belong to the gamma grid')
    d = float(manifest['window_seconds'])
    if not math.isfinite(d) or d <= 0:
        raise ValueError('Window must be finite and positive')
    if manifest['dispatch'] not in {'rbatch', 'rbatch2'}:
        raise ValueError('Only RBAT and periodic RBAT are supported')
    if manifest['dispatch'] == 'rbatch2':
        period = float(manifest['tau_s'])
        if not math.isfinite(period) or not 0 < period <= d:
            raise ValueError('Periodic interval must lie in (0, window]')
    if manifest['shadow'] != 'pb' or manifest['tau'] != 0 or manifest['matching_scope'] != 'citywide':
        raise ValueError('Expected citywide PB with tau zero')


def historical_descriptors(data, data_dir, fold, d):
    """Only the other days enter counts, exposure or opportunities."""
    history = history_days(data, fold)
    provenance_path = data_dir / 'opportunities_provenance.json'
    opportunity_window = read_json(provenance_path)['window_seconds'] if provenance_path.exists() else 60
    with (data_dir / 'area_daily_arrivals.csv').open() as handle:
        counts = {row['da_id']: row for row in csv.DictReader(handle)}
    with (data_dir / 'area_day_summary.csv').open() as handle:
        daily = {(row['da_id'], row['day']): row for row in csv.DictReader(handle)}
    exposure = sum(data['days'][day]['exposure_seconds'] for day in history)
    result = {}
    for area in data['areas']:
        arrivals = sum(int(counts[area][day]) for day in history)
        eligible, opportunities = 0, 0.0
        for day in history:
            row = daily.get((area, day))
            if row:
                eligible += int(row['jobs'])
                opportunities += int(row['jobs']) * float(row['mean_poolable_all'])
            elif int(counts[area][day]):
                raise ValueError(f'Missing opportunity history for area {area}, day {day}')
        result[area] = {'historical_arrivals': arrivals, 'historical_exposure_seconds': exposure,
                        'observed_days': len(history), 'days_with_arrivals': sum(int(counts[area][day]) > 0 for day in history),
                        'arrival_rate_per_second': arrivals / exposure, 'nominal_density': arrivals / exposure * d,
                        'opportunity_eligible_jobs': eligible,
                        'mean_poolable_all': opportunities / eligible if eligible else None,
                        'opportunity_window_seconds': opportunity_window,
                        'opportunities_match_waiting_window': opportunity_window == d,
                        'fallback_reason': 'no_historical_arrivals' if not arrivals else None}
    return result


def output_dir(root, manifest):
    return root / 'results' / manifest['experiment_id']


def result_path(root, manifest, stage, day, candidate, fold='', step=0):
    scope = 'baseline' if stage == 'baseline' else f'fold{fold}/{stage}{step:02d}'
    return output_dir(root, manifest) / scope / f'day{day}_g{candidate:02d}.json'


def expected_identity(manifest, data, stage, day, candidate, vector, fold='', step=0):
    return {'manifest_id': manifest['manifest_id'], 'dataset_id': data['dataset_id'],
            'code_id': manifest['code_id'], 'day': day, 'jobs_sha256': data['days'][day]['sha256'],
            'stage': stage, 'target_day': fold, 'step': step, 'pass': step // len(manifest['area_order']) + 1,
            'area': manifest['area_order'][step % len(manifest['area_order'])] if stage == 'coordinate' else '',
            'candidate_index': candidate, 'candidate_gamma': gamma_grid(manifest)[candidate] if stage != 'evaluation' else None,
            'seed': manifest['seed'], 'd': manifest['window_seconds'], 'tau': 0,
            'shadow': 'pb', 'dispatch': manifest['dispatch'], 'matching_scope': 'citywide',
            **({'tau_s': manifest['tau_s'], 'tick_origin': 'loaded_timestamp_zero'} if manifest['dispatch'] == 'rbatch2' else {}),
            'coefficient_vector': vector, 'vector_id': identity(vector),
            'effective_coefficients': {area: gamma / 2 for area, gamma in vector.items()}}


def validated_result(path, expected):
    row = read_json(path)
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Result provenance mismatch: {path}')
    for key in ['savings', 'time_s', 'direct_distance_total', 'mean_wait_seconds', 'pooled_pct']:
        if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
            raise ValueError(f'Invalid {key}: {path}')
    if (not isinstance(row.get('n'), int) or 2 * row['pairs'] + row['solos'] != row['n']
            or not math.isclose(row['pooled_pct'], 200 * row['pairs'] / row['n'], abs_tol=1e-9)):
        raise ValueError(f'Incomplete result accounting: {path}')
    if row['result_id'] != identity({k: v for k, v in row.items() if k != 'result_id'}):
        raise ValueError(f'Result content identity mismatch: {path}')
    return row


def baseline_rows(root, m, data):
    rows = []
    for day in data['days']:
        for index in baseline_candidates(m):
            gamma = gamma_grid(m)[index]
            vector = dict.fromkeys(data['areas'], gamma)
            expected = expected_identity(m, data, 'baseline', day, index, vector)
            row = validated_result(result_path(root, m, 'baseline', day, index), expected)
            if row['n'] != data['days'][day]['jobs']:
                raise ValueError('Baseline population mismatch')
            rows.append(row)
    if len({identity(row['runtime']) for row in rows}) != 1:
        raise ValueError('Baseline numerical runtimes differ; do not mix environments')
    return rows


def global_fit(root, m, data, fold):
    # Require every day/grid candidate, even when fitting a single fold.
    rows = baseline_rows(root, m, data)
    historical = [row for row in rows if row['day'] != fold]
    if uses_fixed_reference(m):
        selected = fixed_gamma(m)
        return selected, {'historical_days': history_days(data, fold), 'selected_gamma': selected,
                          'initialization': 'fixed_reference', 'selection_rule': 'predeclared; no global gamma search',
                          'historical_result_ids': [row['result_id'] for row in historical]}
    scores = {g: sum(row['savings'] for row in historical if row['candidate_gamma'] == g) for g in gamma_grid(m)}
    selected = select_gamma(scores, m['global_tie_reference'], tolerance=m['tie_tolerance'], grid=gamma_grid(m))
    return selected, {'historical_days': history_days(data, fold), 'scores': {str(k): v for k, v in scores.items()},
                      'historical_result_ids': [row['result_id'] for row in historical], 'selected_gamma': selected}


def current_state(root, m, data, fold, step):
    global_gamma, fit = global_fit(root, m, data, fold)
    vector = dict.fromkeys(data['areas'], global_gamma)
    dependency = identity(fit)
    for previous in range(step):
        path = output_dir(root, m) / f'fold{fold}/selection{previous:02d}.json'
        state = read_json(path)
        if (state['input_vector'] != vector or state['dependency_id'] != dependency
                or state['manifest_id'] != m['manifest_id'] or state['fold'] != fold or state['step'] != previous
                or state['selection_id'] != identity({k: v for k, v in state.items() if k != 'selection_id'})):
            raise ValueError('Coordinate selection dependency mismatch')
        vector = state['selected_vector']
        dependency = state['selection_id']
    return vector, global_gamma, dependency, fit


def coefficient_table(m, data, fold, vector, global_gamma, dependency, descriptors):
    return {'schema_version': 1, 'grouping': 'da_id', 'matching_scope': 'citywide', 'shadow': 'pb',
            'dispatch': m['dispatch'], 'tau': 0, 'window_seconds': m['window_seconds'],
            **({'tau_s': m['tau_s']} if m['dispatch'] == 'rbatch2' else {}), 'target_day': fold,
            'historical_days': history_days(data, fold), 'dataset_days': list(data['days']), 'dataset_id': data['dataset_id'],
            'training_manifest_id': dependency, 'coefficients': vector, 'fallback_gamma': global_gamma,
            'missing_area_policy': 'historical_global', 'support': descriptors,
            'effective_coefficients': {area: gamma / 2 for area, gamma in vector.items()}}


def candidate_vectors(m, data, vector, descriptors, step, global_gamma):
    area = m['area_order'][step % len(m['area_order'])]
    return [{**vector, area: gamma if descriptors[area]['historical_arrivals'] else global_gamma} for gamma in gamma_grid(m)]


def prepare_day(root, m, data, day, jobs, runtime):
    if not m['with_lp'] and not m['with_opt']:
        return PreparedPB(jobs, m['window_seconds'], dispatch=m['dispatch'], tau_s=m.get('tau_s', 30))
    path = output_dir(root, m) / 'benchmarks' / f'day{day}.json'
    expected = {'jobs_sha256': data['days'][day]['sha256'], 'code_id': m['code_id'],
                'd': m['window_seconds'], 'runtime': runtime, 'dispatch': m['dispatch'], 'tau_s': m.get('tau_s'),
                'with_lp': m['with_lp'], 'with_opt': m['with_opt']}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        # A concurrent request exits instead of duplicating an expensive solve.
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Benchmarks already being prepared for day {day}; retry after completion') from exc
        if path.exists():
            bounds = read_json(path)
            if (any(bounds.get(k) != v for k, v in expected.items())
                    or bounds['bounds_id'] != identity({k: v for k, v in bounds.items() if k != 'bounds_id'})):
                raise ValueError('Cached benchmark identity mismatch')
            prepared = PreparedPB(jobs, m['window_seconds'], dispatch=m['dispatch'], tau_s=m.get('tau_s', 30))
            prepared.lp, prepared.opt = bounds['lp_total'], bounds['opt_total']
            return prepared
        prepared = PreparedPB(jobs, m['window_seconds'], with_lp=m['with_lp'], with_opt=m['with_opt'], dispatch=m['dispatch'], tau_s=m.get('tau_s', 30))
        bounds = {**expected, 'lp_total': prepared.lp, 'opt_total': prepared.opt,
                  'preparation_s': prepared.preparation_s}
        bounds['bounds_id'] = identity(bounds)
        atomic_json(path, bounds)
        return prepared


_POLICY_METRICS = (
    'n', 'savings', 'pairs', 'solos', 'pooled_pct', 'direct_distance_total',
    'savings_fraction', 'mean_wait_seconds', 'max_wait_seconds', 'cross_area_pairs',
    'cross_area_pair_fraction', 'time_s', 'lp_total', 'opt_total', 'ratio_lp', 'ratio_opt',
)


def run_cached(root, m, data, day, vector, prepared, runtime, *, gamma=None, table=None):
    """Reuse an exact day/vector policy evaluation across historical folds."""
    spec = {'code_id': m['code_id'], 'dataset_id': data['dataset_id'],
            'jobs_sha256': data['days'][day]['sha256'], 'day': day,
            'window_seconds': m['window_seconds'], 'tau_s': m.get('tau_s'),
            'dispatch': m['dispatch'], 'shadow': 'pb', 'tau': 0, 'reward': 'pooling',
            'seed': m['seed'], 'coefficient_vector': vector, 'runtime': runtime,
            'with_lp': m['with_lp'], 'with_opt': m['with_opt']}
    cache_id = identity(spec)
    path = output_dir(root, m) / 'vector_cache' / f'day{day}' / f'{cache_id}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        # Other folds may request exactly the same simulation simultaneously.
        # Wait for its atomic cache write; if the owner fails, this lock releases.
        fcntl.flock(lock, fcntl.LOCK_EX)
        hit = path.exists()
        if hit:
            saved = read_json(path)
            if (saved['spec'] != spec or saved['payload_id'] != identity({k: v for k, v in saved.items() if k != 'payload_id'})
                    or set(saved['metrics']) != set(_POLICY_METRICS)):
                raise ValueError('Simulation cache provenance/content mismatch')
        else:
            row, result = prepared.run(gamma=gamma, gamma_table=table, seed=m['seed'])
            saved = {'spec': spec, 'metrics': {key: row[key] for key in _POLICY_METRICS},
                     'decision_id': identity({'pairs': result['pairs'], 'solos': result['solos'],
                                              'dispatch_times': result['dispatch_times']})}
            saved['payload_id'] = identity(saved)
            atomic_json(path, saved)
    _, provenance = resolve_area_gammas(prepared.jobs, table, d=m['window_seconds'], gamma=gamma,
                                       dispatches=(m['dispatch'],), tau_s=m.get('tau_s', 30))
    return {**saved['metrics'], **provenance, 'gamma': gamma, 'simulation_cache_id': cache_id,
            'simulation_payload_id': saved['payload_id'], 'decision_id': saved['decision_id'],
            'simulation_cache_hit': hit}


def run_task(manifest_path, stage, day, fold='', step=0, candidate_index=None):
    root, m, data, data_dir = load_experiment(manifest_path)
    if day not in data['days'] or stage not in {'baseline', 'coordinate', 'evaluation'}:
        raise ValueError('Unknown stage/day')
    if stage == 'coordinate' and not 0 <= step < len(m['area_order']) * m['passes']:
        raise ValueError('Step outside declared search')
    if stage == 'evaluation' and step != len(m['area_order']) * m['passes']:
        raise ValueError('Evaluation requires every declared coordinate selection')
    descriptors, dependency, global_gamma = {}, '', None
    if stage == 'baseline':
        vectors = [dict.fromkeys(data['areas'], gamma) for gamma in gamma_grid(m)]
    else:
        if stage == 'coordinate' and day == fold:
            raise ValueError('Target day cannot enter a conditional training sweep')
        if stage == 'evaluation' and day != fold:
            raise ValueError('Evaluation must use the target day')
        vector, global_gamma, dependency, fit = current_state(root, m, data, fold, step)
        descriptors = historical_descriptors(data, data_dir, fold, m['window_seconds'])
        vectors = (candidate_vectors(m, data, vector, descriptors, step, global_gamma) if stage == 'coordinate'
                   else [v for _, v in evaluation_specs(m, data, vector, global_gamma)])
    indices = ((baseline_candidates(m) if stage == 'baseline' else range(len(vectors)))
               if candidate_index is None else [candidate_index])
    if any(index < 0 or index >= len(vectors) for index in indices):
        raise ValueError('Candidate index outside stage grid')
    if stage == 'baseline' and any(index not in baseline_candidates(m) for index in indices):
        raise ValueError('This experiment requires only the fixed reference gamma')
    job_path = data_dir / data['days'][day]['path']
    if file_identity(job_path) != data['days'][day]['sha256']:
        raise ValueError('Job data identity mismatch')
    runtime = {'python': platform.python_version(), 'numpy': np.__version__, 'networkx': nx.__version__}
    prepared, rows = None, []
    for index in indices:
        vector = vectors[index]
        expected = expected_identity(m, data, stage, day, index, vector, fold, step)
        path = result_path(root, m, stage, day, index, fold, step)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'Candidate already running: {path}') from exc
            if path.exists():
                row = validated_result(path, expected)
                if row['runtime'] != runtime:
                    raise ValueError('Cannot resume candidates with a different numerical runtime')
                rows.append(row)
                print(f'Resume: {path.name}', flush=True)
                continue
            start = time.perf_counter()
            if stage == 'evaluation' and index < len(vectors) - 1:
                scalar_gamma = next(iter(vector.values()))
                source = next(r for r in baseline_rows(root, m, data)
                              if r['day'] == day and r['candidate_gamma'] == scalar_gamma)
                if source['runtime'] != runtime:
                    raise ValueError('Evaluation and baseline numerical runtimes differ')
                row = {k: v for k, v in source.items() if k != 'result_id'}
                row['reused_result_id'] = source['result_id']
                preparation_s = 0.
            else:
                if prepared is None:
                    jobs = load_jobs_from_csv(job_path)
                    if len(jobs) != data['days'][day]['jobs']:
                        raise ValueError('Job population mismatch')
                    prepared = prepare_day(root, m, data, day, jobs, runtime)
                if stage == 'baseline':
                    row = run_cached(root, m, data, day, vector, prepared, runtime, gamma=gamma_grid(m)[index])
                else:
                    table = coefficient_table(m, data, fold, vector, global_gamma, dependency, descriptors)
                    row = run_cached(root, m, data, day, vector, prepared, runtime, table=table)
                preparation_s = prepared.preparation_s
            row.update(expected)
            row.update(runtime=runtime, preparation_s=preparation_s,
                       candidate_wall_s=time.perf_counter() - start, dependency_id=dependency)
            if stage == 'evaluation':
                row['strategy'] = evaluation_specs(m, data, vector, global_gamma)[index][0]
            row['result_id'] = identity(row)
            atomic_json(path, row)
            rows.append(validated_result(path, expected))
            print(f'{stage} day={day} fold={fold} step={step} candidate={index}: savings={row["savings"]:.9f}, policy={row["time_s"]:.2f}s', flush=True)
    # CSVs are review conveniences; hashed candidate JSON files are authoritative.
    scope = 'baseline' if stage == 'baseline' else f'fold{fold}/{stage}{step:02d}'
    suffix = '' if candidate_index is None else f'_g{candidate_index:02d}'
    atomic_csv(output_dir(root, m) / scope / f'day{day}{suffix}.csv',
               [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows])


def select_step(manifest_path, fold, step):
    root, m, data, data_dir = load_experiment(manifest_path)
    if not 0 <= step < len(m['area_order']) * m['passes']:
        raise ValueError('Step outside declared search')
    current, global_gamma, dependency, fit = current_state(root, m, data, fold, step)
    descriptors = historical_descriptors(data, data_dir, fold, m['window_seconds'])
    vectors = candidate_vectors(m, data, current, descriptors, step, global_gamma)
    rows, scores = [], {}
    for index, vector in enumerate(vectors):
        scores[gamma_grid(m)[index]] = 0.
        for day in history_days(data, fold):
            expected = expected_identity(m, data, 'coordinate', day, index, vector, fold, step)
            row = validated_result(result_path(root, m, 'coordinate', day, index, fold, step), expected)
            if row['dependency_id'] != dependency or row['n'] != data['days'][day]['jobs']:
                raise ValueError('Conditional curve dependency/population mismatch')
            scores[gamma_grid(m)[index]] += row['savings']
            rows.append(row)
    if len({identity(row['runtime']) for row in rows + baseline_rows(root, m, data)}) != 1:
        raise ValueError('Candidate numerical runtimes differ')
    area = m['area_order'][step % len(m['area_order'])]
    selected = select_gamma(scores, current[area], tolerance=m['tie_tolerance'], grid=gamma_grid(m))
    vector = {**current, area: selected}
    table = coefficient_table(m, data, fold, vector, global_gamma, dependency, descriptors)
    state = {'manifest_id': m['manifest_id'], 'fold': fold, 'step': step, 'area': area,
             'pass': step // len(m['area_order']) + 1, 'dependency_id': dependency,
             'input_vector': current, 'selected_vector': vector,
             'scores': {str(k): v for k, v in scores.items()}, 'selected_gamma': selected,
             'current_gamma': current[area], 'historical_days': history_days(data, fold),
             'historical_result_ids': [r['result_id'] for r in rows], 'global_fit': fit,
             'training_gain': scores[selected] - scores[current[area]],
             'near_optimal_gammas': [g for g, score in scores.items() if max(scores.values()) - score <= m['near_optimal_absolute']],
             'coefficient_table': table}
    state['selection_id'] = identity(state)
    path = output_dir(root, m) / f'fold{fold}/selection{step:02d}.json'
    if path.exists() and read_json(path) != state:
        raise ValueError('Refusing to overwrite a different coordinate selection')
    atomic_json(path, state)
    table = {**table, 'training_manifest_id': state['selection_id']}
    atomic_json(path.with_name(f'coefficients{step:02d}.json'), table)
    atomic_csv(path.with_name(f'curve{step:02d}.csv'),
               [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows])
    print(f'fold={fold}, step={step}, area={area}: {current[area]:g} -> {selected:g}; historical citywide gain={state["training_gain"]:.9f}')
    return state


def fit_fold(manifest_path, fold, workers=1):
    """Fit all declared areas in order; parallelize only the historical days.

    The baseline must already be complete. Each child holds at most one day's
    PreparedPB state, reusing it across that day's candidate gammas. Every step
    joins all children and validates its complete curve before selecting gamma.
    """
    root, m, data, _ = load_experiment(manifest_path)
    if fold not in m['folds']:
        raise ValueError('Fold is not declared in this experiment')
    days = history_days(data, fold)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= len(days):
        raise ValueError('Workers must be between one and the number of historical days')
    slots = os.environ.get('NSLOTS')
    if slots is not None and workers > int(slots):
        raise ValueError('Workers exceed allocated NSLOTS; request matching --grid_ncpus')
    baseline_rows(root, m, data)
    folder = output_dir(root, m) / f'fold{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'fit.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Fold {fold} is already running') from exc
        # Spawn avoids inherited native numerical-library/thread state.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for step in range(len(m['area_order']) * m['passes']):
                print(f'Fold {fold}: starting step {step}, area {m["area_order"][step % len(m["area_order"])]}', flush=True)
                futures = [pool.submit(run_task, str(manifest_path), 'coordinate', day, fold, step) for day in days]
                for future in futures:
                    future.result()
                select_step(manifest_path, fold, step)
        run_task(manifest_path, 'evaluation', fold, fold, len(m['area_order']) * m['passes'])
    print(f'Fold {fold}: fitting and held-out evaluation complete', flush=True)


def array_fold(manifest_path):
    """Map the scheduler's one-based task ID to the manifest's fold order."""
    _, m, _, _ = load_experiment(manifest_path)
    try:
        task_id = int(os.environ.get('SGE_TASK_ID', ''))
    except ValueError as exc:
        raise ValueError('fit-fold --array requires a numeric SGE_TASK_ID') from exc
    if not 1 <= task_id <= len(m['folds']):
        raise ValueError('SGE_TASK_ID outside the declared folds')
    return m['folds'][task_id - 1]


def generate(root, dataset_manifest, experiment_id, areas, folds, passes=1, with_lp=False, with_opt=False,
             *, window=60, dispatch='rbatch', tau_s=30, gammas=GRID, reference_gamma=.5,
             initialization='historical_global'):
    root = Path(root).resolve()
    dataset_path = (root / dataset_manifest).resolve()
    data = read_json(dataset_path)
    if not experiment_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in experiment_id):
        raise ValueError('Use letters, digits, underscore and hyphen in experiment ID')
    if areas == ['all']:
        areas = list(data['areas'])
    if folds == ['all']:
        folds = list(data['days'])
    if not areas or len(set(areas)) != len(areas) or not set(areas) <= set(data['areas']):
        raise ValueError('Area order must contain distinct known IDs')
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= set(data['days']) or passes < 1:
        raise ValueError('Invalid folds or passes')
    m = {'schema_version': 1, 'experiment_id': experiment_id, 'dataset_manifest': str(dataset_path.relative_to(root)),
         'dataset_manifest_sha256': file_identity(dataset_path), 'code_id': code_identity(),
         'window_seconds': window, 'seed': 0, 'area_order': areas, 'folds': folds, 'passes': passes,
         'with_lp': with_lp, 'with_opt': with_opt, 'gamma_grid': list(gammas),
         'tie_tolerance': 1e-12, 'near_optimal_absolute': 1e-6,
         'selection_objective': 'total citywide historical savings', 'global_tie_reference': reference_gamma, 'fixed_gamma': reference_gamma,
         'matching_scope': 'citywide', 'shadow': 'pb', 'dispatch': dispatch, 'tau': 0,
         **({'initialization': initialization} if initialization != 'historical_global' else {}),
         **({'tau_s': tau_s, 'tick_origin': 'loaded_timestamp_zero', 'deadline_rule': 'dispatch_by_next_tick'} if dispatch == 'rbatch2' else {})}
    validate_spec(m)
    m['manifest_id'] = identity(m)
    manifest_path = root / 'configs' / f'{experiment_id}.json'
    if manifest_path.exists() and read_json(manifest_path) != m:
        raise ValueError('Experiment ID already has a different manifest; choose a new ID')
    atomic_json(manifest_path, m)
    relative = str(manifest_path.relative_to(root))
    def task(stage, day, fold='', step=0):
        return {'experiment_manifest': relative, 'experiment_stage': stage, 'experiment_day': day,
                'experiment_fold': fold, 'experiment_step': step, 'experiment_candidate': ''}
    reference_name = 'reference' if uses_fixed_reference(m) else 'baseline'
    configs = {reference_name: [task('baseline', day) for day in data['days']],
               reference_name + '_probe': [{**task('baseline', next(iter(data['days']))), 'experiment_candidate': gamma_grid(m).index(reference_gamma)}]}
    for step in range(len(areas) * passes):
        configs[f'pilot_step{step:02d}'] = [task('coordinate', day, fold, step)
                                          for fold in folds for day in history_days(data, fold)]
    configs['evaluation'] = [task('evaluation', fold, fold, len(areas) * passes) for fold in folds]
    for name, rows in configs.items():
        path = root / 'configs' / f'{experiment_id}_{name}.csv'
        atomic_csv(path, rows)
        print(f'{path}: {len(rows)} tasks')
    return manifest_path


def summarize(manifest_path):
    root, m, data, _ = load_experiment(manifest_path)
    baseline = baseline_rows(root, m, data)
    atomic_csv(output_dir(root, m) / ('reference_summary.csv' if uses_fixed_reference(m) else 'baseline_summary.csv'),
               [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in baseline])
    comparisons, curves, coefficients = [], [], []
    for fold in m['folds']:
        vector, global_gamma, _, _ = current_state(root, m, data, fold, len(m['area_order']) * m['passes'])
        for step in range(len(m['area_order']) * m['passes']):
            state = read_json(output_dir(root, m) / f'fold{fold}/selection{step:02d}.json')
            for gamma in gamma_grid(m):
                curves.append({'target_day': fold, 'area': state['area'], 'step': step, 'pass': state['pass'],
                               'candidate_gamma': gamma, 'effective_coefficient': gamma / 2,
                               'historical_citywide_savings': state['scores'][str(gamma)],
                               'selected': gamma == state['selected_gamma'], 'current_gamma': state['current_gamma'],
                               'historical_days': json.dumps(state['historical_days']),
                               **state['coefficient_table']['support'][state['area']]})
        for area, gamma in vector.items():
            coefficients.append({'target_day': fold, 'area': area, 'gamma': gamma,
                                 'effective_coefficient': gamma / 2, 'historical_global_gamma': global_gamma,
                                 'swept': area in m['area_order']})
        specs = evaluation_specs(m, data, vector, global_gamma)
        vectors = [v for _, v in specs]
        rows = [validated_result(result_path(root, m, 'evaluation', fold, i, fold, len(m['area_order']) * m['passes']),
                                 expected_identity(m, data, 'evaluation', fold, i, v, fold, len(m['area_order']) * m['passes']))
                for i, v in enumerate(vectors)]
        scores = {label: row['savings'] for (label, _), row in zip(specs, rows)}
        comparison = {'day': fold, **scores,
                      f'area_minus_fixed_{fixed_gamma(m):g}': scores['historical_area'] - scores[f'fixed_global_{fixed_gamma(m):g}']}
        if not uses_fixed_reference(m):
            comparison.update(area_minus_historical_global=scores['historical_area'] - scores['historical_global'])
            comparison['area_minus_plain_' + m['dispatch']] = scores['historical_area'] - scores['plain_' + m['dispatch']]
        comparisons.append(comparison)
    atomic_csv(output_dir(root, m) / 'paired_day_savings.csv', comparisons)
    atomic_csv(output_dir(root, m) / 'area_gamma_curves.csv', curves)
    atomic_csv(output_dir(root, m) / 'selected_area_gammas.csv', coefficients)
    print(f'Validated {len(baseline)} baseline evaluations and {len(comparisons)} held-out folds')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--root', type=Path, default=Path.cwd())
    gen.add_argument('--dataset-manifest', default='data/meituan_area_gamma_v1/dataset.json')
    gen.add_argument('--experiment-id', default='meituan_area_gamma_60s_pilot_v1')
    gen.add_argument('--areas', default='22,5,9')
    gen.add_argument('--folds', default='0')
    gen.add_argument('--window', type=float, default=60)
    gen.add_argument('--dispatch', choices=['rbatch', 'rbatch2'], default='rbatch')
    gen.add_argument('--tau-s', type=float, default=30)
    gen.add_argument('--gammas', default=','.join(str(g) for g in GRID))
    gen.add_argument('--reference-gamma', type=float, default=.5)
    gen.add_argument('--initialization', choices=['historical_global', 'fixed_reference'], default='historical_global')
    gen.add_argument('--passes', type=int, default=1)
    gen.add_argument('--with-lp', action='store_true')
    gen.add_argument('--with-opt', action='store_true')
    run = sub.add_parser('run-task')
    run.add_argument('--manifest', required=True)
    run.add_argument('--stage', choices=['baseline', 'coordinate', 'evaluation'], required=True)
    run.add_argument('--day', required=True)
    run.add_argument('--fold', default='')
    run.add_argument('--step', type=int, default=0)
    run.add_argument('--candidate', type=int)
    select = sub.add_parser('select')
    select.add_argument('--manifest', required=True)
    select.add_argument('--fold', required=True)
    select.add_argument('--step', type=int, required=True)
    fit = sub.add_parser('fit-fold')
    fit.add_argument('--manifest', required=True)
    fold_arg = fit.add_mutually_exclusive_group(required=True)
    fold_arg.add_argument('--fold')
    fold_arg.add_argument('--array', action='store_true')
    fit.add_argument('--workers', type=int, default=1)
    summary = sub.add_parser('summarize')
    summary.add_argument('--manifest', required=True)
    a = p.parse_args()
    if a.command == 'generate':
        generate(a.root, a.dataset_manifest, a.experiment_id, a.areas.split(','), a.folds.split(','), a.passes, a.with_lp, a.with_opt, window=a.window, dispatch=a.dispatch, tau_s=a.tau_s, gammas=tuple(float(g) for g in a.gammas.split(',')), reference_gamma=a.reference_gamma, initialization=a.initialization)
    elif a.command == 'run-task':
        run_task(a.manifest, a.stage, a.day, a.fold, a.step, a.candidate)
    elif a.command == 'select':
        select_step(a.manifest, a.fold, a.step)
    elif a.command == 'fit-fold':
        fit_fold(a.manifest, array_fold(a.manifest) if a.array else a.fold, a.workers)
    else:
        summarize(a.manifest)


if __name__ == '__main__':
    main()
