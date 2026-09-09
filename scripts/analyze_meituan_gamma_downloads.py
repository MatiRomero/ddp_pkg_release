"""Audit saved Meituan gamma experiments and export descriptive report inputs.

Never runs a simulation or changes an experiment. Validation uses the pure
validation/selection functions from each experiment's hash-verified archive.
Partial curves are excluded from fitted comparisons until all seven days and
six candidates, plus the saved selection dependency chain, are verified.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

import numpy as np

NAMES = ["meituan_area_gamma_60s_pilot_v1", "meituan_rbatch2_pb_120s_g6_v1",
         "meituan_area5_rbatch2_pb_120s_g6_v1"]


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def table(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def require(ok, description):
    if not ok:
        raise ValueError(description)


def frozen_functions(root, name, manifest):
    archive = root / 'results' / (name + '_upload.tar.gz')
    with tarfile.open(archive) as tar:
        prefix = ('experiment_src/' + name + '/ddp/') if 'area5' in name else 'src/ddp/'
        source = {member.name[len(prefix):]: tar.extractfile(member).read()
                  for member in tar.getmembers()
                  if member.isfile() and member.name.startswith(prefix) and member.name.endswith('.py')}
        hashes = {key: hashlib.sha256(value).hexdigest() for key, value in source.items()}
        require(identity(hashes) == manifest['code_id'], name + ': frozen source identity')
        archived_manifest = json.loads(tar.extractfile('configs/' + name + '.json').read())
        require(archived_manifest == manifest, name + ': archived manifest differs')
    # Execute only named pure functions, retaining each archived implementation.
    # This avoids importing the solver or using today's workflow for older files.
    namespace = dict(json=json, math=math, Path=Path, hashlib=hashlib, np=np, csv=csv,
                     GRID=tuple(i / 10 for i in range(11)), code_identity=lambda: identity(hashes))
    area_names = {'identity', 'file_identity', 'coefficient', 'select_gamma'}
    workflow_names = {'read_json', 'load_experiment', 'history_days', 'gamma_grid', 'fixed_gamma',
                      'uses_fixed_reference', 'baseline_candidates', 'validate_spec', 'historical_descriptors',
                      'output_dir', 'result_path', 'expected_identity', 'validated_result', 'baseline_rows',
                      'global_fit', 'current_state', 'coefficient_table', 'candidate_vectors'}
    for file, names in [('area_gamma.py', area_names), ('scripts/meituan_area_gamma.py', workflow_names)]:
        tree = ast.parse(source[file])
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(tree, name + '/' + file, 'exec'), namespace)
    return namespace, {'archive_sha256': sha(archive), 'code_id': identity(hashes), 'source_files': hashes}


def summarize_rows(rows):
    n = sum(row['n'] for row in rows)
    direct = sum(row['direct_distance_total'] for row in rows)
    savings = sum(row['savings'] for row in rows)
    pairs = sum(row['pairs'] for row in rows)
    return {'days': len(rows), 'n': n, 'savings': savings, 'direct_distance_total': direct,
            'savings_pct': 100 * savings / direct,
            'pooled_pct': 200 * pairs / n,
            'mean_wait_seconds': sum(row['n'] * row['mean_wait_seconds'] for row in rows) / n,
            'cross_area_pairs_pct': 100 * sum(row['cross_area_pairs'] for row in rows) / pairs,
            'simulation_seconds': sum(row['time_s'] for row in rows)}


def analyze(root, source_root, out):
    out.mkdir(parents=True, exist_ok=True)
    # Byte-identical local snapshot, preserving input paths for provenance.
    inventory = []
    for name in NAMES:
        source_dir = source_root / 'results' / name
        for path in sorted(source_dir.rglob('*')):
            if not path.is_file() or path.suffix not in {'.json', '.csv'}:
                continue
            target = root / 'results' / name / path.relative_to(source_dir)
            content = path.read_bytes()
            if target.exists():
                require(target.read_bytes() == content, 'Snapshot conflict: ' + str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            inventory.append({'experiment': name, 'path': str(path.relative_to(source_root)),
                              'source_path': str(path), 'bytes': len(content), 'sha256': sha(target)})
    dump(out / 'source_inventory.json', inventory)
    summaries, audits = {}, {}
    all_rows, selected_rows, curve_rows, coverage_rows, missing_rows = [], [], [], [], []
    data = read(root / 'data/meituan_area_gamma_v1/dataset.json')
    direct_by_day = {}
    for day, spec in data['days'].items():
        path = root / 'data/meituan_area_gamma_v1' / spec['path']
        require(sha(path) == spec['sha256'], 'Job file SHA256: ' + day)
        with path.open() as handle:
            jobs = list(csv.DictReader(handle))
        require(len(jobs) == spec['jobs'], 'Job population: ' + day)
        direct_by_day[day] = math.fsum(math.hypot(float(r['sender_lat']) - float(r['recipient_lat']),
                                               float(r['sender_lng']) - float(r['recipient_lng'])) for r in jobs)
    for name in NAMES:
        manifest = read(root / 'configs' / (name + '.json'))
        ns, audit = frozen_functions(root, name, manifest)
        _, m, d, data_dir = ns['load_experiment'](root / 'configs' / (name + '.json'))
        grid = list(m.get('gamma_grid', [i / 10 for i in range(11)]))
        baseline = ns['baseline_rows'](root, m, d)
        base_map = {(r['day'], r['candidate_gamma']): r for r in baseline}
        exp_rows = []
        runtime = baseline[0]['runtime']
        cache_ids = set()

        def validate_row(path, expected, dependency=''):
            row = ns['validated_result'](path, expected)
            day = row['day']
            require(row['n'] == d['days'][day]['jobs'], str(path) + ': population')
            require(row['runtime'] == runtime, str(path) + ': numerical runtime')
            require(row.get('dependency_id', '') == dependency, str(path) + ': dependency')
            require(math.isclose(row['direct_distance_total'], direct_by_day[day], rel_tol=1e-12), str(path) + ': direct distances')
            require(math.isclose(row['savings_fraction'], row['savings'] / row['direct_distance_total'], rel_tol=1e-12), str(path) + ': savings normalization')
            require(0 <= row['cross_area_pairs'] <= row['pairs'], str(path) + ': cross-area count')
            require(math.isclose(row['cross_area_pair_fraction'], row['cross_area_pairs'] / row['pairs'], abs_tol=1e-12), str(path) + ': cross-area fraction')
            require(0 <= row['mean_wait_seconds'] <= m['window_seconds'], str(path) + ': wait')
            if 'simulation_cache_id' in row:
                cache_id = row['simulation_cache_id']
                cache = read(root / 'results' / name / 'vector_cache' / ('day' + day) / (cache_id + '.json'))
                require(identity(cache['spec']) == cache_id, str(path) + ': cache spec')
                require(identity({k: v for k, v in cache.items() if k != 'payload_id'}) == cache['payload_id'] == row['simulation_payload_id'], str(path) + ': cache payload')
                expected_spec = {'code_id': m['code_id'], 'dataset_id': d['dataset_id'], 'jobs_sha256': d['days'][day]['sha256'],
                                 'day': day, 'window_seconds': m['window_seconds'], 'tau_s': m.get('tau_s'),
                                 'dispatch': m['dispatch'], 'shadow': 'pb', 'tau': 0, 'reward': 'pooling', 'seed': m['seed'],
                                 'coefficient_vector': row['coefficient_vector'], 'runtime': runtime,
                                 'with_lp': m['with_lp'], 'with_opt': m['with_opt']}
                require(cache['spec'] == expected_spec, str(path) + ': exact cache identity')
                require(all(row[key] == value for key, value in cache['metrics'].items()), str(path) + ': cache metrics')
                require(cache['decision_id'] == row['decision_id'], str(path) + ': decisions')
                cache_ids.add(cache_id)
            row = dict(row, experiment=name, relative_path=str(path.relative_to(root)))
            exp_rows.append(row)
            return row

        for row in baseline:
            vector = dict.fromkeys(d['areas'], row['candidate_gamma'])
            path = ns['result_path'](root, m, 'baseline', row['day'], row['candidate_index'])
            validate_row(path, ns['expected_identity'](m, d, 'baseline', row['day'], row['candidate_index'], vector))
        global_summary = []
        for gamma in sorted({r['candidate_gamma'] for r in baseline}):
            global_summary.append(dict(gamma=gamma, effective_coefficient=gamma / 2,
                                       **summarize_rows([r for r in baseline if r['candidate_gamma'] == gamma])))
        table(out / (name + '_global.csv'), global_summary)
        strategies, day_results = [], []
        if len(global_summary) > 1:
            for gamma in dict.fromkeys([0., .4, m.get('fixed_gamma', .5)]):
                if gamma in grid:
                    strategies.append(dict(strategy='fixed_' + str(gamma), **summarize_rows([base_map[day, gamma] for day in d['days']])))
            selected = []
            oracle = []
            for day in d['days']:
                gamma, fit = ns['global_fit'](root, m, d, day)
                selected.append(base_map[day, gamma])
                oracle_gamma = max(grid, key=lambda g: base_map[day, g]['savings'])
                oracle.append(base_map[day, oracle_gamma])
                ref = base_map[day, m.get('fixed_gamma', .5)]
                day_results.append({'day': day, 'date': d['days'][day]['date'], 'historical_global_gamma': gamma,
                                    'oracle_same_day_gamma': oracle_gamma, 'heldout_savings': selected[-1]['savings'],
                                    'heldout_savings_pct': 100 * selected[-1]['savings_fraction'],
                                    'gain_vs_fixed_reference': selected[-1]['savings'] - ref['savings'],
                                    'gain_vs_plain': selected[-1]['savings'] - base_map[day, 0.]['savings']})
            strategies.append(dict(strategy='leave_one_day_out_global', **summarize_rows(selected)))
            strategies.append(dict(strategy='same_day_oracle', **summarize_rows(oracle)))
        table(out / (name + '_strategies.csv'), strategies)
        table(out / (name + '_heldout_global.csv'), day_results)
        exp_selections = []
        for fold in m['folds']:
            global_gamma, fit = ns['global_fit'](root, m, d, fold)
            vector = dict.fromkeys(d['areas'], global_gamma)
            dependency = identity(fit)
            descriptors = ns['historical_descriptors'](d, data_dir, fold, m['window_seconds'])
            history = ns['history_days'](d, fold)
            history_direct = sum(direct_by_day[day] for day in history)
            initial_score = sum(base_map[day, global_gamma]['savings'] for day in history)
            chain_complete = True
            for step in range(len(m['area_order']) * m['passes']):
                area = m['area_order'][step % len(m['area_order'])]
                rows = []
                for index, gamma in enumerate(grid):
                    for day in history:
                        path = ns['result_path'](root, m, 'coordinate', day, index, fold, step)
                        if not path.exists():
                            missing_rows.append({'experiment': name, 'fold': fold, 'step': step, 'area': area,
                                                 'day': day, 'gamma': gamma, 'path': str(path.relative_to(root))})
                            continue
                        require(chain_complete, str(path) + ': candidate without completed prior chain')
                        candidate_vector = dict(vector, **{area: gamma})
                        rows.append(validate_row(path, ns['expected_identity'](m, d, 'coordinate', day, index, candidate_vector, fold, step), dependency))
                selection_path = root / 'results' / name / ('fold' + fold) / f'selection{step:02d}.json'
                complete = len(rows) == len(grid) * len(history)
                coverage_rows.append({'experiment': name, 'fold': fold, 'step': step, 'area': area,
                                      'found': len(rows), 'expected': len(grid) * len(history),
                                      'complete_curve': complete, 'saved_selection': selection_path.exists()})
                if not complete:
                    require(not selection_path.exists(), str(selection_path) + ': selection despite missing candidates')
                    chain_complete = False
                    continue
                require(selection_path.exists(), str(selection_path) + ': complete curve missing selection')
                # Match the archived coordinate runner's sequential += order.
                # Python 3.12+ sum() uses a different floating-point algorithm.
                scores = dict.fromkeys(grid, 0.)
                for row in rows:
                    scores[row['candidate_gamma']] += row['savings']
                selected_gamma = ns['select_gamma'](scores, vector[area], tolerance=m['tie_tolerance'], grid=grid)
                selected_vector = dict(vector, **{area: selected_gamma})
                coefficient_table = ns['coefficient_table'](m, d, fold, selected_vector, global_gamma, dependency, descriptors)
                state = {'manifest_id': m['manifest_id'], 'fold': fold, 'step': step, 'area': area,
                         'pass': step // len(m['area_order']) + 1, 'dependency_id': dependency,
                         'input_vector': vector, 'selected_vector': selected_vector,
                         'scores': {str(k): v for k, v in scores.items()}, 'selected_gamma': selected_gamma,
                         'current_gamma': vector[area], 'historical_days': history,
                         'historical_result_ids': [r['result_id'] for r in rows], 'global_fit': fit,
                         'training_gain': scores[selected_gamma] - scores[vector[area]],
                         'near_optimal_gammas': [g for g, score in scores.items() if max(scores.values()) - score <= m['near_optimal_absolute']],
                         'coefficient_table': coefficient_table}
                state['selection_id'] = identity(state)
                require(read(selection_path) == state, str(selection_path) + ': independently reconstructed selection')
                require(read(selection_path.with_name(f'coefficients{step:02d}.json')) == dict(coefficient_table, training_manifest_id=state['selection_id']), str(selection_path) + ': coefficient table')
                with selection_path.with_name(f'curve{step:02d}.csv').open() as stream:
                    saved_curve = list(csv.DictReader(stream))
                require([r['result_id'] for r in saved_curve] == [r['result_id'] for r in rows], str(selection_path) + ': CSV curve identities')
                for gamma, score in scores.items():
                    curve_rows.append({'experiment': name, 'fold': fold, 'step': step, 'area': area, 'gamma': gamma,
                                       'score': score, 'gain_vs_current': score - scores[vector[area]],
                                       'gain_vs_current_relative_pct': 100 * (score - scores[vector[area]]) / scores[vector[area]],
                                       'gain_vs_current_savings_pp': 100 * (score - scores[vector[area]]) / history_direct,
                                       'selected_gamma': selected_gamma, 'current_gamma': vector[area]})
                selection_summary = {'experiment': name, 'fold': fold, 'step': step, 'area': area,
                                     'selected_gamma': selected_gamma, 'effective_coefficient': selected_gamma / 2,
                                     'current_gamma': vector[area], 'global_gamma': global_gamma,
                                     'training_gain': state['training_gain'],
                                     'training_gain_relative_pct': 100 * state['training_gain'] / scores[vector[area]],
                                     'training_gain_savings_pp': 100 * state['training_gain'] / history_direct,
                                     'cumulative_training_gain_relative_pct': 100 * (scores[selected_gamma] - initial_score) / initial_score,
                                     'curve_range_relative_pct': 100 * (max(scores.values()) - min(scores.values())) / scores[vector[area]],
                                     'runner_up_gap_relative_pct': 100 * (sorted(scores.values(), reverse=True)[0] - sorted(scores.values(), reverse=True)[1]) / scores[vector[area]],
                                     'near_optimal_gammas': state['near_optimal_gammas'], **descriptors[area]}
                selected_rows.append(selection_summary)
                exp_selections.append(selection_summary)
                vector, dependency = selected_vector, state['selection_id']
        # Fail on extra/unrecognized candidate results rather than silently ignore them.
        actual_candidates = {str(p.relative_to(root)) for p in (root / 'results' / name).rglob('*.json') if 'result_id' in read(p)}
        require(actual_candidates == {r['relative_path'] for r in exp_rows}, name + ': unvalidated candidate files')
        # Validate even cache entries whose candidate output has not yet arrived.
        cache_times = []
        cache_by_day_vector = {}
        cache_files = list((root / 'results' / name / 'vector_cache').rglob('*.json'))
        for path in cache_files:
            payload = read(path)
            require(identity(payload['spec']) == path.stem, str(path) + ': cache filename')
            require(identity({k: v for k, v in payload.items() if k != 'payload_id'}) == payload['payload_id'], str(path) + ': cache hash')
            require(payload['spec']['code_id'] == m['code_id'] and payload['spec']['dataset_id'] == d['dataset_id'], str(path) + ': cache experiment')
            cache_times.append(payload['metrics']['time_s'])
            cache_by_day_vector[payload['spec']['day'], identity(payload['spec']['coefficient_vector'])] = (path, payload)
        # A held-out policy can be evaluated from an exact cached day/vector
        # simulation even when the separate evaluation-stage row is absent.
        # Selection still excludes this fold's day; cache creation under another
        # fold changes no policy inputs or test-day outcome.
        prefix_rows, prefix_summary = [], []
        for selection in exp_selections:
            fold, step = selection['fold'], selection['step']
            state = read(root / 'results' / name / ('fold' + fold) / f'selection{step:02d}.json')
            require(fold not in state['historical_days'], 'Held-out selection exclusion')
            cached = cache_by_day_vector.get((fold, identity(state['selected_vector'])))
            if cached is None:
                continue
            path, payload = cached
            # Require a separately validated candidate referencing these exact
            # metrics/spec/decisions, in addition to the cache content hash.
            references = [r for r in exp_rows if r.get('simulation_cache_id') == path.stem]
            require(bool(references), 'Held-out cache has no validated candidate')
            reference = base_map[fold, selection['global_gamma']]
            metrics = payload['metrics']
            prefix_rows.append({'experiment': name, 'fold': fold, 'date': d['days'][fold]['date'],
                                'completed_coordinates': step + 1, 'latest_area': state['area'],
                                'selection_id': state['selection_id'], 'historical_days': state['historical_days'],
                                'selected_vector': state['selected_vector'], 'cache_path': str(path.relative_to(root)),
                                'simulation_payload_id': payload['payload_id'],
                                'source_result_ids': [r['result_id'] for r in references],
                                **metrics, 'reference_gamma': selection['global_gamma'],
                                'reference_savings': reference['savings'],
                                'heldout_gain': metrics['savings'] - reference['savings'],
                                'heldout_gain_relative_pct': 100 * (metrics['savings'] - reference['savings']) / reference['savings'],
                                'heldout_gain_savings_pp': 100 * (metrics['savings'] - reference['savings']) / metrics['direct_distance_total']})
        for count in sorted({r['completed_coordinates'] for r in prefix_rows}):
            rows = [r for r in prefix_rows if r['completed_coordinates'] == count]
            ref_savings = sum(r['reference_savings'] for r in rows)
            gain = sum(r['heldout_gain'] for r in rows)
            prefix_summary.append({'completed_coordinates': count, **summarize_rows(rows),
                                   'heldout_gain': gain, 'heldout_gain_relative_pct': 100 * gain / ref_savings,
                                   'heldout_gain_savings_pp': 100 * gain / sum(r['direct_distance_total'] for r in rows),
                                   'positive_days': sum(r['heldout_gain'] > 1e-12 for r in rows),
                                   'negative_days': sum(r['heldout_gain'] < -1e-12 for r in rows),
                                   'unchanged_days': sum(abs(r['heldout_gain']) <= 1e-12 for r in rows)})
        table(out / (name + '_reconstructed_heldout_prefix.csv'), prefix_rows)
        audit.update(manifest_id=m['manifest_id'], dataset_id=d['dataset_id'], candidate_results=len(exp_rows),
                     selected_coordinates=len(exp_selections), numerical_runtime=runtime,
                     vector_cache_entries=len(cache_files), referenced_vector_cache_entries=len(cache_ids),
                     unique_cached_simulation_hours=sum(cache_times) / 3600,
                     validation='passed: manifests, frozen source, dataset, diagnostics, result identities, metrics, runtime, cache, historical selections and dependency chains')
        audits[name] = audit
        summaries[name] = {'manifest': m, 'global': global_summary, 'strategies': strategies,
                           'heldout_global': day_results, 'selected': exp_selections,
                           'reconstructed_heldout_prefix': prefix_rows, 'heldout_prefix_summary': prefix_summary,
                           'counts': dict(Counter((r['stage'] + (str(r['step']) if r['stage'] == 'coordinate' else '')) for r in exp_rows)),
                           'cache_runtime_median_seconds': statistics.median(cache_times) if cache_times else None,
                           'audit': {k: v for k, v in audit.items() if k != 'source_files'}}
        all_rows.extend(exp_rows)
    # Same simulator semantics in the full and reduced periodic experiments.
    full = {(r['day'], r['candidate_gamma']): r for r in all_rows if r['experiment'] == NAMES[1] and r['stage'] == 'baseline'}
    reduced = [r for r in all_rows if r['experiment'] == NAMES[2] and r['stage'] == 'baseline']
    comparisons = []
    for row in reduced:
        other = full[row['day'], .4]
        fields = ['savings', 'direct_distance_total', 'pairs', 'solos', 'mean_wait_seconds', 'max_wait_seconds', 'decision_id']
        require(all(row[k] == other[k] for k in fields), 'Periodic gamma .4 baseline cross-check: ' + row['day'])
        comparisons.append({'day': row['day'], 'metrics_and_decisions_identical': True})
    table(out / 'candidate_results.csv', all_rows)
    table(out / 'selected_areas.csv', selected_rows)
    table(out / 'complete_conditional_curves.csv', curve_rows)
    table(out / 'coverage.csv', coverage_rows)
    table(out / 'missing_coordinate_results.csv', missing_rows)
    dump(out / 'validation.json', {'experiments': audits, 'gamma_04_cross_check': comparisons, 'direct_distance_by_day': direct_by_day})
    dump(out / 'analysis.json', summaries)
    for name, result in summaries.items():
        print(name, result['counts'])
        print('global:', [(r['gamma'], round(r['savings_pct'], 6)) for r in result['global']])
        print('strategies:', [(r['strategy'], round(r['savings_pct'], 6)) for r in result['strategies']])
        for area in result['manifest']['area_order']:
            selected = [r for r in result['selected'] if r['area'] == area]
            if selected:
                print('area', area, 'folds', len(selected), 'gammas', dict(Counter(r['selected_gamma'] for r in selected)),
                      'training relative % mean/range', statistics.mean(r['training_gain_relative_pct'] for r in selected),
                      (min(r['training_gain_relative_pct'] for r in selected), max(r['training_gain_relative_pct'] for r in selected)))
    print('Validated all available candidate results; report inputs:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    analyze(args.root, args.source_root, args.output or args.root / 'results/meituan_gamma_analysis_20260909')
