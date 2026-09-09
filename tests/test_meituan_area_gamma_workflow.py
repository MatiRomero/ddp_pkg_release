import csv
import json
from pathlib import Path
import os
import subprocess
import sys
from unittest import mock

import pytest

from ddp.area_gamma import atomic_json, file_identity, identity
from ddp.scripts import meituan_area_gamma as workflow


def make_dataset(root):
    directory = root / 'data/toy'
    directory.mkdir(parents=True)
    areas = ['22', '5', '9']
    days = {}
    for day in range(8):
        rows = []
        for index in range(12):
            rows.append({'sender_lat': 0, 'sender_lng': 0, 'recipient_lat': (index % 4 + 1) / 10,
                         'recipient_lng': .2 + day / 100, 'platform_order_time': f'2020-01-0{day+1}T10:30:{index*4:02d}',
                         'da_id': areas[index % 3], 'job_id': f'{day}:{index}', 'day': str(day), 'dataset_id': 'toy'})
        path = directory / f'day{day}.csv'
        workflow.atomic_csv(path, rows)
        days[str(day)] = {'path': path.name, 'sha256': file_identity(path), 'jobs': len(rows), 'exposure_seconds': 10800}
    workflow.atomic_csv(directory / 'area_daily_arrivals.csv', [{'da_id': area, **{str(day): 4 for day in range(8)}} for area in areas])
    workflow.atomic_csv(directory / 'area_day_summary.csv', [{'da_id': area, 'day': day, 'jobs': 4, 'mean_poolable_all': day + 1} for area in areas for day in range(8)])
    data = {'dataset_id': 'toy', 'days': days, 'areas': areas,
            'diagnostic_files': {name: file_identity(directory / name) for name in ['area_daily_arrivals.csv', 'area_day_summary.csv']}}
    atomic_json(directory / 'dataset.json', data)
    return directory, data


def test_complete_staged_pilot_and_resume(tmp_path):
    directory, data = make_dataset(tmp_path)
    manifest = workflow.generate(tmp_path, 'data/toy/dataset.json', 'toy', ['22', '5', '9'], ['0'])
    root, m, _, _ = workflow.load_experiment(manifest)
    with pytest.raises(FileNotFoundError):
        workflow.run_task(manifest, 'coordinate', '1', '0', 0)
    for day in data['days']:
        workflow.run_task(manifest, 'baseline', day)
    assert len(workflow.baseline_rows(root, m, data)) == 88
    global_gamma, fit = workflow.global_fit(root, m, data, '0')
    baseline_file = workflow.result_path(root, m, 'baseline', '0', 0)
    target = workflow.read_json(baseline_file)
    changed = {**target, 'savings': target['savings'] + 1000}
    changed['result_id'] = identity({k: v for k, v in changed.items() if k != 'result_id'})
    atomic_json(baseline_file, changed)
    assert workflow.global_fit(root, m, data, '0') == (global_gamma, fit)
    atomic_json(baseline_file, target)
    with mock.patch.object(workflow.PreparedPB, 'run', side_effect=AssertionError('resume reran simulation')):
        workflow.run_task(manifest, 'baseline', '0')
    with pytest.raises(ValueError, match='Target day'):
        workflow.run_task(manifest, 'coordinate', '0', '0', 0)
    for step in range(3):
        with pytest.raises(FileNotFoundError):
            workflow.select_step(manifest, '0', step)
        for day in workflow.history_days(data, '0'):
            workflow.run_task(manifest, 'coordinate', day, '0', step)
        state = workflow.select_step(manifest, '0', step)
        assert state['training_gain'] >= -1e-12
        assert state['historical_days'] == [str(i) for i in range(1, 8)]
        assert len(state['historical_result_ids']) == 77
        assert state == workflow.select_step(manifest, '0', step)
    workflow.run_task(manifest, 'evaluation', '0', '0', 3)
    workflow.summarize(manifest)
    assert (root / 'results/toy/paired_day_savings.csv').exists()
    # All independent task output identities are unique.
    outputs = list((root / 'results/toy').rglob('day*_g*.json'))
    assert len(outputs) == 88 + 3 * 77 + 4
    assert len({workflow.read_json(p)['result_id'] for p in outputs}) == len(outputs)
    corrupt = workflow.read_json(baseline_file); corrupt['savings'] += 1
    atomic_json(baseline_file, corrupt)
    with pytest.raises(ValueError, match='content identity'):
        workflow.baseline_rows(root, m, data)


def test_historical_exposure_opportunities_and_no_history(tmp_path):
    directory, data = make_dataset(tmp_path)
    before = workflow.historical_descriptors(data, directory, '0', 60)
    assert before['22']['historical_arrivals'] == 28
    assert before['22']['nominal_density'] == pytest.approx(28 / (7 * 10800) * 60)
    assert before['22']['mean_poolable_all'] == 5
    counts = list(csv.DictReader((directory / 'area_daily_arrivals.csv').open()))
    summary = list(csv.DictReader((directory / 'area_day_summary.csv').open()))
    for row in counts:
        row['0'] = 999999
    for row in summary:
        if row['day'] == '0':
            row['mean_poolable_all'] = 999999
    workflow.atomic_csv(directory / 'area_daily_arrivals.csv', counts)
    workflow.atomic_csv(directory / 'area_day_summary.csv', summary)
    assert workflow.historical_descriptors(data, directory, '0', 60) == before
    for row in counts:
        if row['da_id'] == '22':
            for day in range(1, 8):
                row[str(day)] = 0
    summary = [row for row in summary if row['da_id'] != '22' or row['day'] == '0']
    workflow.atomic_csv(directory / 'area_daily_arrivals.csv', counts)
    workflow.atomic_csv(directory / 'area_day_summary.csv', summary)
    descriptors = workflow.historical_descriptors(data, directory, '0', 60)
    assert descriptors['22']['fallback_reason'] == 'no_historical_arrivals'
    assert descriptors['22']['observed_days'] == 7
    assert descriptors['22']['mean_poolable_all'] is None
    vectors = workflow.candidate_vectors({'area_order': ['22']}, data, dict.fromkeys(data['areas'], .4), descriptors, 0, .4)
    assert all(v['22'] == .4 for v in vectors)


def test_config_fields_and_outside_checkout_resolution(tmp_path):
    directory, _ = make_dataset(tmp_path)
    manifest = workflow.generate(tmp_path, 'data/toy/dataset.json', 'toy', ['22'], ['0'])
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(repo / 'src'), 'SGE_TASK_ID': '8'}
    cmd = [sys.executable, '-m', 'ddp.scripts.run_from_config', '--config', str(manifest.with_name('toy_baseline.csv')), '--dry-run']
    result = subprocess.run(cmd, cwd=tmp_path, env=env, text=True, capture_output=True, check=True)
    assert '--day 7' in result.stdout and str(manifest) in result.stdout
    config = tmp_path / 'configs/single.csv'
    workflow.atomic_csv(config, [{'jobs_csv': 'data/toy/day0.csv', 'save_csv': 'results/single.csv',
                                 'd': 60, 'shadows': 'pb', 'dispatch': 'rbatch', 'gamma_table': 'data/table.json',
                                 'skip_lp': 1, 'tau_s': 30}])
    env['SGE_TASK_ID'] = '1'
    result = subprocess.run([*cmd[:3], '--config', str(config), '--dry-run'], cwd=tmp_path, env=env, text=True, capture_output=True, check=True)
    assert f'--gamma-table {tmp_path}/data/table.json' in result.stdout
    assert '--skip-lp' in result.stdout and '--tau_s 30' in result.stdout


def test_bound_cache_reused_after_resume(tmp_path):
    directory, data = make_dataset(tmp_path)
    manifest = workflow.generate(tmp_path, 'data/toy/dataset.json', 'bounds', ['22'], ['0'], with_lp=True, with_opt=True)
    root, m, _, _ = workflow.load_experiment(manifest)
    from ddp.scripts.csv_loader import load_jobs_from_csv
    jobs = load_jobs_from_csv(directory / 'day0.csv')
    with mock.patch('ddp.pb_experiment.compute_lp_relaxation', return_value={'total_upper': 99}) as lp, \
         mock.patch('ddp.pb_experiment.compute_opt', return_value={'total_reward': 98}) as opt:
        first = workflow.prepare_day(root, m, data, '0', jobs, {'test': 'runtime'})
        second = workflow.prepare_day(root, m, data, '0', jobs, {'test': 'runtime'})
        assert (first.lp, first.opt) == (second.lp, second.opt) == (99, 98)
        lp.assert_called_once(); opt.assert_called_once()


def test_table_config_runs_with_labeled_jobs(tmp_path):
    directory, data = make_dataset(tmp_path)
    manifest = workflow.generate(tmp_path, 'data/toy/dataset.json', 'cli', ['22'], ['0'])
    _, m, _, _ = workflow.load_experiment(manifest)
    table = workflow.coefficient_table(m, data, '0', dict.fromkeys(data['areas'], .5), .5, 'unit-history', {})
    atomic_json(tmp_path / 'data/table.json', table)
    config = tmp_path / 'configs/single.csv'
    workflow.atomic_csv(config, [{'jobs_csv': 'data/toy/day0.csv', 'save_csv': 'results/single.csv',
                                 'd': 60, 'shadows': 'pb', 'dispatch': 'rbatch', 'gamma_table': 'data/table.json', 'skip_lp': 1}])
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(repo / 'src'), 'SGE_TASK_ID': '1'}
    subprocess.run([sys.executable, '-m', 'ddp.scripts.run_from_config', '--config', str(config)],
                   cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    row, = csv.DictReader((tmp_path / 'results/single.csv').open())
    assert row['gamma_mode'] == 'area_table' and row['job_day'] == '0'
    assert json.loads(row['coefficient_vector']) == table['coefficients']
    assert row['code_id'] == workflow.code_identity()


def periodic_experiment(root, experiment_id='periodic', folds=None):
    return workflow.generate(root, 'data/toy/dataset.json', experiment_id, ['all'], folds or ['0', '1'],
                             window=180, dispatch='rbatch2', tau_s=30,
                             gammas=(0, .2, .4, .6, .8, 1), reference_gamma=.4)


def test_periodic_parallel_fold_runner_and_resume(tmp_path, monkeypatch):
    directory, data = make_dataset(tmp_path)
    manifest = periodic_experiment(tmp_path)
    root, m, _, _ = workflow.load_experiment(manifest)
    monkeypatch.setenv('NSLOTS', '2')
    with pytest.raises(FileNotFoundError):
        workflow.fit_fold(manifest, '0', workers=2)
    for day in data['days']:
        workflow.run_task(manifest, 'baseline', day)
    baseline = workflow.baseline_rows(root, m, data)
    assert len(baseline) == 48
    assert {r['dispatch'] for r in baseline} == {'rbatch2'}
    assert all(r['tau_s'] == 30 and r['d'] == 180 and r['max_wait_seconds'] <= 180 for r in baseline)
    for fold in m['folds']:
        workflow.fit_fold(manifest, fold, workers=2)
        for step, area in enumerate(data['areas']):
            state = workflow.read_json(root / f'results/periodic/fold{fold}/selection{step:02d}.json')
            assert state['area'] == area
            assert set(map(float, state['scores'])) == set(m['gamma_grid'])
            assert len(state['historical_result_ids']) == 42
            assert fold not in state['historical_days'] and len(state['historical_days']) == 7
            assert state['training_gain'] >= -1e-12
            assert state['coefficient_table']['dispatch'] == 'rbatch2'
            assert state['coefficient_table']['tau_s'] == 30
            assert set(state['selected_vector'].values()) <= set(m['gamma_grid'])
    workflow.summarize(manifest)
    paired = list(csv.DictReader((root / 'results/periodic/paired_day_savings.csv').open()))
    assert len(paired) == 2 and 'fixed_global_0.4' in paired[0] and 'plain_rbatch2' in paired[0]
    outcomes = {str(p): p.read_bytes() for p in (root / 'results/periodic').rglob('day*_g*.json')}
    assert len(outcomes) == 48 + 2 * 3 * 42 + 2 * 4
    workflow.fit_fold(manifest, '0', workers=2)
    assert all(Path(p).read_bytes() == payload for p, payload in outcomes.items())
    with pytest.raises(ValueError, match='NSLOTS'):
        workflow.fit_fold(manifest, '0', workers=3)
    monkeypatch.setenv('SGE_TASK_ID', '2')
    assert workflow.array_fold(manifest) == '1'
    monkeypatch.setenv('SGE_TASK_ID', 'undefined')
    with pytest.raises(ValueError, match='numeric SGE_TASK_ID'):
        workflow.array_fold(manifest)
    descriptors = workflow.historical_descriptors(data, directory, '0', 180)
    assert descriptors['22']['nominal_density'] == pytest.approx(28 / (7 * 10800) * 180)
    assert descriptors['22']['opportunity_window_seconds'] == 60
    assert descriptors['22']['opportunities_match_waiting_window'] is False


def test_vector_cache_reuses_decisions_but_keeps_fold_provenance(tmp_path):
    directory, data = make_dataset(tmp_path)
    manifest = periodic_experiment(tmp_path)
    root, m, _, _ = workflow.load_experiment(manifest)
    from ddp.scripts.csv_loader import load_jobs_from_csv
    jobs = load_jobs_from_csv(directory / 'day2.csv')
    prepared = workflow.PreparedPB(jobs, 180, dispatch='rbatch2', tau_s=30)
    vector = {'22': .2, '5': .6, '9': 1.}
    table0 = workflow.coefficient_table(m, data, '0', vector, .4, 'fold0-history', {})
    table1 = workflow.coefficient_table(m, data, '1', vector, .4, 'fold1-history', {})
    runtime = {'test': 'same-environment'}
    first = workflow.run_cached(root, m, data, '2', vector, prepared, runtime, table=table0)
    assert not first['simulation_cache_hit']
    with mock.patch.object(prepared, 'run', side_effect=AssertionError('recomputed identical decisions')):
        second = workflow.run_cached(root, m, data, '2', vector, prepared, runtime, table=table1)
    assert second['simulation_cache_hit']
    assert first['decision_id'] == second['decision_id']
    assert first['savings'] == second['savings']
    assert first['gamma_table_id'] != second['gamma_table_id']
    assert first['target_day'] == '0' and second['target_day'] == '1'
    assert first['historical_days'] != second['historical_days']
    path = root / f'results/periodic/vector_cache/day2/{first["simulation_cache_id"]}.json'
    corrupt = workflow.read_json(path)
    corrupt['metrics']['savings'] += 1
    atomic_json(path, corrupt)
    with pytest.raises(ValueError, match='cache provenance/content'):
        workflow.run_cached(root, m, data, '2', vector, prepared, runtime, table=table1)


def test_fixed_reference_subset_pilot(tmp_path, monkeypatch):
    _, data = make_dataset(tmp_path)
    manifest = workflow.generate(tmp_path, 'data/toy/dataset.json', 'fixed', ['22', '5'], ['0', '1'],
                                 window=120, dispatch='rbatch2', tau_s=30,
                                 gammas=(0., .2, .4, .6, .8, 1.), reference_gamma=.4,
                                 initialization='fixed_reference')
    root, m, _, _ = workflow.load_experiment(manifest)
    assert workflow.baseline_candidates(m) == [2]
    assert manifest.with_name('fixed_reference.csv').exists()
    assert not manifest.with_name('fixed_baseline.csv').exists()
    with pytest.raises(ValueError, match='only the fixed reference'):
        workflow.run_task(manifest, 'baseline', '0', candidate_index=0)
    for day in data['days']:
        workflow.run_task(manifest, 'baseline', day)
    assert len(workflow.baseline_rows(root, m, data)) == 8
    initial, global_gamma, _, fit = workflow.current_state(root, m, data, '0', 0)
    assert initial == dict.fromkeys(data['areas'], .4) and global_gamma == .4
    assert fit['initialization'] == 'fixed_reference'
    assert '0' not in fit['historical_days']
    reference = workflow.result_path(root, m, 'baseline', '0', 2)
    target = workflow.read_json(reference)
    changed = {**target, 'savings': 1e9}
    changed['result_id'] = identity({k:v for k,v in changed.items() if k != 'result_id'})
    atomic_json(reference, changed)
    assert workflow.global_fit(root, m, data, '0') == (global_gamma, fit)
    atomic_json(reference, target)
    monkeypatch.setenv('NSLOTS', '2')
    for fold in m['folds']:
        workflow.fit_fold(manifest, fold, workers=2)
        for step in range(2):
            state = workflow.read_json(root / f'results/fixed/fold{fold}/selection{step:02d}.json')
            assert len(state['historical_result_ids']) == 42
            assert set(map(float, state['scores'])) == set(m['gamma_grid'])
            assert state['selected_vector']['9'] == .4
            if step:
                previous = workflow.read_json(root / f'results/fixed/fold{fold}/selection00.json')
                assert state['input_vector'] == previous['selected_vector']
        saved = {str(p): p.read_bytes() for p in (root / f'results/fixed/fold{fold}').rglob('*.json')}
        workflow.fit_fold(manifest, fold, workers=2)
        assert all(Path(p).read_bytes() == value for p,value in saved.items())
    workflow.summarize(manifest)
    outcomes = list((root / 'results/fixed').rglob('day*_g*.json'))
    assert len(outcomes) == 8 + 2 * 2 * 42 + 2 * 2
    paired = list(csv.DictReader((root / 'results/fixed/paired_day_savings.csv').open()))
    assert len(paired) == 2 and 'historical_global' not in paired[0]
    for row in paired:
        assert float(row['area_minus_fixed_0.4']) == pytest.approx(float(row['historical_area']) - float(row['fixed_global_0.4']))
    selected = list(csv.DictReader((root / 'results/fixed/selected_area_gammas.csv').open()))
    assert all(float(r['gamma']) == .4 and r['swept'] == 'False' for r in selected if r['area'] == '9')
    assert (root / 'results/fixed/reference_summary.csv').exists()
