from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from ddp.area_gamma import resolve_area_gammas, select_gamma, GRID, identity
from ddp.model import Job
from ddp.scripts.csv_loader import load_jobs_from_csv
from ddp.scripts.run import run_instance, run_once, make_weight_fn
from ddp.pb_experiment import PreparedPB
from ddp.engine.sim import simulate


def jobs():
    rng = np.random.default_rng(9)
    return [Job(tuple(rng.random(2)), tuple(rng.random(2)), float(i * 4),
                str(i % 3), str(i), '0', 'dataset') for i in range(24)]


def table(gamma=.5):
    return {'schema_version': 1, 'grouping': 'da_id', 'matching_scope': 'citywide',
            'shadow': 'pb', 'dispatch': 'rbatch', 'tau': 0, 'window_seconds': 60,
            'target_day': '0', 'historical_days': [str(i) for i in range(1, 8)],
            'dataset_id': 'dataset', 'dataset_days': [str(i) for i in range(8)], 'training_manifest_id': 'training',
            'coefficients': {str(i): gamma for i in range(3)},
            'fallback_gamma': .4, 'missing_area_policy': 'error'}


@pytest.mark.parametrize('gamma', [0, .1, .5, 1])
def test_constant_table_scalar_equivalence(gamma):
    kwargs = dict(d=60, shadows=('pb',), dispatches=('rbatch',), seed=3,
                  skip_lp=True, print_table=False, return_details=True)
    scalar = run_instance(jobs(), gamma=gamma, **kwargs)
    local = run_instance(jobs(), gamma_table=table(gamma), **kwargs)
    assert scalar['details'] == local['details']  # pairs, solos, exact dispatch times
    for key in ['savings', 'pairs', 'solos', 'pooled_pct']:
        assert scalar['rows'][0][key] == local['rows'][0][key]
    prepared = PreparedPB(jobs())
    row, detail = prepared.run(gamma_table=table(gamma), seed=3)
    assert row['savings'] == scalar['rows'][0]['savings']
    assert detail['dispatch_times'] == next(iter(local['details'].values()))['dispatch_times']
    once = run_once(len(jobs()), 60, 3, 'pb', 'rbatch', jobs=jobs(), gamma_table=table(gamma), skip_lp=True)
    assert once['savings'] == row['savings']
    if gamma == 0:
        plain = run_instance(jobs(), d=60, shadows=('naive',), dispatches=('rbatch',),
                             seed=3, skip_lp=True, print_table=False, return_details=True)
        assert next(iter(plain['details'].values())) == next(iter(local['details'].values()))
        assert plain['rows'][0]['savings'] == row['savings']


def test_cross_area_endpoint_and_critical_addback():
    js = [Job((0, 0), (2, 0), 0, '0', 'a', '0', 'dataset'),
          Job((0, 0), (2, 0), 30, '1', 'b', '0', 'dataset')]
    t = table(); t['coefficients'] = {'0': .2, '1': .8}
    gammas, _ = resolve_area_gammas(js, t, d=60)
    shadows = gammas * np.array([j.length for j in js]) / 2
    reward = lambda i, j, jobs: 2.
    weight = make_weight_fn(reward, shadows)
    assert weight(0, 1, js) == pytest.approx(1.)
    res = simulate(js, None, reward, time_window=60, policy='rbatch', weight_fn=weight, shadow=shadows)
    assert res['pairs'][0][2] == pytest.approx(1.2)  # 2 - .2 - .8 + .2
    assert res['dispatch_times'] == {0: 60., 1: 60.}


@pytest.mark.parametrize('key,value', [('window_seconds', 30), ('grouping', 'h3'),
    ('dataset_id', 'wrong'), ('dispatch', 'rbatch+'), ('matching_scope', 'within_area'),
    ('historical_days', ['0','1']), ('historical_days', ['1','1']), ('tau', 1),
    ('fallback_gamma', float('nan')), ('fallback_gamma', 2), ('training_manifest_id', ''),
    ('coefficients', {'0': float('inf')}), ('coefficients', {'0': -.1}), ('coefficients', {'0': True})])
def test_table_validation(key, value):
    t = table(); t[key] = value
    with pytest.raises(ValueError):
        resolve_area_gammas(jobs(), t, d=60)


@pytest.mark.parametrize('kwargs', [{'gamma': .5}, {'dispatches': ('rbatch+',)},
                                   {'shadows': ('naive',)}, {'tau': .1}, {'reward_type': 'rewardB'}])
def test_unsupported_table_combinations(kwargs):
    with pytest.raises(ValueError):
        resolve_area_gammas(jobs(), table(), d=60, **kwargs)


def test_metadata_alignment_and_fallback():
    text = '''sender_lat,sender_lng,recipient_lat,recipient_lng,platform_order_time,da_id,job_id,day,dataset_id
0,0,1,1,2020-01-01T12:00:10,2,b,0,dataset
0,0,2,2,2020-01-01T12:00:00,0,a,0,dataset
0,0,3,3,2020-01-01T12:00:00,1,c,0,dataset
'''
    js = load_jobs_from_csv(StringIO(text))
    assert [j.job_id for j in js] == ['a', 'c', 'b']
    assert [j.timestamp for j in js] == [0, 0, 10]
    assert js[0].original_timestamp == '2020-01-01T12:00:00'
    t = table(); t['coefficients'] = {'0': .1, '1': .7}
    with pytest.raises(ValueError, match='Missing area'):
        resolve_area_gammas(js, t, d=60)
    t['missing_area_policy'] = 'historical_global'
    values, provenance = resolve_area_gammas(js, t, d=60)
    np.testing.assert_array_equal(values, [.1, .7, .4])
    assert provenance['fallback_jobs'] == 1
    assert json.loads(provenance['fallback_areas']) == ['2']
    for invalid in [text.replace(',2,b,', ',,b,'), text.replace(',2,b,', ',2,a,'), text.replace('0,0,1,1,', ',0,1,1,')]:
        with pytest.raises(ValueError):
            load_jobs_from_csv(StringIO(invalid))
    with pytest.raises(ValueError, match='identities'):
        resolve_area_gammas([replace(js[0], job_id=None)], t, d=60)


def test_duplicate_table_keys(tmp_path):
    path = tmp_path / 'table.json'
    path.write_text(json.dumps(table()).replace('"0": 0.5', '"0": 0.5, "0": 0.7'))
    with pytest.raises(ValueError, match='Duplicate'):
        resolve_area_gammas(jobs(), path, d=60)


def test_ties_and_incomplete_grid():
    flat = dict.fromkeys(GRID, 1.)
    assert select_gamma(flat, .7) == .7
    curve = dict.fromkeys(GRID, 0.); curve[.2] = curve[.8] = 2
    assert select_gamma(curve, .5) == .2
    curve[.6] = 2
    assert select_gamma(curve, .5) == .6
    del curve[0]
    with pytest.raises(ValueError, match='all 11'):
        select_gamma(curve, .5)


def test_bounds_are_opt_in_and_reused():
    with mock.patch('ddp.pb_experiment.compute_lp_relaxation', return_value={'total_upper': 99}) as lp, \
         mock.patch('ddp.pb_experiment.compute_opt', return_value={'total_reward': 98}) as opt:
        p = PreparedPB(jobs())
        p.run(gamma=0); p.run(gamma=.5)
        lp.assert_not_called(); opt.assert_not_called()
        p = PreparedPB(jobs(), with_lp=True, with_opt=True)
        p.run(gamma=0); p.run(gamma=1)
        lp.assert_called_once(); opt.assert_called_once()
    with mock.patch('ddp.scripts.run.compute_lp_relaxation') as lp:
        run_instance(jobs(), 60, shadows=('pb',), dispatches=('rbatch',), skip_lp=True, print_table=False)
        lp.assert_not_called()
    with pytest.raises(ValueError, match='HD requires'):
        run_instance(jobs(), 60, shadows=('hd',), skip_lp=True)


def test_incomplete_history_and_false_effective_coefficient():
    t = table(); t['historical_days'] = ['1', '2']
    with pytest.raises(ValueError, match='exactly the other'):
        resolve_area_gammas(jobs(), t, d=60)
    t = table(); t['effective_coefficients'] = {'0': .5}
    with pytest.raises(ValueError, match='gamma / 2'):
        resolve_area_gammas(jobs(), t, d=60)
    t = table(); t['support'] = {'0': {'historical_arrivals': 0}}
    with pytest.raises(ValueError, match='without history'):
        resolve_area_gammas(jobs(), t, d=60)
    curve = dict.fromkeys(GRID, 0.); curve[.4] = curve[.6] = 1
    assert select_gamma(curve, .5) == .4


def test_legacy_csv_with_incidental_area_columns_keeps_scalar_loading():
    content = "sender_lat,sender_lng,recipient_lat,recipient_lng,platform_order_time,da_id,day\n0,0,1,1,2020-01-01T10:30:00,2,0\n"
    js = load_jobs_from_csv(StringIO(content))
    assert len(js) == 1 and js[0].da_id == '2' and js[0].day == '0'
    with pytest.raises(ValueError, match='identities'):
        resolve_area_gammas(js, table(), d=60)
