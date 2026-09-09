from dataclasses import replace
from unittest import mock
import numpy as np
import pytest

from ddp.area_gamma import resolve_area_gammas, select_gamma
from ddp.pb_experiment import PreparedPB
from ddp.scripts.run import run_instance, run_once, make_weight_fn
from ddp.engine.sim import simulate
from test_area_gamma import jobs, table

GRID6 = (0., .2, .4, .6, .8, 1.)


def periodic_table(gamma=.4, window=180, period=30):
    result = table(gamma)
    result.update(dispatch='rbatch2', window_seconds=window, tau_s=period)
    return result


@pytest.mark.parametrize('gamma', GRID6)
def test_periodic_table_and_scalar_equivalence(gamma):
    js = [replace(j, timestamp=float(i * 17)) for i,j in enumerate(jobs())]
    params = dict(d=180, shadows=('pb',), dispatches=('rbatch2',), tau_s=30,
                  seed=0, print_table=False, return_details=True, skip_lp=True)
    scalar = run_instance(js, gamma=gamma, **params)
    local = run_instance(js, gamma_table=periodic_table(gamma), **params)
    assert scalar['details'] == local['details']
    assert scalar['rows'][0]['savings'] == local['rows'][0]['savings']
    once = run_once(len(js), 180, 0, 'pb', 'rbatch2', jobs=js, gamma_table=periodic_table(gamma), skip_lp=True)
    assert once['savings'] == scalar['rows'][0]['savings']
    row, detail = PreparedPB(js, 180, dispatch='rbatch2', tau_s=30).run(gamma_table=periodic_table(gamma))
    expected = next(iter(scalar['details'].values()))
    assert [(i,j) for i,j,*_ in detail['pairs']] == expected['pairs']
    assert detail['dispatch_times'] == expected['dispatch_times']
    assert row['savings'] == scalar['rows'][0]['savings']
    assert row['tau_s'] == 30 and row['max_wait_seconds'] <= 180
    for i,t in detail['dispatch_times'].items():
        assert t % 30 == 0 and js[i].timestamp <= t <= js[i].timestamp + 180
    if gamma == 0:
        plain = run_instance(js, 180, shadows=('naive',), dispatches=('rbatch2',), tau_s=30,
                             print_table=False, return_details=True, skip_lp=True)
        assert next(iter(plain['details'].values())) == expected


def test_periodic_critical_shadow_addback_and_early_dispatch():
    js = [replace(jobs()[i], timestamp=t) for i,t in enumerate([0,20])]
    shadows = np.array([.2,.8])
    reward = lambda i,j,js: 2.
    result = simulate(js, None, reward, time_window=60, policy='rbatch2', tau_s=30,
                      weight_fn=make_weight_fn(reward,shadows), shadow=shadows)
    assert result['dispatch_times'] == {0:30.,1:30.}
    assert result['pairs'][0][2] == pytest.approx(1.2)
    # With no profitable match, the deadline-180 job leaves at tick 150.
    result = simulate([js[0]], None, reward, time_window=180, policy='rbatch2', tau_s=30)
    assert result['dispatch_times'] == {0:150.}


@pytest.mark.parametrize('period', [0, -1, 181, float('nan')])
def test_periodic_interval_validation(period):
    with pytest.raises(ValueError, match='interval'):
        PreparedPB(jobs(),180,dispatch='rbatch2',tau_s=period)
    with pytest.raises(ValueError, match='interval'):
        resolve_area_gammas(jobs(), periodic_table(period=period), d=180, dispatches=('rbatch2',),tau_s=period)


def test_table_policy_and_period_must_match():
    with pytest.raises(ValueError, match='interval mismatch'):
        resolve_area_gammas(jobs(), periodic_table(), d=180, dispatches=('rbatch2',),tau_s=60)
    with pytest.raises(ValueError, match='dispatch mismatch'):
        resolve_area_gammas(jobs(), periodic_table(), d=180)
    scores=dict.fromkeys(GRID6, 1.)
    assert select_gamma(scores,.4,grid=GRID6)==.4
    del scores[.2]
    with pytest.raises(ValueError, match='all 6'):
        select_gamma(scores,.4,grid=GRID6)
