import os
from pathlib import Path
import subprocess

import pytest


def setup_submission(tmp_path, bad_reply='0'):
    name = 'meituan_area5_rbatch2_pb_120s_g6_v1'
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs' / (name + '.json')).write_text('{}')
    (tmp_path / 'configs' / (name + '_reference.csv')).write_text('test fixture')
    source = tmp_path / 'experiment_src' / name / 'ddp/scripts'
    source.mkdir(parents=True)
    (source / 'meituan_area_gamma.py').write_text('# no simulation in submission test')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    wrapper = bin_dir / 'anapy3'
    wrapper.write_text('''#!/bin/bash
set -e
printf '%s|%s\n' "$PYTHONPATH" "$*" >> "$DDP_CALL_LOG"
ddp_count=$(wc -l < "$DDP_CALL_LOG" | tr -d ' ')
if [[ "$ddp_count" == "$DDP_BAD_REPLY" ]]; then
  echo 'Submission reply without a recognizable job ID'
  exit 0
fi
case "$ddp_count" in
  1) echo 'Your job-array 8887723.1-8:1 ("python") has been submitted' ;;
  2) echo 'Your job-array 8887724.1-8:1 ("python") has been submitted' ;;
  3) echo 'Your job 8887725 ("python") has been submitted' ;;
esac
''')
    wrapper.chmod(0o755)
    return {**os.environ, 'DDP_REPO': str(tmp_path), 'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'],
            'DDP_CALL_LOG': str(tmp_path / 'calls.txt'), 'DDP_BAD_REPLY': bad_reply, 'DDP_DRY_RUN': '0'}


def test_all_stages_have_correct_dependencies_and_isolated_source(tmp_path):
    env = setup_submission(tmp_path)
    helper = Path(__file__).resolve().parents[1] / 'scripts/submit_meituan_area5_120s.sh'
    result = subprocess.run(['bash', str(helper), 'all'], env=env, text=True, capture_output=True, check=True)
    calls = (tmp_path / 'calls.txt').read_text().splitlines()
    assert len(calls) == 3
    assert all(line.startswith(str(tmp_path / 'experiment_src/meituan_area5_rbatch2_pb_120s_g6_v1') + '|') for line in calls)
    assert '--grid_hold' not in calls[0] and '_reference.csv' in calls[0]
    assert '--grid_hold=8887723' in calls[1] and '--grid_ncpus=7' in calls[1]
    assert 'fit-fold' in calls[1] and '--workers 7' in calls[1]
    assert '--grid_hold=8887724' in calls[2] and '--grid_array' not in calls[2]
    assert 'summarize' in calls[2]
    assert 'Queued reference 8887723 -> five-area fit 8887724 -> summary 8887725' in result.stdout
    assert len(list((tmp_path / 'logs').glob('*/job_id.txt'))) == 3


@pytest.mark.parametrize('bad_reply', ['1', '2'])
def test_unknown_job_id_stops_dependent_submission(tmp_path, bad_reply):
    env = setup_submission(tmp_path, bad_reply)
    helper = Path(__file__).resolve().parents[1] / 'scripts/submit_meituan_area5_120s.sh'
    result = subprocess.run(['bash', str(helper), 'all'], env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert 'No following stage was submitted' in result.stderr
    assert len((tmp_path / 'calls.txt').read_text().splitlines()) == int(bad_reply)
