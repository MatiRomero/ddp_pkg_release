import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ddp.engine.sim import simulate
from ddp.model import Job
from ddp.scripts.run import make_local_score


def test_greedyx_prescreens_negative_reward_partners():
    jobs = [
        Job((0.0, 0.0), (0.0, 1.0), timestamp=0.0),
        Job((0.0, 0.0), (10.0, 0.0), timestamp=0.0),
        Job((0.0, 0.0), (0.0, 2.0), timestamp=0.0),
    ]

    reward_lookup = {
        (0, 1): -0.1,
        (1, 0): -0.1,
        (0, 2): 0.2,
        (2, 0): 0.2,
        (1, 2): -0.2,
        (2, 1): -0.2,
    }

    def reward_fn(i: int, j: int, _jobs):
        return reward_lookup[(i, j)]

    sp = np.array([0.0, -10.0, 0.0])
    score_fn = make_local_score(reward_fn, sp)

    greedy = simulate(
        jobs,
        score_fn,
        reward_fn,
        "naive",
        time_window=0.0,
        policy="score",
        weight_fn=None,
        shadow=None,
        seed=1,
    )
    greedyx = simulate(
        jobs,
        score_fn,
        reward_fn,
        "prescreen",
        time_window=0.0,
        policy="score",
        weight_fn=None,
        shadow=None,
        seed=1,
    )

    assert greedy["pairs"] == []
    assert greedyx["pairs"]
    matched = {greedyx["pairs"][0][0], greedyx["pairs"][0][1]}
    assert matched == {0, 2}
