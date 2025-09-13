import numpy as np
from ddp.engine.sim import simulate

def reward_fn(i, j, theta):
    return min(theta[i], theta[j])

def score_pb(i, j, theta):
    # s_j = theta[j]/2  (simple potential)
    return reward_fn(i, j, theta) - 0.5 * theta[j]

def main():
    theta = np.array([0.9, 0.2, 0.8, 0.1], dtype=float)
    timestamps = np.arange(len(theta), dtype=float)
    d = 1.0

    for rule in ("naive", "threshold"):
        res = simulate(
            theta, score_fn=score_pb, reward_fn=reward_fn,
            decision_rule=rule, timestamps=timestamps, time_window=d,
            policy="score", seed=0
        )
        pairs_only = [(i, j) for (i, j, _, _) in res["pairs"]]
        print(f"{rule.upper():<10} pairs={pairs_only}  savings={res['total_savings']:.3f}  pooled%={res['pooled_pct']:.1f}%")

if __name__ == "__main__":
    main()
