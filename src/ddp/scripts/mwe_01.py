import argparse

import numpy as np

from ddp.model import Job
from ddp.scripts.run import run_instance


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run the minimal working example for delivery dispatching."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate matplotlib plots for the resulting matches.",
    )
    args = parser.parse_args(argv)

    plot = bool(args.plot)

    # Hand-crafted instance
    theta = np.array([0.2, 0.1, 0.8, 1, 0.1, 0.9, 1, 0.1], dtype=float)
    timestamps = np.arange(len(theta), dtype=float)  # arrivals 0,1,2,...
    jobs = [
        Job(origin=(0.0, 0.0), dest=(float(length), 0.0), timestamp=float(ts))
        for length, ts in zip(theta, timestamps)
    ]
    d = 3  # time window (periods)

    # shadows = ("naive", "pb", "hd")
    shadows = ("naive", "pb")
    # dispatches = ("greedy", "greedy+", "batch", "rbatch")
    dispatches = ("greedy", "batch", "rbatch")

    # Ask the core runner to return detailed matches and also print them
    result = run_instance(
        jobs=jobs,
        d=d,
        shadows=shadows,
        dispatches=dispatches,
        seed=0,
        with_opt=True,          # keep True if you want R/OPT in the table
        save_csv="",           # or "mwe_01_results.csv"
        print_table=True,
        return_details=True,    # <-- get pairs/solos per algorithm back
        print_matches=False,     # <-- also print pairs/solos after the table
    )

    # Pretty-print final matches with arrival periods (for quick debugging)
    # --- OPT summary (if computed) ---
    opt_pairs_raw = result.get("opt_pairs")
    opt_total = result.get("opt_total")
    if opt_pairs_raw:
        # result['opt_pairs'] is [(i, j, weight), ...] -> strip weights
        opt_pairs = [(i, j) for (i, j, _) in opt_pairs_raw]
        print(f"\nOPT        total={opt_total:.3f}  pairs={opt_pairs}")

    details = result.get("details", {})
    print("\nFinal matches per algorithm (pairs i-j) with arrivals and solos:")
    for sh in shadows:
        for disp in dispatches:
            key = (sh, disp)
            info = details.get(key)
            if not info:
                continue
            pairs = info["pairs"]
            solos = info["solos"]
            print(f"{sh.upper():<10} {disp:<12} pairs={pairs}  solos={solos}")

    if not plot:
        return

    # === Plot: one figure per (shadow, dispatch) ===
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        msg = (
            "Matplotlib is required for plotting. Install the optional dependencies "
            "with `pip install ddp[plot]` or, for editable installs, `pip install -e .[plot]`."
        )
        raise SystemExit(msg) from exc

    theta_arr = np.array([job.length for job in result["jobs"]], dtype=float)
    t_arr = result["timestamps"]

    # Use a deterministic palette: same color within a figure, different across algorithms
    palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', [
        'C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'
    ])

    def alg_color(sh, disp):
        i = list(shadows).index(sh)
        j = list(dispatches).index(disp)
        return palette[(i * len(dispatches) + j) % len(palette)]

    def plot_one(sh, disp):
        info = details.get((sh, disp))
        if not info:
            return
        pairs = info["pairs"]
        col = alg_color(sh, disp)

        plt.figure()
        # jobs as points (arrival on x, "type/location" on y)
        plt.scatter(t_arr, theta_arr, s=50, label="jobs", color="#666666", alpha=0.8)
        # matched pairs as line segments — force same color within this figure
        for (i, j) in pairs:
            plt.plot([t_arr[i], t_arr[j]], [theta_arr[i], theta_arr[j]], linewidth=2, alpha=0.95, color=col)
        plt.xlabel("arrival / period")
        plt.ylabel("job type / location (theta)")
        plt.title(f"{sh.upper()} + {disp}")
        plt.tight_layout()

    for sh in shadows:
        for disp in dispatches:
            plot_one(sh, disp)


    # --- Plot OPT (offline maximum) as a separate figure ---
    opt_pairs_raw = result.get("opt_pairs")
    if opt_pairs_raw:
        opt_pairs = [(i, j) for (i, j, _) in opt_pairs_raw]
        plt.figure()
        # jobs as points
        plt.scatter(t_arr, theta_arr, s=50, label="jobs", color="#666666", alpha=0.8)
        # matched pairs (single consistent color)
        for (i, j) in opt_pairs:
            plt.plot([t_arr[i], t_arr[j]], [theta_arr[i], theta_arr[j]],
                    linewidth=2, alpha=0.95, color="#000000")
        plt.xlabel("arrival / period")
        plt.ylabel("job type / location (theta)")
        plt.title("OPT (offline maximum)")
        plt.tight_layout()
    plt.show()  # or comment out if you're saving files only


if __name__ == "__main__":
    main()
