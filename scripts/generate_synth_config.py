"""Generate configuration CSVs for synthetic job archives.

This helper enumerates seeds and ``d`` values, pairing each seed with a
pre-generated ``.npz`` instance. The resulting CSV feeds directly into
``ddp.scripts.run_from_config`` so cluster launches can select rows via
``$SGE_TASK_ID``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


_DEFAULT_SHADOWS: Sequence[str] = ["naive", "pb", "hd", "ad"]


def _parse_ints(payload: str) -> list[int]:
    """Parse comma-separated integers or range like '1:100:1'."""
    payload = payload.strip()
    if ":" in payload:
        # Range format: start:stop:step
        parts = payload.split(":")
        if len(parts) == 2:
            start, stop = int(parts[0]), int(parts[1])
            return list(range(start, stop + 1))
        elif len(parts) == 3:
            start, stop, step = int(parts[0]), int(parts[1]), int(parts[2])
            return list(range(start, stop + 1, step))
        else:
            raise ValueError(f"Invalid range format: {payload}")
    return [int(token.strip()) for token in payload.replace(";", ",").split(",") if token.strip()]


def _parse_floats(payload: str) -> list[float]:
    return [float(token.strip()) for token in payload.replace(";", ",").split(",") if token.strip()]


def _sanitise(value: object) -> str:
    return str(value).strip().replace("/", "_")


def _iter_rows(
    *,
    seeds: Iterable[int],
    d_values: Iterable[float],
    shadows: Iterable[str],
    jobs_pattern: str | None,
    dispatch: str,
    results_dir: Path,
    include_job_details: bool,
    generate_on_fly: bool = False,
    n: int | None = None,
    fix_origin_zero: bool = False,
    flatten_axis: str | None = None,
    ad_data_dir: str | None = None,
    ad_base_path: str | None = None,
    with_opt: bool = False,
    beta_alpha: float = 1.0,
    beta_beta: float = 1.0,
    ad_mapping: str | None = None,
) -> Iterable[dict[str, str]]:
    for seed in seeds:
        for d in d_values:
            for shadow in shadows:
                base_name = f"seed{seed}_d{_sanitise(d)}_{shadow}_{dispatch}"
                save_path = results_dir / f"{base_name}.csv"
                job_details_path = results_dir / f"{base_name}_jobs.csv"
                
                if generate_on_fly:
                    if n is None:
                        raise ValueError("--n required when --generate-on-fly is used")
                    row = {
                        "n": str(n),
                        "d": str(d),
                        "shadows": shadow,
                        "dispatch": dispatch,
                        "seed": str(seed),
                        "save_csv": str(save_path),
                        "save_job_csv": str(job_details_path) if include_job_details else "",
                    }
                    if fix_origin_zero:
                        row["fix_origin_zero"] = "1"
                    if flatten_axis:
                        row["flatten_axis"] = flatten_axis
                    # Always include beta parameters (defaults to 1.0, 1.0 for uniform)
                    row["beta_alpha"] = str(beta_alpha)
                    row["beta_beta"] = str(beta_beta)
                else:
                    if jobs_pattern is None:
                        raise ValueError("--jobs-pattern required when not using --generate-on-fly")
                    jobs_path = Path(jobs_pattern.format(seed=seed))
                    row = {
                        "jobs": str(jobs_path),
                        "d": str(d),
                        "shadows": shadow,
                        "dispatch": dispatch,
                        "seed": str(seed),
                        "save_csv": str(save_path),
                        "save_job_csv": str(job_details_path) if include_job_details else "",
                    }
                
                # Add AD parameters if shadow is 'ad'
                if shadow == "ad" and ad_data_dir:
                    d_int = int(d)
                    n_value = n if n is not None else 1000  # Default to 1000 if n not specified
                    # Determine suffix based on geometry
                    if fix_origin_zero and flatten_axis:
                        suffix = "_1d_common_origin"
                    elif fix_origin_zero:
                        suffix = "_common_origin"
                    else:
                        suffix = ""
                    if ad_base_path:
                        ad_duals_path = f"{ad_base_path}/{ad_data_dir}/ad_uniform_grid_n{n_value}{suffix}_d{d_int}.csv"
                    else:
                        ad_duals_path = f"{ad_data_dir}/ad_uniform_grid_n{n_value}{suffix}_d{d_int}.csv"
                    row["ad_duals"] = ad_duals_path
                    row["ad_mapping"] = ad_mapping if ad_mapping is not None else "ddp.mappings.uniform_grid:job_mapping"
                else:
                    row["ad_duals"] = ""
                    row["ad_mapping"] = ""
                
                # Add with_opt if requested
                if with_opt:
                    row["with_opt"] = "1"
                
                yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Label for the generated CSV and results directory (defaults to timestamp)",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs"),
        help="Directory to store the generated config CSV",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "synth_runs",
        help="Base directory for run outputs (a subdir using --config-name is created)",
    )
    parser.add_argument(
        "--jobs-pattern",
        default=None,
        help="Path template for the synthetic job archives (must include {seed}). Required unless --generate-on-fly is used.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma- or semicolon-separated seeds to enumerate, or range like '1:100:1'",
    )
    parser.add_argument(
        "--d-values",
        default="30,60,90",
        help="Comma- or semicolon-separated d values to sweep",
    )
    parser.add_argument(
        "--shadows",
        default=",".join(_DEFAULT_SHADOWS),
        help="Comma-separated list of shadow policies (or use --policies for shadow+dispatch combinations)",
    )
    parser.add_argument(
        "--dispatch",
        default="batch",
        help="Dispatch policy for every row (or comma-separated list for multiple dispatches). Ignored if --policies is used.",
    )
    parser.add_argument(
        "--policies",
        help="Comma-separated list of shadow+dispatch combinations (e.g., 'naive+greedy,pb+greedy'). Overrides --shadows and --dispatch.",
    )
    parser.add_argument(
        "--generate-on-fly",
        action="store_true",
        help="Generate config with 'n' column instead of 'jobs' paths (generates jobs on the fly)",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Number of jobs to generate (required when --generate-on-fly is used)",
    )
    parser.add_argument(
        "--fix-origin-zero",
        action="store_true",
        help="Set every generated job origin to the depot at (0, 0)",
    )
    parser.add_argument(
        "--flatten-axis",
        choices=["x", "y"],
        help="Project all jobs onto a single axis by zeroing the chosen coordinate",
    )
    parser.add_argument(
        "--beta-alpha",
        type=float,
        default=1.0,
        help="Alpha parameter for Beta distribution (default: 1.0, which gives uniform distribution)",
    )
    parser.add_argument(
        "--beta-beta",
        type=float,
        default=1.0,
        help="Beta parameter for Beta distribution (default: 1.0, which gives uniform distribution)",
    )
    parser.add_argument(
        "--no-job-details",
        dest="include_job_details",
        action="store_false",
        help="Leave the save_job_csv column empty",
    )
    parser.add_argument(
        "--ad-data-dir",
        default="data",
        help="Directory containing AD lookup files (relative to repo root or --ad-base-path)",
    )
    parser.add_argument(
        "--ad-base-path",
        default=None,
        help="Base path prefix for absolute AD file paths (e.g., /user/mer2262/ddp_pkg_release). If not provided, uses relative paths.",
    )
    parser.add_argument(
        "--ad-mapping",
        default=None,
        help="Module:function specification for AD mapping (default: ddp.mappings.uniform_grid:job_mapping). Use ddp.mappings.uniform_grid:fine_job_mapping for fine grid.",
    )
    parser.add_argument(
        "--with-opt",
        action="store_true",
        help="Compute OPT (offline optimal) for regret/ratio calculations",
    )
    parser.set_defaults(include_job_details=True)

    args = parser.parse_args()

    # Validate arguments
    if args.generate_on_fly:
        if args.n is None:
            parser.error("--n is required when --generate-on-fly is used")
        if args.jobs_pattern:
            parser.error("--jobs-pattern cannot be used with --generate-on-fly")
    else:
        if args.jobs_pattern is None:
            parser.error("--jobs-pattern is required when --generate-on-fly is not used")
        if args.n is not None:
            parser.warning("--n is ignored when --generate-on-fly is not used")

    configs_dir: Path = args.configs_dir
    configs_dir.mkdir(parents=True, exist_ok=True)

    results_dir: Path = Path(args.results_dir) / args.config_name
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_ints(args.seeds)
    d_values = _parse_floats(args.d_values)
    
    # Parse policies: either from --policies (shadow+dispatch) or from --shadows + --dispatch
    if args.policies:
        # Parse shadow+dispatch combinations
        policy_combos = []
        for policy_str in args.policies.split(","):
            policy_str = policy_str.strip()
            if "+" in policy_str:
                shadow, dispatch = policy_str.split("+", 1)
                policy_combos.append((shadow.strip(), dispatch.strip()))
            else:
                parser.error(f"Invalid policy format '{policy_str}'. Expected 'shadow+dispatch' (e.g., 'naive+greedy')")
        
        rows = []
        for shadow, dispatch in policy_combos:
            rows.extend(
                _iter_rows(
                    seeds=seeds,
                    d_values=d_values,
                    shadows=[shadow],
                    jobs_pattern=args.jobs_pattern,
                    dispatch=dispatch,
                    results_dir=results_dir,
                    include_job_details=args.include_job_details,
                    generate_on_fly=args.generate_on_fly,
                    n=args.n,
                    fix_origin_zero=args.fix_origin_zero,
                    flatten_axis=args.flatten_axis,
                    ad_data_dir=args.ad_data_dir,
                    ad_base_path=args.ad_base_path,
                    with_opt=args.with_opt,
                    beta_alpha=args.beta_alpha,
                    beta_beta=args.beta_beta,
                    ad_mapping=args.ad_mapping,
                )
            )
    else:
        shadows = [token.strip() for token in args.shadows.split(",") if token.strip()]
        dispatches = [token.strip() for token in args.dispatch.split(",") if token.strip()]

        rows = []
        for dispatch in dispatches:
            rows.extend(
                _iter_rows(
                    seeds=seeds,
                    d_values=d_values,
                    shadows=shadows,
                    jobs_pattern=args.jobs_pattern,
                    dispatch=dispatch,
                    results_dir=results_dir,
                    include_job_details=args.include_job_details,
                    generate_on_fly=args.generate_on_fly,
                    n=args.n,
                    fix_origin_zero=args.fix_origin_zero,
                    flatten_axis=args.flatten_axis,
                    ad_data_dir=args.ad_data_dir,
                    ad_base_path=args.ad_base_path,
                    with_opt=args.with_opt,
                    beta_alpha=args.beta_alpha,
                    beta_beta=args.beta_beta,
                    ad_mapping=args.ad_mapping,
                )
            )

    config_path = configs_dir / f"config_synth_{args.config_name}.csv"
    
    # Build headers dynamically based on mode
    base_headers = ["d", "shadows", "dispatch", "seed", "save_csv", "save_job_csv"]
    if args.generate_on_fly:
        headers = ["n"] + base_headers
        if args.fix_origin_zero:
            headers.append("fix_origin_zero")
        if args.flatten_axis:
            headers.append("flatten_axis")
        # Always include beta parameters (defaults to 1.0, 1.0 for uniform)
        headers.extend(["beta_alpha", "beta_beta"])
    else:
        headers = ["jobs"] + base_headers
    
    # Add AD columns if any row has them (check for key existence, not just non-empty values)
    if rows and ("ad_duals" in rows[0] or "ad_mapping" in rows[0]):
        headers.extend(["ad_duals", "ad_mapping"])
    
    # Add with_opt column if requested
    if args.with_opt:
        headers.append("with_opt")

    with config_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {config_path} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
