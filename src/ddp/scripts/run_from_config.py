# src/ddp/scripts/run_from_config.py
import os, sys, csv, subprocess, argparse, pathlib

from ddp.results.append_csv import append_csv_locked

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.csv", help="Path to config CSV")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved command and exit")
    parser.add_argument(
        "--combined-csv",
        default=None,
        help=(
            "Optional path to a shared CSV that will receive the per-task results "
            "after a successful run."
        ),
    )
    parser.add_argument(
        "--keep-individual",
        action="store_true",
        help="Keep each task's CSV output after appending to the combined file.",
    )
    # Everything else (fixed flags) is forwarded to ddp.scripts.run:
    args, forward = parser.parse_known_args()

    # Pick row by SGE task id (1-based)
    sge_task_id = int(os.environ.get("SGE_TASK_ID", "1"))
    row_idx = sge_task_id - 1

    with open(args.config, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows found in {args.config}")
    if not (0 <= row_idx < len(rows)):
        raise SystemExit(f"SGE_TASK_ID={sge_task_id} out of range for {len(rows)} rows in {args.config}")

    row = rows[row_idx]
    d        = str(row["d"]).strip()
    jobs_csv = row["jobs_csv"].strip()
    save_csv = row["save_csv"].strip()
    shadow   = row.get("shadow", "hd").strip()  # default if missing
    combined_csv_row = str(row.get("combined_csv", "") or "").strip()

    combined_csv_arg = args.combined_csv or ""
    combined_csv = combined_csv_arg.strip()
    if not combined_csv:
        combined_csv = combined_csv_row

    pathlib.Path(save_csv).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "ddp.scripts.run",
        *forward,
        "--jobs-csv", jobs_csv,
        "--save_csv", save_csv,
        "--d", d,
        "--shadows", shadow,          # run one shadow per task
    ]

    print(f"[INFO] SGE_TASK_ID={sge_task_id} -> row {row_idx+1}/{len(rows)}")
    print("[INFO] Running:", " ".join(cmd))
    if args.dry_run:
        return

    res = subprocess.run(cmd)

    if combined_csv and res.returncode == 0:
        try:
            append_csv_locked(save_csv, combined_csv)
        except Exception:
            print(
                f"[ERROR] Failed to append {save_csv} to combined CSV {combined_csv}",
                file=sys.stderr,
            )
            raise
        if not args.keep_individual:
            try:
                os.remove(save_csv)
            except FileNotFoundError:
                pass

    raise SystemExit(res.returncode)

if __name__ == "__main__":
    main()
