# Plan for a Config-driven Meituan Shadow Sweep Orchestrator

This document sketches the structure for a new helper script that keeps the existing
batch utilities untouched while composing them into a more automated pipeline. The
goal is to accept the same command-line surface as `scripts/run_meituan_shadow_sweep.py`
(the older `meituan_sweep_param` entry point), export an explicit configuration CSV,
fan out runs one configuration at a time, and finally collapse the independent
results back into a single aggregated report.

## 1. Interface parity with `run_meituan_shadow_sweep`

* Mirror the current CLI flags: `--day` (repeatable) and `--d` (repeatable), with
  defaults matching the original sweep (`DAYS = range(8)`, `D_VALUES = [10, 20, 30]`).
* Accept `--gamma-values` and `--tau-values` arguments (comma lists or
  start:stop:step grids) mirroring the package CLI so the orchestrator can
  enumerate every gamma/tau combination.
* Add opt-in knobs for the output layout:
  - `--config-out` (path for the generated CSV; default under `configs/`).
  - `--results-dir` (base directory to store the per-run CSV files; defaults to
    `results/`).
  - `--final-out` for the combined CSV (defaults to
    `results/meituan_sweep_final.csv`).
  - `--timestamp-column`, `--jobs-template`, etc. remain fixed as in
    `run_meituan_shadow_sweep`; expose them only if required.

## 2. Config CSV emission (`generate_config` parity)

* Reuse the base column layout from `src/ddp/scripts/generate_config.py`
  (`day`, `d`, `shadow`, `jobs_csv`, `save_csv`) so the table remains compatible
  with `run_from_config`.
* Extend the schema with `gamma` and `tau` columns to cover
  `meituan_shadow_sweep` inputs. Emit one row for every `(day, d, gamma, tau)`
  combination so the downstream run has everything it needs to pick the correct
  shadow adjustments.
* Restrict `shadows` to the default (`hd`) to mirror the original script's
  defaults; keep the column so the CSV continues to interoperate with the
  generic runner if necessary.
* Materialise the `jobs_csv` path by plugging the day into the standard filename
  pattern.
* Derive `save_csv` with a fully descriptive suffix (e.g.
  `meituan_day{day}_d{d}_gamma{gamma}_tau{tau}_run.csv` when gamma/tau are
  supplied) so the aggregated filenames remain unique and self-explanatory.

## 3. Sequential execution (`run_from_config` blueprint)

* After emitting the CSV, load it back via `csv.DictReader` (same approach as
  `run_from_config`), iterate deterministically over every row, and build the
  `ddp.scripts.meituan_shadow_sweep` command.
* Accept additional passthrough flags (use `parse_known_args`) so callers can
  enrich the base command (e.g. `--with_opt`, `--export-npz`).
* For each configuration:
  - Ensure the target results directory exists (`pathlib.Path(...).mkdir`).
  - Compose a descriptive filename for the run log or stdout to aid debugging.
  - Inject `--gamma-values`/`--tau-values` flags using the single-entry grids
    derived from each CSV row so every invocation mirrors the behaviour of
    `meituan_shadow_sweep`.
  - Invoke `subprocess.run(cmd, check=True)` and surface failures immediately.

## 4. Combined CSV assembly

* Track the successfully produced CSV paths in a list as the fan-out loop runs.
* Reuse `ddp.results.append_csv.append_csv_locked` to append each individual file
  into the final aggregate, preserving the header on the first append.
* Provide a `--keep-individual` flag (default True) to mirror
  `run_from_config` behaviour; optionally clean up intermediates afterwards.
* Derive per-run filenames with a consistent naming pattern so downstream
  consumers can reconstruct the configuration (include `gamma`/`tau` tokens when
  available to distinguish shadow sweep results).

## 5. Optional metadata/artifacts

* Emit a lightweight manifest (JSON or CSV) listing each configuration, its
  output path, and the return code to help external tooling monitor progress.
* Consider a `--dry-run` mode that prints the resolved commands without
  executing them, matching `run_from_config`'s inspectability.

## 6. Result reuse

* Because the final CSV is a straight concatenation of the per-run outputs,
  existing utilities (`combine_meituan_results.py`, plotting helpers) can ingest
  it directly. The dedicated filenames make it trivial to load a subset of runs
  without rerunning the sweep.

## 7. Implementation outline

1. Parse arguments (primary + passthrough).
2. Resolve the day/d grid (defaults or CLI overrides).
3. Build the configuration rows and write them to the CSV path.
4. Iterate rows:
   * Build the command using the same ordering as `run_from_config`.
   * Execute and capture success/failure.
   * Collect the paths of successful CSV outputs.
5. Merge collected CSVs into the final aggregate.
6. Optionally clean up intermediates based on `--keep-individual`.

This staged workflow keeps the existing scripts untouched while providing a
single entry point that documents and automates the process from configuration
materialisation to aggregated results.
