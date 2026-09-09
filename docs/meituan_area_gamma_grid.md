# Meituan area gamma: 60-second baseline and conditional pilot

Archived experiment instructions, updated 2026-09-09. **All 88 scalar baseline evaluations are complete and validated.** The proposed three-area 60-second pilot was superseded before fitting. The current task is the [PB periodic 180-second, six-gamma experiment for all 23 areas](meituan_periodic_180s_grid.md). The commands below describe the old workflow and require its frozen source archive; they are not the next submission steps. Old results and the old archive are preserved.

The policy is `shadow=pb`, `dispatch=rbatch`, `tau=0`, with job shadow `gamma[da_id] * direct_distance / 2`. All matching remains citywide, including cross-area pairs. The existing critical-job add-back is unchanged. `rbatch+` is a separate late-arrival policy. There is no monotonicity constraint on gamma.

## Prepared artifacts

The generated CSVs and manifest live **directly under `configs/`**. Do not run a config copy from a log subdirectory: `run_from_config` resolves repository-relative paths from its config's parent directory.

| File prefix `meituan_area_gamma_60s_pilot_v1_` | GRID rows | Work |
| --- | ---: | --- |
| `baseline_probe.csv` | 1 | Day 0, gamma 0.5; optional GRID timing probe |
| `baseline.csv` | 8 | One day per task, all 11 gammas: 88 evaluations |
| `pilot_step00.csv` | 7 | Fold 0, area 22, all 11 gammas per historical day |
| `pilot_step01.csv` | 7 | Fold 0, area 5, conditional on selection 00 |
| `pilot_step02.csv` | 7 | Fold 0, area 9, conditional on selection 01 |
| `evaluation.csv` | 1 | Fold 0 final area table; three scalar comparators reused from baseline |

The common manifest is `configs/meituan_area_gamma_60s_pilot_v1.json`. Pilot order is **22 → 5 → 9**, one pass, held-out day 0. Area 22 has sparse demand; areas 5 and 9 have similar density and substantially different opportunity counts. This is a workload and workflow pilot, not the final eight-day estimate of improvement.

Each coordinate has 77 saved results: 11 candidate gammas × seven historical days. The next coordinate cannot run until its predecessor has a complete selection artifact. Successive coordinate updates are never submitted together. The configs for later coordinates are dependency-bound instructions; their actual vectors are resolved only after selection.

The upload bundle `results/meituan_area_gamma_60s_pilot_v1_upload.tar.gz` contains the frozen Python source, all six CSVs and manifest, labeled inputs, preparation/benchmark scripts, and documentation. It excludes the raw dispatch data, synthetic outputs and local timing results. Upload it to `/user/mer2262/ddp_pkg_release`, then extract it there. Alternatively upload those same files individually. Preserve the existing research outputs and keep the uploaded source fixed throughout the experiment.

## Next GRID commands

After uploading the bundle to the checkout, run:

```bash
cd /user/mer2262/ddp_pkg_release
tar -xzf meituan_area_gamma_60s_pilot_v1_upload.tar.gz
export PYTHONPATH="/user/mer2262/ddp_pkg_release/src${PYTHONPATH:+:$PYTHONPATH}"
```

Define these two submission helpers in the GRID shell. Each creates a timestamped log directory and keeps the authoritative config in `configs/`.

```bash
ddp_area_submit() {
  local ddp_stage="$1"
  local ddp_tasks="$2"
  local ddp_repo="/user/mer2262/ddp_pkg_release"
  local ddp_config="$ddp_repo/configs/meituan_area_gamma_60s_pilot_v1_${ddp_stage}.csv"
  local ddp_log="$ddp_repo/logs/meituan-area-${ddp_stage}-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$ddp_log"
  cp "$ddp_config" "$ddp_log/submitted_config.csv"
  (
    cd "$ddp_log" || exit 1
    export PYTHONPATH="$ddp_repo/src${PYTHONPATH:+:$PYTHONPATH}"
    local ddp_manifest="$ddp_repo/configs/meituan_area_gamma_60s_pilot_v1.json"
    local ddp_array_args=()
    local ddp_python_args=()
    if [ "$ddp_tasks" -eq 1 ]; then
      case "$ddp_stage" in
        baseline_probe)
          ddp_python_args=(-m ddp.scripts.meituan_area_gamma run-task
            --manifest "$ddp_manifest" --stage baseline --day 0 --candidate 5)
          ;;
        evaluation)
          ddp_python_args=(-m ddp.scripts.meituan_area_gamma run-task
            --manifest "$ddp_manifest" --stage evaluation --day 0 --fold 0 --step 3)
          ;;
        *) echo "Unknown single-job stage: $ddp_stage"; exit 1 ;;
      esac
    else
      ddp_array_args=(--grid_array="1-${ddp_tasks}/8")
      ddp_python_args=(-m ddp.scripts.run_from_config --config "$ddp_config")
    fi
    anapy3 --grid_submit=batch --grid_mem=8G --grid_ncpus=1 \
      "${ddp_array_args[@]}" "${ddp_python_args[@]}"
  )
}

ddp_area_select() {
  local ddp_step="$1"
  local ddp_repo="/user/mer2262/ddp_pkg_release"
  local ddp_log="$ddp_repo/logs/meituan-area-select${ddp_step}-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$ddp_log"
  (
    cd "$ddp_log" || exit 1
    export PYTHONPATH="$ddp_repo/src${PYTHONPATH:+:$PYTHONPATH}"
    anapy3 --grid_submit=batch --grid_mem=2G --grid_ncpus=1 \
      -m ddp.scripts.meituan_area_gamma select \
      --manifest "$ddp_repo/configs/meituan_area_gamma_60s_pilot_v1.json" \
      --fold 0 --step "$ddp_step"
  )
}
```

Single-job correction: the GRID wrapper rejected `--grid_array=1-1`. The probe, evaluation, selection and summary therefore use ordinary batch submissions without an array option. Single simulation jobs invoke `meituan_area_gamma run-task` directly, so they do not depend on the scheduler supplying a numeric `SGE_TASK_ID`. The multi-task baseline and coordinate arrays keep their existing syntax. Source, data and config files need no changes.

First run the single candidate probe, inspect its successful result and scheduler accounting, then run the baseline:

```bash
ddp_area_submit baseline_probe 1
qstat -u mer2262
# After the probe finishes successfully:
ddp_area_submit baseline 8
```

The baseline reuses the probe result and evaluates the other 87 candidates. It accepts only matching data, code and numerical-runtime provenance. The 8 GB request is an initial resource choice; local peak memory was not recorded for the completed development timings. Check GRID memory and wall time before increasing concurrency. A cap such as `/8` or `/1000` applies to **that array only**, not across separate submissions.

After all baseline tasks finish, run the pilot in this order. **Wait for each submitted array or selection job to finish successfully before executing the next command.** An absent dependency fails explicitly; it does not schedule or wait for another array.

```bash
ddp_area_submit pilot_step00 7
# Wait for all seven tasks, then:
ddp_area_select 0
# Wait for selection00.json, then:
ddp_area_submit pilot_step01 7
# Wait for all seven tasks, then:
ddp_area_select 1
# Wait for selection01.json, then:
ddp_area_submit pilot_step02 7
# Wait for all seven tasks, then:
ddp_area_select 2
# Wait for selection02.json, then:
ddp_area_submit evaluation 1
```

Selections are saved in `results/meituan_area_gamma_60s_pilot_v1/fold0/selection00.json`, `selection01.json`, and `selection02.json`. Each records the full input and selected vectors, the objective at every gamma, each historical result identity, support summaries, and deterministic tie handling. `coefficients02.json` is the final pilot table, usable by the ordinary `run.py --gamma-table` entry point.

After evaluation, produce the summary with a short job:

```bash
(
  DDP_REPO="/user/mer2262/ddp_pkg_release"
  DDP_LOG="$DDP_REPO/logs/meituan-area-summary-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$DDP_LOG"
  cd "$DDP_LOG" || exit 1
  export PYTHONPATH="$DDP_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
  anapy3 --grid_submit=batch --grid_mem=2G --grid_ncpus=1 \
    -m ddp.scripts.meituan_area_gamma summarize \
    --manifest "$DDP_REPO/configs/meituan_area_gamma_60s_pilot_v1.json"
)
```

Download the entire experiment result directory, including JSON files. `baseline_summary.csv` retains the 88 scalar outcomes; `fold0/curve00.csv` through `curve02.csv` retain each historical day's conditional curve; `paired_day_savings.csv` compares plain RBAT, fixed gamma 0.5, historically tuned global gamma, and historically tuned area gamma. The tuned global gamma is the primary comparator. The pilot has one held-out day and cannot establish eight-day performance.

## Completed GRID probe, 2026-09-09

The downloaded `baseline/day0_g05.json` and CSV agree on every field. The result hash, frozen source/config/data identities, all 23 gamma values, job accounting and metric identities passed validation. The probe ran with Python 3.13.5, NumPy 2.1.3 and NetworkX 3.4.2. Continue the GRID baseline with that numerical environment; the local development Python environment is different and should not be mixed into the saved candidate grid.

Policy time was **255.036 seconds (4 minutes 15 seconds)**; candidate wall time was 255.303 seconds. It processed 22,535 jobs, saved 86.36150431132887 distance units, formed 7,791 pairs, dispatched 6,953 solos, and selected 3,073 cross-area pairs. Every recorded outcome metric exactly matches the local gamma-0.5 reference. The downloaded artifacts contain summary metrics, not selected job-pair identities or individual dispatch times. Scheduler accounting and peak memory were not included.

The user subsequently completed all 88 baseline candidates. Their hashes, provenance and job accounting passed validation before periodic source changes; see `results/meituan_area_gamma_benchmark_v1/completed_baseline_validation.json`. The probe-only audit remains in `grid_probe_validation.json`.

## Selection and historical information

For a target day, global selection sums the saved citywide savings for each gamma over the other seven days. A global tie uses 0.5 as its current reference. Every area starts at that selected global gamma. Each coordinate tries all 11 grid values while keeping the other 22 coefficients fixed, then maximizes total **citywide** savings across exactly those historical days.

Within an absolute tolerance of `1e-12` savings units, ties retain the current coefficient; otherwise select the nearest maximizer, then the smaller gamma. Equal grid distances are rounded to 12 decimals to avoid floating-point direction bias. The manifest also predeclares a `1e-6` absolute near-optimal range. The complete curve is preserved; flat curves should not be interpreted as evidence for a density relationship.

Historical arrivals use the same valid jobs as simulation. Nominal density is `historical_arrivals / (7 * 10800) * 60`; zero-count observed days remain in the denominator. Historical opportunity means weight each other day's mean by its number of complete-window focal jobs. They reuse the completed exact diagnostic, which counts strictly future positive-reward arrivals from every area and excludes incomplete windows. No opportunity calculation was repeated. Both descriptors exclude the held-out day.

A group with no historical arrivals retains the historical global gamma and receives a `no_historical_arrivals` flag. A completely missing table entry is either an error or an explicit historical-global fallback, as declared in the table. Sparse support and opportunity-eligible job counts are retained without imposing an unannounced support threshold.

## Preparation, reuse and provenance

`Job` now has optional `da_id`, `job_id`, `day`, `dataset_id`, and `original_timestamp` metadata. Enriched CSVs require complete, unique job identities, preserve timestamp ties in source order, and reject invalid rows rather than silently dropping labels. Ordinary scalar CSV loading retains its existing filtering and sorting behavior. The Meituan preparation uses the validated accepted-record join on microdegree coordinates, `dt`, and both timestamps. It never uses the snapshot's exported index as a raw-record key. Duplicate identical snapshot keys receive an occurrence suffix only for job identity after area recovery.

The prepared corpus contains 190,875 jobs, eight days and 23 areas. Its manifest records source hashes, per-day enriched-file hashes, the join audit and 10:30–13:30 exposure. Enriched files retain the original snapshot coordinate strings. The original research directory was only read. Raw records are needed for preparation locally, not for GRID simulation.

Coefficient JSON validation covers schema, area grouping, dataset identity, exact other-day history, target exclusion, matching scope, policy, tau, window, finite values in [0,1], gamma/2 effective coefficients, and fallback behavior. `run_instance` and `run_once(jobs=...)` use the same resolver. Scalar and table gamma arguments are mutually exclusive; tables currently support only PB-RBAT with pooling reward and tau zero. Enriched NPZ export is rejected because the existing NPZ format omits labels. Keep labeled datasets in CSV.

`run_from_config` explicitly recognizes `gamma_table`, `skip_lp`, `tau_s`, and the grouped experiment fields. Scalar defaults remain unchanged, including the existing LP behavior; `--skip-lp` is opt-in for ordinary runs and invalid for HD. The new grouped PB workflow requests no bounds by default. It loads each day and prepares lengths/potentials once per task, with a bounded cache of pair rewards shared across candidates. It leaves matching logic unchanged. Requested LP/OPT benchmarks are cached once per day and reused across candidates, resume attempts and folds. Enable them only in a new manifest with `--with-lp` / `--with-opt`; their existing citywide solvers can be much more costly than the PB policy.

Each candidate JSON is written atomically and includes the entire vector, vector/table/experiment identity, fold, area, pass, step, candidate, day, seed, code hash, data hash, numerical runtime, savings, actual direct-distance total, pooled fraction, waits and cross-area pair counts. Integer candidate indices `g00` through `g10` avoid filename collisions. Policy time and preparation time are separate. Reused scalar evaluation rows point to their baseline result identity; their policy time is the original baseline timing, not new evaluation work.

Completed candidates are validated and skipped on resubmission. Exclusive per-candidate advisory locks reject simultaneous duplicate work and release on process exit; the empty `.lock` files can remain. A failed task may leave some complete candidates and no final day CSV; rerun the same config row to finish it. Candidate JSONs are authoritative. Incomplete grids, invalid hashes, altered manifests and mixed numerical runtimes fail before fitting. Keep source and inputs frozen while an array is active. Use a new experiment ID for any actual specification/code/data change.

## Local reproduction and validation

From this worktree with the original checkout's Python environment:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
DDP_PYTHON="/Users/mer2262/Documents/Columbia/Research/Dynamic Delivery Pooling/ddp_pkg_release/.venv/bin/python"

"$DDP_PYTHON" scripts/prepare_meituan_area_gamma.py \
  --reference-root "/Users/mer2262/Documents/Columbia/Research/Dynamic Delivery Pooling/ddp_pkg_release" \
  --raw "/Users/mer2262/Documents/Columbia/Research/Dynamic Delivery Pooling/data/all_waybill_info_meituan_0322.csv"

"$DDP_PYTHON" -m ddp.scripts.meituan_area_gamma generate
```

Preparation reuses an existing dataset only after verifying its source and output identities. Generation is deterministic and refuses to change an existing experiment manifest. The versioned generator is the source of truth; generated `data/`, `configs/`, and `results/` remain ignored by Git, so transfer or retain the upload bundle explicitly.

Completed development timing probes on day 0 (22,535 jobs, d=60):

| Probe | Policy seconds | Total savings | Pairs | Cross-area pairs |
| --- | ---: | ---: | ---: | ---: |
| All gamma 0.5 | 357.792 | 86.3615043113 | 7,791 | 3,073 |
| Area 5 gamma 0.1; others 0.5 | 377.980 | 85.8207640319 | 7,855 | 3,122 |

These use the simulator's existing Euclidean coordinate-distance units. Preparation was about 0.07 seconds. They ran locally during development, without LP/OPT and without peak-memory capture, and are separate from the frozen experiment namespace. The second vector was **unfitted** and measures candidate cost, not historical tuning or held-out improvement. They suggest roughly an hour of work per eleven-candidate day task, but GRID timing and cache warmth can differ. Reproduce fresh timing and process peak RSS with:

```bash
"$DDP_PYTHON" scripts/benchmark_meituan_area_gamma.py --mode scalar \
  --output results/meituan_fresh_scalar_benchmark.json
"$DDP_PYTHON" scripts/benchmark_meituan_area_gamma.py --mode area \
  --output results/meituan_fresh_area_benchmark.json
```

Correctness checks cover exact constant-table/scalar pairs, savings and dispatch times at gamma 0, 0.1, 0.5 and 1; zero-table/plain-RBAT equivalence; cross-area endpoint subtraction and critical add-back; malformed tables and metadata; stable job alignment; missing-history fallbacks; ties; target exclusion; optional bound reuse; config execution from an unrelated directory; atomic/resumable output; and a complete reduced-data 88 + 231 + 4 result workflow. This exercises the conditional pilot machinery without spending the full citywide fitting budget locally.

The full suite reports 74 passed and six pre-existing failures. The six unrelated repository test failures (five plot-label/style tests and one average-dual cache signature test) also reproduce on an untouched `HEAD` source archive. They are not fixed by this feature.

## Original follow-up plan (superseded)

Inspect conditional curves, total task wall time, GRID memory and the paired held-out result before preparing the larger fit. The generator supports an explicit area order, folds and passes, but this milestone generates only the small pilot. The final design is all 23 areas and eight held-out days, retaining complete conditional curves and paired day-level differences. First analysis figures relate selected gamma to historical arrivals, nominal density and historical all-area opportunities. Additional windows follow validation of this 60-second workflow.

The independent heterogeneous-origin 2D synthetic experiment is unchanged. Its 66-result pilot remains validated in the original checkout, and completion of the remaining 13,134 GRID tasks is still unconfirmed.
