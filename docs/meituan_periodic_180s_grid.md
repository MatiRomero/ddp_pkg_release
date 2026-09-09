# Meituan periodic RBAT: PB, 180-second window, six gammas

Prepared on 2026-09-09. The new experiment is `meituan_rbatch2_pb_180s_g6_v1`. All configs and the source manifest are ready; the full experiment has not been submitted to GRID.

## What is tested

- **PB only**, `dispatch=rbatch2`, waiting window **180 seconds**, matching every **30 seconds**, `tau=0`, seed 0. LP and offline OPT are disabled.
- All **23 area IDs**, in numeric order 0 through 22, with every job retained in **citywide matching**. Cross-area pairs remain possible.
- Each area tries gamma **0, 0.2, 0.4, 0.6, 0.8, 1.0**. The job shadow is `gamma[area] * direct_distance / 2`, so the full distance coefficient is **0, 0.1, 0.2, 0.3, 0.4, 0.5**.
- Eight leave-one-day-out folds. For each target day, initialize every area at the best global gamma on the other seven days. Then visit each area once, try all six values with the other areas held fixed, and retain the value maximizing total historical **citywide** savings. Finally evaluate the fitted vector on the held-out day.

This is one conditional coordinate pass: every area gets its six-value curve, conditional on the current values of the other areas. It does not enumerate all `6^23` joint vectors or establish a global optimum. Each fold can select a different vector. There is no monotonicity constraint. Deterministic ties retain the current gamma, then prefer the nearest maximizer, then the smaller gamma. The fixed scalar comparator and initial global tie reference are **0.4**, which belongs to the new grid. The historically tuned global gamma is the primary comparator.

The previous completed 60-second run contained **88 scalar PB-RBAT evaluations**: 11 uniform gammas × eight days. All 23 areas participated in each run, but they shared one gamma. It was event-driven `rbatch`, and it did not fit separate area coefficients. Those outputs and their frozen source archive are preserved. They cannot initialize this different periodic/180-second experiment.

## Periodic dispatch convention

The existing `rbatch2` simulator is unchanged. At each tick it solves one matching over the available jobs. Jobs whose deadlines are at or before the next tick are eligible for dispatch; their shadow costs are added back. Selected pairs with at least one eligible endpoint leave, eligible unmatched jobs leave solo, and pairs with neither endpoint eligible are not dispatched yet.

Ticks are measured from the loader's time zero, the day's first arrival, rather than a wall-clock phase. Thus an unmatched job arriving at zero with deadline 180 leaves at tick 150. All dispatches occur on 30-second ticks and within the 180-second waiting limit. The full day-0 probe's largest observed wait was 179 seconds.

## Upload and run

Upload the single archive **`meituan_rbatch2_pb_180s_g6_v1_upload.tar.gz`** from this worktree's `results/` directory into `/user/mer2262/ddp_pkg_release`. The archive is a bundle of source files, new configs, labeled input data, tests and instructions; extraction places them in their proper subdirectories. Uploading individual files is unnecessary. The archive does not contain experiment results. It updates `src/`, so keep the old 60-second archive if you need to restore its frozen source later.

Paste this block on GRID to extract and submit the one-candidate probe:

```bash
cd /user/mer2262/ddp_pkg_release
tar -xzf meituan_rbatch2_pb_180s_g6_v1_upload.tar.gz
bash scripts/submit_meituan_periodic_180s.sh probe
qstat -u mer2262
```

Extraction is normally silent. The helper resets `PYTHONPATH` to the uploaded `src/`, creates a timestamped log directory, and submits through `anapy3`. It avoids the rejected `--grid_array=1-1` syntax. The probe runs day 0 at gamma 0.4; its saved output is `results/meituan_rbatch2_pb_180s_g6_v1/baseline/day0_g02.json`.

After each preceding stage finishes successfully, run the next line:

```bash
bash scripts/submit_meituan_periodic_180s.sh baseline
# Wait for all eight baseline tasks to finish successfully, then:
bash scripts/submit_meituan_periodic_180s.sh fit
# Wait for all eight fitting tasks to finish successfully, then:
bash scripts/submit_meituan_periodic_180s.sh summarize
```

The baseline reuses the GRID probe and computes the remaining candidates. Each fitting array task owns one held-out fold, requests **seven CPUs**, and runs its seven historical days concurrently. It automatically completes all 23 area selections in order and then evaluates the held-out day. There is no need to submit 23 separate rounds manually. Both arrays are capped at eight concurrent tasks, so fitting can request up to **56 CPUs** in total. The initial memory request is 8 GB per submitted task via `--grid_mem=8G`; inspect scheduler accounting before increasing it or concurrency. No local `anapy3` installation was available to test scheduler submission itself.

Optionally, the helper accepts a numeric dependency job ID as its second argument, passed to the documented `--grid_hold` option. This permits queueing fitting behind the baseline and summarization behind fitting. A hold waits for completion; the Python workflow independently requires successful, complete predecessor artifacts and rejects incomplete results.

```bash
# Replace 123456 with the actual baseline job ID printed by anapy3:
bash scripts/submit_meituan_periodic_180s.sh fit 123456
```

Resubmit the same stage after a failure: completed candidates are checked and reused. To rerun just one failed fold, use an ordinary batch job with an explicit fold, replacing `0` as needed:

```bash
(
  set -e
  DDP_REPO="/user/mer2262/ddp_pkg_release"
  DDP_LOG="$DDP_REPO/logs/periodic-fold-retry-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$DDP_LOG"
  cd "$DDP_LOG"
  export PYTHONPATH="$DDP_REPO/src"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  anapy3 --grid_submit=batch --grid_mem=8G --grid_ncpus=7 \
    -m ddp.scripts.meituan_area_gamma fit-fold \
    --manifest "$DDP_REPO/configs/meituan_rbatch2_pb_180s_g6_v1.json" \
    --fold 0 --workers 7
)
```

The individual `pilot_step00.csv` through `pilot_step22.csv` configs remain available for staged recovery: each has 56 rows, with six candidates per row. The `pilot_` prefix is retained for compatibility; this manifest covers all 23 areas and eight folds. Use either the fold runner or staged tasks for a fold, avoiding overlapping submissions. Per-fold and per-candidate locks reject conflicting work.

## Workload and measured cost

| Stage | Saved policy records | Scheduled work |
| --- | ---: | --- |
| Scalar baseline | 48 | Eight day tasks, six gammas each |
| Conditional curves | 7,728 | 23 areas × six gammas × seven historical days × eight folds |
| Held-out comparisons | 32 | Four strategies × eight days; 24 scalar records reuse the baseline |

Identical day/full-vector simulations are cached across candidates and folds. In particular, each area's unchanged candidate can reuse the previous selected vector. Cache keys include input hashes, the full vector, code, policy, window, period and numerical runtime. Reused decisions receive fresh fold-specific table provenance, so sharing a computation does not share the training-day selection. The six candidate records are always retained even when computations are reused.

A full citywide development probe on day 0 (22,535 jobs), gamma 0.4, took **420.3 seconds (7.0 minutes)** of policy time and approximately **165.5 MiB** process peak RSS locally. It formed 9,534 pairs and 3,467 solos, saved 112.257934 distance units, and had mean wait 130.870 seconds. It is a timing probe, not a fitted result, and lives separately in `results/meituan_periodic_180s_benchmark_v1/`.

The earlier local 60-second event-driven scalar probe took 357.8 seconds. These differ in window and gamma as well as policy, so they do not isolate the effect of periodic matching. The larger waiting population can outweigh the reduction in matching rounds. The new GRID probe is needed for a cluster estimate.

At a uniform seven minutes per simulation, 7,784 potential fresh simulations (48 baseline + 7,728 training + eight held-out area evaluations) would represent roughly **908 CPU-hours**, or **16 hours with 56 fully occupied CPUs**, before cache reuse, queue time, unequal task lengths and hardware differences. These are arithmetic planning figures, not a promised completion time. Full fitting is an hours-scale experiment despite the reduced grid.

## Outputs and interpretation

Download the whole `results/meituan_rbatch2_pb_180s_g6_v1/` folder after summarization. The main review files are:

- `baseline_summary.csv`: 48 scalar outcomes.
- `area_gamma_curves.csv`: 1,104 rows, six historical candidate totals for each area/fold; includes effective distance coefficients and the selected value.
- `selected_area_gammas.csv`: 184 fitted gamma values and effective coefficients, one per area/fold.
- `paired_day_savings.csv`: eight held-out comparisons against gamma zero, fixed gamma 0.4, and the historically tuned global gamma.
- `fold0/` through `fold7/`: all candidate JSONs, selections, daily curves, held-out evaluations and coefficient tables. `coefficients22.json` is each fold's final fitted table.
- `vector_cache/`: exact reusable simulation metrics and decision hashes. Keep these to resume without recomputation. JSONs, including hashes and provenance, are authoritative; CSVs are review exports.

The dataset retains all 190,875 jobs and the original 10:30–13:30 exposure. Historical counts and nominal density use only the other seven days; density is arrivals divided by historical exposure, multiplied by 180 seconds. Existing opportunity diagnostics were calculated for **60 seconds**. They are retained only as explicitly labeled 60-second descriptive covariates (`opportunity_window_seconds=60`, `opportunities_match_waiting_window=false`), not reported as 180-second opportunities and not used to select gamma.

## Reproduction and verification

The generator can create later windows under distinct experiment IDs. The present manifest was generated with:

```bash
python -m ddp.scripts.meituan_area_gamma generate \
  --experiment-id meituan_rbatch2_pb_180s_g6_v1 \
  --areas all --folds all --window 180 --dispatch rbatch2 --tau-s 30 \
  --gammas 0,0.2,0.4,0.6,0.8,1.0 --reference-gamma 0.4
```

Source and data identities are frozen in the manifest. Keep them unchanged while running; create a new experiment ID for changed settings or code. Numerical-runtime checks prevent mixing local development candidates with GRID candidates. The original 60-second archive remains the source snapshot for its results.

Local verification covers exact periodic scalar/table decisions at all six gammas, zero-gamma/plain-periodic equivalence, critical-endpoint shadow add-back, tick alignment, waiting limits, period mismatch rejection, historical target exclusion, cross-fold cache provenance and corruption, and a complete reduced-data two-fold/three-area fit with multiprocessing, selection, evaluation and byte-identical resume. All **49 area-workflow tests pass**. The full suite reports **88 passed and six previously established failures** in unrelated plotting labels and an average-dual cache signature.
