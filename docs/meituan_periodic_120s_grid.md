# Meituan periodic RBAT: PB, 120-second window, six gammas

The current experiment is `meituan_rbatch2_pb_120s_g6_v1`. It uses **120 seconds** of maximum waiting, **30 seconds** between matching rounds, PB only, and `rbatch2`. All **23 areas** test gamma **0, 0.2, 0.4, 0.6, 0.8, 1.0**, corresponding to distance coefficients **0, 0.1, 0.2, 0.3, 0.4, 0.5**. Matching remains citywide. Eight held-out-day folds and one area sweep per fold are retained. The 180-second manifest and archive remain available separately.

## Why use held-out days?

Selecting gamma on a set of days and reporting the best score on those same days can overstate its performance on new data. A held-out day separates those two jobs: choose all coefficients using seven days, freeze them, then evaluate on the eighth. Repeating this with each of the eight days held out gives eight comparisons and shows variation across days.

All eight repetitions are a validation choice, not a requirement for running the matching policy or inspecting exploratory gamma curves. One fixed held-out day would be a cheaper pilot: it needs one eighth as many conditional training records, although actual computing time also depends on cache reuse and parallelism. It provides less evidence about day-to-day variation. This package retains the existing eight-fold design.

These are leave-one-day-out checks across the observed dataset, not chronological forecasts. Some training days may occur after the held-out day. A strict forward-in-time deployment test would require a different split.

## Why update one area at a time?

Jobs can pair across area boundaries, so an area's coefficient affects matching opportunities in other areas. We first select the best uniform gamma using the seven training days. For area 0, try all six gammas while holding the other 22 values fixed, then retain the best value. Repeat for area 1 using that updated vector, and continue through area 22. Every trial evaluates total citywide savings on the same seven training days.

This is a practical way to tune a complete vector. Testing every joint combination would require `6^23` vectors. A single coordinate pass uses six candidates per area, but can depend on area order and does not guarantee the best joint combination. Separate sweeps against one common fixed vector would answer a different question: how each area behaves near that fixed setting, without progressively fitting the whole vector.

## Upload and submit

Upload **`meituan_rbatch2_pb_120s_g6_v1_upload.tar.gz`** from this worktree's `results/` directory into `/user/mer2262/ddp_pkg_release`. It contains the source, 120-second configs, labeled inputs and instructions; no individual-file upload is needed. Then paste:

```bash
cd /user/mer2262/ddp_pkg_release
tar -xzf meituan_rbatch2_pb_120s_g6_v1_upload.tar.gz
bash scripts/submit_meituan_periodic_120s.sh probe
qstat -u mer2262
```

The probe evaluates day 0 at gamma 0.4. Its output is `results/meituan_rbatch2_pb_120s_g6_v1/baseline/day0_g02.json`. Extraction is normally silent. The submission helper sets `PYTHONPATH`, creates a timestamped log directory, and uses an ordinary batch job for the probe, avoiding the rejected `--grid_array=1-1` syntax.

After each stage finishes successfully, run the next one:

```bash
bash scripts/submit_meituan_periodic_120s.sh baseline
# Wait for all eight baseline tasks to finish successfully, then:
bash scripts/submit_meituan_periodic_120s.sh fit
# Wait for all eight fitting tasks to finish successfully, then:
bash scripts/submit_meituan_periodic_120s.sh summarize
```

The baseline has eight tasks with six gammas each. Fitting has eight tasks, one per held-out day. Each fitting task requests seven CPUs and runs the seven training days concurrently, completing all 23 area updates automatically before evaluating its held-out day. The array permits eight concurrent tasks, up to 56 CPUs. The helper requests `--grid_mem=8G` for simulation/fitting and 2 GB for summarization. These initial requests follow the 180-second workflow; inspect GRID accounting after the probe.

The helper also accepts a numeric predecessor job ID as its second argument, passed to `--grid_hold`. Fitting and summarization validate their required predecessor results. Failed or interrupted stages can be resubmitted; completed candidates and exact full-vector simulations are reused. Run either the fold runner or the individual staged configs for a given fold, avoiding overlapping submissions.

## Workload and results

The smaller window reduces the potential number of jobs waiting for each matching round; the number of gamma trials is unchanged: **48 scalar baseline evaluations**, **7,728 conditional training records**, and eight fresh held-out area evaluations. Identical day/vector evaluations share a cache. There are also 24 held-out scalar comparator records reused from the baseline. Queue time, numerical environment and each day's demand affect elapsed time.

The fixed comparator and global tie reference are gamma 0.4. The primary comparison is the fitted area vector versus the historically selected uniform gamma on each held-out day. LP, HD and offline OPT are disabled. Gamma zero gives the zero-shadow periodic rule.

Download the whole `results/meituan_rbatch2_pb_120s_g6_v1/` folder after summarization. Its main files are:

- `baseline_summary.csv`: the 48 scalar outcomes.
- `area_gamma_curves.csv`: 1,104 historical candidate totals, six per area/fold.
- `selected_area_gammas.csv`: 184 fitted gammas and effective distance coefficients.
- `paired_day_savings.csv`: eight held-out comparisons.
- `fold0/` through `fold7/`: candidate results, selections, daily curves and final `coefficients22.json` tables.
- `vector_cache/`: reusable exact simulation metrics and decision hashes.

The policy and validation machinery are unchanged from the [180-second workflow](meituan_periodic_180s_grid.md). Ticks begin from the loader's first-arrival time zero. Jobs due by the next tick are eligible now, so an unmatched job arriving at zero with deadline 120 leaves at tick 90. The runner checks that every job leaves within 120 seconds and on a 30-second tick.

Historical density is recomputed as historical arrivals divided by seven days' exposure, multiplied by **120** seconds. Reused opportunity diagnostics still describe **60-second** opportunities and are explicitly labeled as such. They do not select gamma.

## Reproduce the configuration

```bash
python -m ddp.scripts.meituan_area_gamma generate \
  --experiment-id meituan_rbatch2_pb_120s_g6_v1 \
  --areas all --folds all --window 120 --dispatch rbatch2 --tau-s 30 \
  --gammas 0,0.2,0.4,0.6,0.8,1.0 --reference-gamma 0.4
```

This change uses the already-tested simulation and fitting source without modification. Config coverage, source/data hashes, submission arguments and extracted archive contents are checked separately. The prior source verification reported 49 relevant tests passing; the full suite had 88 passing and six established unrelated failures. No GRID job has been submitted from this workspace.
