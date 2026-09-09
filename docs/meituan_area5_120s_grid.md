# Five-area PB-PRBAT pilot at 120 seconds

Experiment: `meituan_area5_rbatch2_pb_120s_g6_v1`.

The question is whether changing selected areas' gamma values improves citywide performance relative to **gamma 0.4 everywhere**. Figure 21(a), page 50 of [Dynamic Delivery Pooling](</Users/mer2262/Documents/Columbia/Research/Dynamic Delivery Pooling/Dynamic_Delivery_Pooling.pdf>), plots gamma 0.4 for PB-PRBAT at a 120-second window. We use this as a predeclared reference and initialization. This pilot does not repeat the search for a global gamma.

## Areas and settings

The conditional sweep visits **22 -> 6 -> 12 -> 5 -> 9** once, in that order.

| Area | Average jobs/day | Reason for inclusion |
| --- | ---: | --- |
| 22 | 29.6 | Very sparse demand; tests whether the gamma curve is flat or noisy at low volume |
| 6 | 1,700.6 | No cross-area positive-reward future opportunities in the existing diagnostic |
| 12 | 656.0 | Most measured opportunities involve other areas |
| 5 | 2,629.4 | High demand with many compatible future jobs |
| 9 | 2,649.6 | Almost the same demand as area 5, with substantially fewer compatible future jobs |

Selection uses the existing descriptive area diagnostics, which measured **60-second** future opportunities. For example, the mean all-area opportunity count was 8.96 for area 5 and 3.37 for area 9; area 12 had 5.07 cross-area opportunities out of 6.08 total. These are selection guides, not claims about 120-second opportunities or fitted gamma. Areas were chosen purposively using observed metadata, so this is an exploratory pilot rather than a blind test of how an area-selection rule generalizes.

- PB only; periodic RBAT (`rbatch2`, called PB-PRBAT in the paper).
- Maximum waiting **120 seconds**, matching every **30 seconds**, tau 0, seed 0.
- Six candidates per fitted area: gamma **0, 0.2, 0.4, 0.6, 0.8, 1.0**.
- Job shadow is `gamma[area] * direct_distance / 2`, giving effective distance coefficients **0, 0.1, 0.2, 0.3, 0.4, 0.5**.
- Every simulation includes **all 23 areas and all citywide jobs**, including cross-area pairs. The 18 areas outside the pilot retain gamma **0.4** throughout.
- Eight leave-one-day-out folds are retained. Each fold starts all coefficients at 0.4, selects each area's gamma using total citywide savings on the other seven days, then evaluates the completed vector on its held-out day.
- Earlier selected areas keep their updated values as later areas are swept. This is one conditional coordinate pass, with order-dependent results and no guarantee of a global optimum.
- LP, HD and offline OPT are disabled. The simulation and matching implementation are unchanged.

The held-out comparisons assess the incremental area fitting conditional on the preselected areas and gamma-0.4 reference. They do not independently validate the original global-gamma selection from the paper. As before, leaving out one day uses the other observed days, not a strictly chronological training split.

## Work and expected time

| Stage | Work |
| --- | --- |
| Fixed reference | **Eight simulations**, gamma 0.4 only, one per day |
| Area fitting | **1,680 candidate records**: five areas x six gammas x seven training days x eight folds |
| Held-out evaluation | Eight area-vector simulations; eight reference comparisons reuse the fixed reference |
| Summary | Validate and export saved results |

The reference is a measurement of the agreed comparator, not another global coefficient sweep. This pilot has no dependency on the old 48-result baseline. Its own fixed-reference computations also seed the exact-vector cache. The unchanged candidate at subsequent steps and identical vectors across folds are reused automatically.

This is **78.3% fewer conditional records** than the 23-area experiment (1,680 versus 7,728). With the existing day-0 timing of about 249 seconds per simulation, allow roughly **1-3 hours once the fitting resources are allocated**, plus queue time. This is a planning estimate; day/gamma variation and cache reuse affect it. The eight reference jobs should take approximately one simulation's elapsed time when they run concurrently. The pilot has not been run on GRID yet.

## Upload and run all stages

This is a small **add-on archive**, `meituan_area5_rbatch2_pb_120s_g6_v1_upload.tar.gz`. It uses the labeled data already uploaded with the 120-second package. Upload it into `/user/mer2262/ddp_pkg_release`.

The archive installs an isolated source snapshot under `experiment_src/meituan_area5_rbatch2_pb_120s_g6_v1/`. It does not replace the existing GRID `src/`, old configs or old outputs. The helper selects the snapshot through `PYTHONPATH`, and the pilot writes to a separate result directory.

Paste:

```bash
cd /user/mer2262/ddp_pkg_release
tar -xzf meituan_area5_rbatch2_pb_120s_g6_v1_upload.tar.gz
bash scripts/submit_meituan_area5_120s.sh all
qstat -u mer2262
```

The helper automatically captures GRID's job IDs and queues **reference -> five-area fitting -> summary** using dependencies. It prints and saves each submission reply; no manual job-ID entry is needed. If a submission fails or its job ID cannot be recognized, the helper stops before submitting dependent stages. Read the printed log path to recover instead of repeating the entire submission blindly.

Reference: eight tasks, one CPU per task, 8 GB requested. Fitting: eight tasks, seven CPUs and 8 GB requested per task, at most eight tasks concurrently (**56 CPUs total**). Summary: one ordinary job, one CPU and 2 GB. Each fold completes its five steps internally, running seven historical days before each selection; there is no manual step-by-step submission.

If the previous full fitting and summary jobs **8887724** and **8887725** are still listed in `qstat` and this pilot replaces that run, cancel them to release their allocations:

```bash
qdel 8887724 8887725
```

This leaves baseline job 8887723 alone and preserves completed files. No GRID jobs have been canceled or submitted from this local workspace.

To retry a failed stage, use `reference`, `fit`, or `summarize` instead of `all`. An optional second numeric argument declares a predecessor job ID. Completed candidates are validated and reused. Do not overlap an old and new submission of the same pilot stage.

## Download and interpret

Download `results/meituan_area5_rbatch2_pb_120s_g6_v1/` in full, including JSONs and the vector cache.

- `area_gamma_curves.csv`: **240 rows**, six historical candidate totals for each of five areas and eight folds; includes the selected value and effective coefficient.
- `paired_day_savings.csv`: eight comparisons of the fitted area vector against gamma 0.4 everywhere; `area_minus_fixed_0.4` is the main improvement measure.
- `selected_area_gammas.csv`: all 23 coefficients for every fold; the **40 fitted entries** have `swept=True`, while the 144 background entries remain 0.4 and have `swept=False`.
- `reference_summary.csv`: the eight fixed-gamma outcomes.
- `fold0/` through `fold7/`: full daily curves, candidate results, selections and final `coefficients04.json` tables.

Compare effect sizes and stability across folds, including whether several gamma values perform nearly equally. The area-22 sample is especially small. A selected gamma is a conditional optimizer for this pilot, not an intrinsic property of the area. Opportunity descriptors remain explicitly labeled as 60-second diagnostics; nominal arrival density is recomputed for the 120-second window.

## Reproduction and validation

The updated workflow adds a `fixed_reference` initialization mode. Ordinary historically tuned workflows retain their behavior. The fixed-reference mode requires only the gamma-0.4 reference rows, fixes all unswept coefficients at 0.4, and reports only the fixed reference and fitted area vector in the held-out comparison.

```bash
export PYTHONPATH="$PWD/experiment_src/meituan_area5_rbatch2_pb_120s_g6_v1"
python -m ddp.scripts.meituan_area_gamma generate \
  --experiment-id meituan_area5_rbatch2_pb_120s_g6_v1 \
  --areas 22,6,12,5,9 --folds all \
  --window 120 --dispatch rbatch2 --tau-s 30 \
  --gammas 0,0.2,0.4,0.6,0.8,1.0 \
  --reference-gamma 0.4 --initialization fixed_reference
```

Tests exercise a complete reduced-data fixed-reference fit, all six candidates, unchanged background areas, historical target exclusion, conditional step dependencies, held-out reference reuse, exact resume, and the prior workflow. Submission checks use a stub with the exact job/job-array response format seen on GRID, verifying dependency IDs, source isolation and failure handling without submitting real jobs. Package checks validate every config row, the frozen source, existing dataset identities, and extraction.
