# Area-specific gamma for potential-augmented rolling batching

Design agreed 2026-09-08; implementation milestone completed 2026-09-09. Validated area coefficient tables, labeled job metadata, grouped 60-second scalar baselines, and a three-area conditional pilot are implemented. See [the concrete GRID workflow](meituan_area_gamma_grid.md) for prepared files, validation evidence, measured candidate costs, and submission commands. The full citywide historical fitting and held-out evaluation remain to be run on GRID.

The selected first Meituan design is one coefficient per job-level `da_id`, with citywide matching retained. A job's coefficient is uniquely determined by its area ID; no origin/destination geographic assignment is required for this version. The independent synthetic workflow is finalized for heterogeneous origins and destinations in 2D and can run immediately using [the prepared GRID workflow](synthetic_gamma_grid.md).

## Preliminary zoning analysis

The [initial diagnostics](meituan_zone_diagnostics.md) cover all 190,875 jobs in the eight city lunch snapshots. Area labels were recovered from the original accepted dispatch records using coordinates and timestamps, with no missing or conflicting labels. The raw dataset supplies one `da_id` per order, not separate origin/destination area labels or geographic boundaries.

Compare area ID (23 labels), H3 resolution 7 (38 origin cells), and resolution 8 (155 origin cells). Median daily arrivals per assigned/origin zone are approximately 739, 203, and 51 respectively. At resolutions 7 and 8, 63.0% and 89.5% of orders cross from their origin cell to another destination cell.

Order OD crossings and matching across job groups are different quantities. In a uniform sample of 100,000 same-day candidate pairs arriving within 60 seconds, 3,153 have positive reward under the current simulator's distance model. Of these, 53.5% have different area IDs, 45.1% different resolution-7 origin cells, and 75.2% different resolution-8 origin cells. These are potential matches, not selected matches or estimates of the performance loss from partitioning.

The area-ID choice is now settled for the first experiment. Treat the supplied label as an input for each arriving job. Cross-area poolability does not obstruct coefficient assignment: each endpoint of a matching edge contributes its own shadow. Geographic H3 alternatives and endpoint boundary recovery are optional later comparisons.

The [exact density sanity check](area_density_poolability.md) counts future poolable arrivals from every area for each job over 60 seconds. On 190,419 jobs with complete windows, area-level historical density and mean all-area opportunity count have Pearson correlation 0.614 and rank correlation 0.597. The rank correlation is positive on every day. The densest quarter of areas averages 5.77 opportunities per job, versus 1.93 in the least dense quarter. Similar-density areas can nevertheless differ: areas 5 and 9 have density about 14.7 and mean opportunities 8.96 and 3.37 respectively. Retain nominal density and historical effective opportunity counts as complementary descriptors.

## Research questions

1. At a fixed pooling window, how does the best gamma vary across areas, and how much of that variation is explained by historical arrival density or historical pooling opportunities?
2. Do area-specific gammas improve performance on a held-out day relative to a single gamma selected using the same historical days?
3. When the pooling window changes, does the best gamma depend primarily on arrival rate times window length, or do rate and window have separate effects?
4. Does the gamma–density relationship also appear in synthetic instances with controlled spatial and arrival distributions?

The upward trend in Figure 17(a) motivates these questions. It does not establish that the relationship is monotone within every area or that heterogeneous coefficients will improve performance.

## Definitions and proposed first experiment

For job j, let L_j be its direct delivery distance and h(j) its supplied area ID. Keep the existing potential and introduce a local multiplier:

```
base potential:                  p_j = L_j / 2
gamma grid:                      G = {0, 0.1, ..., 1.0}
local shadow:                    s_j = gamma[h(j), d] * p_j
effective coefficient on L_j:    c[h, d] = gamma[h, d] / 2
```

Thus gamma 0.2 means a coefficient of 0.1 on direct distance. Store and label gamma and c separately. Fix the additive parameter tau at zero initially.

The first-experiment specification is:

| Choice | First experiment |
| --- | --- |
| Dataset | All eight downloaded Meituan city lunch snapshots |
| Job's coefficient group | Its supplied `da_id`, one gamma per area |
| Matching scope | Citywide, including jobs with different area IDs |
| Pooling window | 60 seconds |
| Policy | `shadow=pb`, `dispatch=rbatch`, `tau=0` |
| Candidate gammas | 0 through 1 in increments of 0.1 |
| Historical sample | Other seven days, leaving out the evaluation day |
| Selection objective | Total historical distance saved |

The current snapshots contain 23 area IDs, all observed on every day. Median daily arrivals per area are about 739. The area table will assign one gamma to each label; it does not require a geographic polygon for that label.

For target day t, estimate density using historical counts and known observation exposure:

```
lambda_hat[h, -t] = sum(N[h, k] for k != t) / sum(T[k] for k != t)
rho_hat[h, -t, d] = lambda_hat[h, -t] * d
```

N counts valid jobs from that area under the same filtering used in simulation. T is observed time in seconds: 10,800 for each complete 10:30–13:30 snapshot, as verified in the diagnostics. Include observed days with zero jobs in the denominator; do not treat missing observations as zeros. Do not estimate exposure from the first and last arrivals within an area.

Call rho *nominal density*: the expected number of subsequent arrivals over the allowed waiting window under a constant-rate model. Actual dispatch can occur before the deadline when a job is selected as another job's partner. Actual waiting time and arrivals during that wait are policy-dependent diagnostics, not interchangeable with lambda times d. Exclude the focal job when counting subsequent arrivals. Record boundary truncation and keep end-of-snapshot handling consistent across policies.

Plot historical total arrivals as requested, and also show arrival rate and rho. Historical counts are directly comparable only when the observation duration and number of historical days agree. Time-varying lunch demand can be a later extension using the arrival intensity integrated over each job's window.

## Citywide policy and per-area tuning

Keep all jobs and the existing citywide matching opportunities. Each job supplies its own shadow to an edge. For the existing RBAT rule, the critical job's shadow is added back as it is today. A cross-area edge uses the two jobs' own gammas without averaging them or selecting one gamma for the pair.

Cross-area matching is compatible with the policy. It only means the tuning objective depends on the full coefficient vector. A practical proposed fitting method is an exhaustive conditional sweep: initialize all areas to the historically selected global gamma, then try all 11 values for one area's gamma while holding the others fixed. Select using historical citywide total savings, update that coordinate, and move to the next area. The first pilot should assess one pass, with further passes as an optional extension.

Each coordinate update includes its current value, so selecting the best candidate cannot reduce the training objective when all candidates use the same instances and deterministic simulation setup. Improvements on held-out data remain to be measured. Record the area order and entire current vector. This is exhaustive search for each coordinate conditional on the others, not an exhaustive joint search over 11^23 vectors.

Isolated-area sweeps can still be useful diagnostic curves or initializations, but they are optional and do not replace evaluation with citywide matching. Choosing the group no longer depends on making cross-area opportunities disappear.

## Historical fitting and evaluation

1. Run the citywide scalar-gamma grid once per day/window and retain individual day-level metrics. Reuse these rows to choose each fold's global gamma on the other seven days.
2. Starting from that fold's historical global gamma, run the conditional per-area sweeps above. Aggregate total citywide savings over the same seven days, with common job instances and simulation seeds for all candidates. Preserve each day's row and the full vector identifying the run.
3. Cache an instance/vector result only when jobs, window, vector, policy, reward, code version, and seed all agree. Different historical folds can generate different coordinate paths, so the old independent-cell sweep reuse does not apply automatically to citywide fitting.
4. Use an explicit fallback to the historical global gamma for areas with no usable history. Flag sparse histories and retain support counts, days observed, and conditional reward curves. If adding a minimum-support threshold, declare it in advance or tune it entirely within historical data.
5. Break ties deterministically: retain the current coordinate if it is a maximizer, otherwise prefer a maximizer nearest it and then the smaller gamma. Report flat or nearly flat curves and a predeclared near-optimal range so arbitrary grid maximizers do not become an apparent density trend.
6. Evaluate the chosen table on the held-out day. All counts, gamma choices, fallback values, and fitted density rules must exclude that day. The same day's outcome may be used in the historical sample of another fold.

For the citywide evaluation, include:

- Plain RBAT: gamma 0.
- A fixed global reference, including the current PB-RBAT default gamma 0.5.
- A historical-data-tuned global gamma: the primary comparator.
- Historical-data-tuned per-area gammas.
- Later, a shared gamma rule using nominal density or historical mean poolable-arrival counts, fitted only on historical data.

A same-day best gamma can be shown as a clearly labeled oracle diagnostic, never as deployable performance. Use paired day-level performance differences and show all eight held-out days. Do not treat areas or overlapping seven-day training folds as independent replications for uncertainty calculations.

Primary output: held-out total distance saved and its difference from the tuned global baseline. Also record normalized savings using the actual sum of direct distances, pooled fraction, waiting times, fraction of matches between different area IDs, and online runtime separately from fitting cost. Use an OPT ratio only when OPT has been computed for the same jobs, window, and matching scope; reuse valid existing benchmarks or compute them once per instance.

## Figures and extensions

**Fixed 60-second window:** scatter plots of selected gamma against historical arrivals, rho, and historical effective opportunity counts; representative conditional savings-versus-gamma curves from low-, medium-, and high-density areas; paired held-out gains over the tuned global coefficient.

**Multiple windows:** repeat for d = 30, 60, 90, 120, 150, 180 seconds. Plot gamma versus arrival rate with one panel or color per window, then gamma versus rho with the same window labels. Examine whether curves align. A density-only model is a hypothesis to assess, not a constraint to impose. Compare it with separate rate/window effects using historical fitting and held-out evaluation. Start with unregularized cell estimates as diagnostics; a pooled density rule may be more stable in sparse cells.

**Independent synthetic calibration:** the current generator uses timestamps 1, 2, ..., n, so lambda is one job per time unit and nominal rho equals d away from finite-horizon boundaries. The [new synthetic config generator and GRID instructions](synthetic_gamma_grid.md) implement the initial sweep independently of the Meituan feature. The downloaded older synthetic config collection did not contain an organized best-gamma sweep.

The finalized starting geometry has heterogeneous origins and destinations independently distributed uniformly in the two-dimensional unit square, n = 1,000, d = 5, 10, ..., 30, and the same 11 gammas. The prepared configs explicitly set `--het-origins --dimension 2` and Beta(1,1) coordinates. Reuse the same generated jobs across candidate gammas. Training seeds are 1–100 and evaluation seeds 101–200. Select a global gamma per density on training seeds and evaluate it on fresh seeds. Save full curves as well as maximizers.

Additional spatial distributions are optional later extensions. Separately varying arrival rate and window can be useful, but rescaling regularly spaced arrival times and deadlines by the same factor leaves event-driven RBAT essentially the same problem. Such a check validates units; it is not independent evidence for a density law. A stochastic-arrival extension can add further controlled experiments later.

## Code changes

The matching engine already accepts a job-aligned shadow vector. Extend the preparation and experiment layers before considering any engine changes.

| Area | Planned change |
| --- | --- |
| Dataset preparation | Restore job-level area IDs through the validated raw-data join, not the snapshot's exported row index. Save stable job identities, day, area, counts, observation exposure, and original time reference. H3 conversion is optional for later comparisons. |
| New coefficient-table utility | Load a table for a specific target day, window, area grouping, and training manifest; resolve it to a job-aligned gamma vector and explicit fallback diagnostics. Validate values and reject mismatched metadata. |
| `src/ddp/scripts/run.py` | Add mutually exclusive scalar-gamma and table-based inputs for PB-RBAT, using a shared resolver across `run_instance` and `run_once`. Compute each job's shadow exactly once, preserving scalar defaults. Reject unsupported combinations rather than silently ignoring the table. Extend to PB-`rbatch2` after the first experiment. |
| `src/ddp/scripts/shadow_sweep.py` / new area-sweep runner | Keep individual day/area/candidate/vector records; prepare jobs and potentials once per task. Avoid solving the LP and extracting HD duals for every PB-only gamma candidate. Compute requested bounds once per instance; reuse pair rewards where memory permits. |
| New historical-fit step | Derive all leave-one-day-out coefficient tables and density summaries from saved sweep rows; require complete expected candidate results before selection. |
| `src/ddp/scripts/run_from_config.py` and config generator | Register new table/mapping fields explicitly; the current runner ignores unknown CSV columns. Add `tau_s` when supporting periodic configs. Provide config stages for sweeps, fitting, and evaluation. |
| Result aggregation and plotting | Separate scalar, local-table, and density-rule strategies. Group folds by a shared experiment ID while preserving each fold's table ID as provenance. Calculate saving fractions from each instance's actual direct-distance total. |

Use `pb` plus `rbatch` for this feature. The policy named `rbatch+` currently implements a different late-arrival adjustment; it is not the name for potential augmentation.

Suggested artifacts:

- Density table: day, area, valid arrivals, exposure seconds, historical mean poolable arrivals where requested.
- Sweep table: day, target fold, tuned area, full coefficient-vector ID, pass, window, candidate gamma, savings, pooling/waiting diagnostics, seed, data/version identifiers.
- Coefficient table: target day, historical days, area, window, selected gamma, effective coefficient, historical counts/rate/density/opportunities, support and fallback status.
- Evaluation table: target day, window, strategy, matching scope, metrics, shared experiment ID, coefficient-table ID.
- Optional job audit: job identity, area, assigned gamma, density, shadow, arrival/deadline/dispatch time, partner area.

## GRID organization

Run a `(day, window)` citywide global-gamma sweep once and reuse it across folds. For a conditional area sweep, parallelize the historical days and candidate gammas under the same current vector, then aggregate and choose the coordinate before advancing to the next area. Different folds can also run independently. Preserve this dependency structure in GRID manifests.

Benchmark a citywide candidate before generating a large tuning experiment: these are substantially larger instances than isolated areas or the synthetic n=1000 runs. The fitting runner should reuse input preparation and LP/OPT benchmarks rather than multiply that cost by candidates and folds. Parameter manifests record the target fold, full vector, area order, pass, matching scope, window, gamma grid, tau, policy, observation interval, seed, code version, and data identity.

Use unique integer task IDs or canonical parameter encodings, atomic per-task output, and completeness checks for resuming. Avoid float-to-filename transformations that collapse gamma 0.1 and 1.0 to the same name. Store a versioned config generator and experiment specification even if generated CSV configs remain ignored by Git.

## Implementation sequence and acceptance checks

0. **Completed design and diagnostics.** Area ID is the selected coefficient group, cross-area matching is retained, and the density sanity check is complete. The independent heterogeneous-origin 2D synthetic configurations are ready to run.
1. **Data and scalar-equivalence foundation — implemented and locally checked.** Add indexing/density metadata and the gamma resolver. A constant coefficient table must reproduce scalar-run pairs, savings, and dispatch times. Gamma zero must reproduce plain RBAT. Check per-job shadow subtraction on cross-cell edges and critical-job add-back.
2. **Fixed-window tuning pilot — runner/configs prepared, reduced-data workflow exercised, full-city candidate costs measured; GRID fitting pending.** Select a few areas spanning density/opportunity levels, run conditional gamma sweeps within the citywide instance, exercise historical selection, and inspect curves and runtime. Test zero-history fallbacks, ties, count/exposure units, and target-day exclusion. Changing the held-out day's outcomes must not alter its fitted table.
3. **Complete area-coefficient GRID experiment.** Generate the 60-second configs with the chosen conditional fitting protocol; verify serial/task-runner agreement, unique outputs, complete candidate grids, and resume behavior. Produce the first gamma–density and gamma–opportunity plots.
4. **Held-out citywide evaluation.** Compare per-area gammas with the historical citywide global baseline while retaining cross-area matching. This establishes whether heterogeneous coefficients improve the intended policy.
5. **Analyze the independent synthetic runs when available.** Use the same candidate grid and separation of fitting and evaluation. GRID simulation can start before any Meituan feature implementation.
6. **Window and model extensions.** Add the remaining windows, fit density rules, assess sparse-cell regularization, and then consider periodic RBAT or citywide coordinate tuning. For periodic experiments, declare the tick interval and validate deadline handling for the chosen windows.

The first milestone is a reproducible fixed-window experiment and an honest held-out comparison, regardless of whether local gammas improve savings. It should answer both whether coefficients vary meaningfully with density and whether that variation is useful outside the historical sample.

## Concrete milestone, 2026-09-09

The first pilot is fold 0, one pass, in area order 22 → 5 → 9. All 23 coefficients start at the global gamma selected from the other seven days. The baseline is eight grouped day tasks (88 evaluations), followed by seven tasks for each conditional coordinate (77 evaluations) and a selection barrier between coordinates. The scalar evaluation comparators reuse the baseline results. No GRID submission or full historical fit has occurred locally. See [the workflow](meituan_area_gamma_grid.md) for exact commands and the distinction between the unfitted runtime probes and the eventual held-out comparison.

The user subsequently completed the GRID day-0/gamma-0.5 probe. Its downloaded JSON/CSV passed frozen provenance and accounting checks and exactly matched all local reference outcome metrics; policy runtime was 255.036 seconds. One baseline candidate is complete and the eight-task baseline array can resume it while computing the remaining 87 candidates. Full historical fitting remains pending.
