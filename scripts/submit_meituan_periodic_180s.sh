#!/usr/bin/env bash
# Submit one stage; simulation and fitting always run inside GRID allocations.
set -euo pipefail

DDP_REPO="${DDP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
DDP_EXPERIMENT="meituan_rbatch2_pb_180s_g6_v1"
DDP_MANIFEST="$DDP_REPO/configs/$DDP_EXPERIMENT.json"
DDP_STAGE="${1:-}"
DDP_HOLD="${2:-}"
DDP_GRID_ARGS=(--grid_submit=batch --grid_mem=8G --grid_ncpus=1)
DDP_PYTHON_ARGS=()

case "$DDP_STAGE" in
  probe)
    # No 1-1 array: anapy3 rejects a single-task array range.
    DDP_PYTHON_ARGS=(-m ddp.scripts.meituan_area_gamma run-task
      --manifest "$DDP_MANIFEST" --stage baseline --day 0 --candidate 2)
    ;;
  baseline)
    DDP_GRID_ARGS+=(--grid_array=1-8/8)
    DDP_PYTHON_ARGS=(-m ddp.scripts.run_from_config
      --config "$DDP_REPO/configs/${DDP_EXPERIMENT}_baseline.csv")
    ;;
  fit)
    DDP_GRID_ARGS=(--grid_submit=batch --grid_mem=8G --grid_ncpus=7 --grid_array=1-8/8)
    DDP_PYTHON_ARGS=(-m ddp.scripts.meituan_area_gamma fit-fold
      --manifest "$DDP_MANIFEST" --array --workers 7)
    ;;
  summarize)
    DDP_GRID_ARGS=(--grid_submit=batch --grid_mem=2G --grid_ncpus=1)
    DDP_PYTHON_ARGS=(-m ddp.scripts.meituan_area_gamma summarize --manifest "$DDP_MANIFEST")
    ;;
  *)
    echo "Usage: bash $0 {probe|baseline|fit|summarize} [dependency_job_id]" >&2
    exit 2
    ;;
esac

if [[ -n "$DDP_HOLD" ]]; then
  if [[ ! "$DDP_HOLD" =~ ^[0-9]+$ ]]; then
    echo "Dependency must be the numeric GRID job ID." >&2
    exit 2
  fi
  DDP_GRID_ARGS+=(--grid_hold="$DDP_HOLD")
fi
if [[ ! -f "$DDP_MANIFEST" ]]; then
  echo "Missing manifest: $DDP_MANIFEST. Extract the upload archive in the repository first." >&2
  exit 1
fi
export PYTHONPATH="$DDP_REPO/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "${DDP_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' anapy3 "${DDP_GRID_ARGS[@]}" "${DDP_PYTHON_ARGS[@]}"
  printf '\n'
  exit 0
fi
DDP_LOG="$DDP_REPO/logs/${DDP_EXPERIMENT}-${DDP_STAGE}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$DDP_LOG"
cp "$DDP_MANIFEST" "$DDP_LOG/submitted_manifest.json"
if [[ "$DDP_STAGE" == baseline ]]; then
  cp "$DDP_REPO/configs/${DDP_EXPERIMENT}_baseline.csv" "$DDP_LOG/submitted_config.csv"
fi
cd "$DDP_LOG"
printf 'Submitting %s; logs: %s\n' "$DDP_STAGE" "$DDP_LOG"
anapy3 "${DDP_GRID_ARGS[@]}" "${DDP_PYTHON_ARGS[@]}"
