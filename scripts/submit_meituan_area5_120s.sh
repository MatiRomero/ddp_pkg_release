#!/usr/bin/env bash
# Fixed-gamma reference -> five-area fit -> summary. Uses an isolated source copy.
set -euo pipefail
DDP_REPO="${DDP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
DDP_EXPERIMENT="meituan_area5_rbatch2_pb_120s_g6_v1"
DDP_MANIFEST="$DDP_REPO/configs/$DDP_EXPERIMENT.json"
DDP_SOURCE="$DDP_REPO/experiment_src/$DDP_EXPERIMENT"
[[ -f "$DDP_MANIFEST" && -f "$DDP_SOURCE/ddp/scripts/meituan_area_gamma.py" ]] || {
  echo "Missing pilot files. Extract the pilot archive in $DDP_REPO first." >&2
  exit 1
}
export PYTHONPATH="$DDP_SOURCE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DDP_SUBMITTED_JOB=""

ddp_submit() {
  local ddp_stage="$1" ddp_hold="${2:-}" ddp_log ddp_output ddp_id
  local ddp_grid=(--grid_submit=batch --grid_mem=8G --grid_ncpus=1)
  local ddp_python=()
  case "$ddp_stage" in
    reference)
      ddp_grid+=(--grid_array=1-8/8)
      ddp_python=(-m ddp.scripts.run_from_config --config "$DDP_REPO/configs/${DDP_EXPERIMENT}_reference.csv")
      ;;
    fit)
      ddp_grid=(--grid_submit=batch --grid_mem=8G --grid_ncpus=7 --grid_array=1-8/8)
      ddp_python=(-m ddp.scripts.meituan_area_gamma fit-fold --manifest "$DDP_MANIFEST" --array --workers 7)
      ;;
    summarize)
      ddp_grid=(--grid_submit=batch --grid_mem=2G --grid_ncpus=1)
      ddp_python=(-m ddp.scripts.meituan_area_gamma summarize --manifest "$DDP_MANIFEST")
      ;;
    *) echo "Unknown stage: $ddp_stage" >&2; return 2 ;;
  esac
  if [[ -n "$ddp_hold" ]]; then
    [[ "$ddp_hold" =~ ^[0-9]+$ ]] || { echo "Invalid dependency job ID." >&2; return 2; }
    ddp_grid+=(--grid_hold="$ddp_hold")
  fi
  if [[ "${DDP_DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' anapy3 "${ddp_grid[@]}" "${ddp_python[@]}"
    printf '\n'
    DDP_SUBMITTED_JOB=1000
    return
  fi
  ddp_log="$DDP_REPO/logs/${DDP_EXPERIMENT}-${ddp_stage}-$(date +%Y%m%d-%H%M%S)-$$"
  mkdir -p "$ddp_log"
  cp "$DDP_MANIFEST" "$ddp_log/submitted_manifest.json"
  if [[ "$ddp_stage" == reference ]]; then
    cp "$DDP_REPO/configs/${DDP_EXPERIMENT}_reference.csv" "$ddp_log/submitted_config.csv"
  fi
  printf 'Submitting %s; logs: %s\n' "$ddp_stage" "$ddp_log"
  if ! ddp_output="$(cd "$ddp_log" && anapy3 "${ddp_grid[@]}" "${ddp_python[@]}" 2>&1)"; then
    printf '%s\n' "$ddp_output" | tee "$ddp_log/submission.txt"
    return 1
  fi
  printf '%s\n' "$ddp_output" | tee "$ddp_log/submission.txt"
  # This is the job/job-array response observed from the user's GRID wrapper.
  ddp_id="$(printf '%s\n' "$ddp_output" | awk '$1 == "Your" && ($2 == "job" || $2 == "job-array") {split($3, a, "."); print a[1]}')"
  [[ "$ddp_id" =~ ^[0-9]+$ ]] || {
    echo "Could not identify one submitted job ID. No following stage was submitted; inspect $ddp_log/submission.txt before retrying." >&2
    return 1
  }
  DDP_SUBMITTED_JOB="$ddp_id"
  printf '%s\n' "$ddp_id" > "$ddp_log/job_id.txt"
}

case "${1:-all}" in
  all)
    [[ $# -le 1 ]] || { echo "all takes no dependency argument." >&2; exit 2; }
    ddp_submit reference
    DDP_REFERENCE_JOB="$DDP_SUBMITTED_JOB"
    ddp_submit fit "$DDP_REFERENCE_JOB"
    DDP_FIT_JOB="$DDP_SUBMITTED_JOB"
    ddp_submit summarize "$DDP_FIT_JOB"
    printf 'Queued reference %s -> five-area fit %s -> summary %s\n' "$DDP_REFERENCE_JOB" "$DDP_FIT_JOB" "$DDP_SUBMITTED_JOB"
    ;;
  reference|fit|summarize) ddp_submit "$1" "${2:-}" ;;
  *) echo "Usage: bash $0 {all|reference|fit|summarize} [dependency_job_id]" >&2; exit 2 ;;
esac
