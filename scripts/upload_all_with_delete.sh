#!/usr/bin/env bash
# Upload all ONNX bundles to Hugging Face with stale sidecar deletion.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HF_ORG="${HF_ORG:-hari31416}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Reduce bundle size after ONNX post-export layout optimization}"
PYTHON="${PYTHON:-.venv/bin/python}"
LOG="${LOG:-scratch/upload_all_with_delete.log}"

mkdir -p "$(dirname "$LOG")"
: >"$LOG"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

delete_stale_sidecars() {
  local repo="$1"
  log "DELETE stale sidecars on $repo"
  if hf repo-files delete "$repo" --repo-type model \
    decoder_model.onnx.data decoder_with_past_model.onnx.data \
    --commit-message "Remove obsolete per-decoder weight sidecars" >>"$LOG" 2>&1; then
    log "DELETE ok $repo"
  else
    log "DELETE skipped or failed $repo (files may already be gone)"
  fi
}

upload_bundle() {
  local upload_script="$1"
  local folder="$2"
  local repo_base="$3"
  local precision_suffix="$4"

  local dir="scratch/${folder}"
  local repo="${HF_ORG}/${repo_base}${precision_suffix}"

  if [[ ! -d "$dir" ]]; then
    log "SKIP missing directory: $dir"
    return 0
  fi

  log "=== START $folder -> $repo ==="

  "$PYTHON" "$upload_script" \
    --model-dir "$dir" \
    --repo-id "$repo" \
    --dry-run >>"$LOG" 2>&1

  hf upload "$repo" "$dir" \
    --repo-type model \
    --exclude '.git/*' \
    --commit-message "$COMMIT_MESSAGE" >>"$LOG" 2>&1

  delete_stale_sidecars "$repo"

  log "=== DONE $repo ==="
}

log "Upload started (HF_ORG=$HF_ORG)"

declare -a DIRECTIONS_200M=(
  "en-indic:indictrans2-en-indic-dist-200M-ONNX"
  "indic-en:indictrans2-indic-en-dist-200M-ONNX"
  "indic-indic:indictrans2-indic-indic-dist-320M-ONNX"
)

declare -a DIRECTIONS_1B=(
  "en-indic:indictrans2-en-indic-1B-ONNX"
  "indic-en:indictrans2-indic-en-1B-ONNX"
  "indic-indic:indictrans2-indic-indic-1B-ONNX"
)

declare -a PRECISIONS=(fp32 fp16 int8 q4f16)

for entry in "${DIRECTIONS_200M[@]}"; do
  direction="${entry%%:*}"
  repo_base="${entry#*:}"
  for precision in "${PRECISIONS[@]}"; do
    if [[ "$precision" == "fp32" ]]; then
      folder="${direction}-onnx"
      suffix=""
    else
      folder="${direction}-onnx-${precision}"
      suffix="-${precision}"
    fi
    upload_bundle "src/05_upload_hf.py" "$folder" "$repo_base" "$suffix"
  done
done

for entry in "${DIRECTIONS_1B[@]}"; do
  direction="${entry%%:*}"
  repo_base="${entry#*:}"
  for precision in "${PRECISIONS[@]}"; do
    if [[ "$precision" == "fp32" ]]; then
      folder="${direction}-1b-onnx"
      suffix=""
    else
      folder="${direction}-1b-onnx-${precision}"
      suffix="-${precision}"
    fi
    upload_bundle "src/v2/05_upload_hf.py" "$folder" "$repo_base" "$suffix"
  done
done

log "All uploads finished successfully."
