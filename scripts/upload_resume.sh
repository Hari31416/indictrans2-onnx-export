#!/usr/bin/env bash
# Resume HF uploads from en-indic-1b-fp16 onward (after interrupted batch run).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HF_ORG="${HF_ORG:-hari31416}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Reduce bundle size after ONNX post-export layout optimization}"
PYTHON="${PYTHON:-.venv/bin/python}"
LOG="${LOG:-scratch/upload_resume.log}"

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
    log "DELETE skipped $repo"
  fi
}

upload_bundle() {
  local upload_script="$1"
  local folder="$2"
  local repo_base="$3"
  local precision_suffix="$4"

  local dir="scratch/${folder}"
  local repo="${HF_ORG}/${repo_base}${precision_suffix}"

  [[ -d "$dir" ]] || { log "SKIP missing $dir"; return 0; }

  log "=== START $folder -> $repo ==="
  "$PYTHON" "$upload_script" --model-dir "$dir" --repo-id "$repo" --dry-run >>"$LOG" 2>&1
  hf upload "$repo" "$dir" --repo-type model --exclude '.git/*' \
    --commit-message "$COMMIT_MESSAGE" >>"$LOG" 2>&1
  delete_stale_sidecars "$repo"
  log "=== DONE $repo ==="
}

log "Resume upload started"

declare -a REMAINING=(
  "src/v2/05_upload_hf.py:en-indic-1b-onnx-fp16:indictrans2-en-indic-1B-ONNX:-fp16"
  "src/v2/05_upload_hf.py:en-indic-1b-onnx-int8:indictrans2-en-indic-1B-ONNX:-int8"
  "src/v2/05_upload_hf.py:en-indic-1b-onnx-q4f16:indictrans2-en-indic-1B-ONNX:-q4f16"
  "src/v2/05_upload_hf.py:indic-en-1b-onnx:indictrans2-indic-en-1B-ONNX:"
  "src/v2/05_upload_hf.py:indic-en-1b-onnx-fp16:indictrans2-indic-en-1B-ONNX:-fp16"
  "src/v2/05_upload_hf.py:indic-en-1b-onnx-int8:indictrans2-indic-en-1B-ONNX:-int8"
  "src/v2/05_upload_hf.py:indic-en-1b-onnx-q4f16:indictrans2-indic-en-1B-ONNX:-q4f16"
  "src/v2/05_upload_hf.py:indic-indic-1b-onnx:indictrans2-indic-indic-1B-ONNX:"
  "src/v2/05_upload_hf.py:indic-indic-1b-onnx-fp16:indictrans2-indic-indic-1B-ONNX:-fp16"
  "src/v2/05_upload_hf.py:indic-indic-1b-onnx-int8:indictrans2-indic-indic-1B-ONNX:-int8"
  "src/v2/05_upload_hf.py:indic-indic-1b-onnx-q4f16:indictrans2-indic-indic-1B-ONNX:-q4f16"
)

for entry in "${REMAINING[@]}"; do
  IFS=':' read -r script folder base suffix <<< "$entry"
  upload_bundle "$script" "$folder" "$base" "$suffix"
done

log "Resume upload finished."
