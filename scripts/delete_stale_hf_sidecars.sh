#!/usr/bin/env bash
# Remove obsolete per-decoder weight sidecars from all HF model repos.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HF_ORG="${HF_ORG:-hari31416}"
LOG="${LOG:-scratch/delete_stale_sidecars.log}"

mkdir -p "$(dirname "$LOG")"
: >"$LOG"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

delete_repo() {
  local repo="$1"
  log "DELETE $repo"
  if hf repo-files delete "$repo" --repo-type model \
    decoder_model.onnx.data decoder_with_past_model.onnx.data \
    --commit-message "Remove obsolete per-decoder weight sidecars" >>"$LOG" 2>&1; then
    log "OK $repo"
  else
    log "SKIP $repo (nothing to delete or repo missing)"
  fi
}

declare -a REPOS=(
  indictrans2-en-indic-dist-200M-ONNX
  indictrans2-en-indic-dist-200M-ONNX-fp16
  indictrans2-en-indic-dist-200M-ONNX-int8
  indictrans2-en-indic-dist-200M-ONNX-q4f16
  indictrans2-indic-en-dist-200M-ONNX
  indictrans2-indic-en-dist-200M-ONNX-fp16
  indictrans2-indic-en-dist-200M-ONNX-int8
  indictrans2-indic-en-dist-200M-ONNX-q4f16
  indictrans2-indic-indic-dist-320M-ONNX
  indictrans2-indic-indic-dist-320M-ONNX-fp16
  indictrans2-indic-indic-dist-320M-ONNX-int8
  indictrans2-indic-indic-dist-320M-ONNX-q4f16
  indictrans2-en-indic-1B-ONNX
  indictrans2-en-indic-1B-ONNX-fp16
  indictrans2-en-indic-1B-ONNX-int8
  indictrans2-en-indic-1B-ONNX-q4f16
  indictrans2-indic-en-1B-ONNX
  indictrans2-indic-en-1B-ONNX-fp16
  indictrans2-indic-en-1B-ONNX-int8
  indictrans2-indic-en-1B-ONNX-q4f16
  indictrans2-indic-indic-1B-ONNX
  indictrans2-indic-indic-1B-ONNX-fp16
  indictrans2-indic-indic-1B-ONNX-int8
  indictrans2-indic-indic-1B-ONNX-q4f16
)

log "Deleting stale sidecars (HF_ORG=$HF_ORG)"
for base in "${REPOS[@]}"; do
  delete_repo "${HF_ORG}/${base}"
done
log "Finished."
