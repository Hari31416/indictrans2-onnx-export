# IndicTrans2 ONNX Export Pipeline

Standalone tooling to export [IndicTrans2](https://huggingface.co/collections/ai4bharat/indictrans2-664ccb91d23bbae0d681c3ca) seq2seq models to browser-ready ONNX bundles (encoder + decoder + decoder_with_past).

## Why this exists

These exports support **[local-voice-chat](https://github.com/Hari31416/local-voice-chat)** — a local, in-browser voice chat project with live translation for Indian languages. The voice app stays lightweight (TypeScript/Vite); this repo holds the heavy Python export pipeline and publishes weights to Hugging Face. Code lives here, ONNX artifacts live in `scratch/` (gitignored) and on HF.

Consumer integration (IndicProcessor TS port, translation worker, model catalog) is in the voice chat repo under `src/lib/translation/`.

## Repo layout

```
indictrans2-onnx-export/
├── src/                 # export scripts (run via Makefile)
├── fixtures/            # golden sentences + parity reports (committed)
├── scratch/             # fp32 ONNX bundles (gitignored — upload to HF)
├── Makefile             # pipeline targets
├── pyproject.toml       # Python deps (uv)
├── EXPORT_ISSUES.md     # fp32 export blockers and fixes
└── ROADMAP.md           # fp16 / int8 / bf16 / q4f16 plan
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- ~8 GB disk for scratch artifacts
- GPU recommended for export validation (CPU works, slower)
- `make` (optional)

```bash
git clone <this-repo>
cd indictrans2-onnx-export
make setup
make help
make indic-en               # full pipeline: export → tokenizers → validate
make en-indic
make indic-indic
```

Manual setup:

```bash
uv sync
source .venv/bin/activate
```

## Export targets

| Direction   | PyTorch base model                            | HF output repo (`HF_ORG`)                      | fp32 status   |
| ----------- | --------------------------------------------- | ---------------------------------------------- | ------------- |
| en→indic    | `ai4bharat/indictrans2-en-indic-dist-200M`    | `{org}/indictrans2-en-indic-dist-200M-ONNX`    | ✅ validated  |
| indic→en    | `ai4bharat/indictrans2-indic-en-dist-200M`    | `{org}/indictrans2-indic-en-dist-200M-ONNX`    | ✅ validated  |
| indic→indic | `ai4bharat/indictrans2-indic-indic-dist-320M` | `{org}/indictrans2-indic-indic-dist-320M-ONNX` | ✅ validated  |

Reference I/O layout: [naklitechie/indictrans2-en-indic-dist-200M-ONNX](https://huggingface.co/naklitechie/indictrans2-en-indic-dist-200M-ONNX)

See also:

- [EXPORT_ISSUES.md](./EXPORT_ISSUES.md) — problems hit during fp32 export and fixes
- [ROADMAP.md](./ROADMAP.md) — planned fp16, int8, bf16, q4f16 work

## Pipeline

Override paths with env vars, e.g. `make export-indic-en HF_ORG=my-org`.

### 1. Export ONNX graphs

Manual `torch.onnx.export` with wrappers matching naklitechie I/O (Optimum does not support IndicTrans).

```bash
make export-indic-en
make export-en-indic
make export-indic-indic
```

### 2. Build fast tokenizers

```bash
make tokenizers-indic-en
```

`SpmConverter` + dict remap → `tokenizer_src.json`, `tokenizer_tgt.json`, `tokenizer_meta.json`. Inline encode parity vs slow HF tokenizer must hit 100%.

### 3. Parity validation

```bash
make validate-indic-en
```

PyTorch vs ONNX greedy decode on direction-specific fixtures. **Pass criteria:** ≥ 99% token-exact match.

### 4. Optional INT8 (after fp32 passes)

```bash
make quantize-indic-en
```

See [ROADMAP.md](./ROADMAP.md) for fp16 / q4f16 plans — not implemented yet.

### 5. Upload to Hugging Face

Uploads the local ONNX bundles to Hugging Face Hub using the new `hf upload` CLI. It automatically generates and includes a model card `README.md` inside the repository.

```bash
make upload-en-indic HF_ORG=your-hf-org COMMIT_MESSAGE="Initial release of en-indic ONNX bundle"
make upload-indic-en HF_ORG=your-hf-org COMMIT_MESSAGE="Initial release of indic-en ONNX bundle"
make upload-indic-indic HF_ORG=your-hf-org COMMIT_MESSAGE="Initial release of indic-indic ONNX bundle"
```

Local bundles: `scratch/en-indic-onnx/` (~1.7 GB), `scratch/indic-en-onnx/` (~1.2 GB), `scratch/indic-indic-onnx/` (~1.9 GB).

After upload, set bundle paths in [local-voice-chat `translation-models.ts`](https://github.com/Hari31416/local-voice-chat/blob/main/src/lib/translation-models.ts).

### 6. Smoke-Test Verification & Translation Matrix

To verify that the exported ONNX models translate correctly across target languages and to generate translation matrices, run:

```bash
# Verify model download and translate a sample sentence
.venv/bin/python src/test_hf_models.py --repo-id your-hf-org/indictrans2-indic-en-dist-200M-ONNX

# Generate translation matrix of smoke-test sentences
.venv/bin/python src/generate_translation_matrix.py --repo-id scratch/en-indic-onnx
.venv/bin/python src/generate_translation_matrix.py --repo-id scratch/indic-en-onnx
.venv/bin/python src/generate_translation_matrix.py --repo-id scratch/indic-indic-onnx
```

Input sentences are loaded from `fixtures/smoke-test/test_sentences_<lang>.json`, and the resulting matrices are saved to `fixtures/smoke-test/translation_matrix_<direction>.json`.

## fp32 parity summary

| Direction   | Fixtures | Token | Text |
| ----------- | -------- | ----- | ---- |
| en→indic    | 8        | 100%  | 100% |
| indic→en    | 8        | 100%  | 100% |
| indic→indic | 12       | 100%  | 100% |

## Known issues (fp32)

| Issue | Status |
| ----- | ------ |
| Optimum unsupported for IndicTrans | Fixed — `src/01_export_encoder_decoder.py` |
| `decoder_with_past` dynamic axes + mask in graph | Fixed |
| Fast tokenizer SPM ≠ dict IDs | Fixed — `src/02_build_fast_tokenizers.py` |
| `model.generate()` broken | Workaround — manual greedy loop in `src/03_validate_parity.py` |
| Cross-attention skipped in step 2+ | Fixed — `src/it2_onnx_wrappers.py` |

Full list: [EXPORT_ISSUES.md](./EXPORT_ISSUES.md)

## Fixtures

```bash
make capture-fixtures-indic-en
```

Golden files: `fixtures/*-golden.jsonl`. Expand to 11+ languages × 12 sentences for production confidence.

## Documentation and ONNX Bundle Guide

An interactive, self-contained HTML guide explaining the internals of the exported ONNX Translation Bundle is available in [onnx-components.html](./onnx-components.html).

### Previewing Locally

To run a local web server and view the guide in your browser:

```bash
make preview
```

Then open your browser and navigate to [http://localhost:8000/onnx-components.html](http://localhost:8000/onnx-components.html).

### GitHub Pages Deployment

The guide is configured to deploy automatically to GitHub Pages on every push to the `main` branch via a GitHub Actions workflow.

To enable the deployment:

- Go to your GitHub repository settings at `https://github.com/Hari31416/indictrans2-onnx-export/settings/pages`.
- Under **Build and deployment** -> **Source**, select **GitHub Actions** (instead of "Deploy from a branch").

Once enabled, the deployment workflow will run, and the documentation will be live at `https://hari31416.github.io/indictrans2-onnx-export/`.

