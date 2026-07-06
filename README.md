# IndicTrans2 ONNX Export Pipeline

Standalone tooling to export [IndicTrans2](https://huggingface.co/collections/ai4bharat/indictrans2-664ccb91d23bbae0d681c3ca) seq2seq models to browser-ready ONNX bundles (encoder + decoder + decoder_with_past).

## Why this exists

These exports support **[local-voice-chat](https://github.com/Hari31416/local-voice-chat)** — a local, in-browser voice chat project with live translation for Indian languages. The voice app stays lightweight (TypeScript/Vite); this repo holds the heavy Python export pipeline and publishes weights to Hugging Face. Code lives here, ONNX artifacts live in `scratch/` (gitignored) and on HF.

Consumer integration (IndicProcessor TS port, translation worker, model catalog) is in the voice chat repo under `src/lib/translation/`.

## Repo layout

```
indictrans2-onnx-export/
├── browser-lab/         # local ONNX bundle tester (load scratch/ in browser)
├── src/                 # export scripts (run via Makefile)
│   └── onnx_bundle_optimize.py  # post-export size optimizations
├── fixtures/            # golden sentences + parity reports (committed)
├── scratch/             # fp32 ONNX bundles (gitignored — upload to HF)
├── Makefile             # pipeline targets
└── pyproject.toml       # Python deps (uv)
```

> [!TIP]
>Try out the translation models directly in your browser:
>
>- **Hugging Face Space**: [indictrans2-onnx-browser-demo](https://huggingface.co/spaces/hari31416/indictrans2-onnx-browser-demo)
>- **GitHub Pages**: [indictrans2-onnx-browser-demo](https://hari31416.github.io/indictrans2-onnx-browser-demo/)
>
>The code is available in [indictrans2-onnx-browser-demo](https://github.com/Hari31416/indictrans2-onnx-browser-demo). The application loads ONNX models on-demand and runs execution client-side via WebGPU or WebAssembly.

To test **local exports** from `scratch/` before publishing, use the browser lab:

```bash
make browser-lab   # http://127.0.0.1:8010 — pick your scratch/ bundle folder
```

See [browser-lab/README.md](./browser-lab/README.md).

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

### 200M/320M Models (Distilled)

| Direction   | PyTorch base model                            | HF output repo (`HF_ORG`)                      | fp32 status   |
| ----------- | --------------------------------------------- | ---------------------------------------------- | ------------- |
| en→indic    | `ai4bharat/indictrans2-en-indic-dist-200M`    | `{org}/indictrans2-en-indic-dist-200M-ONNX`    | ✅ validated  |
| indic→en    | `ai4bharat/indictrans2-indic-en-dist-200M`    | `{org}/indictrans2-indic-en-dist-200M-ONNX`    | ✅ validated  |
| indic→indic | `ai4bharat/indictrans2-indic-indic-dist-320M` | `{org}/indictrans2-indic-indic-dist-320M-ONNX` | ✅ validated  |

Reference I/O layout: [naklitechie/indictrans2-en-indic-dist-200M-ONNX](https://huggingface.co/naklitechie/indictrans2-en-indic-dist-200M-ONNX)

### 1B Models (Full)

| Direction   | PyTorch base model                    | HF output repo (`HF_ORG`)              | fp32 status   |
| ----------- | ------------------------------------- | -------------------------------------- | ------------- |
| en→indic    | `ai4bharat/indictrans2-en-indic-1B`    | `{org}/indictrans2-en-indic-1B-ONNX`    | ✅ validated  |
| indic→en    | `ai4bharat/indictrans2-indic-en-1B`    | `{org}/indictrans2-indic-en-1B-ONNX`    | ✅ validated  |
| indic→indic | `ai4bharat/indictrans2-indic-indic-1B` | `{org}/indictrans2-indic-indic-1B-ONNX` | ✅ validated  |

## Pipeline

Override paths with env vars, e.g. `make export-indic-en HF_ORG=my-org`.

### 1. Export ONNX graphs

Manual `torch.onnx.export` with wrappers matching naklitechie I/O (Optimum does not support IndicTrans).

```bash
make export-indic-en
make export-en-indic
make export-indic-indic
```

Each export produces three ONNX graphs plus tokenizer/config artifacts. After export, the pipeline runs size optimizations (tied-weight dedup, ORT graph fusion, weight externalization, shared decoder sidecar). A typical **200M en→indic fp32** bundle layout:

| File | Role |
| ---- | ---- |
| `encoder_model.onnx` | Small graph proto |
| `encoder_model.onnx.data` | Encoder weights (~294 MB fp32) |
| `decoder_model.onnx` | First decode step (small proto) |
| `decoder_with_past_model.onnx` | Autoregressive steps 2+ (small proto) |
| `decoder_shared.onnx.data` | **Shared** decoder weights for both decoder graphs |
| `tokenizer_*.json`, `config.json`, … | Tokenizer + HF config |

> Both decoder graphs reference `decoder_shared.onnx.data` — upload and ship all `.onnx` + `.data` files together. FP16 / INT8 / Q4F16 conversion steps re-apply the same layout automatically.

1B exports use `src/v2/01_export_encoder_decoder.py` (same artifact layout; larger sidecars).

### ONNX bundle size optimizations

Raw `torch.onnx.export` traces produce functionally correct bundles that are ~40% larger than necessary. After each export (and again after FP16 / INT8 / Q4F16 conversion), the pipeline runs automatic post-processing via `src/onnx_bundle_optimize.py`.

| Optimization | What it does | fp32 impact |
| ------------ | ------------ | ----------- |
| Tied-weight dedup | IndicTrans2 ties `decoder.embed_tokens` and `lm_head` on **en→indic** and **indic→en**; export used to serialize both. Post-export, embedding `Gather` routes through `lm_head.weight` when weights are verified tied. **Skipped on indic→indic** (separate embed / lm_head matrices) | ~−500 MB (tied models only) |
| ORT graph fusion | ONNX Runtime graph optimizer fuses redundant ops (e.g. 3,696 → 1,662 nodes on `decoder_model`) | Moderate proto size; faster load |
| Weight externalization | Inline weights above 100 MB move to `.onnx.data` sidecars; graph protos stay small for HF browsing | Layout only for encoder |
| Shared decoder sidecar | `decoder_model` and `decoder_with_past` share one `decoder_shared.onnx.data` via content-addressed offsets instead of two full copies | ~−550 MB |

**Before vs after (200M en→indic fp32):** ~1.9 GB → ~1.05 GB on disk, with 100% greedy-decode parity preserved. Indic→indic (320M) keeps separate embed / lm_head weights and a larger fp32 bundle (~1.3 GB after shared-decoder and ORT fusion).

The optimizations run in this order:

1. Export three graphs with `src/it2_onnx_wrappers.py` (standard `input_ids` decode path — required for IndicTrans correctness)
2. `optimize_export_bundle()` — dedup, ORT fusion, externalize, share decoder weights
3. `finalize_bundle_layout()` — re-applied after `05_convert_fp16.py`, `04_quantize_int8.py`, and `06_quantize_q4f16.py` so quantized tiers keep the same sidecar layout

For the full size analysis, comparison against the [naklitechie reference bundle](https://huggingface.co/naklitechie/indictrans2-en-indic-dist-200M-ONNX), and future optimization ideas, see [ONNX_SIZE_OPTIMIZATION.md](./ONNX_SIZE_OPTIMIZATION.md).

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

### 5. Upload to Hugging Face

Uploads the local ONNX bundles to Hugging Face Hub using the new `hf upload` CLI. It automatically generates and includes a model card `README.md` inside the repository.

```bash
make upload-en-indic HF_ORG=your-hf-org COMMIT_MESSAGE="Initial release of en-indic ONNX bundle"
make upload-indic-en HF_ORG=your-hf-org COMMIT_MESSAGE="Initial release of indic-en ONNX bundle"
make upload-indic-indic HF_ORG=your-hf-org COMMIT_MESSAGE="Initial release of indic-indic ONNX bundle"
```

Local bundles: `scratch/en-indic-onnx/` (~1.1 GB), `scratch/indic-en-onnx/` (~0.8 GB), `scratch/indic-indic-onnx/` (~1.1 GB).

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

## Parity and Benchmark Reports

Parity validation (`03_validate_parity.py`) and quantization benchmarks (`07_benchmark_precision.py`) run **batched greedy decode** via `src/it2_inference.py`. Fixtures are grouped by `(src_lang, tgt_lang)` and processed in batches (default batch size **16**). Override with `--batch-size N` or `make validate-en-indic EVAL_BATCH_SIZE=32`.

### FP32 Parity Summary

| Direction | Fixtures | Token Pass Rate | Text Pass Rate | Model Size | Validation Time |
| - | - | - | - | - | - |
| en-indic | 1100 | 100.0% | 100.0% | ~1.06 GB | 210s |
| indic-en | 1100 | 100.0% | 100.0% | ~0.89 GB | — |
| indic-indic | 1100 | 100.0% | 100.0% | ~1.25 GB | — |

### Quantization Benchmarks (200M/320M Models)

Compared against the FP32 ONNX oracle on the same 1100 golden fixtures.

| Direction | Format | Token Match | Text Match | Model Size | FP32 Latency | Quant Latency | Speedup |
| - | - | - | - | - | - | - | - |
| en-indic | FP16 | 99.64% | 99.64% | 559.6 MB | 18.3ms | 24.8ms | 0.736x |
| en-indic | INT8 | 74.36% | 74.36% | 302.7 MB | 21.0ms | 13.2ms | 1.594x |
| en-indic | Q4F16 | 55.18% | 55.64% | 380.6 MB | 19.2ms | 27.3ms | 0.705x |
| indic-en | FP16 | 99.91% | 99.91% | 471.4 MB | 12.2ms | 14.3ms | 0.849x |
| indic-en | INT8 | 85.64% | 85.64% | 257.3 MB | 11.8ms | 10.1ms | 1.171x |
| indic-en | Q4F16 | 73.36% | 73.36% | 292.4 MB | 11.7ms | 15.8ms | 0.742x |
| indic-indic | FP16 | 99.82% | 99.82% | 671.9 MB | 23.0ms | 27.4ms | 0.840x |
| indic-indic | INT8 | 72.18% | 72.36% | 370.9 MB | 24.3ms | 16.5ms | 1.475x |
| indic-indic | Q4F16 | 45.91% | 46.36% | 492.9 MB | 23.5ms | 28.3ms | 0.829x |

### Quantization Benchmarks (1B Models)

Compared against the FP32 ONNX oracle on the same 1100 golden fixtures.

| Direction | Format | Token Match | Text Match | Model Size | FP32 Latency | Quant Latency | Speedup |
| - | - | - | - | - | - | - | - |
| en-indic | FP16 | 99.73% | 99.73% | 2.11 GB | 69.5ms | 74.3ms | 0.935x |
| en-indic | INT8 | 89.55% | 89.55% | 1.08 GB | 66.7ms | 31.4ms | 2.125x |
| en-indic | Q4F16 | 82.45% | 82.55% | 1.01 GB | 69.3ms | 58.4ms | 1.186x |
| indic-en | FP16 | 99.82% | 99.82% | 1.94 GB | 49.0ms | 49.7ms | 0.987x |
| indic-en | INT8 | 94.45% | 94.45% | 1020.1 MB | 47.9ms | 25.2ms | 1.900x |
| indic-en | Q4F16 | 88.55% | 88.55% | 861.5 MB | 46.2ms | 42.7ms | 1.080x |
| indic-indic | FP16 | 99.82% | 99.82% | 2.31 GB | 94.7ms | 108.3ms | 0.874x |
| indic-indic | INT8 | 83.64% | 83.73% | 1.19 GB | 97.8ms | 43.7ms | 2.240x |
| indic-indic | Q4F16 | 73.18% | 73.18% | 1.21 GB | 102.3ms | 94.2ms | 1.087x |

> [!NOTE]
> **Quantization performance of 1B vs 200M models**: Larger models are more robust to quantization. On 200M/320M bundles, INT8 and Q4F16 can drop sharply (e.g. en→indic Q4F16 at 55% token match, indic→indic INT8 at 72%). The 1B variants hold up much better (e.g. indic→en INT8 at 94%, en→indic Q4F16 at 82%). FP16 remains lossless (≥99%) across all directions and sizes.

### Detailed Benchmarks & Visualizations

Detailed language-level and category-level charts and tables for all quantization tiers are available in:
- **[BENCHMARKS.md (200M/320M Models)](./BENCHMARKS.md)**
- **[BENCHMARKS_1B.md (1B Models)](./BENCHMARKS_1B.md)**

For direction-specific visualizations, see:
- **[EN-INDIC Benchmarks (200M)](./BENCHMARKS.md#en-indic-model-performance)** (Plots: [Overall](./fixtures/en_indic_overall.png), [Languages](./fixtures/en_indic_languages.png), [Categories](./fixtures/en_indic_categories.png))
- **[EN-INDIC Benchmarks (1B)](./BENCHMARKS_1B.md#en-indic-model-performance)** (Plots: [Overall](./fixtures/en_indic_1b_overall.png), [Languages](./fixtures/en_indic_1b_languages.png), [Categories](./fixtures/en_indic_1b_categories.png))
- **[INDIC-EN Benchmarks (200M)](./BENCHMARKS.md#indic-en-model-performance)** (Plots: [Overall](./fixtures/indic_en_overall.png), [Languages](./fixtures/indic_en_languages.png), [Categories](./fixtures/indic_en_categories.png))
- **[INDIC-EN Benchmarks (1B)](./BENCHMARKS_1B.md#indic-en-model-performance)** (Plots: [Overall](./fixtures/indic_en_1b_overall.png), [Languages](./fixtures/indic_en_1b_languages.png), [Categories](./fixtures/indic_en_1b_categories.png))
- **[INDIC-INDIC Benchmarks (320M)](./BENCHMARKS.md#indic-indic-model-performance)** (Plots: [Overall](./fixtures/indic_indic_overall.png), [Languages](./fixtures/indic_indic_languages.png), [Categories](./fixtures/indic_indic_categories.png))
- **[INDIC-INDIC Benchmarks (1B)](./BENCHMARKS_1B.md#indic-indic-model-performance)** (Plots: [Overall](./fixtures/indic_indic_1b_overall.png), [Languages](./fixtures/indic_indic_1b_languages.png), [Categories](./fixtures/indic_indic_1b_categories.png))

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
