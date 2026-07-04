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
└── pyproject.toml       # Python deps (uv)
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

## Parity and Benchmark Reports

### FP32 Parity Summary

| Direction | Fixtures | Token Pass Rate | Text Pass Rate | Model Size | Validation Time |
| - | - | - | - | - | - |
| en-indic | 264 | 100.0% | 100.0% | 1.74 GB | 48.03s |
| indic-en | 264 | 100.0% | 100.0% | 1.22 GB | 37.85s |
| indic-indic | 264 | 100.0% | 100.0% | 1.91 GB | 52.34s |

### Quantization Benchmarks (200M/320M Models)

Compared against the FP32 ONNX oracle on the same 264 golden fixtures.

| Direction | Format | Token Match | Text Match | Model Size | FP32 Latency | Quant Latency | Speedup |
| - | - | - | - | - | - | - | - |
| en-indic | FP16 | 100.0% | 100.0% | 892.0 MB | 61.5ms | 66.6ms | 0.923x |
| en-indic | INT8 | 79.55% | 79.92% | 452.9 MB | 63.1ms | 37.7ms | 1.674x |
| en-indic | Q4F16 | 63.64% | 64.39% | 623.3 MB | 62.7ms | 59.4ms | 1.055x |
| indic-en | FP16 | 99.62% | 99.62% | 627.2 MB | 37.6ms | 44.1ms | 0.853x |
| indic-en | INT8 | 81.44% | 81.44% | 319.7 MB | 39.2ms | 33.0ms | 1.187x |
| indic-en | Q4F16 | 65.91% | 65.91% | 358.6 MB | 39.7ms | 43.1ms | 0.922x |
| indic-indic | FP16 | 100.0% | 100.0% | 980.2 MB | 64.7ms | 70.5ms | 0.918x |
| indic-indic | INT8 | 78.41% | 79.17% | 497.1 MB | 65.0ms | 40.3ms | 1.612x |
| indic-indic | Q4F16 | 56.06% | 57.58% | 711.6 MB | 64.1ms | 65.3ms | 0.983x |

### Quantization Benchmarks (1B Models)

Compared against the FP32 ONNX oracle on the same 264 golden fixtures.

| Direction | Format | Token Match | Text Match | Model Size | FP32 Latency | Quant Latency | Speedup |
| - | - | - | - | - | - | - | - |
| en-indic | FP16 | 99.73% | 99.73% | 3.36 GB | 244.4ms | 259.8ms | 0.941x |
| en-indic | INT8 | 89.64% | 89.73% | 1.71 GB | 244.4ms | 112.6ms | 2.228x |
| en-indic | Q4F16 | 82.27% | 82.36% | 1.71 GB | 244.4ms | 143.3ms | 1.673x |
| indic-en | FP16 | 99.91% | 99.91% | 2.84 GB | 171.8ms | 180.5ms | 0.952x |
| indic-en | INT8 | 94.18% | 94.18% | 1.45 GB | 171.8ms | 76.4ms | 2.196x |
| indic-en | Q4F16 | 87.55% | 87.55% | 1.19 GB | 171.8ms | 85.5ms | 1.962x |
| indic-indic | FP16 | 99.82% | 99.82% | 3.55 GB | 251.7ms | 270.4ms | 0.931x |
| indic-indic | INT8 | 84.36% | 84.36% | 1.82 GB | 251.7ms | 109.5ms | 2.292x |
| indic-indic | Q4F16 | 73.73% | 73.73% | 1.90 GB | 251.7ms | 151.0ms | 1.666x |

> [!NOTE]
> **Quantization Performance of 1B vs 200M Models**: Larger models possess greater representation capacity and are significantly more robust to quantization. While the 200M/320M model formats experience notable accuracy drops under INT8 and Q4F16 (e.g., indic-indic exact token match drops to 73.0% and 44.9%), the 1B configurations maintain exceptionally high parity (exact token match of 84.4% for INT8 and 73.7% for Q4F16 on indic-indic, and up to 94.2% for INT8 on indic-en).

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
