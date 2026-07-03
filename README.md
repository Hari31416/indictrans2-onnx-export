# IndicTrans2 ONNX Export Pipeline

Dev/CI tooling to export browser-ready ONNX bundles for IndicTrans2 variants.
Not shipped in the Vite bundle.

## Prerequisites

- Python 3.10+
- ~8 GB disk for scratch artifacts
- GPU recommended for export validation (CPU works, slower)
- `make` (optional — convenience wrapper for pipeline steps)

```bash
cd indictrans2-onnx-export   # standalone repo — weights in scratch/, publish to HF
make setup                  # create venv + install deps
make help                   # list all targets
make indic-en               # full P0 pipeline (export → tokenizers → validate)
make en-indic               # full en→indic pipeline
make indic-indic            # full P1 pipeline (indic→indic 320M)
```

Manual venv setup (equivalent to `make setup`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Export targets

| Direction   | PyTorch base model                            | Output repo (set `--hf-org`)                   | Priority                      |
| ----------- | --------------------------------------------- | ---------------------------------------------- | ----------------------------- |
| indic→en    | `ai4bharat/indictrans2-indic-en-dist-200M`    | `{org}/indictrans2-indic-en-dist-200M-ONNX`    | P0                            |
| indic→indic | `ai4bharat/indictrans2-indic-indic-dist-320M` | `{org}/indictrans2-indic-indic-dist-320M-ONNX` | P1                            |
| en→indic    | `ai4bharat/indictrans2-en-indic-dist-200M`    | `{org}/indictrans2-en-indic-dist-200M-ONNX`    | done (also on naklitechie HF) |

Reference implementation: [naklitechie/indictrans2-en-indic-dist-200M-ONNX](https://huggingface.co/naklitechie/indictrans2-en-indic-dist-200M-ONNX)

See also: [EXPORT_ISSUES.md](./EXPORT_ISSUES.md) — full list of blockers and fixes.

## Pipeline

Individual steps map to `make` targets — see `make help`. Override paths with env vars, e.g. `make export-indic-en HF_ORG=my-org`.

### 1. Export ONNX graphs (encoder + decoder + decoder_with_past)

Uses **manual `torch.onnx.export`** (Optimum does not support the custom IndicTrans
architecture). Wrappers match the naklitechie I/O layout.

```bash
make export-indic-en
make export-en-indic
```
### 2. Build fast `tokenizer.json` files

```bash
make tokenizers-indic-en
```

Converts `model.SRC` / `model.TGT` via Hugging Face `SpmConverter`, then remaps
vocabulary IDs to match `dict.SRC.json` / `dict.TGT.json` (SPM-native indices
differ from Fairseq dictionary IDs). Language tags are registered as atomic
added tokens. Encode parity against the slow tokenizer is validated inline
(8 samples, must hit 100%).

### 3. Parity validation

```bash
make validate-indic-en
```

Validates ONNX graphs against PyTorch using the **fast tokenizers** for encoder
input and the slow tokenizer for decode post-processing (100% token match on 8
indic→en fixtures as of latest export).

**Pass criteria:** ≥ 99% token-exact match vs PyTorch greedy decode.

### 4. Optional INT8 quantization (after fp32 passes)

```bash
make quantize-indic-en
```

### 5. Upload to Hugging Face

```bash
make upload-en-indic HF_ORG=your-hf-org
make upload-indic-en HF_ORG=your-hf-org
make upload-indic-indic HF_ORG=your-hf-org
```

Bundles: `scratch/en-indic-onnx/` (~1.7 GB), `scratch/indic-en-onnx/` (~1.2 GB), `scratch/indic-indic-onnx/` (~1.9 GB).

After upload, update `IT2_ONNX_ORG` in `src/lib/translation-models.ts`.

## Known issues

| Issue                                                                      | Status                                                              |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Optimum `ORTModelForSeq2SeqLM` unsupported for IndicTrans                  | Fixed — manual export in `01_export_encoder_decoder.py`             |
| `decoder_with_past` needs dynamic axes + `encoder_attention_mask` in graph | Fixed                                                               |
| Fast tokenizer SPM indices ≠ dict.SRC/TGT IDs                              | Fixed — SpmConverter + vocab remap in `02_build_fast_tokenizers.py` |
| `model.generate()` broken on IndicTrans custom code                        | Workaround — manual greedy loop in `03_validate_parity.py`          |

## Fixtures

```bash
make capture-fixtures-indic-en
```

Golden fixtures: `fixtures/indic-en-golden.jsonl` (indic→en pairs only).

Categories: generic, politics, numerals, lexicon — expand to 11+ languages × 12 sentences for full parity suites.

## indic→indic notes (320M)

```bash
make indic-indic            # export + tokenizers + validate
```

- Bundle size: **~1.9 GB fp32** (`scratch/indic-indic-onnx/`)
- 462 translation directions via `src_lang` / `tgt_lang` tags
- Parity: **12/12 (100%)** on `fixtures/indic-indic-golden.jsonl` (Devanagari, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Assamese pairs)
- Fast tokenizer encode parity: 100% (same SpmConverter + dict remap pipeline)
