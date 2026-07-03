# IndicTrans2 ONNX Export — Issues and Fixes

Notes from exporting all three IndicTrans2 directions to browser-ready ONNX bundles.
Standalone export repo. Scratch ONNX bundles live in `scratch/` (gitignored — publish to Hugging Face).

## Summary

| Direction   | Model                                         | Bundle size | Parity       | Status           |
| ----------- | --------------------------------------------- | ----------- | ------------ | ---------------- |
| en→indic    | `ai4bharat/indictrans2-en-indic-dist-200M`    | ~1.7 GB     | 8/8 (100%)   | Exported locally |
| indic→en    | `ai4bharat/indictrans2-indic-en-dist-200M`    | ~1.2 GB     | 8/8 (100%)   | Exported locally |
| indic→indic | `ai4bharat/indictrans2-indic-indic-dist-320M` | ~1.9 GB     | 12/12 (100%) | Exported locally |

Reference bundle (pre-existing): [naklitechie/indictrans2-en-indic-dist-200M-ONNX](https://huggingface.co/naklitechie/indictrans2-en-indic-dist-200M-ONNX) — used as I/O layout reference; our en→indic export replicates it.

---

## 1. Optimum does not support IndicTrans

**Symptom:** `ValueError: custom IndicTrans architecture` when using `ORTModelForSeq2SeqLM` or `optimum.exporters`.

**Fix:** Bypass Optimum entirely. Use manual `torch.onnx.export` with PyTorch wrappers in `it2_onnx_wrappers.py` that match the naklitechie I/O contract:

- Encoder: `input_ids`, `attention_mask` → `last_hidden_state`
- Decoder step 1: `input_ids`, `encoder_hidden_states`, `encoder_attention_mask` → `logits` + `present.{i}.{decoder,encoder}.{key,value}`
- Decoder step 2+: `input_ids`, `encoder_attention_mask`, `past_key_values.*` → `logits` + presents (no `encoder_hidden_states` input)

**File:** `01_export_encoder_decoder.py`

---

## 2. Missing `onnxscript` dependency

**Symptom:** Export fails with import error for `onnxscript` when PyTorch tries the dynamo exporter.

**Fix:** Add `onnxscript>=0.1.0` to `requirements.txt`. Pass `dynamo=False` to `torch.onnx.export` to use the legacy tracer (more reliable for this model).

---

## 3. `save_model` import path

**Symptom:** `ImportError` when externalizing large weight sidecars.

**Fix:** Use `from onnx import save_model` (not `from onnx.save_model import save_model`).

---

## 4. `decoder_with_past` past KV shape mismatch

**Symptom:** ONNX graph traced with wrong past sequence length; runtime decode fails or produces garbage after step 1.

**Root cause:** Dummy past tensors used seq len 2 during tracing, but greedy decode feeds seq len 1 per step.

**Fix:** Trace `decoder_with_past` with past decoder KV tensors of shape `(batch, heads, 1, head_dim)`.

---

## 5. Fixed encoder sequence length in past KV

**Symptom:** Graph only works for the traced encoder length (8 tokens); longer/shorter inputs fail.

**Fix:** Add dynamic axes on all `past_key_values.*` and `present.*` tensors for encoder sequence dimension.

---

## 6. `encoder_attention_mask` dropped from decoder graph

**Symptom:** ONNX optimizer or tracer elides `encoder_attention_mask` because it appears unused in the forward pass.

**Fix:** Force the input to remain in the graph via a zero-cost dependency in the wrapper:

```python
logits = logits + encoder_attention_mask.sum() * 0.0
```

**File:** `it2_onnx_wrappers.py`

---

## 7. `model.generate()` broken on IndicTrans custom code

**Symptom:** `AttributeError` related to `use_cache` when calling `model.generate()` on the HF IndicTrans model.

**Fix:** Implement manual greedy decode in `03_validate_parity.py` for both PyTorch and ONNX paths. Merge decoder self-attn KV with carried encoder cross-attn KV between steps (same logic as export wrappers).

---

## 8. Wrong parity fixtures for indic→en

**Symptom:** 0% parity when running indic→en ONNX against en→indic golden fixtures (mixed directions).

**Fix:** Create direction-specific fixture files:

- `fixtures/en-indic-golden.jsonl` — `eng_Latn` → indic targets
- `fixtures/indic-en-golden.jsonl` — indic → `eng_Latn`
- `fixtures/indic-indic-golden.jsonl` — indic → indic pairs

---

## 9. Fast tokenizer swap approach (failed)

**Symptom:** Swapping naklitechie en→indic `tokenizer_src.json` / `tokenizer_tgt.json` for indic→en gave 0% token match vs slow HF tokenizer.

**Root cause:** indic→en `dict.SRC.json` interleaves language tags at different IDs than the en→indic bundle. Vocab sizes and token ID mappings are not simple mirrors.

**Fix:** Build fast tokenizers from scratch per model (see #10). Removed swap logic from `02_build_fast_tokenizers.py`.

---

## 10. Fast tokenizer SPM indices ≠ dict IDs

**Symptom:** `SpmConverter` output uses SentencePiece-native token IDs, which do not match `dict.SRC.json` / `dict.TGT.json` IDs expected by the model.

**Fix (ported from `phase2/gemini` branch):**

1. Convert `model.SRC` / `model.TGT` via Hugging Face `SpmConverter`
2. Remap every entry in `model.vocab` to the ID from the corresponding dict JSON
3. Register language tags (`hin_Deva`, etc.) as `added_tokens` with correct dict IDs (`single_word: true`, `special: false` — matching naklitechie)
4. Add `TemplateProcessing` post-processor to append `</s>` (id 2)
5. Write `tokenizer_meta.json` with `src_dict_size`, `tgt_dict_size`, `unk_id: 3`
6. Validate inline: 8 encode samples must hit 100% vs slow tokenizer

**File:** `02_build_fast_tokenizers.py`

---

## 11. Optimum exporter normalization (gemini branch — not used in final pipeline)

**Symptom:** `Could not find the attribute named "hidden_size" in the normalized config` for M2M100-style normalization.

**Attempted fix:** Custom `IndicTransNormalizedConfig` mapping `encoder_embed_dim` → `hidden_size`, etc.

**Outcome:** Abandoned in favor of manual `torch.onnx.export` (#1), which is simpler and matches naklitechie exactly.

---

## 12. Large protobuf files

**Symptom:** Decoder ONNX protos exceed 2 GB or are unwieldy in git/LFS.

**Fix:** `convert_model_to_external_data` in `_externalize_if_large()` — weights go to `.onnx.data` sidecars. Observed for indic→indic 320M decoders (~769 MB and ~696 MB sidecars).

---

## 13. Network / HF cache in sandbox

**Symptom:** `ProxyError: Tunnel connection failed: 403` when downloading models in restricted sandbox.

**Fix:** Run export commands with `full_network` permission, or rely on locally cached HF snapshots under `~/.cache/huggingface/`.

---

## Pipeline commands

```bash
cd indictrans2-onnx-export
make setup
make en-indic        # en→indic 200M
make indic-en        # indic→en 200M
make indic-indic     # indic→indic 320M
```

Upload after validation:

```bash
make upload-en-indic HF_ORG=your-org
make upload-indic-en HF_ORG=your-org
make upload-indic-indic HF_ORG=your-org
```

## 14. Wrong slow tokenizer loaded during ONNX text decode

**Symptom:** 100% token parity but 0% text parity on en→indic. ONNX decoded output looked like English gibberish (`"do Be was [that 420..."`) despite identical token IDs.

**Root cause:** `onnx_greedy_decode` loaded the slow tokenizer using `config.json` `_name_or_path`, which is absent in exported bundles. Fell back to `indictrans2-indic-en-dist-200M`, so indic target token IDs were decoded with the wrong vocabulary.

**Fix:** Pass `pytorch_model` from the validation CLI into `onnx_greedy_decode` so decode always uses the matching HF tokenizer.

**File:** `03_validate_parity.py`

---

## 15. Double-wrapped postprocess input (minor)

**Symptom:** `postprocess_batch([decoded], ...)` where `decoded` is already a `list[str]` from `batch_decode`.

**Fix:** Pass `decoded` directly: `postprocess_batch(decoded, lang=...)`.

**File:** `src/03_validate_parity.py`

---

## 16. Cross-attention skipped in decoder during step 2+

**Symptom:** ONNX model outputs correct translation for the first step but outputs repetitive or drifted garbage at steps 2+.

**Root cause:** In the wrapper `IndicTransDecoderWithPastWrapper`, `encoder_hidden_states` was passed as `None` to the decoder module to signify autoregressive run. Inside AI4Bharat's custom `modeling_indictrans.py`, there is a check `if encoder_hidden_states is not None:` surrounding the cross-attention block. Passing `None` caused the exporter to completely skip compilation of the cross-attention block in the ONNX graph.

**Why validation passed before:** The validation script `03_validate_parity.py` manual PyTorch greedy decoding loop had the exact same bug where it passed `None` for `encoder_hidden_states` at step 1+, so both paths generated the exact same incorrect translations, achieving 100% false-positive parity.

**Fix:** Update the wrapper to construct a dummy `encoder_hidden_states` tensor (matching the encoder sequence length dynamically) and pass the full 4-element `past_key_values` cache. Also update the validation script's manual PyTorch decode path similarly. This triggers the cross-attention block to compile and correctly use the cached cross-attention key/value states at runtime.

**Files:** `src/it2_onnx_wrappers.py`, `src/03_validate_parity.py`

---

| File                                | Role                                            |
| ----------------------------------- | ----------------------------------------------- |
| `src/01_export_encoder_decoder.py`  | Manual ONNX export (encoder + 2 decoder graphs) |
| `src/02_build_fast_tokenizers.py`   | SpmConverter + dict remap + validation          |
| `src/03_validate_parity.py`         | Greedy decode parity (PyTorch vs ONNX)          |
| `src/04_quantize_int8.py`           | Optional INT8 quantization                      |
| `src/05_upload_hf.py`               | Model card README generator and HF Hub uploader |
| `src/generate_translation_matrix.py`| Multi-language smoke-test matrix generator      |
| `src/test_hf_models.py`             | Inference test downloading directly from HF     |
| `src/it2_onnx_wrappers.py`          | PyTorch wrappers matching naklitechie I/O       |
| `fixtures/smoke-test/`              | Direction-specific test sentences and matrices  |
| `fixtures/parity-report-*.json`     | Validation results                              |
