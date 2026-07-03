# Quantization & precision roadmap

Plan for shrinking IndicTrans2 ONNX bundles beyond fp32. fp32 exports are **done** and validated; everything below is **planned experimentation**.

## Current status (fp32)

| Direction   | Bundle (approx.) | Token parity | Text parity  | Location                    |
| ----------- | ---------------- | ------------ | ------------ | --------------------------- |
| en→indic    | ~1.7 GB          | 8/8 (100%)   | 8/8 (100%)   | `scratch/en-indic-onnx/`    |
| indic→en    | ~1.2 GB          | 8/8 (100%)   | 8/8 (100%)   | `scratch/indic-en-onnx/`    |
| indic→indic | ~1.9 GB          | 12/12 (100%) | 12/12 (100%) | `scratch/indic-indic-onnx/` |

**Pass bar for any new precision:** ≥ 99% token-exact greedy decode vs PyTorch on direction-specific fixtures. Text parity tracked separately (IndicProcessor postprocess).

---

## Recommended order

```
fp32 (done) → fp16 → q4f16 → int8 → bf16 (server-only, skip browser)
```

fp16 is the best quality/size tradeoff to try first. INT8 is a separate “preview tier” with known regression. bf16 is not targeted for browser ORT.

---

## 1. fp16

**Goal:** ~50% smaller bundles (~600 MB–1 GB) with near-fp32 quality on WebGPU.

### How to proceed

1. **Post-convert fp32 → fp16** ✅
   - `src/05_convert_fp16.py` + `make convert-fp16-*` targets implemented
   - Uses `onnxconverter-common.convert_float_to_float16` with `keep_io_types=True`
   - Start with **encoder only**, run parity, then decoder, then `decoder_with_past`

2. **Export-time fp16** (if post-convert fails parity)
   - Load PyTorch model as `torch.float16`, trace with same wrappers in `src/it2_onnx_wrappers.py`
   - May need fp16-safe dummy tensors and attention mask dtypes

3. **Validate**
   - Re-run `make validate-*` against fp16 bundle (may need ORT session option `preferredOutputLocation` / fp16 EP)
   - Test in browser with `onnxruntime-web` WebGPU (Chrome 121+)

### Expected issues

| Issue                                                          | Mitigation                                                                             |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| naklitechie reported **fp16 decoder export failed**            | Try post-convert before re-tracing; decoder_with_past is the fragile graph             |
| **KV cache dtype drift** in autoregressive steps               | Keep past/present tensors consistently fp16; watch for Cast nodes inserted incorrectly |
| **WebGPU fp16 overflow** on some models (seen with Gemma 3)    | Run full fixture suite; compare logits at step 1 vs step N                             |
| **Encoder/decoder dtype mismatch** if only one graph converted | Convert all three graphs together                                                      |
| Slightly different greedy argmax on borderline logits          | Accept if text parity stays ≥ 99%; document any fixture regressions                    |

### Ship as

`{org}/indictrans2-*-ONNX-fp16` — separate HF repo, document WebGPU requirement.

---

## 2. INT8

**Goal:** ~4× smaller (~300–500 MB), fast WASM CPU inference, “preview” tier.

### How to proceed

1. Run existing step after fp32 passes:
   ```bash
   make quantize-en-indic    # src/04_quantize_int8.py
   make quantize-indic-en
   make quantize-indic-indic
   ```
2. Re-run parity on int8 output dir (`scratch/*-onnx-int8/`)
3. Expand fixtures before trusting int8 (current suite is small)
4. Upload as separate HF repos (`*-ONNX-int8`)

### Expected issues

| Issue                                                         | Mitigation                                                                                                        |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **~80% exact text match** (naklitechie benchmark)             | Document as research/preview only; not default in voice chat app                                                  |
| **Dynamic quantization** only weights (activations stay fp32) | Acceptable for size; limited speedup vs full quant                                                                |
| **External data sidecars** copied but not re-quantized        | Verify `.onnx.data` handling in `04_quantize_int8.py`                                                             |
| **WebGPU int8 bugs** in older transformers.js                 | Prefer WASM for int8 tier; see [transformers.js#1512](https://github.com/huggingface/transformers.js/issues/1512) |
| Per-channel vs per-tensor tradeoffs                           | Script uses `per_channel=True`; tune if quality drops                                                             |

### Ship as

`{org}/indictrans2-*-ONNX-int8` with README warning: not production quality.

---

## 3. bf16

**Goal:** Evaluate for **server/GPU** inference only — **not** a browser target.

### How to proceed

1. Export or convert fp32 → bf16 for ONNX Runtime **CUDA/CPU** (not onnxruntime-web WebGPU)
2. Useful if running translation on a backend service alongside the voice chat app
3. Skip for in-browser IndicTrans2 until ORT Web adds bf16 WebGPU support

### Expected issues

| Issue                                                         | Mitigation                                      |
| ------------------------------------------------------------- | ----------------------------------------------- |
| **No bf16 in onnxruntime-web WebGPU**                         | Do not bundle for browser; server-only artifact |
| **No bf16 in WASM CPU path**                                  | Same — not for client-side                      |
| PyTorch bf16 works on Apple Silicon but ONNX path is separate | Export from fp32 baseline, convert weights      |
| IndicTrans2 200M may not benefit much over fp16 on client     | Deprioritize unless backend deployment needs it |

### Ship as

Optional `*-ONNX-bf16` for server use only; out of scope for [local-voice-chat](https://github.com/Hari31416/local-voice-chat) Phase 2 browser worker.

---

## 4. q4f16 (4-bit weights + fp16 activations)

**Goal:** ~200 MB class bundles for Transformers.js / aggressive mobile download.

### How to proceed

1. Start from **fp16** graphs (or fp32 if fp16 blocked)
2. Use ORT-native `MatMul4BitsQuantizer` (Path A) ✅
   - `src/06_quantize_q4f16.py` + `make quantize-q4f16-*` targets implemented
   - `accuracy_level=2` (fp16 scales + fp16 input A = true q4f16)
   - `block_size=32` default (tunable via `Q4F16_BLOCK_SIZE=`)
   - embed_tokens / lm_head excluded by default to protect translation quality
3. Validate with same greedy loop + expanded fixtures
4. May require different runtime path than custom 3-graph ORT loop (Transformers.js vs raw onnxruntime-web)

### Expected issues

| Issue                                                                                      | Mitigation                                                    |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| **fp16 decoder export must work first** (or q4 on fp32 with care)                          | Complete fp16 milestone before q4f16                          |
| **Custom 3-graph layout** may not map cleanly to Transformers.js single-model expectations | May need ORT-native q4 or keep custom loader                  |
| **Quality unknown** for Indic scripts                                                      | Run indic + en fixture suites; watch Devanagari normalization |
| **Quantized embedding / LM head** sensitivity                                              | Try weight-only q4 first; keep logits computation fp16        |
| Gemma-style **WebGPU overflow** with q4f16                                                 | Test early on target browsers (Chrome macOS/Windows)          |

### Ship as

`{org}/indictrans2-*-ONNX-q4f16` — experimental; link from voice chat app as optional low-bandwidth tier.

---

## Validation checklist (all precisions)

- [ ] Token parity ≥ 99% vs PyTorch greedy decode
- [ ] Text parity after IndicProcessor postprocess (track %)
- [ ] Encoder-only smoke test (single forward)
- [ ] Full autoregressive loop (10+ steps, KV cache)
- [ ] At least 2 language pairs per script family (indic→indic)
- [ ] Browser smoke test on WebGPU (if targeting client)
- [ ] Parity report JSON committed under `fixtures/`
- [ ] README on HF repo documents precision, size, and known regressions

---

## Repo layout (post-restructure)

```
indictrans2-onnx-export/
├── src/                    # Python pipeline
│   ├── 01_export_encoder_decoder.py
│   ├── 02_build_fast_tokenizers.py
│   ├── 03_validate_parity.py
│   ├── 04_quantize_int8.py
│   ├── 05_convert_fp16.py      # fp32 → fp16 (onnxconverter-common)
│   ├── 06_quantize_q4f16.py   # fp16 → q4f16 (MatMul4BitsQuantizer)
│   └── it2_onnx_wrappers.py
├── fixtures/               # golden inputs + parity reports (in git)
├── scratch/                # fp32/int8/fp16 bundles (gitignored → HF)
├── Makefile
├── pyproject.toml          # uv-managed deps
├── EXPORT_ISSUES.md        # blockers hit during fp32 export
└── ROADMAP.md              # this file
```

---

## References

- [naklitechie/indictrans2-en-indic-dist-200M-ONNX](https://huggingface.co/naklitechie/indictrans2-en-indic-dist-200M-ONNX) (fp32 reference I/O layout)
- [naklitechie/indictrans2-en-indic-dist-200M-ONNX-int8](https://huggingface.co/naklitechie/indictrans2-en-indic-dist-200M-ONNX-int8) (~80% match)
- [onnxruntime-web WebGPU fp16](https://github.com/microsoft/onnxruntime/blob/main/js/web/README.md) (Chrome 121+)
- [local-voice-chat](https://github.com/Hari31416/local-voice-chat) — consumer app for these bundles
