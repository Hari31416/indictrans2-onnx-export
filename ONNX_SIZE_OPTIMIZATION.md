# IndicTrans2 ONNX Bundle — Size & Optimization Report

Notes on why the export bundle is large, how it compares to the [naklitechie reference bundle](https://huggingface.co/naklitechie/indictrans2-en-indic-dist-200M-ONNX), and what can be optimized beyond the initial export fixes.

Reference bundles:

- **Ours:** [hari31416/indictrans2-en-indic-dist-200M-ONNX](https://huggingface.co/hari31416/indictrans2-en-indic-dist-200M-ONNX) (~1.91 GB)
- **Reference:** [naklitechie/indictrans2-en-indic-dist-200M-ONNX](https://huggingface.co/naklitechie/indictrans2-en-indic-dist-200M-ONNX) (~1.4 GB)

---

## 1. Why three ONNX graphs?

Sequence-to-sequence models cannot be exported as a single monolithic graph because of autoregressive decoding. ONNX graphs have a **fixed input signature** at export time. PyTorch handles this with one module and `past_key_values=None` vs a populated tuple; ONNX cannot express that cleanly.

| Step         | Graph                          | Inputs                                                         | What happens                                                      |
| ------------ | ------------------------------ | -------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Step 1**   | `decoder_model.onnx`           | `input_ids`, `encoder_hidden_states`, `encoder_attention_mask` | Full cross-attention over encoder output; builds initial KV cache |
| **Steps 2+** | `decoder_with_past_model.onnx` | `input_ids`, `encoder_attention_mask`, `past_key_values.*`     | Reuses cached encoder K/V; appends decoder self-attn K/V          |

This is the standard seq2seq ONNX pattern (same as Hugging Face Optimum and the naklitechie I/O layout). See `src/it2_onnx_wrappers.py` and `EXPORT_ISSUES.md`.

The `decoder_with_past` wrapper passes a **dummy `encoder_hidden_states`** tensor (zeros) so the tracer compiles the cross-attention block. Without a non-`None` tensor, the exporter strips cross-attention entirely. The real encoder K/V live in `past_key_values.*.encoder.*` after step 1.

---

## 2. Are decoder weights duplicated?

**Yes.** Both `decoder_model` and `decoder_with_past` are exported from the same PyTorch `decoder` + `lm_head`:

```python
_export_decoder(model.model.decoder, model.lm_head, ...)
_export_decoder_with_past(model.model.decoder, model.lm_head, ...)
```

Each `torch.onnx.export` serializes a full copy of every weight tensor into that graph's `.onnx` / `.onnx.data` sidecar. ONNX has no built-in cross-file weight sharing. At runtime, both sessions may be loaded, so duplication can affect disk **and** memory.

This duplication is **unavoidable** with the current two-graph autoregressive pattern unless you pursue shared external weight files or a single merged graph (see §6).

---

## 3. Size comparison: our export vs naklitechie

### 3.1 File sizes (en→indic 200M, FP32)

| Component                   | Ours                    | naklitechie              | Delta                  |
| --------------------------- | ----------------------- | ------------------------ | ---------------------- |
| Encoder weights             | 294 MB (inside `.onnx`) | 294 MB (in `.onnx.data`) | **0** (packaging only) |
| `decoder_model` weights     | 805 MB                  | 554 MB                   | **+251 MB**            |
| `decoder_with_past` weights | 767 MB                  | 517 MB                   | **+250 MB**            |
| **ONNX weights total**      | **~1.87 GB**            | **~1.37 GB**             | **~+500 MB**           |
| **Repo total**              | 1.91 GB                 | 1.4 GB                   | +510 MB                |

Both repos duplicate decoder weights across two graphs. The ~500 MB gap comes from **how** the decoder graphs are exported, not from having two decoders.

### 3.2 Root cause A: duplicate tied embed + lm_head weights (~251 MB per decoder)

IndicTrans2 ties `decoder.embed_tokens` and `lm_head` — they share the same weight matrix. Our export stores **both copies**:

| Our `decoder_model.onnx`               | naklitechie's `decoder_model.onnx` |
| -------------------------------------- | ---------------------------------- |
| `decoder.embed_tokens.weight` — 251 MB | *(not present)*                    |
| `onnx::MatMul_*` (512×122672) — 251 MB | `lm_head.weight` — 251 MB          |
| **502 MB for one tied matrix**         | **251 MB — stored once**           |

Our wrapper calls both the decoder (embedding lookup) and `lm_head` (output projection). `torch.onnx.export` traces both paths separately and serializes the tied weights twice. naklitechie's bundle reuses `lm_head.weight` for **both** token embedding (via `Gather`) and logits (via `MatMul`).

Verified by comparing initializer layouts:

- **Ours:** `Gather` from `decoder.embed_tokens.weight`; separate 251 MB `onnx::MatMul_*` for lm_head
- **naklitechie:** `Gather` from `lm_head.weight`; no separate `embed_tokens` initializer

Difference per decoder file: 805 − 554 ≈ **251 MB**, 767 − 517 ≈ **250 MB**.

### 3.3 Root cause B: unoptimized graph structure (~2.8× more nodes)

| Graph               | Our nodes | naklitechie nodes |
| ------------------- | --------- | ----------------- |
| `decoder_model`     | 3,696     | 1,333             |
| `decoder_with_past` | 3,418     | 1,193             |
| `encoder`           | 2,201     | 776               |

Our export is a raw `torch.onnx.export` trace with `onnx::MatMul_*` initializers. naklitechie's graphs appear post-processed (fused ops, `val_*` naming, fewer nodes). Same math, leaner serialization — but the embedding duplication is the dominant size factor.

### 3.4 Root cause C: encoder packaging (cosmetic)

Our encoder is 294 MB embedded in `encoder_model.onnx` because it sits below the 512 MB externalization threshold in `_externalize_if_large`. naklitechie externalizes it to `encoder_model.onnx` (2 MB) + `encoder_model.onnx.data` (294 MB). Same weight bytes, different file layout.

### 3.5 Minor: extra repo files (~10 MB)

Our HF repo also includes parity charts, `translate.py`, and a larger `tokenizer_tgt.json` (23.9 MB vs 17.7 MB). Small compared to the ONNX gap.

### 3.6 Impact summary

| Factor                                         | Impact                                                                  |
| ---------------------------------------------- | ----------------------------------------------------------------------- |
| Two decoder graphs (both repos)                | ~2× decoder weights — expected with current ONNX autoregressive pattern |
| Tied embed + lm_head stored twice (our export) | **+~500 MB** — fixable                                                  |
| Unfused graph (our export)                     | Smaller protos, slightly less efficient runtime                         |
| Encoder not externalized (our export)          | No weight change, one large `.onnx` file                                |
| Extra HF artifacts                             | ~10 MB                                                                  |

Our models remain **functionally equivalent** at 100% parity — this is a serialization efficiency issue, not a correctness issue.

---

## 4. Initial export fixes (items 1–3)

These address the largest FP32 bundle inefficiencies identified above.

### 4.1 Post-export dedup of tied weights

Remove `decoder.embed_tokens.weight` from decoder graphs and route embedding `Gather` through `lm_head.weight` (or the reverse), since they are tied. Expected savings: **~500 MB** on FP32 bundles.

### 4.2 Graph optimization

Run `onnxsim` or ORT's optimizer to fuse MatMuls, drop redundant nodes, and produce layouts closer to naklitechie's `val_*` graphs. Our graphs have ~2.8× more nodes; fusion improves both file size and runtime.

### 4.3 Lower externalization threshold

Externalize the encoder at e.g. 100 MB for consistency with naklitechie's layout (`encoder_model.onnx` + `.data`). Does not reduce total size, but improves HF browsing and keeps protos small.

---

## 5. Additional optimizations

### 5.1 Export-time

#### 4. Fix weight tying at the wrapper, not just post-export

Instead of deduplicating after export, change the wrapper so the tracer only sees one weight tensor:

```python
# Before export: physically tie weights so tracer sees one tensor
self.lm_head.weight = self.decoder.embed_tokens.weight

# Or: custom forward that embeds via F.embedding(input_ids, self.lm_head.weight)
```

This prevents the 251 MB duplicate from ever being serialized. Cleaner than patching the ONNX graph afterward.

#### 5. Shared `.onnx.data` across both decoder graphs

`decoder_model` and `decoder_with_past` share ~95% of weights (same layers, different I/O). Today each `.data` file is independent. Post-processing both graphs to reference **one shared sidecar** could cut decoder weight storage roughly in half on disk after tied-weight dedup.

ORT does not guarantee this out of the box — it requires manual ONNX surgery.

#### 6. Export with `use_external_data=True` during trace

The v2 export script (`src/v2/01_export_encoder_decoder.py`) explores exporting large decoders directly to external data instead of embed-then-externalize. This can avoid intermediate 800 MB protobufs and sometimes produces cleaner initializer layouts.

#### 7. Opset and operator fusion choices

We export at opset 17. Newer opsets plus ORT's optimizer can fuse `LayerNorm + MatMul`, `Gelu + MatMul`, etc. Choosing export settings that produce fusible patterns (avoiding ops that block fusion) matters at trace time.

### 5.2 Precision tiers (pipeline already supports — tune for deployment)

#### 8. FP16 as the default shipping format

Benchmarks show FP16 at ~50% size with 99.6–100% parity:

| Direction   | FP32    | FP16   | Parity |
| ----------- | ------- | ------ | ------ |
| en-indic    | 1.74 GB | 892 MB | 100%   |
| indic-en    | 1.22 GB | 627 MB | 99.6%  |
| indic-indic | 1.91 GB | 980 MB | 100%   |

For browser deployment, FP16 is often the best size/quality tradeoff. FP32 is mainly a validation oracle. See `src/05_convert_fp16.py` and README parity tables.

#### 9. Q4F16 with tuned `block_size` and `accuracy_level`

`src/06_quantize_q4f16.py` excludes `embed_tokens` / `lm_head` by default. Further tuning:

- **`block_size=16`** instead of 32 → better parity, slightly larger file
- **`accuracy_level=2`** for WebGPU bundles, **`4`** for WASM (see `quantization_issues.md`)
- After tied-weight dedup, quantizing more layers becomes safer because the bundle starts smaller

Current q4f16 parity is 56–66% on 200M models — room to improve with selective exclusion of sensitive cross-attention layers.

#### 10. Per-layer mixed precision

Instead of all-or-nothing quantization:

- Keep **cross-attention** and **lm_head** in FP16
- Quantize FFN and self-attention to INT8 or Q4

`06_quantize_q4f16.py` supports `--exclude-nodes` and `--list-nodes`. Layer-selective exclusion often beats global INT8 (~80% parity) or global Q4F16 (~60%).

### 5.3 Bundle / artifact packaging

#### 11. Slim the tokenizer payload (~7–30 MB)

Our `tokenizer_tgt.json` is **23.9 MB** vs naklitechie's **17.7 MB**. The browser only needs:

- `tokenizer_src.json`, `tokenizer_tgt.json`, `tokenizer_meta.json`
- `config.json`, `generation_config.json`

It does **not** need `model.SRC`, `model.TGT`, `dict.*.json`, `tokenization_indictrans.py`, parity charts, or `__pycache__` for inference. Ship a `browser/` subset on HF or use upload filters.

#### 12. ORT format (`.ort`) instead of raw ONNX

ORT's serialized format can be smaller and faster to load than ONNX protobuf + external data. Worth benchmarking for the 3-session load pattern (live benchmarks show 2.4–3.6 s load times). See `LIVE_BENCHMARKS.md`.

#### 13. Compress artifacts for download, not inference

For HF distribution: zstd/tar or HF's Xet storage helps transfer. The browser still needs uncompressed weights at runtime, but download size affects perceived bundle size.

### 5.4 Runtime / memory (browser — often bigger wins than export)

#### 14. Session lifecycle: unload `decoder_model` after step 1

Three sessions are loaded simultaneously. After the first decode step, `decoder_model` is never used again. **Unload it and free its weights** before running `decoder_with_past`. On WASM this can mean ~800 MB freed — critical near the 4 GB ceiling.

#### 15. Lazy session loading

Load order: encoder → decoder (step 1) → decoder_with_past (step 2+). Don't initialize all three at startup. Live benchmarks show load time is 2.4–6 s — lazy loading improves time-to-first-translation.

#### 16. Provider-specific bundle variants

`LIVE_BENCHMARKS.md` shows the split:

| Target    | Best format       | Why                                        |
| --------- | ----------------- | ------------------------------------------ |
| WASM CPU  | INT8              | 39.5 t/s; WebGPU INT8 unsupported          |
| WebGPU    | Q4F16 or FP16     | INT8 shaders broken on WebGPU              |
| 1B models | Q4F16 WebGPU only | WASM 4 GB limit; FP32 hits 2 GB buffer cap |

Ship **two HF repos/tags** (`-wasm-int8`, `-webgpu-q4f16`) instead of one bundle for all targets.

#### 17. KV cache dtype

Present/past KV tensors are FP32 in the export. At runtime, storing cache as **FP16** halves KV memory for long sequences (18 layers × 4 tensors × seq_len). Does not shrink the model file but reduces peak RAM during generation.

### 5.5 Architectural (harder, higher ceiling)

#### 18. Single merged decoder graph with optional past inputs

A single graph with `past_key_values` as optional (empty tensor = step 1) would eliminate the second weight copy entirely. Research territory — naklitechie did not do this either.

#### 19. Encoder embedding

Encoder is clean: 66 MB `embed_tokens`, no duplicate tied-weight issue. No action needed.

#### 20. Direction-specific vocab pruning

Each direction ships a full vocab (~122K target tokens for en→indic). For single-direction deployment, unused vocab rows could theoretically be pruned from the embedding/lm_head matrix (251 MB per decoder). Risky and direction-specific.

#### 21. Model scale selection

Distilled 200M/320M models are the practical sweet spot for browser voice chat. The 1B models need Q4F16 + WebGPU and still hit engine limits (4 GB WASM, 2 GB WebGPU buffer). See `LIVE_BENCHMARKS.md`.

---

## 6. Priority matrix

| #   | Optimization                       | Size impact             | Effort | Quality risk |
| --- | ---------------------------------- | ----------------------- | ------ | ------------ |
| 1   | Post-export tied-weight dedup      | **−500 MB** fp32        | Medium | None         |
| 2   | Graph optimization (onnxsim / ORT) | Moderate                | Low    | Low          |
| 3   | Encoder externalization            | 0 (layout only)         | Low    | None         |
| 4   | Wrapper-level weight tying         | **−500 MB** fp32        | Low    | None         |
| 5   | Shared decoder `.data`             | **−550 MB** fp32        | High   | Low          |
| 8   | Ship FP16 as default               | **−50%**                | Done   | ~0%          |
| 10  | Per-layer mixed quant              | **−30–50%**             | Medium | Low–medium   |
| 11  | Slim browser bundle                | **−30 MB**              | Low    | None         |
| 14  | Unload decoder after step 1        | 0 disk, **−800 MB RAM** | Low    | None         |
| 16  | Provider-specific variants         | Better perf/size fit    | Medium | None         |
| 12  | ORT format                         | **−10–20%** load time   | Medium | Low          |

**Highest ROI after items 1–3:** fix tying in the wrapper (#4), ship FP16 by default (#8), unload `decoder_model` at runtime (#14), then explore shared decoder weights (#5) if FP32 bundles must stay under ~1 GB.

---

## 7. Related files

| File                               | Role                                           |
| ---------------------------------- | ---------------------------------------------- |
| `src/01_export_encoder_decoder.py` | FP32 export pipeline                           |
| `src/it2_onnx_wrappers.py`         | Encoder / decoder / decoder_with_past wrappers |
| `src/05_convert_fp16.py`           | FP16 conversion                                |
| `src/04_quantize_int8.py`          | INT8 dynamic quantization                      |
| `src/06_quantize_q4f16.py`         | Q4F16 weight-only quantization                 |
| `EXPORT_ISSUES.md`                 | Export debugging log                           |
| `quantization_issues.md`           | Quantization debugging log                     |
| `LIVE_BENCHMARKS.md`               | Browser runtime constraints and benchmarks     |
| `README.md`                        | Parity and quantization summary tables         |
