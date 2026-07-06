# IndicTrans2 ONNX Browser Lab

Standalone web UI to load **local ONNX export bundles** and run translation in the browser. Use this to test exports from `scratch/` before uploading to Hugging Face.

Inference logic is adapted from [indictrans2-onnx-browser-demo](https://github.com/Hari31416/indictrans2-onnx-browser-demo).

## Quick start

1. Export a bundle (example):

```bash
make en-indic
# or: make export-en-indic validate-en-indic
```

2. Start the lab server:

```bash
make browser-lab
# serves browser-lab/ at http://127.0.0.1:8010
```

3. Open [http://127.0.0.1:8010](http://127.0.0.1:8010) in Chrome or Edge (WebGPU) or any modern browser (WASM).

4. Load your bundle using one of:

- **Option A — Local folder:** click *Choose folder*, select e.g. `scratch/en-indic-onnx`, then *Load from folder*.
- **Option B — Local HTTP URL:** serve the bundle separately and paste the base URL:

```bash
# from repo root — exposes scratch/ at :8000
python3 -m http.server 8000
```

Then load `http://127.0.0.1:8000/scratch/en-indic-onnx`.

## Required bundle files

| File | Role |
| ---- | ---- |
| `encoder_model.onnx` | Encoder graph |
| `decoder_model.onnx` | First decode step |
| `decoder_with_past_model.onnx` | Autoregressive steps |
| `tokenizer_src.json` | Source fast tokenizer |
| `tokenizer_tgt.json` | Target fast tokenizer |
| `tokenizer_meta.json` | Dict sizes, `unk_id` |
| `generation_config.json` | `decoder_start_token_id`, `eos_token_id` |
| `*.onnx.data` | Weight sidecars (if present) |

Set **translation direction** to match the bundle (en→indic, indic→en, or indic→indic).

## Execution providers

| Precision | Recommended provider |
| --------- | -------------------- |
| FP32 / FP16 | WebGPU (Chrome 121+) |
| INT8 / Q4F16 | WASM |

## Notes

- Folder upload reads all files into browser memory. Large 1B FP32 bundles may be slow or hit memory limits — use Q4F16/FP16 or the HTTP URL option.
- External weight sidecars (`.onnx.data`) are passed to ONNX Runtime via the `externalData` session option; WASM thread count is set to 1 when sidecars are present.
- No build step — static HTML/JS served over HTTP (ES modules require a server, not `file://`).
