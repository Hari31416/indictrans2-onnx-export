#!/usr/bin/env python3
"""Helper script to generate README.md, copy plots, and upload ONNX model directories to Hugging Face Hub."""

from __future__ import annotations

import argparse
import logging
import subprocess
import shutil
import json
from pathlib import Path
from typing import Final

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger: Final[logging.Logger] = logging.getLogger(__name__)

# Fallback sizes if directories not found
MODEL_SIZES_FALLBACK = {
    "en-indic": {"fp32": 1740.0, "fp16": 892.0, "int8": 452.9, "q4f16": 623.3},
    "indic-en": {"fp32": 1220.0, "fp16": 627.2, "int8": 319.7, "q4f16": 358.6},
    "indic-indic": {"fp32": 1910.0, "fp16": 980.2, "int8": 497.1, "q4f16": 711.6},
}

def load_direction_benchmarks(direction: str) -> dict[str, dict]:
    """Load all benchmark json files for a given direction."""
    benchmarks = {}
    for p in ["fp16", "int8", "q4f16"]:
        path = Path(f"fixtures/benchmark-{direction}-{p}.json")
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    benchmarks[p] = json.load(f)
            except Exception as e:
                logger.warning("Could not parse benchmark file %s: %s", path, e)
    return benchmarks

def generate_benchmark_markdown(direction: str, precision: str, benchmarks: dict[str, dict]) -> str:
    """Generate Markdown comparison tables and charts references if benchmark data exists."""
    if not benchmarks or (precision != "fp32" and precision not in benchmarks):
        return "\n*(Detailed benchmarks not generated yet for this model/precision)*\n"
        
    md = []
    
    # 1. Performance Visualizations
    md.extend([
        "## Performance Visualizations",
        "",
        "These charts show overall tradeoffs, language-level parity, and category breakdown.",
        "",
        "![Overall Tradeoffs](./overall.png)",
        "![Language-Level Parity](./languages.png)",
        "![Category breakdown](./categories.png)",
        ""
    ])
    
    # 2. Overall comparison table
    md.extend([
        "## Performance Tradeoffs & Size Comparison",
        "",
        "Compared against the FP32 ONNX oracle on the golden evaluation fixtures.",
        "",
        "| Format | Model Size | Exact Match (Token) | Exact Match (Text) | SacreBLEU (Raw) | Latency (Mean) | Speedup vs. FP32 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])
    
    # FP32 Baseline (Synthesized)
    ref_key = list(benchmarks.keys())[0]
    fp32_lat = f"{benchmarks[ref_key]['fp32_avg_latency_ms']:.1f} ms"
    size_fp32 = MODEL_SIZES_FALLBACK.get(direction, {}).get("fp32", 1000.0)
    size_fp32_str = f"{size_fp32/1024:.2f} GB" if size_fp32 >= 1000 else f"{size_fp32:.1f} MB"
    md.append(f"| FP32 | {size_fp32_str} | 100.00% | 100.00% | 100.00 | {fp32_lat} | 1.000x |")
    
    for p in ["fp16", "int8", "q4f16"]:
        if p in benchmarks:
            item = benchmarks[p]
            size_mb = MODEL_SIZES_FALLBACK.get(direction, {}).get(p, 0.0)
            size_str = f"{size_mb:.1f} MB"
            tok_match = f"{item['token_exact_rate']:.2f}%"
            txt_match = f"{item['text_exact_rate']:.2f}%"
            bleu = f"{item.get('sacrebleu_bleu', 0.0):.2f}"
            latency = f"{item['cmp_avg_latency_ms']:.1f} ms"
            speedup = f"{item['speedup_vs_fp32']:.3f}x"
            md.append(f"| {p.upper()} | {size_str} | {tok_match} | {txt_match} | {bleu} | {latency} | {speedup} |")
    md.append("")
    
    # 3. Precision specific reports
    if precision in benchmarks:
        report = benchmarks[precision]
        
        # Language table
        if "metrics_by_language" in report:
            md.extend([
                f"## Language-Level Parity ({precision.upper()})",
                "",
                "Exact match rates and translation quality (SacreBLEU / chrF) per language pair under this precision:",
                "",
                "| Language Code | Total Fixtures | Token Match Rate | Text Match Rate | SacreBLEU | SacreBLEU (chrF) |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |"
            ])
            for lang, m in sorted(report["metrics_by_language"].items()):
                tok = f"{m['token_exact_rate']:.1f}%"
                txt = f"{m['text_exact_rate']:.1f}%"
                bleu = f"{m['sacrebleu_bleu']:.2f}"
                chrf = f"{m['sacrebleu_chrf']:.2f}"
                md.append(f"| **{lang}** | {m['total_fixtures']} | {tok} | {txt} | {bleu} | {chrf} |")
            md.append("")
            
        # Category table
        if "metrics_by_category" in report:
            md.extend([
                f"## Category-Level Parity ({precision.upper()})",
                "",
                "Exact match rates and translation quality grouped by category types:",
                "",
                "| Category | Total Fixtures | Token Match Rate | Text Match Rate | SacreBLEU | SacreBLEU (chrF) |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |"
            ])
            for cat, m in sorted(report["metrics_by_category"].items()):
                tok = f"{m['token_exact_rate']:.1f}%"
                txt = f"{m['text_exact_rate']:.1f}%"
                bleu = f"{m['sacrebleu_bleu']:.2f}"
                chrf = f"{m['sacrebleu_chrf']:.2f}"
                md.append(f"| **{cat.capitalize()}** | {m['total_fixtures']} | {tok} | {txt} | {bleu} | {chrf} |")
            md.append("")
            
        # Mismatches list
        mismatches_path = Path(f"fixtures/benchmark-{direction}-{precision}-mismatches.json")
        if mismatches_path.exists():
            try:
                with open(mismatches_path, encoding="utf-8") as f:
                    mismatches = json.load(f)
            except Exception:
                mismatches = []
                
            if mismatches:
                md.extend([
                    "## Translation Mismatch Examples",
                    "",
                    "Here is a sample of up to 5 translation mismatches compared to the FP32 oracle. Many mismatches represent minor synonym differences or spacing variations.",
                    ""
                ])
                for idx, m in enumerate(mismatches[:5]):
                    category = m.get("fixture", {}).get("category", "generic").capitalize()
                    src_lang = m.get("fixture", {}).get("src_lang", "")
                    tgt_lang = m.get("fixture", {}).get("tgt_lang", "")
                    src_text = m.get("fixture", {}).get("text", "")
                    fp32_text = m.get("fp32_text", "")
                    cmp_text = m.get(f"{precision}_text", "")
                    
                    md.extend([
                        f"### Mismatch #{idx + 1} (Category: {category})",
                        f"- **Source ({src_lang} → {tgt_lang})**: `{src_text}`",
                        f"- **Expected (FP32)**: `{fp32_text}`",
                        f"- **Actual ({precision.upper()})**: `{cmp_text}`",
                        ""
                    ])
                    
    return "\n".join(md)

def generate_default_readme(direction: str, repo_id: str, precision: str) -> str:
    """Generate default README content matching the model's direction and precision."""
    indic_languages = [
        "as", "bn", "brx", "doi", "gu", "hi", "kn", "ks", "kok", "mai",
        "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"
    ]
    if direction == "en-indic":
        base_model = "ai4bharat/indictrans2-en-indic-dist-200M"
        direction_title = "en→indic"
        languages = ["en"] + indic_languages
        example_text = "eng_Latn hin_Deva Who will win the election?"
    elif direction == "indic-en":
        base_model = "ai4bharat/indictrans2-indic-en-dist-200M"
        direction_title = "indic→en"
        languages = indic_languages + ["en"]
        example_text = "hin_Deva eng_Latn चुनाव कौन जीतेगा?"
    else:  # indic-indic
        base_model = "ai4bharat/indictrans2-indic-indic-dist-320M"
        direction_title = "indic→indic"
        languages = indic_languages
        example_text = "hin_Deva tam_Taml चुनाव कौन जीतेगा?"

    # Precision details
    precision_labels = {
        "fp32": "FP32 (Full Precision)",
        "fp16": "FP16 (Half Precision, Lossless)",
        "int8": "INT8 (Dynamic Quantization)",
        "q4f16": "Q4F16 (4-bit Block Quantization, Lossy)",
    }
    precision_desc = {
        "fp32": "Baseline full-precision ONNX export.",
        "fp16": "Converts weights and activations to float16. Lossless tier, recommended for Apple Silicon (MPS) and CUDA runtimes.",
        "int8": "Dynamic INT8 quantization of the encoder and decoder. Highly recommended for CPU environments.",
        "q4f16": "4-bit quantization with float16 scale factors and block size of 32. Reduces model size significantly.",
    }

    precision_label = precision_labels.get(precision, precision.upper())
    precision_detail = precision_desc.get(precision, "")

    model_size_text = "320M" if "320m" in base_model else "200M"
    title = f"IndicTrans2 {model_size_text} ({direction_title}) — ONNX bundle [{precision_label}]"

    languages_list = "\n".join(f"  - {lang}" for lang in languages)
    
    # Load and build detailed benchmark markdown
    benchmarks = load_direction_benchmarks(direction)
    benchmark_section = generate_benchmark_markdown(direction, precision, benchmarks)

    # Build model variants callout
    org = repo_id.split("/")[0] if "/" in repo_id else "hari31416"
    base_repos = {
        "en-indic": "indictrans2-en-indic-dist-200M-ONNX",
        "indic-en": "indictrans2-indic-en-dist-200M-ONNX",
        "indic-indic": "indictrans2-indic-indic-dist-320M-ONNX"
    }
    base_name = base_repos.get(direction, "indictrans2-en-indic-dist-200M-ONNX")
    
    precisions_list = {
        "fp32": "FP32 (Full Precision / Base)",
        "fp16": "FP16 (Half Precision)",
        "int8": "INT8 (Dynamic Quantization)",
        "q4f16": "Q4F16 (4-bit Block Quantization)"
    }
    
    callout_lines = [
        "> [!TIP]",
        "> This model is part of a suite of optimized/quantized ONNX versions of the base model.",
        "> Other variants in this direction:"
    ]
    for p, label in precisions_list.items():
        suffix = "" if p == "fp32" else f"-{p}"
        variant_repo = f"{org}/{base_name}{suffix}"
        url = f"https://huggingface.co/{variant_repo}"
        
        if p == precision:
            callout_lines.append(f"> - **{label}**: [`{variant_repo}`]({url}) *(Current)*")
        else:
            callout_lines.append(f"> - **{label}**: [`{variant_repo}`]({url})")
            
    callout = "\n".join(callout_lines)

    return f"""---
license: mit
language:
{languages_list}
tags:
  - translation
  - onnx
  - indic
  - indictrans2
  - browser
  - {precision}
pipeline_tag: translation
library_name: onnx
base_model: {base_model}
---

# {title}

{callout}

ONNX-exported and quantized version of [`{base_model}`](https://huggingface.co/{base_model})
for in-browser and local edge inference.

- **Precision**: {precision_label}
- **Description**: {precision_detail}
- **Source Pipeline & Details**: For pipeline details, benchmarks, and usage instructions, see the [indictrans2-onnx-export GitHub repository](https://github.com/Hari31416/indictrans2-onnx-export).

Built for use with [Transformers.js](https://github.com/huggingface/transformers.js)
and [onnxruntime-web](https://onnxruntime.ai/docs/get-started/with-javascript.html)
in the browser, with fast BPE tokenizer.json files that don't require the
SentencePiece WASM runtime.

{benchmark_section}

## Files

- `encoder_model.onnx` (and optional `.onnx.data` weights sidecar)
- `decoder_model.onnx` (and optional `.onnx.data` weights sidecar)
- `decoder_with_past_model.onnx` (and optional `.onnx.data` weights sidecar)
- Fast tokenizer config files (`tokenizer_src.json`, `tokenizer_tgt.json`, `tokenizer_meta.json`)
- Model configuration configs (`config.json`, `generation_config.json`)

## Usage Example (Python, onnxruntime)

```python
import json, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import snapshot_download

snap = snapshot_download(repo_id="{repo_id}")

src = Tokenizer.from_file(f"{{snap}}/tokenizer_src.json")
tgt = Tokenizer.from_file(f"{{snap}}/tokenizer_tgt.json")
meta = json.load(open(f"{{snap}}/tokenizer_meta.json"))

enc = ort.InferenceSession(f"{{snap}}/encoder_model.onnx")
dec = ort.InferenceSession(f"{{snap}}/decoder_model.onnx")
decp = ort.InferenceSession(f"{{snap}}/decoder_with_past_model.onnx")

# Tokenize example text
text = "{example_text}"
e = src.encode(text)
input_ids = np.array([[i if i < meta["src_dict_size"] else meta["unk_id"] for i in e.ids]], dtype=np.int64)
attn_mask = np.array([e.attention_mask], dtype=np.int64)

# Encoder
enc_h = enc.run(["last_hidden_state"], {{"input_ids": input_ids, "attention_mask": attn_mask}})[0]
```

## License

MIT (preserved from upstream AI4Bharat).
"""

def main() -> None:
    """Main execution entry point."""
    parser = argparse.ArgumentParser(description="Upload ONNX model bundle to Hugging Face Hub")
    parser.add_argument("--model-dir", required=True, type=Path, help="Directory containing ONNX bundle")
    parser.add_argument("--repo-id", required=True, type=str, help="Hugging Face repo ID (e.g. org/repo-name)")
    parser.add_argument("--commit-message", type=str, default="", help="Optional commit message for the upload")
    parser.add_argument("--dry-run", action="store_true", help="Generate README.md and copy plots but do not upload")

    args = parser.parse_args()

    model_dir: Path = args.model_dir.resolve()
    repo_id: str = args.repo_id
    commit_msg: str = args.commit_message

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    # Determine direction
    direction = "en-indic"
    repo_lower = repo_id.lower()
    dir_lower = model_dir.name.lower()
    if "indic-en" in repo_lower or "indic-en" in dir_lower:
        direction = "indic-en"
    elif "indic-indic" in repo_lower or "indic-indic" in dir_lower:
        direction = "indic-indic"

    # Determine precision tier (fp32, fp16, int8, q4f16)
    precision = "fp32"
    if "-int8" in repo_lower or "-int8" in dir_lower:
        precision = "int8"
    elif "-fp16" in repo_lower or "-fp16" in dir_lower:
        precision = "fp16"
    elif "-q4f16" in repo_lower or "-q4f16" in dir_lower:
        precision = "q4f16"

    # Remove any existing PNG files in the directory to clean out stale plots
    for p_file in model_dir.glob("*.png"):
        logger.info("Removing old plot: %s", p_file)
        try:
            p_file.unlink()
        except Exception as e:
            logger.warning("Could not delete old plot %s: %s", p_file, e)

    # Copy plots from fixtures to the upload directory if they exist
    plot_prefix = direction.replace("-", "_")
    plots = ["overall.png", "languages.png", "categories.png"]
    for plot in plots:
        src_filename = f"{plot_prefix}_{plot}"
        src_path = Path(f"fixtures/{src_filename}")
        if src_path.exists():
            dest_path = model_dir / plot
            logger.info("Copying plot %s → %s", src_path, dest_path)
            shutil.copy2(src_path, dest_path)
        else:
            logger.warning("Plot not found in fixtures, skipping copy: %s", src_path)

    # Create/overwrite README.md
    readme_path = model_dir / "README.md"
    logger.info("Generating %s README.md (precision=%s) for repo %s", direction, precision, repo_id)
    readme_content = generate_default_readme(direction, repo_id, precision)
    readme_path.write_text(readme_content, encoding="utf-8")

    if args.dry_run:
        logger.info("Dry-run mode: README.md generated and plots copied for %s. Skipping HF upload.", model_dir.name)
        return

    # Resolve hf path
    hf_cmd = shutil.which("hf") or "hf"

    # Construct and run the command
    cmd: list[str] = [
        hf_cmd, "upload",
        repo_id,
        str(model_dir),
        "--repo-type", "model",
        "--exclude", ".git/*"
    ]

    if commit_msg:
        cmd.extend(["--commit-message", commit_msg])

    logger.info("Running upload command: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        logger.info("Upload to Hugging Face completed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error("Failed to upload model: %s", e)
        raise SystemExit(1) from e

if __name__ == "__main__":
    main()
