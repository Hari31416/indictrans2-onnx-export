#!/usr/bin/env python3
"""Helper script to generate README.md and upload ONNX model directories to Hugging Face Hub."""

from __future__ import annotations

import argparse
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Final

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger: Final[logging.Logger] = logging.getLogger(__name__)


def generate_default_readme(direction: str, repo_id: str) -> str:
    """Generate default README content matching the model's direction."""
    if direction == "en-indic":
        base_model = "ai4bharat/indictrans2-en-indic-dist-200M"
        title = "IndicTrans2 200M (en→indic) — ONNX bundle for in-browser inference"
        languages = ["en", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "ur"]
        example_text = "eng_Latn hin_Deva Who will win the election?"
    elif direction == "indic-en":
        base_model = "ai4bharat/indictrans2-indic-en-dist-200M"
        title = "IndicTrans2 200M (indic→en) — ONNX bundle for in-browser inference"
        languages = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "ur", "en"]
        example_text = "hin_Deva eng_Latn चुनाव कौन जीतेगा?"
    else:  # indic-indic
        base_model = "ai4bharat/indictrans2-indic-indic-dist-320M"
        title = "IndicTrans2 320M (indic→indic) — ONNX bundle for in-browser inference"
        languages = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "ur"]
        example_text = "hin_Deva tam_Taml चुनाव कौन जीतेगा?"

    languages_list = "\n".join(f"  - {lang}" for lang in languages)

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
pipeline_tag: translation
library_name: onnx
base_model: {base_model}
---

# {title}

ONNX-exported version of [`{base_model}`](https://huggingface.co/{base_model})
for in-browser inference.

Built for use with [Transformers.js](https://github.com/huggingface/transformers.js)
and [onnxruntime-web](https://onnxruntime.ai/docs/get-started/with-javascript.html)
in the browser, with fast BPE tokenizer.json files that don't require the
SentencePiece WASM runtime.

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

    # Create README.md if it doesn't exist
    readme_path = model_dir / "README.md"
    if not readme_path.exists():
        logger.info("Generating %s README.md for repo %s", direction, repo_id)
        readme_content = generate_default_readme(direction, repo_id)
        readme_path.write_text(readme_content, encoding="utf-8")
    else:
        logger.info("README.md already exists in %s, skipping generation", model_dir)

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
