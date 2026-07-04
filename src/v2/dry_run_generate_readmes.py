#!/usr/bin/env python3
"""Run 05_upload_hf.py in dry-run mode for all 1B model bundles to generate README.md and copy plots - V2."""

import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRATCH_DIR = Path("scratch")
HF_ORG = "hari31416"

DIRECTIONS = {
    "en-indic": "indictrans2-en-indic-1B-ONNX",
    "indic-en": "indictrans2-indic-en-1B-ONNX",
    "indic-indic": "indictrans2-indic-indic-1B-ONNX",
}

PRECISIONS = ["fp32", "fp16", "int8", "q4f16"]

def main():
    if not SCRATCH_DIR.is_dir():
        logger.error("scratch/ directory does not exist! Please ensure you have exported models.")
        return
        
    for direction, repo_base in DIRECTIONS.items():
        for precision in PRECISIONS:
            precision_suffix = f"-{precision}" if precision != "fp32" else ""
            folder_name = f"{direction}-1b-onnx{precision_suffix}"
            model_dir = SCRATCH_DIR / folder_name
            
            if not model_dir.is_dir():
                logger.warning("Directory not found, skipping: %s", model_dir)
                continue
                
            repo_id = f"{HF_ORG}/{repo_base}{precision_suffix}"
            logger.info("Generating README.md for %s (%s)", folder_name, repo_id)
            
            cmd = [
                ".venv/bin/python", "src/v2/05_upload_hf.py",
                "--model-dir", str(model_dir),
                "--repo-id", repo_id,
                "--dry-run"
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error("Failed to generate for %s: %s", folder_name, e)

if __name__ == "__main__":
    main()
