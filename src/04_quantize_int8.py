#!/usr/bin/env python3
"""INT8 dynamic quantization of fp32 IndicTrans2 ONNX bundle.

Only run after fp32 parity passes (>= 99% token-exact).
Expect ~80% exact text match per en→indic int8 benchmarks.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ONNX_FILES = (
    "encoder_model.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
)


def quantize_int8(input_dir: Path, output_dir: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ONNX_FILES:
        src = input_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing {src}")

        dst = output_dir / name
        logger.info("Quantizing %s → %s", name, dst)
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QInt8,
            per_channel=True,
        )

    # Tokenizer and config files are byte-identical to fp32
    for pattern in (
        "tokenizer_*.json", "config.json", "generation_config.json",
        "dict.*.json", "model.SRC", "model.TGT",
        "tokenization_indictrans.py", "tokenizer_config.json", "special_tokens_map.json",
    ):
        for src in input_dir.glob(pattern):
            shutil.copy2(src, output_dir / src.name)
            logger.info("Copied %s", src.name)

    logger.info("INT8 quantization complete → %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="INT8 quantize IT2 ONNX bundle")
    parser.add_argument("--input", required=True, type=Path, help="fp32 ONNX directory")
    parser.add_argument("--output", required=True, type=Path, help="int8 output directory")
    args = parser.parse_args()
    quantize_int8(args.input, args.output)


if __name__ == "__main__":
    main()
