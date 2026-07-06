#!/usr/bin/env python3
"""Post-convert fp32 IndicTrans2 ONNX graphs to fp16.

Runs after fp32 parity passes (>= 99% token-exact).
Uses onnxruntime.transformers.float16.convert_float_to_float16 (shipped with
onnxruntime) with force_fp16_initializers=True so that bias weights are also
cast to fp16 alongside the weight matrices.

Note: onnxconverter-common's version of convert_float_to_float16 leaves bias
initializers in fp32, producing mixed-type Add nodes that ORT rejects at
session load.  The ORT-bundled version avoids this.

Produces:
  encoder_model.onnx
  decoder_model.onnx
  decoder_with_past_model.onnx

Produces fp16 graphs plus the same sidecar layout (`encoder_model.onnx.data`, `decoder_shared.onnx.data`).

Usage:
  python src/05_convert_fp16.py --input scratch/en-indic-onnx \\
                                 --output scratch/en-indic-onnx-fp16
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from onnx_bundle_optimize import finalize_bundle_layout

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ONNX_FILES = (
    "encoder_model.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
)


def _convert_one(src: Path, dst: Path) -> None:
    """Convert a single ONNX file from fp32 to fp16 using ORT's bundled tool."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16 as ort_fp16

    logger.info("Loading  %s  (%.0f MB)", src.name, src.stat().st_size / 1e6)
    model_fp32 = onnx.load(str(src), load_external_data=True)

    model_fp16 = ort_fp16(
        model_fp32,
        keep_io_types=True,           # int64 input_ids / masks stay fp32 at boundaries
        force_fp16_initializers=True, # also cast bias initializers to fp16; without this
                                      # Add(fp16_matmul, fp32_bias) raises a type error in ORT
        disable_shape_infer=True,     # skip shape-infer pass during conversion;
                                      # ORT re-infers at session load from scratch
    )

    # Belt-and-suspenders: clear any residual intermediate value_info annotations
    # so ORT re-infers all intermediate types cleanly on session load.
    del model_fp16.graph.value_info[:]

    logger.info("Saving   %s", dst)
    onnx.save_model(
        model_fp16,
        str(dst),
        save_as_external_data=False,
    )


def convert_fp16(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ONNX_FILES:
        src = input_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing source graph: {src}")
        dst = output_dir / name
        _convert_one(src, dst)

    # Copy non-ONNX artifacts unchanged (tokenizers, configs, vocab files)
    for pattern in (
        "tokenizer_*.json",
        "config.json",
        "generation_config.json",
        "dict.*.json",
        "model.SRC",
        "model.TGT",
        "tokenization_indictrans.py",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        for src in input_dir.glob(pattern):
            shutil.copy2(src, output_dir / src.name)
            logger.info("Copied  %s", src.name)

    logger.info("Finalizing bundle layout (externalize + shared decoder weights)")
    finalize_bundle_layout(output_dir)

    total_mb = (
        sum(
            f.stat().st_size
            for f in output_dir.iterdir()
            if f.suffix in (".onnx", ".data")
        )
        / 1e6
    )
    logger.info("fp16 conversion complete → %s  (~%.0f MB ONNX)", output_dir, total_mb)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert fp32 IndicTrans2 ONNX bundle to fp16"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing fp32 ONNX bundle",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for fp16 bundle",
    )
    args = parser.parse_args()
    convert_fp16(args.input, args.output)


if __name__ == "__main__":
    main()
