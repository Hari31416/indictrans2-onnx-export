#!/usr/bin/env python3
"""Post-convert fp32 IndicTrans2 ONNX graphs to fp16 - V2.

Supports larger models by checking if serialized model size exceeds the 2GB limit and
saving directly in external weights format if needed.
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


def _convert_one(src: Path, dst: Path) -> None:
    """Convert a single ONNX file from fp32 to fp16 using ORT's bundled tool."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16 as ort_fp16

    logger.info("Loading  %s  (%.0f MB)", src.name, src.stat().st_size / 1e6)
    model_fp32 = onnx.load(str(src), load_external_data=True)

    model_fp16 = ort_fp16(
        model_fp32,
        keep_io_types=True,           # int64 input_ids / masks stay fp32 at boundaries
        force_fp16_initializers=True, # also cast bias initializers to fp16
        disable_shape_infer=True,     # skip shape-infer pass during conversion
    )

    # Belt-and-suspenders: clear any residual intermediate value_info annotations
    del model_fp16.graph.value_info[:]

    logger.info("Saving   %s", dst)
    
    size_bytes = model_fp16.ByteSize()
    save_external = size_bytes >= 2 * 1024 * 1024 * 1024
    
    if save_external:
        data_path = dst.with_suffix(dst.suffix + ".data")
        if data_path.exists():
            data_path.unlink()
        onnx.save_model(
            model_fp16,
            str(dst),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path.name,
            size_threshold=1024,
        )
        logger.info("Saved externalized fp16 weights to %s (protobuf was ~%.0f MB)", data_path.name, size_bytes / 1e6)
    else:
        onnx.save_model(
            model_fp16,
            str(dst),
            save_as_external_data=False,
        )
        # Re-externalize if the proto is large (>512 MB) to stay within protobuf limits
        _externalize_if_large(dst)


def _externalize_if_large(onnx_path: Path, threshold_mb: int = 512) -> None:
    import onnx
    from onnx import save_model as onnx_save
    from onnx.external_data_helper import convert_model_to_external_data

    size_mb = onnx_path.stat().st_size / 1e6
    if size_mb < threshold_mb:
        return

    model = onnx.load(str(onnx_path), load_external_data=True)
    data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if data_path.exists():
        data_path.unlink()

    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
    )
    onnx_save(model, str(onnx_path))
    logger.info(
        "Externalized weights → %s  (proto was ~%.0f MB)", data_path.name, size_mb
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
        for src_file in input_dir.glob(pattern):
            shutil.copy2(src_file, output_dir / src_file.name)
            logger.info("Copied  %s", src_file.name)

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
        description="Convert fp32 IndicTrans2 ONNX bundle to fp16 - V2"
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
