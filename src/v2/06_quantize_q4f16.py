#!/usr/bin/env python3
"""4-bit weight-only quantization (q4f16) of IndicTrans2 ONNX graphs - V2.

Supports larger models by checking if the quantized model size exceeds 2GB and saving with
external weights format if needed.
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

DEFAULT_BLOCK_SIZE = 32
DEFAULT_ACCURACY_LEVEL = 4
DEFAULT_EXCLUDE_SUBSTRINGS = ("embed_tokens", "lm_head", "shared")


def _list_matmul_nodes(onnx_path: Path) -> None:
    """Print all MatMul node names in the graph for exclusion list tuning."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    nodes = [n for n in model.graph.node if n.op_type == "MatMul"]
    logger.info("MatMul nodes in %s (%d total):", onnx_path.name, len(nodes))
    for n in nodes:
        logger.info("  %s", n.name)


def _build_exclude_set(
    onnx_path: Path, exclude_substrings: tuple[str, ...]
) -> list[str]:
    """Return node names containing any of the given substrings."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    excluded: list[str] = []
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        if any(sub in node.name for sub in exclude_substrings):
            excluded.append(node.name)
    if excluded:
        logger.info(
            "  Excluding %d MatMul nodes from q4 in %s: %s",
            len(excluded),
            onnx_path.name,
            excluded,
        )
    return excluded


def _quantize_one(
    src: Path,
    dst: Path,
    block_size: int,
    accuracy_level: int,
    exclude_substrings: tuple[str, ...],
) -> None:
    """Apply MatMulNBitsQuantizer (4-bit weight-only) to a single ONNX file."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    logger.info(
        "Quantizing  %s  (%.0f MB)  block_size=%d  accuracy_level=%d",
        src.name,
        src.stat().st_size / 1e6,
        block_size,
        accuracy_level,
    )

    nodes_to_exclude = _build_exclude_set(src, exclude_substrings)

    quantizer = MatMulNBitsQuantizer(
        model=str(src),
        bits=4,
        block_size=block_size,
        is_symmetric=False,         # UINT4; asymmetric typically better for seq2seq
        accuracy_level=accuracy_level,
        nodes_to_exclude=nodes_to_exclude,
    )
    quantizer.process()

    try:
        size_bytes = quantizer.model.model.ByteSize()
    except AttributeError:
        import onnx
        # Load file size fallback
        size_bytes = onnx.load(str(src), load_external_data=False).ByteSize()

    use_external = size_bytes >= 2 * 1024 * 1024 * 1024
    quantizer.model.save_model_to_file(str(dst), use_external_data_format=use_external)

    src_mb = src.stat().st_size / 1e6
    dst_mb = dst.stat().st_size / 1e6
    logger.info(
        "  %s  %.0f MB → %.0f MB  (%.0f%% of source)",
        dst.name,
        src_mb,
        dst_mb,
        100 * dst_mb / src_mb if src_mb else 0,
    )


def quantize_q4f16(
    input_dir: Path,
    output_dir: Path,
    block_size: int,
    accuracy_level: int,
    exclude_substrings: tuple[str, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ONNX_FILES:
        src = input_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing source graph: {src}")
        dst = output_dir / name
        _quantize_one(src, dst, block_size, accuracy_level, exclude_substrings)

    # Copy non-ONNX artifacts unchanged
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
    logger.info(
        "q4f16 quantization complete → %s  (~%.0f MB ONNX)", output_dir, total_mb
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="q4f16 (4-bit weights + fp16 activations) quantization of IT2 ONNX bundle - V2"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing fp16 (or fp32) ONNX source bundle",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for q4f16 bundle",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"MatMulNBits block size (default {DEFAULT_BLOCK_SIZE}; smaller = higher accuracy)",
    )
    parser.add_argument(
        "--exclude-nodes",
        nargs="*",
        default=list(DEFAULT_EXCLUDE_SUBSTRINGS),
        help="Node name substrings to exclude from quantization "
        f"(default: {list(DEFAULT_EXCLUDE_SUBSTRINGS)})",
    )
    parser.add_argument(
        "--accuracy-level",
        type=int,
        default=DEFAULT_ACCURACY_LEVEL,
        choices=[1, 2, 4],
        help=(
            "MatMulNBits compute path: "
            "1=fp32 (safe, slow), "
            f"2=fp16 (WebGPU only — no CPU kernel), "
            f"4=int32-accum (default; correct on CPU). "
            f"Default: {DEFAULT_ACCURACY_LEVEL}"
        ),
    )
    parser.add_argument(
        "--list-nodes",
        metavar="ONNX_FILE",
        help="Print all MatMul node names in the given file and exit (for tuning --exclude-nodes)",
    )
    args = parser.parse_args()

    if args.list_nodes:
        target = args.input / args.list_nodes
        _list_matmul_nodes(target)
        return

    quantize_q4f16(
        args.input,
        args.output,
        block_size=args.block_size,
        accuracy_level=args.accuracy_level,
        exclude_substrings=tuple(args.exclude_nodes),
    )


if __name__ == "__main__":
    main()
