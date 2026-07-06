#!/usr/bin/env python3
"""Export IndicTrans2 seq2seq model to three ONNX graphs for browser inference - V2.

Supports larger (1B+) models by dynamically checking shape configurations and using
ONNX external weight data format when individual graphs exceed the 2GB Protobuf limit.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM

from it2_onnx_wrappers import (
    IndicTransDecoderWithPastWrapper,
    IndicTransDecoderWrapper,
    IndicTransEncoderWrapper,
    past_input_names,
    present_output_names,
    weights_are_tied,
)
from onnx_bundle_optimize import optimize_export_bundle

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Dummy shapes for tracing
BATCH = 1
ENC_SEQ = 8
DEC_SEQ = 1
EXTERNALIZE_THRESHOLD_MB = 100


def _externalize_if_large(onnx_path: Path, size_threshold_mb: int = 512) -> None:
    """Move weights to .onnx.data sidecar when the protobuf is large."""
    import onnx
    from onnx import save_model
    from onnx.external_data_helper import convert_model_to_external_data

    model = onnx.load(str(onnx_path), load_external_data=True)
    proto_mb = onnx_path.stat().st_size / (1024 * 1024)
    if proto_mb < size_threshold_mb:
        return

    data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if data_path.exists():
        data_path.unlink()

    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
    )
    save_model(model, str(onnx_path))
    logger.info("Externalized weights → %s (proto was ~%.0f MB)", data_path.name, proto_mb)


def _export_and_consolidate(
    wrapper: torch.nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
    use_external: bool,
) -> None:
    import onnx
    from onnx import save_model

    # Determine temporary export path to isolate from filename collisions
    temp_dir = path.parent / f"tmp_{path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / path.name

    logger.info("Exporting to temporary path: %s", temp_path)
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(temp_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
        external_data=use_external,
    )

    # Load and consolidate
    logger.info("Loading and consolidating: %s", path.name)
    model = onnx.load(str(temp_path), load_external_data=True)

    # Save to the final destination with external data if needed
    data_path = path.with_suffix(path.suffix + ".data")
    if data_path.exists():
        data_path.unlink()

    if use_external:
        save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path.name,
            size_threshold=1024,
        )
        logger.info("Saved with external weights: %s and %s", path.name, data_path.name)
    else:
        save_model(
            model,
            str(path),
            save_as_external_data=False,
        )
        # If the file size is large (e.g. >512MB) but we didn't use external_data during tracing,
        # we still externalize it here for safety.
        _externalize_if_large(path)

    # Clean up temporary directory
    shutil.rmtree(temp_dir)


def _export_encoder(encoder: torch.nn.Module, output_dir: Path, opset: int) -> None:
    wrapper = IndicTransEncoderWrapper(encoder).eval()
    input_ids = torch.ones((BATCH, ENC_SEQ), dtype=torch.long)
    attention_mask = torch.ones((BATCH, ENC_SEQ), dtype=torch.long)

    path = output_dir / "encoder_model.onnx"
    logger.info("Exporting encoder → %s", path.name)

    param_size = sum(p.numel() for p in wrapper.parameters()) * 4
    use_external = param_size > 1.5 * 1024 * 1024 * 1024

    _export_and_consolidate(
        wrapper=wrapper,
        dummy_inputs=(input_ids, attention_mask),
        path=path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "encoder_sequence_length"},
            "attention_mask": {0: "batch_size", 1: "encoder_sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "encoder_sequence_length"},
        },
        opset=opset,
        use_external=use_external,
    )


def _export_decoder(
    decoder: torch.nn.Module,
    lm_head: torch.nn.Module,
    num_layers: int,
    embed_dim: int,
    output_dir: Path,
    opset: int,
) -> None:
    wrapper = IndicTransDecoderWrapper(decoder, lm_head).eval()

    input_ids = torch.ones((BATCH, DEC_SEQ), dtype=torch.long)
    encoder_attention_mask = torch.ones((BATCH, ENC_SEQ), dtype=torch.long)
    encoder_hidden_states = torch.randn(BATCH, ENC_SEQ, embed_dim)

    output_names = ["logits", *present_output_names(num_layers)]
    path = output_dir / "decoder_model.onnx"
    logger.info("Exporting decoder (first step) → %s", path.name)

    param_size = sum(p.numel() for p in wrapper.parameters()) * 4
    use_external = param_size > 1.5 * 1024 * 1024 * 1024

    _export_and_consolidate(
        wrapper=wrapper,
        dummy_inputs=(input_ids, encoder_attention_mask, encoder_hidden_states),
        path=path,
        input_names=["input_ids", "encoder_attention_mask", "encoder_hidden_states"],
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
            "encoder_attention_mask": {0: "batch_size", 1: "encoder_sequence_length"},
            "encoder_hidden_states": {0: "batch_size", 1: "encoder_sequence_length"},
            "logits": {0: "batch_size", 1: "decoder_sequence_length"},
        },
        opset=opset,
        use_external=use_external,
    )


def _export_decoder_with_past(
    decoder: torch.nn.Module,
    lm_head: torch.nn.Module,
    num_layers: int,
    embed_dim: int,
    num_heads: int,
    head_dim: int,
    output_dir: Path,
    opset: int,
) -> None:
    wrapper = IndicTransDecoderWithPastWrapper(decoder, lm_head, num_layers, embed_dim).eval()

    input_ids = torch.ones((BATCH, DEC_SEQ), dtype=torch.long)
    encoder_attention_mask = torch.ones((BATCH, ENC_SEQ), dtype=torch.long)

    past_tensors = []
    for _ in range(num_layers):
        past_tensors.extend([
            torch.randn(BATCH, num_heads, 1, head_dim),  # decoder key (seq len 1 at step 2)
            torch.randn(BATCH, num_heads, 1, head_dim),  # decoder value
            torch.randn(BATCH, num_heads, ENC_SEQ, head_dim),  # encoder key
            torch.randn(BATCH, num_heads, ENC_SEQ, head_dim),  # encoder value
        ])

    input_names = ["input_ids", "encoder_attention_mask", *past_input_names(num_layers)]
    output_names = ["logits", *present_output_names(num_layers)]
    path = output_dir / "decoder_with_past_model.onnx"
    logger.info("Exporting decoder_with_past → %s", path.name)

    dynamic_axes: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
        "encoder_attention_mask": {0: "batch_size", 1: "encoder_sequence_length"},
        "logits": {0: "batch_size", 1: "decoder_sequence_length"},
    }
    for i in range(num_layers):
        dynamic_axes[f"past_key_values.{i}.decoder.key"] = {
            0: "batch_size", 2: "past_decoder_sequence_length"
        }
        dynamic_axes[f"past_key_values.{i}.decoder.value"] = {
            0: "batch_size", 2: "past_decoder_sequence_length"
        }
        dynamic_axes[f"past_key_values.{i}.encoder.key"] = {
            0: "batch_size", 2: "encoder_sequence_length"
        }
        dynamic_axes[f"past_key_values.{i}.encoder.value"] = {
            0: "batch_size", 2: "encoder_sequence_length"
        }
        dynamic_axes[f"present.{i}.decoder.key"] = {
            0: "batch_size", 2: "past_decoder_sequence_length_plus_1"
        }
        dynamic_axes[f"present.{i}.decoder.value"] = {
            0: "batch_size", 2: "past_decoder_sequence_length_plus_1"
        }
        dynamic_axes[f"present.{i}.encoder.key"] = {
            0: "batch_size", 2: "encoder_sequence_length"
        }
        dynamic_axes[f"present.{i}.encoder.value"] = {
            0: "batch_size", 2: "encoder_sequence_length"
        }

    param_size = sum(p.numel() for p in wrapper.parameters()) * 4
    use_external = param_size > 1.5 * 1024 * 1024 * 1024

    _export_and_consolidate(
        wrapper=wrapper,
        dummy_inputs=(input_ids, encoder_attention_mask, *past_tensors),
        path=path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset=opset,
        use_external=use_external,
    )


def _copy_hf_artifacts(model_id: str, output_dir: Path) -> None:
    for filename in (
        "config.json",
        "generation_config.json",
        "dict.SRC.json",
        "dict.TGT.json",
        "model.SRC",
        "model.TGT",
        "tokenization_indictrans.py",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        try:
            path = hf_hub_download(repo_id=model_id, filename=filename)
            shutil.copy2(path, output_dir / filename)
            logger.info("Copied %s", filename)
        except Exception:
            logger.debug("Skipped %s (not in repo)", filename)


def export_onnx(model_id: str, output_dir: Path, opset: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model %s", model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    num_layers = model.config.decoder_layers
    embed_dim = model.config.decoder_embed_dim
    num_heads = getattr(model.config, "decoder_attention_heads", 8)
    head_dim = embed_dim // num_heads
    if weights_are_tied(model.model.decoder, model.lm_head):
        logger.info("decoder.embed_tokens and lm_head are tied — post-export dedup will apply")
    else:
        logger.info(
            "decoder.embed_tokens and lm_head are separate — skipping tied-weight dedup"
        )

    logger.info(
        "Model Config Details: num_layers=%d, embed_dim=%d, num_heads=%d, head_dim=%d",
        num_layers, embed_dim, num_heads, head_dim
    )

    _export_encoder(model.model.encoder, output_dir, opset)
    _export_decoder(model.model.decoder, model.lm_head, num_layers, embed_dim, output_dir, opset)
    _export_decoder_with_past(
        model.model.decoder,
        model.lm_head,
        num_layers,
        embed_dim,
        num_heads,
        head_dim,
        output_dir,
        opset,
    )
    _copy_hf_artifacts(model_id, output_dir)

    logger.info("Running post-export optimizations (tied-weight dedup, onnxsim, externalize, shared decoder weights)")
    optimize_export_bundle(output_dir, externalize_threshold_mb=EXTERNALIZE_THRESHOLD_MB)

    expected = [
        "encoder_model.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx",
    ]
    missing = [f for f in expected if not (output_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Export incomplete — missing: {missing}")

    total_mb = sum(
        f.stat().st_size
        for f in output_dir.iterdir()
        if f.suffix in (".onnx", ".data")
    ) / (1024 * 1024)
    logger.info("Export complete. ONNX artifacts: ~%.0f MB", total_mb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export IndicTrans2 to ONNX - V2")
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id, e.g. ai4bharat/indictrans2-en-indic-1B",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for ONNX bundle",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset (default 17, compatible with onnxruntime-web 1.21+)",
    )
    args = parser.parse_args()
    export_onnx(args.model, args.output, args.opset)


if __name__ == "__main__":
    main()
