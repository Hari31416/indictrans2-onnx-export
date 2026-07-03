#!/usr/bin/env python3
"""Benchmark a quantized ONNX bundle (fp16 / q4f16 / int8) against the fp32 ONNX oracle.

Chain of trust:
  PyTorch  →  fp32 ONNX (validated ≥ 99% by 03_validate_parity.py)
                  ↓
              fp16 ONNX  ──── this script ────  q4f16 ONNX
                                                 int8 ONNX

Since fp32 ONNX already matches PyTorch at ≥ 99%, comparing quantized bundles
against fp32 ONNX avoids reloading the heavy PyTorch model for every precision.

Metrics reported per sentence and in aggregate:
  - token_exact_rate : fraction of fixtures where decoded token IDs are identical
  - text_exact_rate  : fraction where post-processed text strings are identical
  - avg_latency_ms   : mean wall-clock time per sentence (ORT CPU, greedy loop)
  - tokens_per_sec   : output tokens / total decode time

Usage:
  # Compare fp16 vs fp32 (same direction):
  python src/07_benchmark_precision.py \\
      --fp32-dir  scratch/en-indic-onnx \\
      --cmp-dir   scratch/en-indic-onnx-fp16 \\
      --fixtures  fixtures/en-indic-golden.jsonl \\
      --pytorch-model ai4bharat/indictrans2-en-indic-dist-200M \\
      --label fp16 \\
      --report fixtures/benchmark-en-indic-fp16.json

  # Compare q4f16 vs fp32:
  python src/07_benchmark_precision.py \\
      --fp32-dir  scratch/en-indic-onnx \\
      --cmp-dir   scratch/en-indic-onnx-q4f16 \\
      --fixtures  fixtures/en-indic-golden.jsonl \\
      --pytorch-model ai4bharat/indictrans2-en-indic-dist-200M \\
      --label q4f16 \\
      --report fixtures/benchmark-en-indic-q4f16.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 128


@dataclass
class Fixture:
    text: str
    src_lang: str
    tgt_lang: str
    category: str = "generic"


def load_fixtures(path: Path) -> list[Fixture]:
    fixtures: list[Fixture] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fixtures.append(
                Fixture(
                    text=row["text"],
                    src_lang=row["src_lang"],
                    tgt_lang=row["tgt_lang"],
                    category=row.get("category", "generic"),
                )
            )
    return fixtures


def _past_feed(past_outputs: list[np.ndarray], num_layers: int) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for i in range(num_layers):
        base = i * 4
        feed[f"past_key_values.{i}.decoder.key"] = past_outputs[base]
        feed[f"past_key_values.{i}.decoder.value"] = past_outputs[base + 1]
        feed[f"past_key_values.{i}.encoder.key"] = past_outputs[base + 2]
        feed[f"past_key_values.{i}.encoder.value"] = past_outputs[base + 3]
    return feed


def _make_sessions(onnx_dir: Path) -> tuple[Any, Any, Any]:
    """Load the three ONNX inference sessions for a bundle directory."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # suppress ORT INFO spam

    enc = ort.InferenceSession(
        str(onnx_dir / "encoder_model.onnx"), sess_options=opts
    )
    dec = ort.InferenceSession(
        str(onnx_dir / "decoder_model.onnx"), sess_options=opts
    )
    dec_past = ort.InferenceSession(
        str(onnx_dir / "decoder_with_past_model.onnx"), sess_options=opts
    )
    return enc, dec, dec_past


@dataclass
class DecodeResult:
    token_ids: list[int]
    text: str
    latency_ms: float  # wall-clock for greedy loop (excl. tokenization)
    n_output_tokens: int  # tokens produced (incl. EOS)


def _greedy_decode_onnx(
    onnx_dir: Path,
    fixtures: list[Fixture],
    pytorch_model: str,
) -> list[DecodeResult]:
    """Run greedy decode on an ONNX bundle, return decoded results + timing."""
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    from IndicTransToolkit import IndicProcessor

    meta = json.loads((onnx_dir / "tokenizer_meta.json").read_text(encoding="utf-8"))
    src_tok = Tokenizer.from_file(str(onnx_dir / "tokenizer_src.json"))
    slow_tok = AutoTokenizer.from_pretrained(pytorch_model, trust_remote_code=True)
    ip = IndicProcessor(inference=True)

    enc, dec, dec_past = _make_sessions(onnx_dir)
    num_layers = (len(dec.get_outputs()) - 1) // 4

    gen_cfg: dict[str, Any] = {}
    gen_config_path = onnx_dir / "generation_config.json"
    if gen_config_path.exists():
        gen_cfg = json.loads(gen_config_path.read_text(encoding="utf-8"))

    decoder_start_id = int(gen_cfg.get("decoder_start_token_id", 2))
    eos_id = int(gen_cfg.get("eos_token_id", 2))

    results: list[DecodeResult] = []

    for fx in fixtures:
        if hasattr(ip, "_placeholder_entity_maps"):
            ip._placeholder_entity_maps.queue.clear()

        batch = ip.preprocess_batch([fx.text], src_lang=fx.src_lang, tgt_lang=fx.tgt_lang)
        prefixed = batch[0]
        encoded = src_tok.encode(prefixed)
        input_ids_list = [
            i if i < meta["src_dict_size"] else meta["unk_id"] for i in encoded.ids
        ]
        input_ids = np.array([input_ids_list], dtype=np.int64)
        attn_mask = np.array([encoded.attention_mask], dtype=np.int64)

        t_start = time.perf_counter()

        enc_out = enc.run(["last_hidden_state"], {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        })[0]

        decoder_input_ids = np.array([[decoder_start_id]], dtype=np.int64)
        output_ids = [decoder_start_id]
        past_outputs: list[np.ndarray] | None = None

        for step in range(MAX_NEW_TOKENS):
            if step == 0:
                dec_out = dec.run(None, {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": enc_out,
                    "encoder_attention_mask": attn_mask,
                })
            else:
                dec_out = dec_past.run(None, {
                    "input_ids": decoder_input_ids,
                    "encoder_attention_mask": attn_mask,
                    **_past_feed(past_outputs, num_layers),
                })

            logits = dec_out[0]
            past_outputs = list(dec_out[1:])
            next_id = int(np.argmax(logits[0, -1, :]))
            output_ids.append(next_id)
            if next_id == eos_id:
                break
            decoder_input_ids = np.array([[next_id]], dtype=np.int64)

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        safe_ids = [i if i < meta["tgt_dict_size"] else meta["unk_id"] for i in output_ids]
        with slow_tok.as_target_tokenizer():
            decoded = slow_tok.batch_decode(
                [safe_ids],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        text = ip.postprocess_batch(decoded, lang=fx.tgt_lang)[0]

        results.append(
            DecodeResult(
                token_ids=output_ids,
                text=text,
                latency_ms=latency_ms,
                n_output_tokens=len(output_ids),
            )
        )

    return results


def benchmark(
    fp32_dir: Path,
    cmp_dir: Path,
    fixtures: list[Fixture],
    pytorch_model: str,
    label: str,
) -> dict[str, Any]:
    """Compare cmp_dir (fp16/q4f16/int8) against fp32_dir as oracle.

    Returns a report dict suitable for JSON serialisation.
    """
    logger.info("=== Decoding with fp32 oracle: %s ===", fp32_dir)
    fp32_results = _greedy_decode_onnx(fp32_dir, fixtures, pytorch_model)

    logger.info("=== Decoding with %s: %s ===", label, cmp_dir)
    cmp_results = _greedy_decode_onnx(cmp_dir, fixtures, pytorch_model)

    token_match = 0
    text_match = 0
    mismatches: list[dict[str, Any]] = []

    total_fp32_latency = sum(r.latency_ms for r in fp32_results)
    total_cmp_latency = sum(r.latency_ms for r in cmp_results)
    total_fp32_tokens = sum(r.n_output_tokens for r in fp32_results)
    total_cmp_tokens = sum(r.n_output_tokens for r in cmp_results)

    for i, (ref, cmp) in enumerate(zip(fp32_results, cmp_results)):
        tok_ok = ref.token_ids == cmp.token_ids
        txt_ok = ref.text == cmp.text
        if tok_ok:
            token_match += 1
        if txt_ok:
            text_match += 1
        if not tok_ok or not txt_ok:
            mismatches.append({
                "index": i,
                "fixture": fixtures[i].__dict__,
                "fp32_tokens": ref.token_ids[:20],
                f"{label}_tokens": cmp.token_ids[:20],
                "fp32_text": ref.text,
                f"{label}_text": cmp.text,
                "tokens_match": tok_ok,
                "text_match": txt_ok,
            })

    n = len(fixtures)
    report: dict[str, Any] = {
        "label": label,
        "oracle": str(fp32_dir),
        "cmp_dir": str(cmp_dir),
        "total_fixtures": n,
        # Quality
        "token_exact_rate": round(token_match / n * 100, 2) if n else 0,
        "text_exact_rate": round(text_match / n * 100, 2) if n else 0,
        "token_exact_count": token_match,
        "text_exact_count": text_match,
        # Latency
        "fp32_avg_latency_ms": round(total_fp32_latency / n, 1) if n else 0,
        "cmp_avg_latency_ms": round(total_cmp_latency / n, 1) if n else 0,
        "speedup_vs_fp32": round(total_fp32_latency / total_cmp_latency, 3)
        if total_cmp_latency
        else 0,
        "fp32_tokens_per_sec": round(total_fp32_tokens / (total_fp32_latency / 1000), 1)
        if total_fp32_latency
        else 0,
        "cmp_tokens_per_sec": round(total_cmp_tokens / (total_cmp_latency / 1000), 1)
        if total_cmp_latency
        else 0,
        # Mismatches
        "mismatches": mismatches,
    }

    # Log summary
    logger.info(
        "[%s] token exact: %d/%d (%.1f%%)  text exact: %d/%d (%.1f%%)",
        label,
        token_match, n, report["token_exact_rate"],
        text_match, n, report["text_exact_rate"],
    )
    logger.info(
        "[%s] latency: fp32=%.0fms  %s=%.0fms  speedup=%.2f×",
        label,
        report["fp32_avg_latency_ms"],
        label,
        report["cmp_avg_latency_ms"],
        report["speedup_vs_fp32"],
    )
    logger.info(
        "[%s] throughput: fp32=%.1f tok/s  %s=%.1f tok/s",
        label,
        report["fp32_tokens_per_sec"],
        label,
        report["cmp_tokens_per_sec"],
    )

    if report["token_exact_rate"] >= 99.0:
        logger.info("[%s] PASS ✓  token_exact_rate ≥ 99%% (lossless tier)", label)
    elif report["token_exact_rate"] >= 80.0:
        logger.info(
            "[%s] PASS ✓  token_exact_rate %.1f%% ≥ 80%% (acceptable for 4-bit weight-only tier)",
            label, report["token_exact_rate"],
        )
    elif report["token_exact_rate"] >= 60.0:
        logger.warning(
            "[%s] WARN  token_exact_rate %.1f%% — mismatches may still be semantic near-matches; "
            "inspect mismatches and run larger fixture set before rejecting",
            label, report["token_exact_rate"],
        )
    else:
        logger.error(
            "[%s] FAIL ✗  token_exact_rate %.1f%% — check quantization settings",
            label, report["token_exact_rate"],
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fp16/q4f16/int8 ONNX bundle against fp32 ONNX oracle"
    )
    parser.add_argument(
        "--fp32-dir",
        required=True,
        type=Path,
        help="fp32 ONNX bundle directory (oracle/reference)",
    )
    parser.add_argument(
        "--cmp-dir",
        required=True,
        type=Path,
        help="Bundle to compare (fp16 / q4f16 / int8 ONNX directory)",
    )
    parser.add_argument(
        "--fixtures",
        required=True,
        type=Path,
        help="JSONL fixture file",
    )
    parser.add_argument(
        "--pytorch-model",
        required=True,
        help="HF model id (used to load slow tokenizer for detokenisation)",
    )
    parser.add_argument(
        "--label",
        default="cmp",
        help="Short label for the compared bundle, e.g. fp16 or q4f16 (default: cmp)",
    )
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        help="Output JSON report path",
    )
    args = parser.parse_args()

    fixtures = load_fixtures(args.fixtures)
    logger.info("Loaded %d fixtures from %s", len(fixtures), args.fixtures)

    report = benchmark(
        fp32_dir=args.fp32_dir,
        cmp_dir=args.cmp_dir,
        fixtures=fixtures,
        pytorch_model=args.pytorch_model,
        label=args.label,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Report written → %s", args.report)


if __name__ == "__main__":
    main()
