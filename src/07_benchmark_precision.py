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
  - avg_latency_ms   : mean amortized wall-clock time per sentence (ORT CPU, greedy loop)
  - tokens_per_sec   : output tokens / total decode time
  - sentences_per_sec: fixtures / total decode time (batch-amortized when batch_size > 1)

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sacrebleu

from it2_inference import DecodeOutput, greedy_decode_onnx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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


def calculate_group_metrics(
    indices: list[int],
    fixtures: list[Fixture],
    fp32_results: list[DecodeOutput],
    cmp_results: list[DecodeOutput],
) -> dict[str, Any]:
    """Calculate all standard metrics for a subset of results specified by indices."""
    if not indices:
        return {}

    token_match = 0
    text_match = 0
    total_fp32_latency = 0.0
    total_cmp_latency = 0.0
    total_fp32_tokens = 0
    total_cmp_tokens = 0

    group_fp32_texts = []
    group_cmp_texts = []

    for idx in indices:
        ref = fp32_results[idx]
        cmp = cmp_results[idx]
        if ref.token_ids == cmp.token_ids:
            token_match += 1
        if ref.text == cmp.text:
            text_match += 1
        total_fp32_latency += ref.latency_ms
        total_cmp_latency += cmp.latency_ms
        total_fp32_tokens += ref.n_output_tokens
        total_cmp_tokens += cmp.n_output_tokens
        group_fp32_texts.append(ref.text)
        group_cmp_texts.append(cmp.text)

    n = len(indices)

    try:
        bleu_score = round(sacrebleu.corpus_bleu(group_cmp_texts, [group_fp32_texts]).score, 2)
    except Exception as e:
        logger.warning("Failed to calculate BLEU score: %s", e)
        bleu_score = 0.0

    try:
        chrf_score = round(sacrebleu.corpus_chrf(group_cmp_texts, [group_fp32_texts]).score, 2)
    except Exception as e:
        logger.warning("Failed to calculate ChrF score: %s", e)
        chrf_score = 0.0

    return {
        "total_fixtures": n,
        "token_exact_rate": round(token_match / n * 100, 2) if n else 0.0,
        "text_exact_rate": round(text_match / n * 100, 2) if n else 0.0,
        "token_exact_count": token_match,
        "text_exact_count": text_match,
        "sacrebleu_bleu": bleu_score,
        "sacrebleu_chrf": chrf_score,
        "fp32_avg_latency_ms": round(total_fp32_latency / n, 1) if n else 0.0,
        "cmp_avg_latency_ms": round(total_cmp_latency / n, 1) if n else 0.0,
        "speedup_vs_fp32": round(total_fp32_latency / total_cmp_latency, 3) if total_cmp_latency else 0.0,
        "fp32_tokens_per_sec": round(total_fp32_tokens / (total_fp32_latency / 1000), 1) if total_fp32_latency else 0.0,
        "cmp_tokens_per_sec": round(total_cmp_tokens / (total_cmp_latency / 1000), 1) if total_cmp_latency else 0.0,
        "fp32_sentences_per_sec": round(n / (total_fp32_latency / 1000), 2) if total_fp32_latency else 0.0,
        "cmp_sentences_per_sec": round(n / (total_cmp_latency / 1000), 2) if total_cmp_latency else 0.0,
    }


def benchmark(
    fp32_dir: Path,
    cmp_dir: Path,
    fixtures: list[Fixture],
    pytorch_model: str,
    label: str,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare cmp_dir (fp16/q4f16/int8) against fp32_dir as oracle.

    Returns a tuple of (report_dict, full_mismatches_list).
    """
    logger.info("=== Decoding with fp32 oracle: %s ===", fp32_dir)
    fp32_results = greedy_decode_onnx(
        fp32_dir,
        fixtures,
        pytorch_model=pytorch_model,
        batch_size=batch_size,
        measure_latency=True,
    )

    logger.info("=== Decoding with %s: %s ===", label, cmp_dir)
    cmp_results = greedy_decode_onnx(
        cmp_dir,
        fixtures,
        pytorch_model=pytorch_model,
        batch_size=batch_size,
        measure_latency=True,
    )

    mismatches: list[dict[str, Any]] = []

    for i, (ref, cmp) in enumerate(zip(fp32_results, cmp_results)):
        tok_ok = ref.token_ids == cmp.token_ids
        txt_ok = ref.text == cmp.text
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
    overall_metrics = calculate_group_metrics(list(range(n)), fixtures, fp32_results, cmp_results)

    by_lang: dict[str, list[int]] = {}
    for i, fx in enumerate(fixtures):
        by_lang.setdefault(fx.tgt_lang, []).append(i)

    metrics_by_language: dict[str, dict[str, Any]] = {}
    for lang, indices in sorted(by_lang.items()):
        metrics_by_language[lang] = calculate_group_metrics(indices, fixtures, fp32_results, cmp_results)

    by_cat: dict[str, list[int]] = {}
    for i, fx in enumerate(fixtures):
        by_cat.setdefault(fx.category, []).append(i)

    metrics_by_category: dict[str, dict[str, Any]] = {}
    for cat, indices in sorted(by_cat.items()):
        metrics_by_category[cat] = calculate_group_metrics(indices, fixtures, fp32_results, cmp_results)

    lang_bleus = [m["sacrebleu_bleu"] for m in metrics_by_language.values()]
    lang_chrfs = [m["sacrebleu_chrf"] for m in metrics_by_language.values()]

    overall_metrics["sacrebleu_bleu_mixed"] = overall_metrics["sacrebleu_bleu"]
    overall_metrics["sacrebleu_chrf_mixed"] = overall_metrics["sacrebleu_chrf"]
    overall_metrics["sacrebleu_bleu"] = round(float(np.median(lang_bleus)), 2) if lang_bleus else 0.0
    overall_metrics["sacrebleu_chrf"] = round(float(np.median(lang_chrfs)), 2) if lang_chrfs else 0.0

    report: dict[str, Any] = {
        "label": label,
        "oracle": str(fp32_dir),
        "cmp_dir": str(cmp_dir),
        "batch_size": batch_size,
        **overall_metrics,
        "metrics_by_language": metrics_by_language,
        "metrics_by_category": metrics_by_category,
        "mismatches": mismatches[:20],
    }

    logger.info(
        "[%s] token exact: %d/%d (%.1f%%)  text exact: %d/%d (%.1f%%)",
        label,
        report["token_exact_count"], n, report["token_exact_rate"],
        report["text_exact_count"], n, report["text_exact_rate"],
    )
    logger.info(
        "[%s] sacrebleu BLEU (median / mixed): %.2f / %.2f  ChrF (median / mixed): %.2f / %.2f",
        label,
        report["sacrebleu_bleu"],
        report["sacrebleu_bleu_mixed"],
        report["sacrebleu_chrf"],
        report["sacrebleu_chrf_mixed"],
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
        "[%s] throughput: fp32=%.1f tok/s  %s=%.1f tok/s  sentences/s: fp32=%.2f  %s=%.2f",
        label,
        report["fp32_tokens_per_sec"],
        label,
        report["cmp_tokens_per_sec"],
        report["fp32_sentences_per_sec"],
        label,
        report["cmp_sentences_per_sec"],
    )

    logger.info("\n=== Target Language Breakdown ===")
    logger.info(f"| Language | Count | Token Match % | Text Match % | BLEU | ChrF | Latency (ms) (FP32 / {label}) | Speedup |")
    logger.info("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for lang, m in sorted(metrics_by_language.items()):
        logger.info(
            "| %s | %d | %.2f%% | %.2f%% | %.2f | %.2f | %.1f / %.1f | %.3fx |",
            lang,
            m["total_fixtures"],
            m["token_exact_rate"],
            m["text_exact_rate"],
            m["sacrebleu_bleu"],
            m["sacrebleu_chrf"],
            m["fp32_avg_latency_ms"],
            m["cmp_avg_latency_ms"],
            m["speedup_vs_fp32"],
        )

    logger.info("\n=== Category Breakdown ===")
    logger.info(f"| Category | Count | Token Match % | Text Match % | BLEU | ChrF | Latency (ms) (FP32 / {label}) | Speedup |")
    logger.info("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for cat, m in sorted(metrics_by_category.items()):
        logger.info(
            "| %s | %d | %.2f%% | %.2f%% | %.2f | %.2f | %.1f / %.1f | %.3fx |",
            cat,
            m["total_fixtures"],
            m["token_exact_rate"],
            m["text_exact_rate"],
            m["sacrebleu_bleu"],
            m["sacrebleu_chrf"],
            m["fp32_avg_latency_ms"],
            m["cmp_avg_latency_ms"],
            m["speedup_vs_fp32"],
        )
    logger.info("")

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

    return report, mismatches


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size for fixtures sharing the same language pair (default: 16)",
    )
    args = parser.parse_args()

    fixtures = load_fixtures(args.fixtures)
    logger.info("Loaded %d fixtures from %s", len(fixtures), args.fixtures)

    report, mismatches = benchmark(
        fp32_dir=args.fp32_dir,
        cmp_dir=args.cmp_dir,
        fixtures=fixtures,
        pytorch_model=args.pytorch_model,
        label=args.label,
        batch_size=args.batch_size,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Report written → %s", args.report)

    if mismatches:
        mismatches_path = args.report.with_name(args.report.stem + "-mismatches.json")
        mismatches_path.write_text(
            json.dumps(mismatches, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Full mismatches written → %s", mismatches_path)


if __name__ == "__main__":
    main()
