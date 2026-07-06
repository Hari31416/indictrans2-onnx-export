#!/usr/bin/env python3
"""Validate ONNX bundle parity against PyTorch baseline - V2.

Compares greedy-decoded token IDs and post-processed text against the
original PyTorch model. Pass criteria: >= 99% token-exact match (fp32).

Supports `--smoke` test mode to run a fast validation on a small subset of the fixtures.
"""

from __future__ import annotations

import _src_path  # noqa: F401

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from it2_inference import greedy_decode_onnx, greedy_decode_pytorch

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
            fixtures.append(Fixture(
                text=row["text"],
                src_lang=row["src_lang"],
                tgt_lang=row["tgt_lang"],
                category=row.get("category", "generic"),
            ))
    return fixtures


def capture_fixtures(
    model_id: str,
    output_path: Path,
    languages: list[tuple[str, str]] | None = None,
    sentences_per_lang: int = 12,
) -> None:
    """Generate golden fixtures from PyTorch model (one-time capture)."""
    import torch
    from IndicTransToolkit import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    seed_path = Path("fixtures/seed_sentences.txt")
    if seed_path.exists():
        logger.info("Loading seed sentences from %s", seed_path)
        seed_sentences = [
            line.strip()
            for line in seed_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sentences_per_lang = len(seed_sentences)
    else:
        seed_sentences = [
            "This is a test sentence.",
            "Who will win the election?",
            "The weather is nice today.",
            "Please send an SMS to 9876543210.",
            "Contact us at newemail123@xyz.com by 15th October, 2023.",
            "The GDP grew by 7.5% last quarter.",
            "Education is the key to progress.",
            "The parliament passed a new bill.",
            "Water is essential for life.",
            "Technology is changing the world.",
            "Health care must be accessible to all.",
            "The festival was celebrated with joy.",
        ]

    official_indic_langs = [
        "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "guj_Gujr", "hin_Deva",
        "kan_Knda", "kas_Arab", "gom_Deva", "mai_Deva", "mal_Mlym", "mar_Deva",
        "mni_Beng", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", "sat_Olck",
        "snd_Arab", "tam_Taml", "tel_Telu", "urd_Arab"
    ]

    repo_lower = model_id.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "en-indic" in repo_lower:
        logger.info("Generating en-indic fixtures...")
        with open(output_path, "w", encoding="utf-8") as f:
            for lang in official_indic_langs:
                for i, sent in enumerate(seed_sentences[:sentences_per_lang]):
                    row = {
                        "text": sent,
                        "src_lang": "eng_Latn",
                        "tgt_lang": lang,
                        "category": ["generic", "politics", "numerals", "lexicon"][i % 4],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %d fixture rows to %s", len(official_indic_langs) * min(sentences_per_lang, len(seed_sentences)), output_path)

    elif "indic-en" in repo_lower or "indic-indic" in repo_lower:
        en_indic_model_id = "ai4bharat/indictrans2-en-indic-dist-200M"
        logger.info("Loading translation model %s to generate Indic source sentences...", en_indic_model_id)

        # Generate a list of temporary Fixture objects to decode
        temp_fixtures = []
        for lang in official_indic_langs:
            for i, sent in enumerate(seed_sentences[:sentences_per_lang]):
                temp_fixtures.append(Fixture(
                    text=sent,
                    src_lang="eng_Latn",
                    tgt_lang=lang,
                    category=["generic", "politics", "numerals", "lexicon"][i % 4]
                ))

        logger.info("Translating seed sentences to Indic languages...")
        decoded_results = greedy_decode_pytorch(
            en_indic_model_id,
            temp_fixtures,
            device=device,
        )

        logger.info("Writing fixtures to %s...", output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            idx = 0
            for lang in official_indic_langs:
                for i in range(min(sentences_per_lang, len(seed_sentences))):
                    res = decoded_results[idx]
                    idx += 1
                    if "indic-en" in repo_lower:
                        row = {
                            "text": res.text,
                            "src_lang": lang,
                            "tgt_lang": "eng_Latn",
                            "category": ["generic", "politics", "numerals", "lexicon"][i % 4],
                        }
                    else:  # indic-indic
                        l_idx = official_indic_langs.index(lang)
                        next_lang = official_indic_langs[(l_idx + 1) % len(official_indic_langs)]
                        row = {
                            "text": res.text,
                            "src_lang": lang,
                            "tgt_lang": next_lang,
                            "category": ["generic", "politics", "numerals", "lexicon"][i % 4],
                        }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %d fixture rows to %s", len(official_indic_langs) * min(sentences_per_lang, len(seed_sentences)), output_path)



def validate_parity(
    onnx_dir: Path,
    pytorch_model: str,
    fixtures_path: Path,
    report_path: Path,
    smoke: bool = False,
    batch_size: int = 16,
) -> dict[str, Any]:
    fixtures = load_fixtures(fixtures_path)
    
    if smoke:
        # Limit to 10 queries per unique (src_lang, tgt_lang) pair
        counts = {}
        smoke_fixtures = []
        for fx in fixtures:
            key = (fx.src_lang, fx.tgt_lang)
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= 10:
                smoke_fixtures.append(fx)
        fixtures = smoke_fixtures
        logger.info("Smoke test enabled: limited to %d fixtures (max 10 per language pair)", len(fixtures))

    logger.info("Loaded %d fixtures from %s", len(fixtures), fixtures_path)

    t0 = time.time()
    pt_results = greedy_decode_pytorch(
        pytorch_model,
        fixtures,
        batch_size=batch_size,
    )
    onnx_results = greedy_decode_onnx(
        onnx_dir,
        fixtures,
        pytorch_model=pytorch_model,
        batch_size=batch_size,
    )
    elapsed = time.time() - t0

    token_pass = 0
    text_pass = 0
    mismatches: list[dict[str, Any]] = []

    for i, (pt, ox) in enumerate(zip(pt_results, onnx_results)):
        tokens_match = pt.token_ids == ox.token_ids
        text_match = pt.text == ox.text
        if tokens_match:
            token_pass += 1
        if text_match:
            text_pass += 1
        if not tokens_match or not text_match:
            mismatches.append({
                "index": i,
                "fixture": fixtures[i].__dict__,
                "pytorch_tokens": pt.token_ids[:20],
                "onnx_tokens": ox.token_ids[:20],
                "pytorch_text": pt.text,
                "onnx_text": ox.text,
                "tokens_match": tokens_match,
                "text_match": text_match,
            })

    total = len(fixtures)
    report = {
        "total": total,
        "batch_size": batch_size,
        "pass_tokens": token_pass,
        "pass_text": text_pass,
        "token_pass_rate": round(token_pass / total * 100, 2) if total else 0,
        "text_pass_rate": round(text_pass / total * 100, 2) if total else 0,
        "mismatches_sample": mismatches[:10],
        "elapsed_seconds": round(elapsed, 2),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Parity: %d/%d tokens (%.1f%%), %d/%d text (%.1f%%)",
        token_pass, total, report["token_pass_rate"],
        text_pass, total, report["text_pass_rate"],
    )

    if report["token_pass_rate"] < 99.0:
        logger.warning("FAIL: token pass rate below 99%% threshold")
    else:
        logger.info("PASS: token pass rate meets >= 99%% threshold")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ONNX vs PyTorch parity - V2")
    parser.add_argument("--onnx-dir", type=Path, help="ONNX bundle directory")
    parser.add_argument("--pytorch-model", required=True, help="PyTorch model id")
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/en-indic-golden.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("fixtures/parity-report.json"))
    parser.add_argument("--capture-fixtures", type=Path, help="Capture fixtures to this path and exit")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on a small subset of fixtures")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size for fixtures sharing the same language pair (default: 16)",
    )
    args = parser.parse_args()

    if args.capture_fixtures:
        capture_fixtures(
            args.pytorch_model,
            args.capture_fixtures,
        )
        return

    if not args.onnx_dir:
        parser.error("--onnx-dir is required unless --capture-fixtures is set")

    validate_parity(
        onnx_dir=args.onnx_dir,
        pytorch_model=args.pytorch_model,
        fixtures_path=args.fixtures,
        report_path=args.report,
        smoke=args.smoke,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
