#!/usr/bin/env python3
"""Build fast tokenizer.json files for browser IndicTrans2 inference.

Converts SentencePiece models (model.SRC / model.TGT) via Hugging Face
SpmConverter, then remaps token IDs to match dict.SRC.json / dict.TGT.json
exactly (the SPM-native indices differ from the Fairseq dictionary IDs).

Ported from phase2/gemini with naklitechie post-processing alignment.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LANGUAGE_TAGS = frozenset({
    "asm_Beng", "awa_Deva", "ben_Beng", "bho_Deva", "brx_Deva", "doi_Deva",
    "eng_Latn", "gom_Deva", "gon_Deva", "guj_Gujr", "hin_Deva", "hne_Deva",
    "kan_Knda", "kas_Arab", "kas_Deva", "kha_Latn", "lus_Latn", "mag_Deva",
    "mai_Deva", "mal_Mlym", "mar_Deva", "mni_Beng", "mni_Mtei", "npi_Deva",
    "ory_Orya", "pan_Guru", "san_Deva", "sat_Olck", "snd_Arab", "snd_Deva",
    "tam_Taml", "tel_Telu", "urd_Arab", "unr_Deva",
})

EOS_POST_PROCESSOR: dict[str, Any] = {
    "type": "TemplateProcessing",
    "single": [
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "</s>", "type_id": 0}},
    ],
    "pair": [
        {"Sequence": {"id": "A", "type_id": 0}},
        {"Sequence": {"id": "B", "type_id": 1}},
    ],
    "special_tokens": {
        "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]},
    },
}

SPECIAL_TOKENS = ("<s>", "<pad>", "</s>", "<unk>")
UNK_ID = 3


def _copy_hf_tokenizer_assets(model_id: str, output_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    cache = Path(snapshot_download(
        repo_id=model_id,
        allow_patterns=[
            "dict.SRC.json", "dict.TGT.json",
            "model.SRC", "model.TGT",
            "config.json", "generation_config.json",
            "tokenizer_config.json", "tokenization_indictrans.py",
            "special_tokens_map.json",
        ],
    ))

    for name in (
        "dict.SRC.json", "dict.TGT.json", "model.SRC", "model.TGT",
        "config.json", "generation_config.json",
        "tokenizer_config.json", "tokenization_indictrans.py", "special_tokens_map.json",
    ):
        src = cache / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def _spm_to_fast_json(spm_path: str | Path, out_path: Path) -> None:
    from transformers.convert_slow_tokenizer import SpmConverter

    class _SpmHolder:
        def __init__(self, vocab_file: str) -> None:
            self.vocab_file = vocab_file

    converter = SpmConverter(_SpmHolder(str(spm_path)))
    converter.converted().save(str(out_path))


def _align_vocab_with_dict(fast_json_path: Path, dict_json_path: Path) -> None:
    """Remap BPE vocab IDs and register language tags as added tokens."""
    fast = json.loads(fast_json_path.read_text(encoding="utf-8"))
    vocab_dict = json.loads(dict_json_path.read_text(encoding="utf-8"))

    # We map tokens in vocab_dict to their correct Fairseq IDs.
    # To keep the vocabulary vector in HuggingFace valid and avoid collisions
    # (which break BPE merges), we map all other tokens to unique IDs starting from len(vocab_dict).
    next_id = len(vocab_dict)
    remapped: dict[str, int] = {}
    for token in fast["model"]["vocab"]:
        if token in vocab_dict:
            remapped[token] = vocab_dict[token]
        else:
            remapped[token] = next_id
            next_id += 1

    for token, token_id in vocab_dict.items():
        if token not in remapped:
            remapped[token] = token_id
    fast["model"]["vocab"] = remapped

    added_tokens: list[dict[str, Any]] = [
        {
            "id": vocab_dict[tok],
            "content": tok,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for tok in SPECIAL_TOKENS
        if tok in vocab_dict
    ]

    for tag in sorted(LANGUAGE_TAGS):
        if tag in vocab_dict:
            added_tokens.append({
                "id": vocab_dict[tag],
                "content": tag,
                "single_word": True,
                "lstrip": False,
                "rstrip": True,
                "normalized": False,
                "special": False,
            })

    fast["added_tokens"] = added_tokens
    fast["post_processor"] = EOS_POST_PROCESSOR
    fast_json_path.write_text(json.dumps(fast, indent=2), encoding="utf-8")


def _load_slow_tokenizer(bundle_dir: Path):
    import sys

    sys.path.insert(0, str(bundle_dir.resolve()))
    from tokenization_indictrans import IndicTransTokenizer

    return IndicTransTokenizer(
        src_vocab_fp=str(bundle_dir / "dict.SRC.json"),
        tgt_vocab_fp=str(bundle_dir / "dict.TGT.json"),
        src_spm_fp=str(bundle_dir / "model.SRC"),
        tgt_spm_fp=str(bundle_dir / "model.TGT"),
    )


def _encode_src_slow(slow_tok, text: str) -> list[int]:
    return slow_tok(text, return_tensors=None)["input_ids"]


def _encode_tgt_slow(slow_tok, text: str) -> list[int]:
    with slow_tok.as_target_tokenizer():
        return slow_tok(text, return_tensors=None)["input_ids"]


def _encode_fast(
    fast_tok,
    text: str,
    dict_size: int,
    unk_id: int = UNK_ID,
) -> list[int]:
    encoded = fast_tok.encode(text)
    return [i if i < dict_size else unk_id for i in encoded.ids]


def validate_fast_tokenizers(
    bundle_dir: Path,
    src_samples: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Compare fast tokenizer encode output against the slow HF tokenizer."""
    from tokenizers import Tokenizer

    meta = json.loads((bundle_dir / "tokenizer_meta.json").read_text(encoding="utf-8"))
    fast_src = Tokenizer.from_file(str(bundle_dir / "tokenizer_src.json"))
    fast_tgt = Tokenizer.from_file(str(bundle_dir / "tokenizer_tgt.json"))
    slow = _load_slow_tokenizer(bundle_dir)

    if src_samples is None:
        src_samples = [
            ("hin_Deva", "eng_Latn", "यह एक परीक्षण वाक्य है।"),
            ("tam_Taml", "eng_Latn", "இது ஒரு சோதனை வாக்கியம்."),
            ("ben_Beng", "eng_Latn", "এটি একটি পরীক্ষার বাক্য।"),
            ("tel_Telu", "eng_Latn", "ఇది ఒక పరీక్ష వాక్యం."),
            ("mar_Deva", "eng_Latn", "हा एक चाचणी वाक्य आहे."),
        ]

    tgt_samples = [
        "This is a test.",
        "Who will win the election?",
        "The weather is nice today.",
    ]

    src_pass = 0
    tgt_pass = 0
    failures: list[dict[str, Any]] = []

    for src_lang, tgt_lang, text in src_samples:
        prefixed = f"{src_lang} {tgt_lang} {text}"
        slow_ids = _encode_src_slow(slow, prefixed)
        fast_ids = _encode_fast(fast_src, prefixed, meta["src_dict_size"], meta["unk_id"])
        if slow_ids == fast_ids:
            src_pass += 1
        else:
            failures.append({
                "side": "src",
                "text": prefixed[:80],
                "slow": slow_ids[:20],
                "fast": fast_ids[:20],
            })

    for text in tgt_samples:
        slow_ids = _encode_tgt_slow(slow, text)
        fast_ids = _encode_fast(fast_tgt, text, meta["tgt_dict_size"], meta["unk_id"])
        if slow_ids == fast_ids:
            tgt_pass += 1
        else:
            failures.append({
                "side": "tgt",
                "text": text,
                "slow": slow_ids[:20],
                "fast": fast_ids[:20],
            })

    total = len(src_samples) + len(tgt_samples)
    passed = src_pass + tgt_pass
    report = {
        "src_pass": src_pass,
        "src_total": len(src_samples),
        "tgt_pass": tgt_pass,
        "tgt_total": len(tgt_samples),
        "pass_rate": round(passed / total * 100, 2) if total else 0.0,
        "failures": failures,
    }
    logger.info(
        "Fast tokenizer encode parity: src %d/%d, tgt %d/%d (%.1f%%)",
        src_pass, len(src_samples),
        tgt_pass, len(tgt_samples),
        report["pass_rate"],
    )
    if failures:
        for fail in failures[:3]:
            logger.warning("Mismatch (%s): %s", fail["side"], fail["text"])
    return report


def build_fast_tokenizers(
    model_id: str,
    output_dir: Path,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_hf_tokenizer_assets(model_id, output_dir)

    logger.info("Loading slow tokenizer from %s", model_id)
    slow_tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    for spm_attr, dict_name, out_name in (
        ("src_spm_fp", "dict.SRC.json", "tokenizer_src.json"),
        ("tgt_spm_fp", "dict.TGT.json", "tokenizer_tgt.json"),
    ):
        spm_path = getattr(slow_tok, spm_attr, None)
        if not spm_path:
            raise RuntimeError(f"Tokenizer missing {spm_attr} for {model_id}")

        out_path = output_dir / out_name
        dict_path = output_dir / dict_name
        logger.info("Converting %s -> %s", Path(spm_path).name, out_name)
        _spm_to_fast_json(spm_path, out_path)
        _align_vocab_with_dict(out_path, dict_path)

    src_dict_size = len(json.loads((output_dir / "dict.SRC.json").read_text(encoding="utf-8")))
    tgt_dict_size = len(json.loads((output_dir / "dict.TGT.json").read_text(encoding="utf-8")))

    meta = {
        "src_dict_size": src_dict_size,
        "tgt_dict_size": tgt_dict_size,
        "unk_id": UNK_ID,
    }
    (output_dir / "tokenizer_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    logger.info(
        "Wrote tokenizer_meta.json (src_dict_size=%d, tgt_dict_size=%d)",
        src_dict_size, tgt_dict_size,
    )

    validation_report: dict[str, Any] | None = None
    if validate:
        validation_report = validate_fast_tokenizers(output_dir)
        if validation_report["pass_rate"] < 100.0:
            raise RuntimeError(
                f"Fast tokenizer encode parity {validation_report['pass_rate']}% < 100%"
            )

    return {
        "meta": meta,
        "validation": validation_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fast IT2 tokenizers")
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip encode parity check against slow tokenizer",
    )
    args = parser.parse_args()
    build_fast_tokenizers(args.model, args.output, validate=not args.no_validate)


if __name__ == "__main__":
    main()
