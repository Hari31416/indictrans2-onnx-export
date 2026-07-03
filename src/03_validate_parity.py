#!/usr/bin/env python3
"""Validate ONNX bundle parity against PyTorch baseline.

Compares greedy-decoded token IDs and post-processed text against the
original PyTorch model. Pass criteria: >= 99% token-exact match (fp32).
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
        decoded_results = pytorch_greedy_decode(en_indic_model_id, temp_fixtures, device=device)

        logger.info("Writing fixtures to %s...", output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            idx = 0
            for lang in official_indic_langs:
                for i in range(min(sentences_per_lang, len(seed_sentences))):
                    res = decoded_results[idx]
                    idx += 1
                    if "indic-en" in repo_lower:
                        row = {
                            "text": res["text"],
                            "src_lang": lang,
                            "tgt_lang": "eng_Latn",
                            "category": ["generic", "politics", "numerals", "lexicon"][i % 4],
                        }
                    else:  # indic-indic
                        l_idx = official_indic_langs.index(lang)
                        next_lang = official_indic_langs[(l_idx + 1) % len(official_indic_langs)]
                        row = {
                            "text": res["text"],
                            "src_lang": lang,
                            "tgt_lang": next_lang,
                            "category": ["generic", "politics", "numerals", "lexicon"][i % 4],
                        }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %d fixture rows to %s", len(official_indic_langs) * min(sentences_per_lang, len(seed_sentences)), output_path)



def pytorch_greedy_decode(
    model_id: str,
    fixtures: list[Fixture],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    import torch
    from IndicTransToolkit import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    ip = IndicProcessor(inference=True)

    gen_cfg = model.generation_config
    decoder_start_id = int(gen_cfg.decoder_start_token_id)
    eos_id = int(gen_cfg.eos_token_id)

    results = []
    for fx in fixtures:
        if hasattr(ip, "_placeholder_entity_maps"):
            ip._placeholder_entity_maps.queue.clear()
        batch = ip.preprocess_batch([fx.text], src_lang=fx.src_lang, tgt_lang=fx.tgt_lang)
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        with torch.inference_mode():
            enc_out = model.model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).last_hidden_state

            decoder_input_ids = torch.tensor([[decoder_start_id]], device=device)
            output_ids = [decoder_start_id]
            past_key_values = None

            for _ in range(128):
                if past_key_values is None:
                    dec_out = model.model.decoder(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=enc_out,
                        encoder_attention_mask=inputs["attention_mask"],
                        use_cache=True,
                    )
                else:
                    batch_size = decoder_input_ids.shape[0]
                    encoder_seq_len = inputs["attention_mask"].shape[1]
                    embed_dim = getattr(model.config, "decoder_embed_dim", 512)
                    dummy_encoder_hidden_states = torch.zeros(
                        batch_size,
                        encoder_seq_len,
                        embed_dim,
                        dtype=torch.float32,
                        device=device
                    )
                    dec_out = model.model.decoder(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=dummy_encoder_hidden_states,
                        encoder_attention_mask=inputs["attention_mask"],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                logits = model.lm_head(dec_out.last_hidden_state)
                next_id = int(logits[0, -1, :].argmax().item())
                output_ids.append(next_id)

                if next_id == eos_id:
                    break

                decoder_input_ids = torch.tensor([[next_id]], device=device)
                past_key_values = dec_out.past_key_values

        token_ids = output_ids
        with tokenizer.as_target_tokenizer():
            decoded = tokenizer.batch_decode(
                [token_ids],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        text = ip.postprocess_batch(decoded, lang=fx.tgt_lang)[0]
        results.append({"token_ids": token_ids, "text": text})

    return results


def _past_feed_from_outputs(past_outputs: list, num_layers: int) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for i in range(num_layers):
        base = i * 4
        feed[f"past_key_values.{i}.decoder.key"] = past_outputs[base]
        feed[f"past_key_values.{i}.decoder.value"] = past_outputs[base + 1]
        feed[f"past_key_values.{i}.encoder.key"] = past_outputs[base + 2]
        feed[f"past_key_values.{i}.encoder.value"] = past_outputs[base + 3]
    return feed


def onnx_greedy_decode(
    onnx_dir: Path,
    fixtures: list[Fixture],
    *,
    pytorch_model: str,
) -> list[dict[str, Any]]:
    """Greedy decode via three ONNX sessions (naklitechie I/O layout)."""
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    from IndicTransToolkit import IndicProcessor as PyIndicProcessor

    meta = json.loads((onnx_dir / "tokenizer_meta.json").read_text(encoding="utf-8"))
    src_tok = Tokenizer.from_file(str(onnx_dir / "tokenizer_src.json"))
    slow_tok = AutoTokenizer.from_pretrained(pytorch_model, trust_remote_code=True)

    enc = ort.InferenceSession(str(onnx_dir / "encoder_model.onnx"))
    dec = ort.InferenceSession(str(onnx_dir / "decoder_model.onnx"))
    dec_past = ort.InferenceSession(str(onnx_dir / "decoder_with_past_model.onnx"))

    num_layers = (len(dec.get_outputs()) - 1) // 4

    gen_cfg = {}
    gen_config_path = onnx_dir / "generation_config.json"
    if gen_config_path.exists():
        gen_cfg = json.loads(gen_config_path.read_text(encoding="utf-8"))

    decoder_start_id = int(gen_cfg.get("decoder_start_token_id", 2))
    eos_id = int(gen_cfg.get("eos_token_id", 2))
    max_new_tokens = 128

    ip = PyIndicProcessor(inference=True)
    results = []

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

        enc_out = enc.run(["last_hidden_state"], {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        })[0]

        decoder_input_ids = np.array([[decoder_start_id]], dtype=np.int64)
        output_ids = [decoder_start_id]
        past_outputs: list | None = None

        for step in range(max_new_tokens):
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
                    **_past_feed_from_outputs(past_outputs, num_layers),
                })

            logits = dec_out[0]
            past_outputs = list(dec_out[1:])
            next_id = int(np.argmax(logits[0, -1, :]))
            output_ids.append(next_id)
            if next_id == eos_id:
                break
            decoder_input_ids = np.array([[next_id]], dtype=np.int64)

        safe_ids = [i if i < meta["tgt_dict_size"] else meta["unk_id"] for i in output_ids]
        with slow_tok.as_target_tokenizer():
            decoded = slow_tok.batch_decode(
                [safe_ids],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        text = ip.postprocess_batch(decoded, lang=fx.tgt_lang)[0]
        results.append({"token_ids": output_ids, "text": text})

    return results


def validate_parity(
    onnx_dir: Path,
    pytorch_model: str,
    fixtures_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    fixtures = load_fixtures(fixtures_path)
    logger.info("Loaded %d fixtures from %s", len(fixtures), fixtures_path)

    t0 = time.time()
    pt_results = pytorch_greedy_decode(pytorch_model, fixtures)
    onnx_results = onnx_greedy_decode(onnx_dir, fixtures, pytorch_model=pytorch_model)
    elapsed = time.time() - t0

    token_pass = 0
    text_pass = 0
    mismatches: list[dict[str, Any]] = []

    for i, (pt, ox) in enumerate(zip(pt_results, onnx_results)):
        tokens_match = pt["token_ids"] == ox["token_ids"]
        text_match = pt["text"] == ox["text"]
        if tokens_match:
            token_pass += 1
        if text_match:
            text_pass += 1
        if not tokens_match or not text_match:
            mismatches.append({
                "index": i,
                "fixture": fixtures[i].__dict__,
                "pytorch_tokens": pt["token_ids"][:20],
                "onnx_tokens": ox["token_ids"][:20],
                "pytorch_text": pt["text"],
                "onnx_text": ox["text"],
                "tokens_match": tokens_match,
                "text_match": text_match,
            })

    total = len(fixtures)
    report = {
        "total": total,
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
    parser = argparse.ArgumentParser(description="Validate ONNX vs PyTorch parity")
    parser.add_argument("--onnx-dir", type=Path, help="ONNX bundle directory")
    parser.add_argument("--pytorch-model", required=True, help="PyTorch model id")
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/en-indic-golden.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("fixtures/parity-report.json"))
    parser.add_argument("--capture-fixtures", type=Path, help="Capture fixtures to this path and exit")
    args = parser.parse_args()

    if args.capture_fixtures:
        capture_fixtures(
            args.pytorch_model,
            args.capture_fixtures,
        )
        return

    if not args.onnx_dir:
        parser.error("--onnx-dir is required unless --capture-fixtures is set")

    validate_parity(args.onnx_dir, args.pytorch_model, args.fixtures, args.report)


if __name__ == "__main__":
    main()
