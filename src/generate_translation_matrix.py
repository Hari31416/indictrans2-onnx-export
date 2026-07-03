#!/usr/bin/env python3
"""Script to generate a translation matrix for a set of test sentences into multiple Indic languages."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Final, Any

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from IndicTransToolkit import IndicProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger: Final[logging.Logger] = logging.getLogger(__name__)


def _past_feed_from_outputs(past_outputs: list[np.ndarray], num_layers: int) -> dict[str, np.ndarray]:
    """Format past outputs for the decoder with past session inputs."""
    feed: dict[str, np.ndarray] = {}
    for i in range(num_layers):
        base = i * 4
        feed[f"past_key_values.{i}.decoder.key"] = past_outputs[base]
        feed[f"past_key_values.{i}.decoder.value"] = past_outputs[base + 1]
        feed[f"past_key_values.{i}.encoder.key"] = past_outputs[base + 2]
        feed[f"past_key_values.{i}.encoder.value"] = past_outputs[base + 3]
    return feed


class Translator:
    """Class to manage ONNX-based translation runtime sessions and tokenizer loading."""

    def __init__(self, onnx_dir: Path, base_model_id: str) -> None:
        self.onnx_dir = onnx_dir
        self.base_model_id = base_model_id

        # Load configurations and tokenizers
        self.meta = json.loads((onnx_dir / "tokenizer_meta.json").read_text(encoding="utf-8"))
        self.src_tok = Tokenizer.from_file(str(onnx_dir / "tokenizer_src.json"))
        self.slow_tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

        # Load ORT sessions
        self.enc = ort.InferenceSession(str(onnx_dir / "encoder_model.onnx"))
        self.dec = ort.InferenceSession(str(onnx_dir / "decoder_model.onnx"))
        self.dec_past = ort.InferenceSession(str(onnx_dir / "decoder_with_past_model.onnx"))

        self.num_layers = (len(self.dec.get_outputs()) - 1) // 4

        gen_cfg: dict[str, Any] = {}
        gen_config_path = onnx_dir / "generation_config.json"
        if gen_config_path.exists():
            gen_cfg = json.loads(gen_config_path.read_text(encoding="utf-8"))

        self.decoder_start_id = int(gen_cfg.get("decoder_start_token_id", 2))
        self.eos_id = int(gen_cfg.get("eos_token_id", 2))
        self.max_new_tokens = 128
        self.ip = IndicProcessor(inference=True)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate a single sentence using greedy decoding."""
        if hasattr(self.ip, "_placeholder_entity_maps"):
            self.ip._placeholder_entity_maps.queue.clear()

        preprocessed = self.ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        prefixed = preprocessed[0]

        # Encode inputs
        encoded = self.src_tok.encode(prefixed)
        input_ids_list = [
            i if i < self.meta["src_dict_size"] else self.meta["unk_id"] for i in encoded.ids
        ]
        input_ids = np.array([input_ids_list], dtype=np.int64)
        attn_mask = np.array([encoded.attention_mask], dtype=np.int64)

        # Run Encoder
        enc_out = self.enc.run(["last_hidden_state"], {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        })[0]

        decoder_input_ids = np.array([[self.decoder_start_id]], dtype=np.int64)
        output_ids = [self.decoder_start_id]
        past_outputs: list[np.ndarray] | None = None

        # Autoregressive generation
        for step in range(self.max_new_tokens):
            if step == 0:
                dec_out = self.dec.run(None, {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": enc_out,
                    "encoder_attention_mask": attn_mask,
                })
            else:
                if past_outputs is None:
                    raise RuntimeError("past_outputs must not be None at step > 0")
                dec_out = self.dec_past.run(None, {
                    "input_ids": decoder_input_ids,
                    "encoder_attention_mask": attn_mask,
                    **_past_feed_from_outputs(past_outputs, self.num_layers),
                })

            logits = dec_out[0]
            past_outputs = [np.array(x) for x in dec_out[1:]]
            next_id = int(np.argmax(logits[0, -1, :]))
            output_ids.append(next_id)

            if next_id == self.eos_id:
                break
            decoder_input_ids = np.array([[next_id]], dtype=np.int64)

        # Decode output
        safe_ids = [i if i < self.meta["tgt_dict_size"] else self.meta["unk_id"] for i in output_ids]
        with self.slow_tok.as_target_tokenizer():
            decoded = self.slow_tok.batch_decode(
                [safe_ids],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        postprocessed = self.ip.postprocess_batch(decoded, lang=tgt_lang)[0]
        return postprocessed


def main() -> None:
    """Parse args and generate the translation matrix."""
    parser = argparse.ArgumentParser(description="Generate a translation matrix of test sentences from HF model")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="hari31416/indictrans2-en-indic-dist-200M-ONNX",
        help="Hugging Face ONNX model repo ID or local directory path",
    )
    parser.add_argument(
        "--src-lang",
        type=str,
        default="",
        help="Source language code (e.g. eng_Latn, hin_Deva). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--input-fixtures",
        type=Path,
        default=None,
        help="Path to input test sentences JSON file. Defaults dynamically to fixtures/smoke-test/.",
    )
    parser.add_argument(
        "--output-matrix",
        type=Path,
        default=None,
        help="Path to write the output translation matrix JSON. Defaults to fixtures/smoke-test/.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="",
        help="Comma-separated list of target language codes. Auto-detected based on direction if not specified.",
    )
    args = parser.parse_args()

    # Resolve base model ID, source language, and default targets based on the repo ID or path
    repo_lower = args.repo_id.lower()
    if "en-indic" in repo_lower:
        direction = "en-indic"
        base_model_id = "ai4bharat/indictrans2-en-indic-dist-200M"
        default_src_lang = "eng_Latn"
        default_languages = "hin_Deva,tam_Taml,ben_Beng,tel_Telu,mar_Deva,guj_Gujr,kan_Knda,mal_Mlym,pan_Guru,urd_Arab"
    elif "indic-en" in repo_lower:
        direction = "indic-en"
        base_model_id = "ai4bharat/indictrans2-indic-en-dist-200M"
        default_src_lang = "hin_Deva"
        default_languages = "eng_Latn"
    else:  # indic-indic
        direction = "indic-indic"
        base_model_id = "ai4bharat/indictrans2-indic-indic-dist-320M"
        default_src_lang = "hin_Deva"
        default_languages = "tam_Taml,ben_Beng,tel_Telu,mar_Deva,guj_Gujr,kan_Knda,mal_Mlym,pan_Guru,urd_Arab"

    src_lang = args.src_lang if args.src_lang else default_src_lang

    # Determine input fixture path if not specified
    if args.input_fixtures:
        input_path = args.input_fixtures
    else:
        if src_lang == "eng_Latn":
            input_path = Path("fixtures/smoke-test/test_sentences_en.json")
        else:
            input_path = Path("fixtures/smoke-test/test_sentences_hi.json")

    # Determine output matrix path if not specified
    if args.output_matrix:
        output_path = args.output_matrix
    else:
        output_path = Path(f"fixtures/smoke-test/translation_matrix_{direction}.json")

    # Load input sentences
    if not input_path.exists():
        raise FileNotFoundError(f"Input fixtures not found: {input_path}")

    sentences: list[str] = json.loads(input_path.read_text(encoding="utf-8"))

    # Determine target languages
    langs_str = args.languages if args.languages else default_languages
    languages: list[str] = [lang.strip() for lang in langs_str.split(",") if lang.strip()]

    local_path = Path(args.repo_id)
    if local_path.is_dir():
        onnx_dir = local_path.resolve()
        logger.info("Using local ONNX directory: %s", onnx_dir)
    else:
        logger.info("Downloading ONNX model snapshot for %s from HF Hub...", args.repo_id)
        try:
            model_dir_str = snapshot_download(repo_id=args.repo_id)
            onnx_dir = Path(model_dir_str)
            logger.info("Downloaded successfully to: %s", onnx_dir)
        except Exception as e:
            logger.error("Failed to download model snapshot: %s", e)
            raise SystemExit(1) from e

    # Initialize translator
    translator = Translator(onnx_dir, base_model_id)

    # Translate and construct matrix
    logger.info("Translating %d sentences (%s) into %d languages...", len(sentences), src_lang, len(languages))
    results: list[dict[str, Any]] = []

    for i, sent in enumerate(sentences):
        logger.info("Processing sentence %d/%d: '%s'", i + 1, len(sentences), sent)
        translations: dict[str, str] = {}
        for lang in languages:
            try:
                trans = translator.translate(sent, src_lang=src_lang, tgt_lang=lang)
                translations[lang] = trans
            except Exception as e:
                logger.error("Failed to translate to %s: %s", lang, e)
                translations[lang] = f"ERROR: {e}"

        results.append({
            "source": sent,
            "translations": translations
        })

    # Save translation matrix
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Translation matrix saved successfully to %s", output_path)

    # Print markdown table
    md_header = "| Source Sentence | " + " | ".join(languages) + " |"
    md_separator = "|---|" + "|---|".join("" for _ in languages) + "|"
    print("\n" + md_header)
    print(md_separator)
    for res in results:
        cols = [res["source"]] + [res["translations"].get(lang, "") for lang in languages]
        print("| " + " | ".join(cols) + " |")
    print()


if __name__ == "__main__":
    main()
