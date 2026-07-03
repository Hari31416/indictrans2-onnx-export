#!/usr/bin/env python3
"""Self-contained IndicTrans2 ONNX inference helper.

This file is the single entry-point for running IndicTrans2 ONNX models from
Python. It is intentionally kept self-contained so it can be:

  - Copied from the **Hugging Face repo** (where it is uploaded alongside the
    ONNX bundle), or
  - Found at the **GitHub source repo**:
    https://github.com/Hari31416/indictrans2-onnx-export/blob/main/src/translate.py

Usage
-----

    from translate import IndicTransONNX

    # Pass a HF repo ID or a local directory path
    model = IndicTransONNX("hari31416/indictrans2-en-indic-dist-200M-ONNX")
    print(model.translate("Who will win the election?", src_lang="eng_Latn", tgt_lang="hin_Deva"))

Dependencies
------------
    pip install onnxruntime tokenizers huggingface-hub

CLI
---
    python translate.py hari31416/indictrans2-en-indic-dist-200M-ONNX \\
        "Who will win the election?" --src-lang eng_Latn --tgt-lang hin_Deva
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _past_feed(past_outputs: list[np.ndarray], num_layers: int) -> dict[str, np.ndarray]:
    """Build the ``past_key_values.*`` input dict for decoder_with_past."""
    feed: dict[str, np.ndarray] = {}
    for i in range(num_layers):
        base = i * 4
        feed[f"past_key_values.{i}.decoder.key"] = past_outputs[base]
        feed[f"past_key_values.{i}.decoder.value"] = past_outputs[base + 1]
        feed[f"past_key_values.{i}.encoder.key"] = past_outputs[base + 2]
        feed[f"past_key_values.{i}.encoder.value"] = past_outputs[base + 3]
    return feed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class IndicTransONNX:
    """Load an IndicTrans2 ONNX bundle and run greedy translation.

    Args:
        model_path: HF repo ID (e.g. ``"hari31416/indictrans2-en-indic-dist-200M-ONNX"``)
                    or a local directory path that contains the ONNX bundle.
        providers:  ONNX Runtime execution providers.
                    Defaults to ``["CPUExecutionProvider"]``.
                    Pass ``["CUDAExecutionProvider", "CPUExecutionProvider"]``
                    for GPU, or ``["CoreMLExecutionProvider", "CPUExecutionProvider"]``
                    on Apple Silicon.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = str(model_path)

        # If the path looks like a HF repo ID (contains '/' but isn't a real
        # local path), download via huggingface_hub.
        if "/" in model_path and not Path(model_path).exists():
            from huggingface_hub import snapshot_download

            logger.info("Downloading snapshot for %s ...", model_path)
            model_path = snapshot_download(repo_id=model_path)

        snap = Path(model_path)
        self._providers = providers or ["CPUExecutionProvider"]

        # Tokenizers
        self._src_tok = Tokenizer.from_file(str(snap / "tokenizer_src.json"))
        self._tgt_tok = Tokenizer.from_file(str(snap / "tokenizer_tgt.json"))
        self._meta: dict = json.loads((snap / "tokenizer_meta.json").read_text(encoding="utf-8"))

        # Generation config (decoder start / eos IDs)
        gen_cfg: dict = {}
        gen_config_path = snap / "generation_config.json"
        if gen_config_path.exists():
            gen_cfg = json.loads(gen_config_path.read_text(encoding="utf-8"))
        self._decoder_start_id: int = int(gen_cfg.get("decoder_start_token_id", 2))
        self._eos_id: int = int(gen_cfg.get("eos_token_id", 2))

        # ONNX sessions
        logger.info("Loading ONNX sessions from %s ...", snap)
        self._enc = ort.InferenceSession(
            str(snap / "encoder_model.onnx"), providers=self._providers
        )
        self._dec = ort.InferenceSession(
            str(snap / "decoder_model.onnx"), providers=self._providers
        )
        self._dec_past = ort.InferenceSession(
            str(snap / "decoder_with_past_model.onnx"), providers=self._providers
        )
        # Number of transformer layers inferred from decoder outputs
        # (1 logits tensor + 4 KV tensors per layer)
        self._num_layers: int = (len(self._dec.get_outputs()) - 1) // 4

    # ------------------------------------------------------------------

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        max_new_tokens: int = 128,
    ) -> str:
        """Translate *text* from *src_lang* to *tgt_lang*.

        Args:
            text:           Input sentence (plain text, no language tags needed).
            src_lang:       BCP-47 style lang code, e.g. ``"eng_Latn"``.
            tgt_lang:       BCP-47 style lang code, e.g. ``"hin_Deva"``.
            max_new_tokens: Maximum tokens to generate (default: 128).

        Returns:
            Translated string.
        """
        # Prepend language tags — the format the model was trained with
        prefixed = f"{src_lang} {tgt_lang} {text}"

        # Tokenize source
        encoded = self._src_tok.encode(prefixed)
        # Clamp out-of-vocab IDs to <unk>
        input_ids = np.array(
            [
                [
                    i if i < self._meta["src_dict_size"] else self._meta["unk_id"]
                    for i in encoded.ids
                ]
            ],
            dtype=np.int64,
        )
        attn_mask = np.array([encoded.attention_mask], dtype=np.int64)

        # Encoder
        enc_out: np.ndarray = self._enc.run(
            ["last_hidden_state"],
            {"input_ids": input_ids, "attention_mask": attn_mask},
        )[0]

        # Greedy decoder loop
        decoder_input_ids = np.array([[self._decoder_start_id]], dtype=np.int64)
        output_ids: list[int] = [self._decoder_start_id]
        past_outputs: list[np.ndarray] | None = None

        for step in range(max_new_tokens):
            if step == 0:
                # First step: full encoder hidden states, no cached KV
                dec_out = self._dec.run(
                    None,
                    {
                        "input_ids": decoder_input_ids,
                        "encoder_hidden_states": enc_out,
                        "encoder_attention_mask": attn_mask,
                    },
                )
            else:
                # Subsequent steps: use cached KV from previous step
                dec_out = self._dec_past.run(
                    None,
                    {
                        "input_ids": decoder_input_ids,
                        "encoder_attention_mask": attn_mask,
                        **_past_feed(past_outputs, self._num_layers),  # type: ignore[arg-type]
                    },
                )

            logits: np.ndarray = dec_out[0]
            past_outputs = list(dec_out[1:])
            next_id = int(np.argmax(logits[0, -1, :]))
            output_ids.append(next_id)

            if next_id == self._eos_id:
                break

            decoder_input_ids = np.array([[next_id]], dtype=np.int64)

        # Decode tokens → text
        # Clamp out-of-vocab target IDs before decoding
        safe_ids = [
            i if i < self._meta["tgt_dict_size"] else self._meta["unk_id"]
            for i in output_ids
        ]
        return self._tgt_tok.decode(safe_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Run IndicTrans2 ONNX greedy translation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model",
        help=(
            "HF repo ID (e.g. hari31416/indictrans2-en-indic-dist-200M-ONNX) "
            "or local ONNX bundle directory"
        ),
    )
    parser.add_argument("text", help="Text to translate")
    parser.add_argument(
        "--src-lang",
        default="eng_Latn",
        help="Source BCP-47 language code (default: eng_Latn)",
    )
    parser.add_argument(
        "--tgt-lang",
        default="hin_Deva",
        help="Target BCP-47 language code (default: hin_Deva)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate (default: 128)",
    )
    args = parser.parse_args()

    model = IndicTransONNX(args.model)
    result = model.translate(
        args.text,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_new_tokens=args.max_new_tokens,
    )
    print(result)


if __name__ == "__main__":
    _main()
