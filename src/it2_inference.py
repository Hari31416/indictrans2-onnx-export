"""Shared batched greedy-decode inference for IndicTrans2 eval scripts."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 128


class FixtureLike(Protocol):
    text: str
    src_lang: str
    tgt_lang: str


@dataclass
class DecodeOutput:
    token_ids: list[int]
    text: str
    latency_ms: float = 0.0
    n_output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.n_output_tokens == 0:
            self.n_output_tokens = len(self.token_ids)


def iter_fixture_batches(
    fixtures: Sequence[FixtureLike],
    batch_size: int,
) -> list[list[int]]:
    """Group fixture indices into batches that share (src_lang, tgt_lang)."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    batches: list[list[int]] = []
    current_key: tuple[str, str] | None = None
    current_group: list[int] = []

    def flush_group() -> None:
        nonlocal current_group
        for start in range(0, len(current_group), batch_size):
            batches.append(current_group[start : start + batch_size])
        current_group = []

    for idx, fx in enumerate(fixtures):
        key = (fx.src_lang, fx.tgt_lang)
        if current_key is not None and key != current_key:
            flush_group()
        current_key = key
        current_group.append(idx)

    if current_group:
        flush_group()

    return batches


def pad_sequences(
    sequences: list[list[int]],
    *,
    pad_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("sequences must not be empty")

    max_len = max(len(seq) for seq in sequences)
    batch_n = len(sequences)
    input_ids = np.full((batch_n, max_len), pad_id, dtype=np.int64)
    attn_mask = np.zeros((batch_n, max_len), dtype=np.int64)
    for row, seq in enumerate(sequences):
        input_ids[row, : len(seq)] = seq
        attn_mask[row, : len(seq)] = 1
    return input_ids, attn_mask


def past_feed_from_outputs(
    past_outputs: list[np.ndarray],
    num_layers: int,
) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for layer in range(num_layers):
        base = layer * 4
        feed[f"past_key_values.{layer}.decoder.key"] = past_outputs[base]
        feed[f"past_key_values.{layer}.decoder.value"] = past_outputs[base + 1]
        feed[f"past_key_values.{layer}.encoder.key"] = past_outputs[base + 2]
        feed[f"past_key_values.{layer}.encoder.value"] = past_outputs[base + 3]
    return feed


def make_onnx_sessions(
    onnx_dir: Path,
    *,
    log_severity: int = 3,
) -> tuple[Any, Any, Any]:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = log_severity

    enc = ort.InferenceSession(
        str(onnx_dir / "encoder_model.onnx"),
        sess_options=opts,
    )
    dec = ort.InferenceSession(
        str(onnx_dir / "decoder_model.onnx"),
        sess_options=opts,
    )
    dec_past = ort.InferenceSession(
        str(onnx_dir / "decoder_with_past_model.onnx"),
        sess_options=opts,
    )
    return enc, dec, dec_past


def _load_generation_ids(onnx_dir: Path) -> tuple[int, int]:
    gen_cfg: dict[str, Any] = {}
    gen_config_path = onnx_dir / "generation_config.json"
    if gen_config_path.exists():
        gen_cfg = json.loads(gen_config_path.read_text(encoding="utf-8"))
    decoder_start_id = int(gen_cfg.get("decoder_start_token_id", 2))
    eos_id = int(gen_cfg.get("eos_token_id", 2))
    return decoder_start_id, eos_id


def _greedy_decode_onnx_chunk(
    *,
    enc: Any,
    dec: Any,
    dec_past: Any,
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    src_tok: Any,
    slow_tok: Any,
    meta: dict[str, int],
    ip: Any,
    num_layers: int,
    decoder_start_id: int,
    eos_id: int,
    max_new_tokens: int,
    measure_latency: bool,
) -> list[DecodeOutput]:
    if hasattr(ip, "_placeholder_entity_maps"):
        ip._placeholder_entity_maps.queue.clear()

    prefixed = ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
    id_lists: list[list[int]] = []
    for text in prefixed:
        encoded = src_tok.encode(text)
        id_lists.append(
            [
                token_id if token_id < meta["src_dict_size"] else meta["unk_id"]
                for token_id in encoded.ids
            ]
        )

    input_ids, attn_mask = pad_sequences(id_lists, pad_id=0)
    batch_n = len(texts)

    t_start = time.perf_counter() if measure_latency else 0.0

    enc_out = enc.run(
        ["last_hidden_state"],
        {"input_ids": input_ids, "attention_mask": attn_mask},
    )[0]

    output_ids: list[list[int]] = [[decoder_start_id] for _ in range(batch_n)]
    finished = np.zeros(batch_n, dtype=bool)
    decoder_input_ids = np.full((batch_n, 1), decoder_start_id, dtype=np.int64)
    past_outputs: list[np.ndarray] | None = None

    for step in range(max_new_tokens):
        if finished.all():
            break

        if step == 0:
            dec_out = dec.run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": enc_out,
                    "encoder_attention_mask": attn_mask,
                },
            )
        else:
            dec_out = dec_past.run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_attention_mask": attn_mask,
                    **past_feed_from_outputs(past_outputs, num_layers),  # type: ignore[arg-type]
                },
            )

        logits = dec_out[0]
        past_outputs = list(dec_out[1:])
        next_ids = np.argmax(logits[:, -1, :], axis=-1)

        for row in range(batch_n):
            if finished[row]:
                continue
            next_id = int(next_ids[row])
            output_ids[row].append(next_id)
            if next_id == eos_id:
                finished[row] = True

        decoder_input_ids = np.where(
            finished[:, None],
            0,
            next_ids[:, None],
        ).astype(np.int64)

    batch_latency_ms = (
        (time.perf_counter() - t_start) * 1000.0 if measure_latency else 0.0
    )
    per_item_latency_ms = batch_latency_ms / batch_n if measure_latency else 0.0

    safe_ids_list = [
        [
            token_id if token_id < meta["tgt_dict_size"] else meta["unk_id"]
            for token_id in row_ids
        ]
        for row_ids in output_ids
    ]
    with slow_tok.as_target_tokenizer():
        decoded = slow_tok.batch_decode(
            safe_ids_list,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    postprocessed = ip.postprocess_batch(decoded, lang=tgt_lang)

    return [
        DecodeOutput(
            token_ids=output_ids[row],
            text=postprocessed[row],
            latency_ms=per_item_latency_ms,
            n_output_tokens=len(output_ids[row]),
        )
        for row in range(batch_n)
    ]


def greedy_decode_onnx(
    onnx_dir: Path,
    fixtures: Sequence[FixtureLike],
    *,
    pytorch_model: str,
    batch_size: int = 1,
    measure_latency: bool = False,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[DecodeOutput]:
    """Greedy-decode fixtures with ONNX, optionally batched by language pair."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    from IndicTransToolkit import IndicProcessor

    meta = json.loads((onnx_dir / "tokenizer_meta.json").read_text(encoding="utf-8"))
    src_tok = Tokenizer.from_file(str(onnx_dir / "tokenizer_src.json"))
    slow_tok = AutoTokenizer.from_pretrained(pytorch_model, trust_remote_code=True)
    ip = IndicProcessor(inference=True)

    enc, dec, dec_past = make_onnx_sessions(onnx_dir)
    num_layers = (len(dec.get_outputs()) - 1) // 4
    decoder_start_id, eos_id = _load_generation_ids(onnx_dir)

    results: list[DecodeOutput | None] = [None] * len(fixtures)
    batches = iter_fixture_batches(fixtures, batch_size)
    logger.info(
        "ONNX greedy decode: %d fixtures in %d batches (batch_size=%d)",
        len(fixtures),
        len(batches),
        batch_size,
    )

    for batch_indices in batches:
        chunk = [fixtures[i] for i in batch_indices]
        src_lang = chunk[0].src_lang
        tgt_lang = chunk[0].tgt_lang
        decoded = _greedy_decode_onnx_chunk(
            enc=enc,
            dec=dec,
            dec_past=dec_past,
            texts=[fx.text for fx in chunk],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_tok=src_tok,
            slow_tok=slow_tok,
            meta=meta,
            ip=ip,
            num_layers=num_layers,
            decoder_start_id=decoder_start_id,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
            measure_latency=measure_latency,
        )
        for idx, result in zip(batch_indices, decoded):
            results[idx] = result

    missing = [i for i, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"Missing decode results for fixture indices: {missing[:5]}")
    return [result for result in results if result is not None]


def _greedy_decode_pytorch_chunk(
    *,
    model: Any,
    tokenizer: Any,
    ip: Any,
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    device: str,
    decoder_start_id: int,
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
) -> list[DecodeOutput]:
    import torch

    if hasattr(ip, "_placeholder_entity_maps"):
        ip._placeholder_entity_maps.queue.clear()

    prefixed = ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(
        prefixed,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    batch_n = len(texts)
    embed_dim = getattr(model.config, "decoder_embed_dim", 512)

    with torch.inference_mode():
        enc_out = model.model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).last_hidden_state

        output_ids: list[list[int]] = [[decoder_start_id] for _ in range(batch_n)]
        finished = torch.zeros(batch_n, dtype=torch.bool, device=device)
        decoder_input_ids = torch.full(
            (batch_n, 1),
            decoder_start_id,
            dtype=torch.long,
            device=device,
        )
        past_key_values = None

        for _ in range(max_new_tokens):
            if finished.all():
                break

            if past_key_values is None:
                dec_out = model.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=enc_out,
                    encoder_attention_mask=inputs["attention_mask"],
                    use_cache=True,
                )
            else:
                encoder_seq_len = inputs["attention_mask"].shape[1]
                dummy_encoder_hidden_states = torch.zeros(
                    batch_n,
                    encoder_seq_len,
                    embed_dim,
                    dtype=torch.float32,
                    device=device,
                )
                dec_out = model.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=dummy_encoder_hidden_states,
                    encoder_attention_mask=inputs["attention_mask"],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = model.lm_head(dec_out.last_hidden_state)
            next_ids = logits[:, -1, :].argmax(dim=-1)

            for row in range(batch_n):
                if finished[row]:
                    continue
                next_id = int(next_ids[row].item())
                output_ids[row].append(next_id)
                if next_id == eos_id:
                    finished[row] = True

            decoder_input_ids = torch.where(
                finished.unsqueeze(1),
                torch.tensor(pad_id, device=device),
                next_ids.unsqueeze(1),
            )
            past_key_values = dec_out.past_key_values

    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    postprocessed = ip.postprocess_batch(decoded, lang=tgt_lang)

    return [
        DecodeOutput(token_ids=output_ids[row], text=postprocessed[row])
        for row in range(batch_n)
    ]


def greedy_decode_pytorch(
    model_id: str,
    fixtures: Sequence[FixtureLike],
    *,
    device: str = "cpu",
    batch_size: int = 1,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[DecodeOutput]:
    """Greedy-decode fixtures with PyTorch, optionally batched by language pair."""
    from IndicTransToolkit import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    ip = IndicProcessor(inference=True)

    gen_cfg = model.generation_config
    decoder_start_id = int(gen_cfg.decoder_start_token_id)
    eos_id = int(gen_cfg.eos_token_id)
    pad_id = int(tokenizer.pad_token_id or 0)

    results: list[DecodeOutput | None] = [None] * len(fixtures)
    batches = iter_fixture_batches(fixtures, batch_size)
    logger.info(
        "PyTorch greedy decode: %d fixtures in %d batches (batch_size=%d)",
        len(fixtures),
        len(batches),
        batch_size,
    )

    for batch_indices in batches:
        chunk = [fixtures[i] for i in batch_indices]
        decoded = _greedy_decode_pytorch_chunk(
            model=model,
            tokenizer=tokenizer,
            ip=ip,
            texts=[fx.text for fx in chunk],
            src_lang=chunk[0].src_lang,
            tgt_lang=chunk[0].tgt_lang,
            device=device,
            decoder_start_id=decoder_start_id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_new_tokens=max_new_tokens,
        )
        for idx, result in zip(batch_indices, decoded):
            results[idx] = result

    missing = [i for i, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"Missing decode results for fixture indices: {missing[:5]}")
    return [result for result in results if result is not None]
