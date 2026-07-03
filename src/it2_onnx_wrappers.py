"""PyTorch wrappers for IndicTrans2 ONNX export (naklitechie I/O layout)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _flatten_past(past_key_values: tuple) -> tuple[torch.Tensor, ...]:
    """Flatten (dec_k, dec_v, enc_k, enc_v) per layer → present.* tensors."""
    flat: list[torch.Tensor] = []
    for layer_past in past_key_values:
        dec_k, dec_v, enc_k, enc_v = layer_past
        flat.extend([dec_k, dec_v, enc_k, enc_v])
    return tuple(flat)


def _unflatten_past(flat: tuple[torch.Tensor, ...]) -> tuple:
    past = []
    for i in range(0, len(flat), 4):
        past.append((flat[i], flat[i + 1], flat[i + 2], flat[i + 3]))
    return tuple(past)


def present_output_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for i in range(num_layers):
        names.extend([
            f"present.{i}.decoder.key",
            f"present.{i}.decoder.value",
            f"present.{i}.encoder.key",
            f"present.{i}.encoder.value",
        ])
    return names


def past_input_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for i in range(num_layers):
        names.extend([
            f"past_key_values.{i}.decoder.key",
            f"past_key_values.{i}.decoder.value",
            f"past_key_values.{i}.encoder.key",
            f"past_key_values.{i}.encoder.value",
        ])
    return names


class IndicTransEncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class IndicTransDecoderWrapper(nn.Module):
    """First decode step — no past KV in, full present KV out + logits."""

    def __init__(self, decoder: nn.Module, lm_head: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        out = self.decoder(
            input_ids=input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        logits = self.lm_head(out.last_hidden_state)
        return (logits, *_flatten_past(out.past_key_values))


class IndicTransDecoderWithPastWrapper(nn.Module):
    """Autoregressive steps 2..N — past KV in/out, no encoder_hidden_states input."""

    def __init__(self, decoder: nn.Module, lm_head: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.num_layers = num_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        *past_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        past_key_values = _unflatten_past(past_flat)
        # HF decoder expects 2-tuple past when encoder_hidden_states is omitted
        hf_past = tuple((layer[0], layer[1]) for layer in past_key_values)

        out = self.decoder(
            input_ids=input_ids,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=hf_past,
            use_cache=True,
        )
        logits = self.lm_head(out.last_hidden_state)
        # Keep encoder_attention_mask in the ONNX graph (cross-attn mask for cached KV)
        logits = logits + encoder_attention_mask.sum() * 0.0

        # Merge updated decoder KV with encoder cross-attn KV carried from input past
        combined: list[tuple[torch.Tensor, ...]] = []
        for i, layer_out in enumerate(out.past_key_values):
            dec_k, dec_v = layer_out
            enc_k, enc_v = past_key_values[i][2], past_key_values[i][3]
            combined.append((dec_k, dec_v, enc_k, enc_v))

        return (logits, *_flatten_past(tuple(combined)))
