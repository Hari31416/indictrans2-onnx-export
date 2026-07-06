"""Post-export ONNX bundle optimizations (ONNX_SIZE_OPTIMIZATION.md priorities 1–3, 5)."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper, save_model
from onnx.external_data_helper import (
    convert_model_to_external_data,
    load_external_data_for_tensor,
)

logger = logging.getLogger(__name__)

EMBED_WEIGHT_SUFFIX = "decoder.embed_tokens.weight"
LM_HEAD_WEIGHT_NAME = "lm_head.weight"
SHARED_DECODER_DATA_NAME = "decoder_shared.onnx.data"
INLINE_WEIGHT_THRESHOLD = 1024
VOCAB_DIM_MIN = 50_000
BUNDLE_GRAPH_FILES = (
    "encoder_model.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
)
DEFAULT_EXTERNALIZE_THRESHOLD_MB = 100
# Protobuf hard limit is 2 GiB; save externally well before that.
EXTERNAL_SAVE_THRESHOLD_BYTES = 512 * 1024 * 1024


def _load_model(onnx_path: Path) -> ModelProto:
    return onnx.load(str(onnx_path), load_external_data=True)


def _initializer_raw_bytes(model: ModelProto) -> int:
    return sum(len(init.raw_data) for init in model.graph.initializer if init.raw_data)


def _external_locations(model: ModelProto) -> set[str]:
    locs: set[str] = set()
    for init in model.graph.initializer:
        for entry in init.external_data:
            if entry.key == "location" and entry.value:
                locs.add(entry.value)
    return locs


def _save_model(model: ModelProto, onnx_path: Path) -> None:
    """Save an ONNX model, using a .onnx.data sidecar when weights are large."""
    data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    base_dir = onnx_path.parent
    locs = _external_locations(model)

    if locs:
        # Weights already live in a sidecar on disk (e.g. decoder_shared.onnx.data).
        if all((base_dir / loc).is_file() for loc in locs):
            save_model(model, str(onnx_path), save_as_external_data=False)
            return
        # External metadata set but blob missing — materialise on save.
        if data_path.exists():
            data_path.unlink()
        save_model(
            model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path.name,
            size_threshold=INLINE_WEIGHT_THRESHOLD,
        )
        return

    if _initializer_raw_bytes(model) >= EXTERNAL_SAVE_THRESHOLD_BYTES:
        if data_path.exists():
            data_path.unlink()
        save_model(
            model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path.name,
            size_threshold=INLINE_WEIGHT_THRESHOLD,
        )
        return

    save_model(model, str(onnx_path), save_as_external_data=False)


def _initializer_map(model: ModelProto) -> dict[str, TensorProto]:
    return {init.name: init for init in model.graph.initializer}


def _vocab_projection_name(model: ModelProto) -> str | None:
    """Return the vocab-sized lm_head MatMul weight (stored transposed for MatMul)."""
    initializers = _initializer_map(model)
    if LM_HEAD_WEIGHT_NAME in initializers:
        init = initializers[LM_HEAD_WEIGHT_NAME]
        if len(init.dims) == 2 and init.dims[1] >= VOCAB_DIM_MIN:
            return LM_HEAD_WEIGHT_NAME

    for node in model.graph.node:
        if node.op_type != "MatMul" or "lm_head" not in node.name:
            continue
        for inp in node.input:
            init = initializers.get(inp)
            if init is None:
                continue
            if len(init.dims) == 2 and init.dims[1] >= VOCAB_DIM_MIN:
                return inp
    return None


def _vocab_gather_weight_name(model: ModelProto) -> str | None:
    """Return the vocab-sized embedding Gather weight ([vocab, hidden])."""
    embed_name = _embed_tokens_name(model)
    if embed_name is not None:
        return embed_name

    initializers = _initializer_map(model)
    if LM_HEAD_WEIGHT_NAME in initializers:
        init = initializers[LM_HEAD_WEIGHT_NAME]
        if len(init.dims) == 2 and init.dims[0] >= VOCAB_DIM_MIN:
            return LM_HEAD_WEIGHT_NAME
    return None


def _embed_tokens_name(model: ModelProto) -> str | None:
    for init in model.graph.initializer:
        if init.name == EMBED_WEIGHT_SUFFIX or init.name.endswith(f".{EMBED_WEIGHT_SUFFIX}"):
            return init.name
    return None


def _initializer_array(tensor: TensorProto, base_dir: Path) -> np.ndarray:
    if tensor.external_data:
        load_external_data_for_tensor(tensor, str(base_dir))
    elif not tensor.raw_data:
        load_external_data_for_tensor(tensor, str(base_dir))
    return numpy_helper.to_array(tensor)


def _embed_and_lm_head_are_tied(
    model: ModelProto,
    base_dir: Path,
    gather_name: str,
    matmul_name: str,
) -> bool:
    """Verify lm_head MatMul weight is the transpose of the embedding matrix."""
    initializers = _initializer_map(model)
    embed = _initializer_array(initializers[gather_name], base_dir)
    matmul = _initializer_array(initializers[matmul_name], base_dir)
    if embed.shape != matmul.T.shape:
        return False
    return bool(np.allclose(embed, matmul.T, rtol=0.0, atol=1e-5))


def dedup_tied_embed_weights(model: ModelProto, base_dir: Path) -> tuple[ModelProto, bool]:
    """Route embedding Gather through lm_head.weight; drop duplicate transposed MatMul weight."""
    gather_name = _vocab_gather_weight_name(model)
    matmul_name = _vocab_projection_name(model)
    if gather_name is None:
        return model, False

    initializers = _initializer_map(model)
    gather_init = initializers[gather_name]
    gather_dims = list(gather_init.dims)

    if matmul_name is None:
        if gather_name == LM_HEAD_WEIGHT_NAME:
            return model, False
        gather_init.name = LM_HEAD_WEIGHT_NAME
        for node in model.graph.node:
            if node.op_type == "Gather":
                for i, inp in enumerate(node.input):
                    if inp == gather_name:
                        node.input[i] = LM_HEAD_WEIGHT_NAME
        logger.info("Renamed tied gather weight → %s", LM_HEAD_WEIGHT_NAME)
        return model, True

    matmul_init = initializers[matmul_name]
    matmul_dims = list(matmul_init.dims)

    if gather_name == matmul_name:
        return model, False

    if gather_dims == matmul_dims:
        logger.warning("Gather and MatMul weights share shape %s — cannot dedup safely", gather_dims)
        return model, False

    if gather_dims != matmul_dims[::-1]:
        logger.warning(
            "Tied weights are not transposes (%s vs %s) — skipping dedup",
            gather_dims,
            matmul_dims,
        )
        return model, False

    if not _embed_and_lm_head_are_tied(model, base_dir, gather_name, matmul_name):
        logger.info(
            "embed_tokens and lm_head weights differ — skipping tied-weight dedup"
        )
        return model, False

    canonical_name = LM_HEAD_WEIGHT_NAME
    if gather_name != canonical_name:
        gather_init.name = canonical_name
        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp == gather_name:
                    node.input[i] = canonical_name

    transpose_output = "/lm_head/weight_transposed"
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[canonical_name],
        outputs=[transpose_output],
        name="/lm_head/weight_transpose",
        perm=[1, 0],
    )

    matmul_index = next(
        i for i, n in enumerate(model.graph.node)
        if n.op_type == "MatMul" and matmul_name in n.input
    )
    matmul_node = model.graph.node[matmul_index]
    for i, inp in enumerate(matmul_node.input):
        if inp == matmul_name:
            matmul_node.input[i] = transpose_output

    model.graph.node.insert(matmul_index, transpose_node)
    model.graph.initializer.remove(matmul_init)

    logger.info(
        "Deduped tied weights: %s + %s → %s with Transpose for MatMul",
        gather_name,
        matmul_name,
        canonical_name,
    )
    return model, True


def simplify_onnx_graph(onnx_path: Path) -> ModelProto:
    """Fuse redundant nodes via ONNX Runtime's graph optimizer (priority 2)."""
    from onnxruntime.transformers.optimizer import optimize_by_onnxruntime

    before = len(_load_model(onnx_path).graph.node)
    optimized_path = onnx_path.with_suffix(".optimized.onnx")
    optimized_data_path = optimized_path.with_suffix(optimized_path.suffix + ".data")
    final_data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    use_external = final_data_path.exists() or onnx_path.stat().st_size > 100 * 1024 * 1024

    for stale in (optimized_path, optimized_data_path):
        if stale.exists():
            stale.unlink()

    try:
        optimize_by_onnxruntime(
            onnx_model=str(onnx_path),
            optimized_model_path=str(optimized_path),
            opt_level=99,
            save_as_external_data=use_external,
            external_data_filename=optimized_data_path.name,
        )
    except Exception as exc:
        logger.warning("ORT graph optimization failed for %s: %s", onnx_path.name, exc)
        for stale in (optimized_path, optimized_data_path):
            if stale.exists():
                stale.unlink()
        return _load_model(onnx_path)

    if not optimized_path.exists():
        logger.warning("ORT optimizer produced no output for %s", onnx_path.name)
        return _load_model(onnx_path)

    # ORT must not write into the live *.onnx.data sidecar — it truncates the file
    # before all weights are read, leaving a 0-byte sidecar and broken external refs.
    model = onnx.load(str(optimized_path), load_external_data=True)
    if final_data_path.exists():
        final_data_path.unlink()

    _save_model(model, onnx_path)

    optimized_path.unlink(missing_ok=True)
    optimized_data_path.unlink(missing_ok=True)

    model = _load_model(onnx_path)
    after = len(model.graph.node)
    logger.info("Graph simplified %s: %d → %d nodes", onnx_path.name, before, after)
    return model


def externalize_if_large(onnx_path: Path, size_threshold_mb: int = 100) -> bool:
    """Move inline weights to a .onnx.data sidecar (priority 3)."""
    proto_mb = onnx_path.stat().st_size / (1024 * 1024)
    if proto_mb < size_threshold_mb:
        return False

    model = _load_model(onnx_path)
    data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if data_path.exists():
        data_path.unlink()

    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
    )
    _save_model(model, onnx_path)
    logger.info("Externalized %s → %s (proto was ~%.0f MB)", onnx_path.name, data_path.name, proto_mb)
    return True


def _tensor_raw_bytes(tensor: TensorProto, base_dir: Path) -> bytes:
    if tensor.external_data:
        info = {entry.key: entry.value for entry in tensor.external_data}
        location = info.get("location", "")
        if location:
            data_file = base_dir / location
            offset = int(info.get("offset", 0))
            length = int(info.get("length", 0))
            with data_file.open("rb") as handle:
                handle.seek(offset)
                return handle.read(length)

    if tensor.raw_data:
        return bytes(tensor.raw_data)

    # Quantized graphs may store scalar scale/zero-point in typed fields only.
    return numpy_helper.to_array(tensor).tobytes()


def _assign_inline_tensor(tensor: TensorProto, raw: bytes) -> None:
    del tensor.external_data[:]
    tensor.data_location = TensorProto.DEFAULT
    tensor.raw_data = raw


def _assign_external_tensor(
    tensor: TensorProto,
    location: str,
    offset: int,
    length: int,
) -> None:
    tensor.ClearField("raw_data")
    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value


def share_decoder_external_data(output_dir: Path) -> bool:
    """Merge decoder sidecars into one shared file (priority 5)."""
    decoder_paths = [
        output_dir / "decoder_model.onnx",
        output_dir / "decoder_with_past_model.onnx",
    ]
    if not all(p.exists() for p in decoder_paths):
        logger.warning("Decoder ONNX files missing — skipping shared weight merge")
        return False

    models = [_load_model(p) for p in decoder_paths]
    shared_path = output_dir / SHARED_DECODER_DATA_NAME

    # content hash → (offset, length)
    blob_store: dict[str, tuple[int, int]] = {}
    shared_bytes = bytearray()

    def intern_bytes(raw: bytes) -> tuple[str, int, int]:
        digest = hashlib.sha256(raw).hexdigest()
        if digest not in blob_store:
            offset = len(shared_bytes)
            shared_bytes.extend(raw)
            blob_store[digest] = (offset, len(raw))
        return digest, *blob_store[digest]

    remapped = 0
    for model, onnx_path in zip(models, decoder_paths, strict=True):
        base_dir = onnx_path.parent
        for tensor in model.graph.initializer:
            raw = _tensor_raw_bytes(tensor, base_dir)
            if not raw:
                continue
            if len(raw) <= INLINE_WEIGHT_THRESHOLD:
                _assign_inline_tensor(tensor, raw)
            else:
                _, offset, length = intern_bytes(raw)
                _assign_external_tensor(tensor, SHARED_DECODER_DATA_NAME, offset, length)
            remapped += 1

    if not shared_bytes:
        logger.info("No decoder weights to share")
        return False

    shared_path.write_bytes(shared_bytes)
    for path, model in zip(decoder_paths, models, strict=True):
        _save_model(model, path)

    for path in decoder_paths:
        sidecar = path.with_suffix(path.suffix + ".data")
        if sidecar.exists() and sidecar != shared_path:
            sidecar.unlink()

    unique_mb = len(shared_bytes) / (1024 * 1024)
    logger.info(
        "Shared decoder weights → %s (%d unique tensors, ~%.0f MB)",
        SHARED_DECODER_DATA_NAME,
        len(blob_store),
        unique_mb,
    )
    return remapped > 0


def optimize_decoder_graph(onnx_path: Path) -> None:
    """Dedup tied embed/lm_head weights and simplify a decoder graph."""
    model = _load_model(onnx_path)
    model, _ = dedup_tied_embed_weights(model, onnx_path.parent)
    _save_model(model, onnx_path)
    simplify_onnx_graph(onnx_path)


def optimize_export_bundle(output_dir: Path, *, externalize_threshold_mb: int = 100) -> None:
    """Run post-export optimizations for priorities 1–3 and 5."""
    encoder = output_dir / "encoder_model.onnx"
    decoder = output_dir / "decoder_model.onnx"
    decoder_past = output_dir / "decoder_with_past_model.onnx"

    for path in (decoder, decoder_past):
        if path.exists():
            logger.info("Optimizing %s", path.name)
            optimize_decoder_graph(path)

    if encoder.exists():
        logger.info("Optimizing %s", encoder.name)
        simplify_onnx_graph(encoder)

    finalize_bundle_layout(output_dir, externalize_threshold_mb=externalize_threshold_mb)


def ensure_bundle_graphs_loadable(output_dir: Path) -> None:
    """Verify each bundle graph can load external weights (fails fast on stale sidecar refs)."""
    for name in BUNDLE_GRAPH_FILES:
        path = output_dir / name
        if not path.exists():
            continue
        try:
            onnx.load(str(path), load_external_data=True)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load external weights for {path.name} in {output_dir}. "
                "Re-run the fp16 conversion step (or re-export the fp32 bundle) to "
                "regenerate a consistent sidecar layout."
            ) from exc


def finalize_bundle_layout(
    output_dir: Path,
    *,
    externalize_threshold_mb: int = DEFAULT_EXTERNALIZE_THRESHOLD_MB,
) -> None:
    """Externalize large protos and share decoder weights (after fp16 / quant steps)."""
    for name in BUNDLE_GRAPH_FILES:
        path = output_dir / name
        if path.exists():
            externalize_if_large(path, externalize_threshold_mb)
    share_decoder_external_data(output_dir)
