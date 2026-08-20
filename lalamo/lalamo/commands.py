import math
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import jax.numpy as jnp
import requests
import thefuzz.fuzz
import thefuzz.process
from huggingface_hub import snapshot_download

from lalamo.model_import import ModelSpec
from lalamo.model_import.common import (
    DownloadingFileEvent,
    FileSpec,
    FinishedDownloadingFileEvent,
    FinishedInitializingModelEvent,
    InitializingModelEvent,
    StatusEvent,
    import_model,
)
from lalamo.model_import.loaders.dflash_loader import load_hf_dflash_draft_model
from lalamo.model_import.loaders.weaver_loader import load_weaver
from lalamo.model_import.remote_registry import RegistryModel, RegistryModelFile
from lalamo.models import SpeculatorModel, SpeculatorModelConfig
from lalamo.modules import DFlashSpeculator, DFlashSpeculatorConfig
from lalamo.utils.sharding import ShardingConfig


@dataclass
class PullCallbacks:
    model_spec: RegistryModel
    output_dir: Path
    overwrite: bool

    def started(self) -> None:
        pass

    def output_dir_exists(self) -> None:
        raise RuntimeError(f"{self.output_dir=} already exists, refusing to overwrite!")

    def downloading(self, file_spec: RegistryModelFile) -> None:
        pass

    def finished_downloading(self, file_spec: RegistryModelFile) -> None:
        pass

    def finished(self) -> None:
        pass


def _download_file(url: str, dest_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def _suggest_similar_models(query: str, repo_ids: list[str], limit: int = 3, min_score: int = 70) -> str:
    ranked_matches = thefuzz.process.extract(query, repo_ids, limit=limit, scorer=thefuzz.fuzz.ratio)
    similar_repos = [repo for repo, score in ranked_matches if score >= min_score]
    if not similar_repos:
        return ""
    return "\n\nDid you mean one of these?\n" + "\n".join(f"  - {repo}" for repo in similar_repos)


def pull(
    model_spec: RegistryModel,
    output_dir: Path,
    callbacks_type: Callable[
        [
            RegistryModel,
            Path,
            bool,
        ],
        PullCallbacks,
    ] = PullCallbacks,
    overwrite: bool = False,
) -> None:
    callbacks = callbacks_type(model_spec, output_dir, overwrite)

    if output_dir.exists():
        callbacks.output_dir_exists()

    callbacks.started()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for file_spec in model_spec.files:
            callbacks.downloading(file_spec)

            # Security: validate filename to prevent path traversal attacks
            safe_name = Path(file_spec.name).name
            if not safe_name or safe_name != file_spec.name:
                raise RuntimeError(
                    f"Invalid filename from registry: {file_spec.name!r}. "
                    f"Filenames must not contain path separators or traversal sequences.",
                )

            file_path = temp_path / safe_name
            try:
                _download_file(file_spec.url, file_path)
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download {safe_name}: {e}") from e

            callbacks.finished_downloading(file_spec)

        output_dir.mkdir(parents=True, exist_ok=True)
        for file_spec in model_spec.files:
            safe_name = Path(file_spec.name).name
            src = temp_path / safe_name
            dst = output_dir / safe_name
            shutil.move(str(src), str(dst))

    callbacks.finished()


class DType(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


@dataclass
class ConversionCallbacks:
    model_spec: ModelSpec
    output_dir: Path
    dtype: DType | None
    context_length: int | None

    def started(self) -> None:
        pass

    def output_dir_exists(self) -> None:
        raise RuntimeError(f"{self.output_dir=} already exists, refusing to overwrite!")

    def downloading(self, file_spec: FileSpec) -> None:
        pass

    def finished_downloading(self, file_spec: FileSpec) -> None:
        pass

    def initializing_model(self) -> None:
        pass

    def finished_initializing_model(self) -> None:
        pass

    def saving_model(self) -> None:
        pass

    def finished_saving_model(self) -> None:
        pass


def convert(
    model_spec: ModelSpec,
    output_dir: Path,
    dtype: DType | None = None,
    context_length: int | None = None,
    callbacks_type: Callable[
        [
            ModelSpec,
            Path,
            DType | None,
            int | None,
        ],
        ConversionCallbacks,
    ] = ConversionCallbacks,
    quantize_embeddings: int | None = None,
    embedding_group_size: int = 64,
    quantize: int | None = None,
    quantization_group_size: int = 64,
) -> None:
    effective_dtype = dtype or DType.BFLOAT16
    callbacks = callbacks_type(
        model_spec,
        output_dir,
        effective_dtype,
        context_length,
    )

    if output_dir.exists():
        callbacks.output_dir_exists()

    callbacks.started()

    def progress_callback(event: StatusEvent) -> None:
        match event:
            case DownloadingFileEvent(file_spec):
                callbacks.downloading(file_spec)
            case FinishedDownloadingFileEvent(file_spec):
                callbacks.finished_downloading(file_spec)
            case InitializingModelEvent():
                callbacks.initializing_model()
            case FinishedInitializingModelEvent():
                callbacks.finished_initializing_model()

    imported_model = import_model(
        model_spec,
        sharding_config=ShardingConfig.replicated(),
        dtype=jnp.dtype(effective_dtype.value),
        context_length=context_length,
        progress_callback=progress_callback,
    )
    model = imported_model.model
    if quantize is not None:
        model = _quantize_weight_matrices(model, bits=quantize, group_size=quantization_group_size)
    if quantize_embeddings is not None:
        model = _quantize_tied_embedding(model, bits=quantize_embeddings, group_size=embedding_group_size)

    callbacks.saving_model()
    model.save(output_dir)

    callbacks.finished_saving_model()


def _quantize_weight_matrices(model, *, bits: int, group_size: int):
    """Quantizes every full-precision weight matrix in the model.

    This is what an external quantizer such as `mlx_lm convert -q` produces,
    done in one step of the conversion instead of two tools: the projections
    of every layer, plus the embedding table, move to group-affine integer
    weights, while normalization scales and anything already compressed are
    left alone. Matrices whose reduction dim does not divide by the group are
    skipped rather than regrouped, so the layout the engine expects holds.
    """
    from lalamo.compressed import MLXSpec
    from lalamo.utils.surgery import map_nodes_of_type
    from lalamo.weight_matrix import FullPrecisionMatrix, WeightMatrix

    def quantize(matrix: WeightMatrix) -> WeightMatrix:
        if not isinstance(matrix, FullPrecisionMatrix):
            return matrix
        if matrix.shape[-1] % group_size != 0:
            return matrix
        original = matrix.astype(jnp.float32).decompress()
        spec = MLXSpec(bits=bits, group_size=group_size, layout=matrix.spec.layout)
        quantized = spec.compress(
            original,
            sharding_config=matrix.sharding_config,
            is_sharded=matrix.is_sharded,
        ).astype(matrix.dtype)
        _reject_degraded_quantization(original, quantized.decompress(), bits=bits, group_size=group_size)
        return quantized

    return map_nodes_of_type(WeightMatrix, quantize, model)


def _quantize_tied_embedding(model, *, bits: int, group_size: int):
    """Quantizes the tied embedding table in place of its full-precision form.

    The tied table is read twice per generated token — as the input lookup
    and as the vocab readout matmul — and in bf16 exports it accounts for
    20-32% of all weight traffic (gemma-3-1b: 604 MB per token out of a
    1.9 GB file). Group-affine int quantization of just this table trades a
    0.68% RMS weight error (1.7e-3 max abs, measured on Qwen3-0.6B at 8 bits
    and group 64) for roughly +8% decode throughput on bandwidth-bound
    devices; the uzu engine loads the quantized table with its existing
    quantized lookup and readout paths.
    """
    import equinox as eqx

    from lalamo.compressed import MLXSpec
    from lalamo.modules.embedding import TiedEmbedding
    from lalamo.weight_matrix import FullPrecisionMatrix

    embedding = model.decoder.embedding
    if not isinstance(embedding, TiedEmbedding):
        raise ValueError(
            "--quantize-embeddings targets the tied embedding table (the vocab readout); "
            f"this model uses {type(embedding).__name__}, whose readout does not share the table."
        )
    matrix = embedding.embedding
    if not isinstance(matrix, FullPrecisionMatrix):
        raise ValueError(
            f"--quantize-embeddings expects a full-precision embedding table, found {type(matrix).__name__}."
        )
    if matrix.shape[-1] % group_size != 0:
        raise ValueError(
            f"embedding dim {matrix.shape[-1]} is not divisible by the embedding group size {group_size}."
        )

    spec = MLXSpec(bits=bits, group_size=group_size, layout=matrix.spec.layout)
    # Quantize from float32. `scale_from_min_max` floors every group scale at
    # `finfo(weights.dtype).eps`, which is 2^-7 for bfloat16 — larger than the
    # scale a group of embedding weights actually needs (~5e-4), so a bfloat16
    # table comes back with every scale clamped and roughly four effective bits
    # instead of eight. The export dtype is applied when the model is saved.
    original = matrix.astype(jnp.float32).decompress()
    quantized = spec.compress(
        original,
        sharding_config=matrix.sharding_config,
        is_sharded=matrix.is_sharded,
    ).astype(matrix.dtype)
    _reject_degraded_quantization(original, quantized.decompress(), bits=bits, group_size=group_size)
    return eqx.tree_at(lambda m: m.decoder.embedding.embedding, model, quantized)


def quantization_error_ratio(
    original: "Float[Array, '*components rows cols']",
    reconstructed: "Float[Array, '*components rows cols']",
    *,
    bits: int,
    group_size: int,
) -> float:
    """How much worse the table came back than the bit width allows.

    A group of `group_size` weights spanning `span` is stored on a grid of
    `2**bits - 1` steps, so rounding to the nearest step leaves an error of
    `step / sqrt(12)` — the standard deviation of a uniform distribution over
    one step. Dividing the achieved error by that floor gives a scale-free
    number: near 1 means the quantizer used the bits it was given, and a large
    ratio means it did not (a coarser grid, a clamped scale, a layout the
    reconstruction does not match).
    """
    # The reconstruction comes back in the export dtype; compare in float32 so
    # the measurement is not itself rounded.
    original = original.astype(jnp.float32)
    reconstructed = reconstructed.astype(jnp.float32)
    groups = original.reshape(*original.shape[:-1], -1, group_size)
    spans = groups.max(axis=-1) - groups.min(axis=-1)
    floor_rms = float(jnp.sqrt(jnp.mean(jnp.square(spans / (2**bits - 1)))) / jnp.sqrt(12.0))
    achieved_rms = float(jnp.sqrt(jnp.mean(jnp.square(reconstructed - original))))
    if floor_rms == 0.0:
        return 1.0
    return achieved_rms / floor_rms


# Rounding the scales to the export dtype costs a little over the theoretical
# floor, so the check has to leave room; three times the floor still catches a
# quantizer that silently dropped bits (a bfloat16 scale floor once cost four
# of eight bits and landed at eleven times the floor).
MAX_QUANTIZATION_ERROR_RATIO = 3.0


def _reject_degraded_quantization(
    original: "Float[Array, '*components rows cols']",
    reconstructed: "Float[Array, '*components rows cols']",
    *,
    bits: int,
    group_size: int,
) -> None:
    ratio = quantization_error_ratio(original, reconstructed, bits=bits, group_size=group_size)
    if ratio > MAX_QUANTIZATION_ERROR_RATIO:
        raise ValueError(
            f"embedding quantization to {bits} bits (group {group_size}) came back "
            f"{ratio:.1f}x worse than the bit width allows, so the table would carry "
            f"roughly {bits - math.log2(ratio):.1f} effective bits. Refusing to write a "
            "model whose quality loss does not match what was asked for."
        )


def convert_speculator(
    dflash_repo_id: str,
    output_dir: Path,
    weaver_repo_id: str | None = None,
    dtype: DType | None = None,
    context_length: int | None = None,
) -> None:
    sharding_config = ShardingConfig.replicated()
    effective_dtype = jnp.dtype((dtype or DType.BFLOAT16).value)

    dflash_path = Path(snapshot_download(dflash_repo_id, allow_patterns=["config.json", "*.safetensors"]))
    draft_model = load_hf_dflash_draft_model(
        dflash_path,
        sharding_config=sharding_config,
        dtype=effective_dtype,
        context_length=context_length,
    )

    weaver = None
    if weaver_repo_id is not None:
        weaver_dir = Path(snapshot_download(weaver_repo_id, allow_patterns=["*.pth"]))
        checkpoints = sorted(weaver_dir.rglob("*.pth"))
        if len(checkpoints) != 1:
            found = ", ".join(str(checkpoint.relative_to(weaver_dir)) for checkpoint in checkpoints) or "none"
            raise ValueError(f"Expected exactly one .pth checkpoint in '{weaver_repo_id}', found: {found}.")
        weaver = load_weaver(checkpoints[0], sharding_config, dtype=effective_dtype)

    speculator = DFlashSpeculator(
        config=DFlashSpeculatorConfig(
            draft_config=draft_model.config,
            weaver_config=weaver.config if weaver is not None else None,
        ),
        sharding_config=sharding_config,
        draft_model=draft_model,
        weaver=weaver,
    )
    model = SpeculatorModel(
        config=SpeculatorModelConfig(speculator_config=speculator.config),
        sharding_config=sharding_config,
        speculator=speculator,
    )
    model.save(output_dir)
