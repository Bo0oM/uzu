import jax.numpy as jnp
import pytest

from lalamo.commands import MAX_QUANTIZATION_ERROR_RATIO, quantization_error_ratio

BITS = 8
GROUP_SIZE = 64


def embedding_table() -> jnp.ndarray:
    key = jnp.arange(512 * GROUP_SIZE, dtype=jnp.float32)
    return (jnp.sin(key * 0.37) * 0.03 + jnp.cos(key * 0.011) * 0.01).reshape(512, GROUP_SIZE)


def quantize(table: jnp.ndarray, *, levels: int) -> jnp.ndarray:
    """Round each group onto a grid of `levels` steps, the way an affine
    quantizer does. Passing fewer levels than the bit width allows models a
    quantizer that dropped bits."""
    low = table.min(axis=-1, keepdims=True)
    step = (table.max(axis=-1, keepdims=True) - low) / levels
    return jnp.round((table - low) / step) * step + low


def test_full_bit_width_passes() -> None:
    table = embedding_table()
    reconstructed = quantize(table, levels=2**BITS - 1)
    ratio = quantization_error_ratio(table, reconstructed, bits=BITS, group_size=GROUP_SIZE)
    assert ratio == pytest.approx(1.0, abs=0.3)
    assert ratio <= MAX_QUANTIZATION_ERROR_RATIO


def test_dropped_bits_are_caught() -> None:
    """The bfloat16 scale floor cost four of eight bits before it was fixed;
    the ratio has to put that well outside the tolerance."""
    table = embedding_table()
    reconstructed = quantize(table, levels=2**4 - 1)
    ratio = quantization_error_ratio(table, reconstructed, bits=BITS, group_size=GROUP_SIZE)
    assert ratio > 4 * MAX_QUANTIZATION_ERROR_RATIO
