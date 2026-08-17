#include <metal_stdlib>
#include "../common/dsl.h"

// Expands an int8 KV cache plane back to the model dtype for the prefill
// attention cores, which keep their full-precision layouts (ADR-8: the
// dequant bridge trades one cheap pass at prefill for untouched GEMM
// kernels).
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(KVCacheDequant) (
    const device char* quantized,
    const device float* scales, // one per (position, kv_head) row
    device T* dequantized,
    const constant uint32_t& head_dim,
    const constant uint32_t& element_count,
    const uint32_t element_idx AXIS(element_count, 256)
) {
  const float scale = scales[element_idx / head_dim];
  dequantized[element_idx] = static_cast<T>(float(quantized[element_idx]) * scale);
}
