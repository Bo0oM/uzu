#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;
#define GRAIN_SIZE 4
// Iterations kept in registers between the two passes; covers head_dim up to 256.
#define CACHED_STEPS 2

// Normalizes one already loaded element and writes it out
template <typename ScaleT, typename OutputT, typename AccumT>
static METAL_FUNC void qkv_norm_write(
    device OutputT* output_data,
    const device ScaleT* scales_data,
    const uint i,
    const AccumT value,
    const AccumT rms_norm,
    const float scale_offset,
    const bool full_layer,
    const bool has_scales
) {
  AccumT normalized_high = value * rms_norm;

  if (!has_scales) {
    output_data[i] = static_cast<OutputT>(normalized_high);
  } else if (full_layer) {
    AccumT scale_value_high = static_cast<AccumT>(scales_data[i]) + static_cast<AccumT>(scale_offset);
    output_data[i] = static_cast<OutputT>(normalized_high * scale_value_high);
  } else {
    OutputT normalized_low = static_cast<OutputT>(normalized_high);
    OutputT scale_value_low =
        static_cast<OutputT>(static_cast<AccumT>(scales_data[i]) + static_cast<AccumT>(scale_offset));
    OutputT product_low = normalized_low * scale_value_low;
    output_data[i] = static_cast<OutputT>(product_low);
  }
}

// QKV norm: normalize per-head vectors (small head_dim) efficiently.
//
// Strategy:
// - One SIMD-group (32 threads) processes one head.
// - One threadgroup (one SIMD-group) is dispatched per head.
template <typename InputT, typename ScaleT, typename OutputT, typename AccumT>
VARIANTS(InputT, float, half, bfloat)
VARIANTS(ScaleT, float, half, bfloat)
VARIANTS(OutputT, float, half, bfloat)
VARIANTS(AccumT, float, half)
PUBLIC KERNEL(QKVNorm)(
    const device InputT* qkv_input OPTIONAL(!in_place),
    const device ScaleT* scales OPTIONAL(has_scales),
    device OutputT* qkv_output,
    constant uint& batch_size,
    constant uint& total_heads,
    constant uint& head_dim,
    constant float& epsilon,
    constant float& scale_offset,
    constant uint& head_offset,
    constant uint& head_count,
    const uint batch_idx GROUPS(batch_size),
    const uint head_idx GROUPS(head_count),
    const uint lane_id THREADS(METAL_SIMD_SIZE),
    const bool in_place SPECIALIZE,
    const bool has_scales SPECIALIZE,
    const bool full_layer SPECIALIZE
) {
  if (in_place) {
    qkv_input = (const device InputT*)qkv_output;
  }

  if (head_count == 0u || head_dim == 0u)
    return;

  const ulong slice_offset =
      (ulong)batch_idx * (ulong)total_heads * (ulong)head_dim + (ulong)(head_offset + head_idx) * (ulong)head_dim;

  const device InputT* input_data = qkv_input + slice_offset;
  const device ScaleT* scales_data = scales;
  device OutputT* output_data = qkv_output + slice_offset;
  const uint element_count = head_dim;

  AccumT partial_sum = static_cast<AccumT>(0.0f);

  const uint lane_stride = METAL_SIMD_SIZE * GRAIN_SIZE;
  AccumT cached[CACHED_STEPS * GRAIN_SIZE];

  // Sum of squares; values stay in registers for pass two.
  METAL_PRAGMA_UNROLL
  for (uint step = 0; step < CACHED_STEPS; ++step) {
    const uint base_i = lane_id * GRAIN_SIZE + step * lane_stride;
    if (base_i < element_count) {
      for (uint j = 0; j < GRAIN_SIZE; ++j) {
        uint i = base_i + j;
        cached[step * GRAIN_SIZE + j] = (i < element_count) ? static_cast<AccumT>(input_data[i]) : 0.0f;
      }
      for (uint j = 0; j < GRAIN_SIZE; ++j) {
        partial_sum += cached[step * GRAIN_SIZE + j] * cached[step * GRAIN_SIZE + j];
      }
    }
  }

  // Elements beyond the register budget are re-read in pass two.
  for (uint base_i = lane_id * GRAIN_SIZE + CACHED_STEPS * lane_stride; base_i < element_count;
       base_i += lane_stride) {
    AccumT vals[GRAIN_SIZE];
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      vals[j] = (i < element_count) ? static_cast<AccumT>(input_data[i]) : 0.0f;
    }
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      partial_sum += vals[j] * vals[j];
    }
  }

  // SIMD-group reduction.
  AccumT total_sum = simd_sum(partial_sum);

  // RMS factor.
  AccumT mean_square = static_cast<AccumT>(total_sum) / static_cast<AccumT>(element_count);
  AccumT rms_norm = rsqrt(mean_square + static_cast<AccumT>(epsilon));

  // Normalize + scale, reusing the values pass one left in registers.
  METAL_PRAGMA_UNROLL
  for (uint step = 0; step < CACHED_STEPS; ++step) {
    const uint base_i = lane_id * GRAIN_SIZE + step * lane_stride;
    if (base_i < element_count) {
      for (uint j = 0; j < GRAIN_SIZE; ++j) {
        uint i = base_i + j;
        if (i >= element_count)
          continue;

        qkv_norm_write(
            output_data,
            scales_data,
            i,
            cached[step * GRAIN_SIZE + j],
            rms_norm,
            scale_offset,
            full_layer,
            has_scales
        );
      }
    }
  }

  for (uint base_i = lane_id * GRAIN_SIZE + CACHED_STEPS * lane_stride; base_i < element_count;
       base_i += lane_stride) {
    AccumT vals[GRAIN_SIZE];
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      vals[j] = (i < element_count) ? static_cast<AccumT>(input_data[i]) : 0.0f;
    }

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i >= element_count)
        continue;

      qkv_norm_write(output_data, scales_data, i, vals[j], rms_norm, scale_offset, full_layer, has_scales);
    }
  }
}
