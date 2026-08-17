#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;

// Bounds the tap register cache; kernel_size is a function constant so the
// loops below fully unroll.
#define CONV_UPDATE_MAX_TAPS 8

// Single-token causal conv1d with SiLU, in-place.
template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(DeltaNetConvUpdate)(
    device const float* conv_weight,
    device const float* bias OPTIONAL(has_bias),
    device T* in_out,
    device float* state,
    constant const uint& conv_dim,
    constant const uint& state_stride,
    const bool has_bias SPECIALIZE,
    const uint kernel_size SPECIALIZE,
    const uint channel_idx AXIS(conv_dim, 256)
) {
  const uint tap_count = kernel_size - 1;
  const uint state_offset = channel_idx * state_stride;
  const device float* weight_row = conv_weight + channel_idx * kernel_size;

  float x = float(in_out[channel_idx]);

  // One pass over the state taps: cached in registers for both the
  // convolution and the shift, halving the device reads.
  float taps[CONV_UPDATE_MAX_TAPS];
  METAL_PRAGMA_UNROLL
  for (uint tap = 0; tap < tap_count; ++tap) {
    taps[tap] = state[state_offset + tap];
  }

  float acc = has_bias ? float(bias[channel_idx]) : 0.0f;
  METAL_PRAGMA_UNROLL
  for (uint tap = 0; tap < tap_count; ++tap) {
    acc += float(weight_row[tap]) * taps[tap];
  }
  acc += float(weight_row[tap_count]) * x;

  in_out[channel_idx] = static_cast<T>(activate_silu(acc));

  METAL_PRAGMA_UNROLL
  for (uint tap = 0; tap + 1 < tap_count; ++tap) {
    state[state_offset + tap] = taps[tap + 1];
  }
  state[state_offset + tap_count - 1] = x;
}
