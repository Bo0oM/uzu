#pragma once

#include "../../common/defines.h"

namespace uzu {
namespace gemm {

template <
    typename BT,
    typename AT,
    typename U,
    uint M_TILE,
    uint VALUES_PER_THREAD,
    uint RESULTS_PER_SIMDGROUP,
    uint K_SPLIT,
    bool INPUT_ALIGNED>
struct FullPrecisionBSource {
  // batch_base is the first batch element of this threadgroup's tile
  // (M_TILE consecutive elements share one weight pass — on the
  // bandwidth-bound fp forms this is what makes a verification pass cost
  // ~one decode pass); the gather path is only reachable at M_TILE = 1.
  static METAL_FUNC void accumulate(
      thread U (&result)[M_TILE * RESULTS_PER_SIMDGROUP],
      const device uint32_t* b,
      const device AT* a,
      const device uint* gather_indices,
      bool gathered,
      uint in_vec_size,
      uint out_vec_size,
      uint out_row,
      uint batch_base,
      uint simd_lane,
      uint k_slice
  ) {
    constexpr uint values_per_thread = VALUES_PER_THREAD;
    constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
    typedef vec<BT, 4> W4;
    typedef vec<AT, 4> I4;

    const uint k_stride = K_SPLIT * block_size;
    const uint k_start = k_slice * block_size;
    const uint thread_k = simd_lane * values_per_thread + k_start;
    const device AT* input = a + batch_base * in_vec_size + thread_k;

    // One advancing weight pointer per output row; base row = out_row (dense) or gather index.
    const uint base_row = gathered ? 0u : out_row;
    const device BT* weights = reinterpret_cast<const device BT*>(b);
    weights += base_row * in_vec_size + thread_k;
    thread const device BT* weight_rows[RESULTS_PER_SIMDGROUP];
    METAL_PRAGMA_UNROLL
    for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
      const uint addr_row = gathered ? gather_indices[batch_base * out_vec_size + out_row + row] : row;
      weight_rows[row] = weights + addr_row * in_vec_size;
    }

    uint k = k_start;
    for (; k + block_size <= in_vec_size; k += k_stride) {
      float4 input_values[M_TILE * (VALUES_PER_THREAD / 4)];
      METAL_PRAGMA_UNROLL
      for (uint bt = 0; bt < M_TILE; bt++) {
        METAL_PRAGMA_UNROLL
        for (uint part = 0; part < VALUES_PER_THREAD / 4; part++) {
          input_values[bt * (VALUES_PER_THREAD / 4) + part] =
              static_cast<float4>(*reinterpret_cast<const device I4*>(input + bt * in_vec_size + 4 * part));
        }
      }
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        METAL_PRAGMA_UNROLL
        for (uint part = 0; part < VALUES_PER_THREAD / 4; part++) {
          const float4 weight_values =
              static_cast<float4>(*reinterpret_cast<const device W4*>(weight_rows[row] + 4 * part));
          METAL_PRAGMA_UNROLL
          for (uint bt = 0; bt < M_TILE; bt++) {
            result[bt * RESULTS_PER_SIMDGROUP + row] +=
                dot(weight_values, input_values[bt * (VALUES_PER_THREAD / 4) + part]);
          }
        }
        weight_rows[row] += k_stride;
      }
      input += k_stride;
    }

    if constexpr (!INPUT_ALIGNED) {
      // Exactly one slice exits the loop with k at the partial tail block
      // (the one whose block index is congruent to the tail's); every other
      // slice exits past the end and computes remaining == 0.
      const uint thread_offset = simd_lane * values_per_thread;
      const int remaining = (k + thread_offset < in_vec_size) ? min(static_cast<int>(in_vec_size - k - thread_offset),
                                                                    static_cast<int>(values_per_thread))
                                                              : 0;
      if (remaining > 0) {
        METAL_PRAGMA_UNROLL
        for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          for (int index = 0; index < remaining; index++) {
            METAL_PRAGMA_UNROLL
            for (uint bt = 0; bt < M_TILE; bt++) {
              result[bt * RESULTS_PER_SIMDGROUP + row] +=
                  static_cast<U>(weight_rows[row][index]) * static_cast<U>(input[bt * in_vec_size + index]);
            }
          }
        }
      }
    }
  }
};

} // namespace gemm
} // namespace uzu
