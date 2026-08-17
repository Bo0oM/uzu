#pragma once

#include "../../common/qdot.h"
#include "../../common/quant_pack.h"
#include "quantized_row_state.h"

namespace uzu {
namespace gemm {

template <
    typename BT,
    typename AT,
    typename U,
    typename MT,
    uint M_TILE,
    GemmBPrologueKind B_PROLOGUE,
    uint GROUP_SIZE,
    uint BITS,
    uint PACKS,
    uint K_SPLIT,
    uint RESULTS_PER_SIMDGROUP,
    bool INPUT_ALIGNED>
struct QuantizedBSource {
  // batch_base is the first batch element of this threadgroup's tile
  // (M_TILE consecutive elements share one weight pass); the gather path is
  // only reachable at M_TILE = 1 (the host never selects a batched tile for
  // gathered dispatches), where batch_base is the plain batch index.
  static METAL_FUNC void accumulate(
      thread U (&result)[M_TILE * RESULTS_PER_SIMDGROUP],
      const device uint32_t* b,
      const device BT* scales,
      const device uint8_t* zero_points,
      const device BT* biases,
      const device AT* a,
      const device uint* gather_indices,
      bool gathered,
      uint in_vec_size,
      uint out_vec_size,
      uint out_row,
      uint batch_base,
      uint simd_lane,
      uint k_slice,
      const bool signed_codes
  ) {
    constexpr uint pack_factor = get_pack_factor<BITS, 32>();
    constexpr uint bytes_per_pack = get_bytes_per_pack<BITS, 32>();
    // PACKS = 1 keeps lane loads dense (32 adjacent words per simd load) and
    // doubles the independent k iterations, which hides dequant latency on
    // shallow-k forms (gemma k=1152: upgate -45%, readout -38%). Deep-k forms
    // lose from the doubled loop overhead (gs128 sweep: +33..40%), so the
    // tile table opts in per shape and PACKS = 2 stays the default.
    constexpr uint packs_per_thread = PACKS;
    constexpr uint values_per_thread = pack_factor * packs_per_thread;
    constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
    constexpr uint scale_step_per_thread = GROUP_SIZE / values_per_thread;
    // Slice s owns K blocks s, s+K_SPLIT, s+2*K_SPLIT, ... block_size is a
    // multiple of GROUP_SIZE and pack_factor, so every slice starts on a
    // group and pack boundary.
    constexpr uint k_stride = K_SPLIT * block_size;
    using RowState = QuantizedRowState<BT, U, B_PROLOGUE, BITS, RESULTS_PER_SIMDGROUP>;
    using RowParams = typename RowState::Params;

    const uint k_start = k_slice * block_size;
    const uint weights_row_stride = in_vec_size * bytes_per_pack / pack_factor;
    const uint group_count = (in_vec_size + GROUP_SIZE - 1) / GROUP_SIZE;
    const uint group_offset = k_start / GROUP_SIZE + simd_lane / scale_step_per_thread;

    // Base row = out_row (dense) or 0 (gather); rows indexed relative to it.
    const uint base_row = gathered ? 0u : out_row;

    const device uint8_t* weights = reinterpret_cast<const device uint8_t*>(b);
    weights += base_row * weights_row_stride + (k_start / pack_factor + simd_lane * packs_per_thread) * bytes_per_pack;

    RowState row_state(scales, zero_points, biases, base_row, group_count, group_offset);

    const device AT* input = a + batch_base * in_vec_size + k_start + simd_lane * values_per_thread;
    thread MT input_values[M_TILE * values_per_thread];
    thread U input_sums[M_TILE];

    uint k = k_start;
    for (; k + block_size <= in_vec_size; k += k_stride) {
      METAL_PRAGMA_UNROLL
      for (uint bt = 0; bt < M_TILE; bt++) {
        input_sums[bt] = load_vector<AT, U, MT, values_per_thread, BITS>(
            input + bt * in_vec_size,
            input_values + bt * values_per_thread,
            signed_codes
        );
      }

      RowParams row_params;
      row_state.load(row_params, gather_indices, gathered, batch_base, out_vec_size, out_row);
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        const uint addr_row = gathered ? gather_indices[batch_base * out_vec_size + out_row + row] : row;
        const device uint8_t* weight_row = weights + addr_row * weights_row_stride;
        if constexpr (M_TILE == 1) {
          result[row] += qdot<U, MT, values_per_thread, BITS>(
              weight_row,
              input_values,
              row_params.scale[row],
              row_params.offset[row],
              input_sums[0],
              signed_codes
          );
        } else {
          qdot_batched<U, MT, M_TILE, RESULTS_PER_SIMDGROUP, values_per_thread, BITS>(
              weight_row,
              input_values,
              input_sums,
              row_params.scale[row],
              row_params.offset[row],
              result + row,
              signed_codes
          );
        }
      }

      weights += k_stride * bytes_per_pack / pack_factor;
      row_state.advance(k_stride / GROUP_SIZE);
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
        for (uint bt = 0; bt < M_TILE; bt++) {
          input_sums[bt] = load_vector_safe<AT, U, MT, values_per_thread>(
              input + bt * in_vec_size,
              input_values + bt * values_per_thread,
              remaining
          );
        }

        RowParams row_params;
        row_state.load(row_params, gather_indices, gathered, batch_base, out_vec_size, out_row);
        METAL_PRAGMA_UNROLL
        for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          const uint addr_row = gathered ? gather_indices[batch_base * out_vec_size + out_row + row] : row;
          const device uint8_t* weight_row = weights + addr_row * weights_row_stride;
          METAL_PRAGMA_UNROLL
          for (uint bt = 0; bt < M_TILE; bt++) {
            result[bt * RESULTS_PER_SIMDGROUP + row] += qdot_safe<U, MT, values_per_thread, BITS>(
                weight_row,
                input_values + bt * values_per_thread,
                row_params.scale[row],
                row_params.offset[row],
                input_sums[bt],
                remaining,
                signed_codes
            );
          }
        }
      }
    }
  }
};

} // namespace gemm
} // namespace uzu
