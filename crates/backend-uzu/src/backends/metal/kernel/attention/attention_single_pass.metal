#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/ring.h"
#include "../generated/trie.h"
#include "mask.h"

#define SEQUENCE_BLOCK_SIZE 32
#define HEAD_BLOCK_SIZE 32

using namespace uzu::ring;
using namespace uzu::trie;

template <typename T, uint HEAD_DIM>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_DIM, 64, 128, 256, 512)
PUBLIC KERNEL(AttentionSinglePass)(
    const device T* queries,
    const device T* keys OPTIONAL(!kv_int8),
    const device T* values OPTIONAL(!kv_int8),
    const device char* keys_q8 OPTIONAL(kv_int8),
    const device char* values_q8 OPTIONAL(kv_int8),
    const device float* key_scales OPTIONAL(kv_int8),
    const device float* value_scales OPTIONAL(kv_int8),
    const constant uint& num_kv_heads OPTIONAL(kv_int8),
    device T* out,
    const constant uint& gqa_factor,
    const constant uint& sequence_length,
    const constant uint& k_head_stride,
    const constant uint& k_seq_stride,
    const constant uint& v_head_stride,
    const constant uint& v_seq_stride,
    const constant RingParams& ring_params OPTIONAL(is_kv_cache_ring),
    const constant float& scale,
    const device TrieNode* trie OPTIONAL(is_trie),
    const constant uint& sliding_window_size OPTIONAL(is_sliding_window),
    const device T* sinks OPTIONAL(has_sinks),
    const constant uint& num_heads,
    const constant uint& suffix_length,
    threadgroup float shared_max_scores[SEQUENCE_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    threadgroup float shared_sum_exp_scores[SEQUENCE_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    threadgroup float shared_outputs[SEQUENCE_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    const bool has_sinks SPECIALIZE,
    const bool kv_int8 SPECIALIZE,
    const bool is_kv_cache_ring SPECIALIZE,
    const bool is_causal SPECIALIZE,
    const bool is_trie SPECIALIZE,
    const bool is_sliding_window SPECIALIZE,
    const ThreadContext thread_context,
    const uint head_idx GROUPS(num_heads),
    const uint q_seq_idx GROUPS(suffix_length),
    const uint tid THREADS(1024)
) {
  constexpr bool query_transposed = false;

  constexpr uint value_dim = HEAD_DIM;
  constexpr uint qk_elements_per_thread = HEAD_DIM / HEAD_BLOCK_SIZE;
  constexpr uint value_elements_per_thread = value_dim / HEAD_BLOCK_SIZE;
  // Width 4 pays off, width 2 measurably regresses, so narrower head dims stay scalar via width 1.
  // KV strides are head_dim multiples (core/single_pass.rs), which is what makes the wide load aligned.
  constexpr uint LOAD_WIDTH = qk_elements_per_thread >= 4 ? 4 : 1;
  constexpr uint KV_LOAD_COUNT = qk_elements_per_thread / LOAD_WIDTH;
  static_assert(HEAD_DIM % (HEAD_BLOCK_SIZE * LOAD_WIDTH) == 0, "head-dim must split into aligned KV vectors");
  typedef vec<T, LOAD_WIDTH> KVVec;
  uint inner_k_stride = SEQUENCE_BLOCK_SIZE * int(k_seq_stride);
  uint inner_v_stride = SEQUENCE_BLOCK_SIZE * int(v_seq_stride);

  typedef float U;

  thread U q[qk_elements_per_thread];
  thread U o[value_elements_per_thread];

  const uint kv_head_idx = head_idx / gqa_factor;
  const uint o_offset = q_seq_idx * num_heads + head_idx;
  const uint q_offset = query_transposed ? num_heads * q_seq_idx + head_idx : head_idx * suffix_length + q_seq_idx;

  const uint prefix_length = sequence_length - suffix_length;

  const uint suffix_position = is_kv_cache_ring ? uint(ring_params.ring_length) : prefix_length;

  const uint query_position = is_trie ? suffix_position + trie[q_seq_idx].height : suffix_position + q_seq_idx;

  queries += q_offset * HEAD_DIM + thread_context.simd_lane_id * qk_elements_per_thread;
  if (kv_int8) {
    keys_q8 += kv_head_idx * k_head_stride + thread_context.simdgroup_index * k_seq_stride +
               thread_context.simd_lane_id * qk_elements_per_thread;
    values_q8 += kv_head_idx * v_head_stride + thread_context.simdgroup_index * v_seq_stride +
                 thread_context.simd_lane_id * value_elements_per_thread;
    key_scales += kv_head_idx;
    value_scales += kv_head_idx;
  } else {
    keys += kv_head_idx * k_head_stride + thread_context.simdgroup_index * k_seq_stride +
            thread_context.simd_lane_id * qk_elements_per_thread;
    values += kv_head_idx * v_head_stride + thread_context.simdgroup_index * v_seq_stride +
              thread_context.simd_lane_id * value_elements_per_thread;
  }

  out += o_offset * value_dim + thread_context.simdgroup_index * value_elements_per_thread;

  // Read the query and 0 the output accumulator
  for (uint i = 0; i < qk_elements_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (uint i = 0; i < value_elements_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;
  if (has_sinks && thread_context.simdgroup_index == 0) {
    const int num_q_heads = static_cast<int>(num_heads);
    int q_head_idx = head_idx % num_q_heads;
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  // For each key
  for (uint i = thread_context.simdgroup_index; i < sequence_length; i += SEQUENCE_BLOCK_SIZE) {
    if (should_use_key(
            ring_params,
            trie,
            sliding_window_size,
            q_seq_idx,
            prefix_length,
            suffix_position,
            query_position,
            i,
            is_kv_cache_ring,
            is_causal,
            is_trie,
            is_sliding_window
        )) {
      // Compute the i-th score straight off the loaded vector
      U score = 0;
      if (kv_int8) {
        METAL_PRAGMA_UNROLL
        for (uint jv = 0; jv < KV_LOAD_COUNT; jv++) {
          const vec<char, LOAD_WIDTH> key_vec = reinterpret_cast<const device vec<char, LOAD_WIDTH>*>(keys_q8)[jv];
          for (uint c = 0; c < LOAD_WIDTH; c++) {
            score += q[jv * LOAD_WIDTH + c] * static_cast<U>(key_vec[c]);
          }
        }
        score *= key_scales[i * num_kv_heads];
      } else {
        METAL_PRAGMA_UNROLL
        for (uint jv = 0; jv < KV_LOAD_COUNT; jv++) {
          const KVVec key_vec = reinterpret_cast<const device KVVec*>(keys)[jv];
          for (uint c = 0; c < LOAD_WIDTH; c++) {
            score += q[jv * LOAD_WIDTH + c] * static_cast<U>(key_vec[c]);
          }
        }
      }
      score = simd_sum(score);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Accumulate straight off the loaded vector
      if (kv_int8) {
        const U weighted = exp_score * static_cast<U>(value_scales[i * num_kv_heads]);
        METAL_PRAGMA_UNROLL
        for (uint jv = 0; jv < KV_LOAD_COUNT; jv++) {
          const vec<char, LOAD_WIDTH> value_vec =
              reinterpret_cast<const device vec<char, LOAD_WIDTH>*>(values_q8)[jv];
          METAL_PRAGMA_UNROLL
          for (uint c = 0; c < LOAD_WIDTH; c++) {
            const uint j = jv * LOAD_WIDTH + c;
            o[j] = o[j] * factor + weighted * static_cast<U>(value_vec[c]);
          }
        }
      } else {
        METAL_PRAGMA_UNROLL
        for (uint jv = 0; jv < KV_LOAD_COUNT; jv++) {
          const KVVec value_vec = reinterpret_cast<const device KVVec*>(values)[jv];
          METAL_PRAGMA_UNROLL
          for (uint c = 0; c < LOAD_WIDTH; c++) {
            const uint j = jv * LOAD_WIDTH + c;
            o[j] = o[j] * factor + exp_score * static_cast<U>(value_vec[c]);
          }
        }
      }
    }

    // Move the pointers to the next kv
    if (kv_int8) {
      keys_q8 += inner_k_stride;
      values_q8 += inner_v_stride;
    } else {
      keys += inner_k_stride;
      values += inner_v_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.
  if (thread_context.simd_lane_id == 0) {
    shared_max_scores[thread_context.simdgroup_index] = max_score;
    shared_sum_exp_scores[thread_context.simdgroup_index] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = shared_max_scores[thread_context.simd_lane_id];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(shared_sum_exp_scores[thread_context.simd_lane_id] * factor);

  // Now we need to aggregate all the outputs
  for (uint i = 0; i < value_elements_per_thread; i++) {
    shared_outputs[thread_context.simd_lane_id * HEAD_BLOCK_SIZE + thread_context.simdgroup_index] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(
               shared_outputs[thread_context.simdgroup_index * HEAD_BLOCK_SIZE + thread_context.simd_lane_id] * factor
           ) /
           sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (thread_context.simd_lane_id == 0) {
    for (uint i = 0; i < value_elements_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
