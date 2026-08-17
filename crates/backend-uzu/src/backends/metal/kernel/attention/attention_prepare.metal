#include "../common/defines.h"
#include "../common/dsl.h"

// TODO: very ugly and wasteful, clean up and optimize

template <typename ElementT, typename RopeT>
METAL_FUNC ElementT apply_rope(
    const device ElementT* head,
    const device RopeT* cosines,
    const device RopeT* sines,
    uint32_t batch_idx,
    uint32_t head_dim_idx,
    uint32_t rope_dim,
    ElementT loaded_element
) {
  const uint32_t half_rope_dim = rope_dim / 2;
  const uint32_t paired_idx =
      head_dim_idx < half_rope_dim ? head_dim_idx + half_rope_dim : head_dim_idx - half_rope_dim;
  // Caller already read this element; only the paired one still needs a load.
  const float input = float(loaded_element);
  const float paired = float(head[paired_idx]);
  const float signed_paired = head_dim_idx < half_rope_dim ? -paired : paired;
  const float cos_val = float(cosines[batch_idx * rope_dim + head_dim_idx]);
  const float sin_val = float(sines[batch_idx * rope_dim + head_dim_idx]);

  return static_cast<ElementT>(input * cos_val + signed_paired * sin_val);
}

// Symmetric int8 with one absmax scale per (token, kv head); matches the
// decode-side dequant `float(q) * scale`.
#define PREPARE_MAX_SIMDGROUPS 8

template <typename ElementT, typename RopeT>
VARIANTS(ElementT, bfloat)
VARIANTS(RopeT, float)
PUBLIC KERNEL(AttentionPrepare) (
  const device ElementT* qkv, // [token, (q, k, v), head_dim]
  device ElementT* queries, // [head_idx, token, head_dim]
  device ElementT* keys OPTIONAL(has_kv && !kv_int8), // [(kv_token_offset + token), head_idx, head_dim]
  device ElementT* values OPTIONAL(has_kv && !kv_int8), // [(kv_token_offset + token), head_idx, head_dim]
  device char* keys_q8 OPTIONAL(has_kv && kv_int8),
  device char* values_q8 OPTIONAL(has_kv && kv_int8),
  device float* key_scales OPTIONAL(has_kv && kv_int8), // [(kv_token_offset + token), head_idx]
  device float* value_scales OPTIONAL(has_kv && kv_int8),
  const device RopeT* cosines OPTIONAL(has_rope),
  const device RopeT* sines OPTIONAL(has_rope),
  const constant uint32_t& num_q_heads,
  const constant uint32_t& num_kv_heads OPTIONAL(has_kv),
  const constant uint32_t& head_dim,
  const constant uint32_t& rope_dim OPTIONAL(has_rope),
  const constant uint32_t& kv_token_offset OPTIONAL(has_kv),
  const constant uint32_t& batch_dim,
  const bool has_kv SPECIALIZE,
  const bool has_rope SPECIALIZE,
  const bool kv_int8 SPECIALIZE,
  threadgroup float shared_absmax[PREPARE_MAX_SIMDGROUPS],
  const uint32_t head_dim_idx AXIS(head_dim, 256),
  const uint32_t head_idx AXIS(num_q_heads + num_kv_heads.unwrap_or(0) * 2, 1),
  const uint32_t batch_idx AXIS(batch_dim, 1)
) {
  const uint32_t total_heads = has_kv ? num_q_heads + num_kv_heads * 2 : num_q_heads;
  const uint32_t qkv_head_idx = batch_idx * total_heads * head_dim + head_idx * head_dim;
  const device ElementT* qkv_head = qkv + qkv_head_idx;
  const bool is_query = !has_kv || head_idx < num_q_heads;
  const bool is_key = has_kv && head_idx >= num_q_heads && head_idx < num_q_heads + num_kv_heads;

  ElementT element = qkv_head[head_dim_idx];
  if (has_rope && head_dim_idx < rope_dim && (is_query || is_key)) {
    element = apply_rope(qkv_head, cosines, sines, batch_idx, head_dim_idx, rope_dim, element);
  }

  if (is_query) {
    const uint32_t q_idx = head_idx * batch_dim * head_dim + batch_idx * head_dim + head_dim_idx;
    queries[q_idx] = element;
    return;
  }
  if (!has_kv) {
    return;
  }

  const uint32_t kv_head = is_key ? head_idx - num_q_heads : head_idx - num_q_heads - num_kv_heads;
  const uint32_t kv_row = (kv_token_offset + batch_idx) * num_kv_heads;
  const uint32_t kv_idx = (kv_row + kv_head) * head_dim + head_dim_idx;

  if (!kv_int8) {
    if (is_key) {
      keys[kv_idx] = element;
    } else {
      values[kv_idx] = element;
    }
    return;
  }

  // The threadgroup covers head_dim positions of one (token, head), so the
  // absmax reduces within it: simd first, then across the simdgroups. Head
  // dims above the 256-thread group are rejected host-side.
  const uint32_t lane = head_dim_idx % 32;
  const uint32_t simdgroup = head_dim_idx / 32;
  const uint32_t simdgroups = (head_dim + 31) / 32;
  float absmax = metal::simd_max(metal::fabs(float(element)));
  if (lane == 0) {
    shared_absmax[simdgroup] = absmax;
  }
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  absmax = shared_absmax[0];
  for (uint32_t group = 1; group < simdgroups; group++) {
    absmax = metal::max(absmax, shared_absmax[group]);
  }

  const float scale = metal::max(absmax, 1e-8f) / INT8_QMAX;
  const char quantized =
      char(metal::clamp(metal::rint(float(element) / scale), -INT8_QMAX, INT8_QMAX));
  if (is_key) {
    keys_q8[kv_idx] = quantized;
    if (head_dim_idx == 0) {
      key_scales[kv_row + kv_head] = scale;
    }
  } else {
    values_q8[kv_idx] = quantized;
    if (head_dim_idx == 0) {
      value_scales[kv_row + kv_head] = scale;
    }
  }
}
