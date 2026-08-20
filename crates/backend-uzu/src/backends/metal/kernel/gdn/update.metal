#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;

#define UPDATE_THREADS 512

// One simd group per output dim dv; 32 lanes cover Dk (coalesced state IO,
// reduction via simd_sum). Activations are model dtype T; state stays float.
// The grid splits each head's dv range over dv_blocks threadgroups so decode
// fills the GPU (num_v_heads alone leaves most cores idle); RMSNorm + gate
// run in the separate DeltaNetNormGate pass, which is latency-hidden.
template <typename T, uint HEAD_K_DIM>
VARIANTS(T, float, bfloat)
VARIANTS(HEAD_K_DIM, 128)
PUBLIC KERNEL(DeltaNetUpdate)(
    device const T* in_proj,
    device const float* a_log,
    device const float* dt_bias,
    device float* state,
    device T* out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& dv_blocks,
    const uint dv_block_idx GROUPS(dv_blocks),
    const uint hv_idx GROUPS(num_v_heads),
    const uint tid THREADS(UPDATE_THREADS)
) {
  static_assert(HEAD_K_DIM % METAL_SIMD_SIZE == 0, "HEAD_K_DIM must be a multiple of METAL_SIMD_SIZE");
  constexpr uint ELEMS = HEAD_K_DIM / METAL_SIMD_SIZE;      // Dk per lane (128/32 = 4)
  constexpr uint NUM_SG = UPDATE_THREADS / METAL_SIMD_SIZE; // simd groups / tg (16)

  const uint lane = tid % METAL_SIMD_SIZE;
  const uint sg = tid / METAL_SIMD_SIZE;

  const uint conv_dim = 2 * key_dim + value_dim;
  const uint hk = hv_idx / (num_v_heads / num_k_heads);

  // Load + normalize q/k; each simd group recomputes the norm to skip a
  // barrier.
  float q[ELEMS];
  float k[ELEMS];
  float q_sq = 0.0f;
  float k_sq = 0.0f;
  for (uint i = 0; i < ELEMS; ++i) {
    const uint dk = lane + METAL_SIMD_SIZE * i;
    q[i] = float(in_proj[hk * HEAD_K_DIM + dk]);
    k[i] = float(in_proj[key_dim + hk * HEAD_K_DIM + dk]);
    q_sq += q[i] * q[i];
    k_sq += k[i] * k[i];
  }
  const float q_inv_norm = rsqrt(simd_sum(q_sq) + 1e-6f);
  const float k_inv_norm = rsqrt(simd_sum(k_sq) + 1e-6f);
  const float q_scale = rsqrt(float(HEAD_K_DIM));
  float kq_partial = 0.0f;
  for (uint i = 0; i < ELEMS; ++i) {
    q[i] *= q_inv_norm * q_scale;
    k[i] *= k_inv_norm;
    kq_partial += q[i] * k[i];
  }
  const float kq_dot = simd_sum(kq_partial);

  // beta / decay (scalar per head)
  const float beta_raw = float(in_proj[conv_dim + value_dim + hv_idx]);
  const float beta = 1.0f / (1.0f + fast::exp(-beta_raw));
  const float a_raw = float(in_proj[conv_dim + value_dim + num_v_heads + hv_idx]);
  const float sp = activate_softplus(a_raw + float(dt_bias[hv_idx]));
  const float decay = fast::exp(-fast::exp(float(a_log[hv_idx])) * sp);

  // Delta rule over the dv owned by this simd group. State is [Hv, Dv, Dk].
  // dv_blocks need not divide head_v_dim: rounding the span up and clamping
  // the end keeps every dv owned by exactly one block, where truncating
  // division would leave the tail unwritten.
  const uint dv_span = (head_v_dim + dv_blocks - 1) / dv_blocks;
  const uint dv_base = dv_block_idx * dv_span;
  const uint dv_end = min(dv_base + dv_span, head_v_dim);
  for (uint dv = dv_base + sg; dv < dv_end; dv += NUM_SG) {
    const uint state_row = (hv_idx * head_v_dim + dv) * HEAD_K_DIM;
    const float v_i = float(in_proj[2 * key_dim + hv_idx * head_v_dim + dv]);

    float s[ELEMS];
    float sq_partial = 0.0f;
    float sk_partial = 0.0f;
    for (uint i = 0; i < ELEMS; ++i) {
      const uint dk = lane + METAL_SIMD_SIZE * i;
      s[i] = state[state_row + dk];
      sq_partial += s[i] * q[i];
      sk_partial += s[i] * k[i];
    }
    const float sq_acc = simd_sum(sq_partial);
    const float sk_acc = simd_sum(sk_partial);

    const float retrieved_i = decay * sk_acc;
    const float delta_i = beta * (v_i - retrieved_i);
    const float o_i = decay * sq_acc + delta_i * kq_dot;

    for (uint i = 0; i < ELEMS; ++i) {
      const uint dk = lane + METAL_SIMD_SIZE * i;
      state[state_row + dk] = decay * s[i] + k[i] * delta_i;
    }

    if (lane == 0) {
      out[hv_idx * head_v_dim + dv] = static_cast<T>(o_i);
    }
  }
}
