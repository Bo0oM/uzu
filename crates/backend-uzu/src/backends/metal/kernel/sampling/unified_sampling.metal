#include "../common/threadgroup_reduce.h"
#include "../common/thread_context.h"
#include "../common/defines.h"
#include "../common/dsl.h"

#include "../rng.h"

#define THREADGROUP_SIZE 1024
#define THREADGROUP_SIZE_IN_SIMDS (THREADGROUP_SIZE / METAL_SIMD_SIZE)
#define BITS_IN_U32 32

#define MAX_ITERS 64

// The pipeline carries token indices and counts between its stages on the same
// float4 lanes as the logit values. They used to travel as bit patterns, which
// made every index below 2^23 a denormal and the inactive-lane marker a NaN
// payload — correct only for as long as nothing on the path flushes a denormal
// or canonicalises a NaN, and wrong by a silently different token if anything
// ever does.
//
// A float holds every integer below 2^24 exactly, and a vocabulary is far
// smaller than that, so the value itself rides the lane and the round trip is
// exact by arithmetic rather than by bit preservation. The marker is a
// negative number, which no count or index can be.
constant float INACTIVE_LANE = -1.0f;

static inline float lane_from_index(uint32_t index) {
  return float(index);
}

static inline uint32_t index_from_lane(float lane) {
  return uint32_t(metal::max(lane, 0.0f));
}


// The vocab is walked in units of the shared gumbel noise block (rng.h), so
// the sliced kernels below and the serial tail all draw identical noise.
#define ELEMS_PER_NOISE_BLOCK GUMBEL_ELEMS_PER_NOISE_BLOCK

struct Logit {
  float value;
  uint32_t index;

  static const constant Logit LOWEST;

  template <typename T>
  static inline Logit load(const device T* logits, uint32_t index) {
    return {.value = float(logits[index]), .index = index};
  }

  inline bool operator>(Logit rhs) const { return value > rhs.value || (value == rhs.value && index < rhs.index); }
};

constexpr constant Logit
    Logit::LOWEST{.value = -numeric_limits<float>::infinity(), .index = numeric_limits<uint32_t>::max()};

struct SimdReduceMaxLogit {
  using value_type = Logit;
  static constant constexpr Logit identity = Logit::LOWEST;

  static Logit simd_reduce(Logit x) {
    METAL_PRAGMA_UNROLL
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      Logit y = {
          .value = simd_shuffle_xor(x.value, offset),
          .index = simd_shuffle_xor(x.index, offset),
      };
      x = y > x ? y : x;
    }
    return x;
  }
};

// Slice bounds in noise-block units; the last slice absorbs the remainder.
static inline uint2 slice_noise_blocks(uint32_t slice_idx, uint32_t num_slices, uint32_t vocab_size) {
  const uint32_t blocks = div_ceil(vocab_size, ELEMS_PER_NOISE_BLOCK);
  const uint32_t per_slice = div_ceil(blocks, num_slices);
  const uint32_t lo = metal::min(slice_idx * per_slice, blocks);
  return uint2(lo, metal::min(lo + per_slice, blocks));
}

struct AboveCandidateStats {
  uint32_t num_above;
  float mass_above;
  Logit next_candidate_post_gumbel;
};

// One rejection-loop iteration over a block range: how much sits above the
// candidate in pre-filter order, and the next post-gumbel candidate among it,
// reduced across the threadgroup. The flag arguments are function constants
// at every call site, so the dead branches still fold away.
template <typename T>
METAL_FUNC AboveCandidateStats scan_above_candidate(
    const device T* logits,
    const device uint32_t* bitmask,
    uint64_t rng_seed,
    float recip_temperature,
    Logit candidate_logit_pre_filter,
    float pre_filter_logit_max,
    float pre_filter_logit_norm,
    uint2 blocks,
    uint32_t vocab_size,
    bool is_stochastic,
    bool has_bitmask,
    bool has_temperature,
    bool has_top_k,
    bool has_top_p,
    threadgroup Logit* shared,
    const thread ThreadContext& thread_context,
    uint32_t thread_idx
) {
  uint32_t thread_num_above = 0;
  float thread_mass_above = 0.0;
  Logit thread_next_candidate = Logit::LOWEST;
  for (uint32_t block = blocks.x + thread_idx; block < blocks.y; block += THREADGROUP_SIZE) {
    PhiloxState rng;
    if (is_stochastic) {
      philox_init(&rng, rng_seed, block);
    }
    METAL_PRAGMA_UNROLL
    for (uint32_t word = 0; word < ELEMS_PER_NOISE_BLOCK; word++) {
      const uint32_t logit_index = block * ELEMS_PER_NOISE_BLOCK + word;
      if (logit_index >= vocab_size) {
        break;
      }
      Logit logit = Logit::load(logits, logit_index);

      if (has_bitmask) {
        bool mask = (bitmask[logit_index / BITS_IN_U32] >> (logit_index % BITS_IN_U32)) & 0b1;
        logit.value = mask ? logit.value : -INFINITY;
      }

      if (has_temperature) {
        logit.value *= recip_temperature;
      }

      bool above_current_pre_filter = logit > candidate_logit_pre_filter;

      if (above_current_pre_filter) {
        if (has_top_k) {
          thread_num_above += 1;
        }
        if (has_top_p) {
          thread_mass_above += exp(logit.value - pre_filter_logit_max) / pre_filter_logit_norm;
        }
      }

      if (is_stochastic) {
        logit.value += -log(-log(uniform_float(&rng)));
      }

      if (above_current_pre_filter) {
        thread_next_candidate = logit > thread_next_candidate ? logit : thread_next_candidate;
      }
    }
  }

  AboveCandidateStats stats;
  stats.num_above = 0;
  if (has_top_k) {
    stats.num_above = threadgroup_cooperative_reduce<SimdReduceSum<uint32_t>, THREADGROUP_SIZE>(
        thread_num_above,
        (threadgroup uint32_t*)shared,
        thread_context
    );
  }
  stats.mass_above = 0.0;
  if (has_top_p) {
    stats.mass_above = threadgroup_cooperative_reduce<SimdReduceSum<float>, THREADGROUP_SIZE>(
        thread_mass_above,
        (threadgroup float*)shared,
        thread_context
    );
  }
  stats.next_candidate_post_gumbel = threadgroup_cooperative_reduce<SimdReduceMaxLogit, THREADGROUP_SIZE>(
      thread_next_candidate,
      shared,
      thread_context
  );
  return stats;
}

// Per-slice pass 1: masked/tempered logit stats and the post-gumbel argmax.
// partials[batch * num_slices + slice] = (pre-filter max, pre-filter norm
// relative to that max, best post-gumbel value, best index bits).
template <typename T>
VARIANTS(T, float, bfloat)
KERNEL(SamplingPartialScan)(
  const device T* logits,
  device float4* partials,
  const device uint64_t* seeds OPTIONAL(is_stochastic),
  const device uint32_t* bitmask OPTIONAL(has_bitmask),
  const constant float& temperature OPTIONAL(has_temperature),
  const constant uint32_t& vocab_size,
  const constant uint32_t& batch_size,
  const constant uint32_t& num_slices,
  const bool is_stochastic SPECIALIZE,
  const bool has_bitmask SPECIALIZE,
  const bool has_temperature SPECIALIZE,
  const bool has_top_p SPECIALIZE,
  const bool has_min_p SPECIALIZE,
  threadgroup Logit shared[THREADGROUP_SIZE_IN_SIMDS],
  const ThreadContext thread_context,
  uint batch_idx GROUPS(batch_size),
  uint slice_idx GROUPS(num_slices),
  uint thread_idx THREADS(THREADGROUP_SIZE)
) {
  logits += vocab_size * batch_idx;
  uint64_t rng_seed = 0;
  if (is_stochastic) {
    rng_seed = seeds[batch_idx];
  }
  if (has_bitmask) {
    bitmask += div_ceil(vocab_size, BITS_IN_U32) * batch_idx;
  }
  float recip_temperature = 0.0;
  if (has_temperature) {
    recip_temperature = 1.0 / temperature;
  }

  const uint2 blocks = slice_noise_blocks(slice_idx, num_slices, vocab_size);
  float thread_pre_filter_logit_max = -INFINITY;
  float thread_pre_filter_logit_norm = 0.0;
  Logit thread_post_gumbel_logit_max = Logit::LOWEST;
  for (uint32_t block = blocks.x + thread_idx; block < blocks.y; block += THREADGROUP_SIZE) {
    PhiloxState rng;
    if (is_stochastic) {
      philox_init(&rng, rng_seed, block);
    }
    METAL_PRAGMA_UNROLL
    for (uint32_t word = 0; word < ELEMS_PER_NOISE_BLOCK; word++) {
      const uint32_t logit_index = block * ELEMS_PER_NOISE_BLOCK + word;
      if (logit_index >= vocab_size) {
        break;
      }
      Logit logit = Logit::load(logits, logit_index);

      if (has_bitmask) {
        bool mask = (bitmask[logit_index / BITS_IN_U32] >> (logit_index % BITS_IN_U32)) & 0b1;
        logit.value = mask ? logit.value : -INFINITY;
      }

      if (has_temperature) {
        logit.value *= recip_temperature;
      }

      if (has_top_p && logit.value != -INFINITY) {
        float new_thread_pre_filter_logit_max = max(thread_pre_filter_logit_max, logit.value);

        thread_pre_filter_logit_norm =
            thread_pre_filter_logit_norm * exp(thread_pre_filter_logit_max - new_thread_pre_filter_logit_max) +
            exp(logit.value - new_thread_pre_filter_logit_max);

        thread_pre_filter_logit_max = new_thread_pre_filter_logit_max;
      } else if (has_min_p) {
        thread_pre_filter_logit_max = max(thread_pre_filter_logit_max, logit.value);
      }

      if (is_stochastic) {
        logit.value += -log(-log(uniform_float(&rng)));
      }

      thread_post_gumbel_logit_max = logit > thread_post_gumbel_logit_max ? logit : thread_post_gumbel_logit_max;
    }
  }

  float pre_filter_logit_max = -INFINITY;
  if (has_top_p || has_min_p) {
    pre_filter_logit_max = threadgroup_cooperative_reduce<SimdReduceMax<float>, THREADGROUP_SIZE>(
        thread_pre_filter_logit_max,
        (threadgroup float*)shared,
        thread_context
    );
  }
  float pre_filter_logit_norm = 0.0;
  if (has_top_p) {
    if (thread_pre_filter_logit_norm != 0.0) {
      thread_pre_filter_logit_norm *= exp(thread_pre_filter_logit_max - pre_filter_logit_max);
    }
    pre_filter_logit_norm = threadgroup_cooperative_reduce<SimdReduceSum<float>, THREADGROUP_SIZE>(
        thread_pre_filter_logit_norm,
        (threadgroup float*)shared,
        thread_context
    );
  }
  Logit post_gumbel_logit_max = threadgroup_cooperative_reduce<SimdReduceMaxLogit, THREADGROUP_SIZE>(
      thread_post_gumbel_logit_max,
      shared,
      thread_context
  );

  if (thread_idx == 0) {
    partials[batch_idx * num_slices + slice_idx] = float4(
        pre_filter_logit_max,
        pre_filter_logit_norm,
        post_gumbel_logit_max.value,
        lane_from_index(post_gumbel_logit_max.index)
    );
  }
}

// Combine per-slice partials. Without filters the argmax is the sample;
// with filters the candidate and the softmax stats are staged for the loop
// kernels. state[batch] = (candidate value, candidate index bits, pre-filter
// max, pre-filter norm).
KERNEL(SamplingCombine)(
  const device float4* partials,
  device uint32_t* output,
  device float4* state,
  const constant uint32_t& batch_size,
  const constant uint32_t& num_slices,
  const bool has_filters SPECIALIZE,
  const bool has_top_p SPECIALIZE,
  uint batch_idx GROUPS(batch_size),
  uint lane THREADS(32)
) {
  const bool active = lane < num_slices;
  const float4 partial = active
      ? partials[batch_idx * num_slices + lane]
      : float4(-INFINITY, 0.0, -INFINITY, INACTIVE_LANE);

  const float pre_filter_logit_max = simd_max(partial.x);
  float pre_filter_logit_norm = 0.0;
  if (has_top_p) {
    pre_filter_logit_norm = simd_sum(partial.y != 0.0 ? partial.y * exp(partial.x - pre_filter_logit_max) : 0.0);
  }
  const Logit candidate = SimdReduceMaxLogit::simd_reduce(Logit{partial.z, index_from_lane(partial.w)});

  if (lane == 0) {
    if (has_filters) {
      state[batch_idx] =
          float4(candidate.value, lane_from_index(candidate.index), pre_filter_logit_max, pre_filter_logit_norm);
    } else {
      output[batch_idx] = candidate.index;
    }
  }
}

// Per-slice stats of the first rejection-loop iteration: how much sits above
// the candidate in pre-filter order, and the next post-gumbel candidate among
// it. partials[batch * num_slices + slice] = (count bits, mass, next value,
// next index bits).
template <typename T>
VARIANTS(T, float, bfloat)
KERNEL(SamplingLoopPartial)(
  const device T* logits,
  const device float4* state,
  device float4* partials,
  const device uint64_t* seeds OPTIONAL(is_stochastic),
  const device uint32_t* bitmask OPTIONAL(has_bitmask),
  const constant float& temperature OPTIONAL(has_temperature),
  const constant uint32_t& vocab_size,
  const constant uint32_t& batch_size,
  const constant uint32_t& num_slices,
  const bool is_stochastic SPECIALIZE,
  const bool has_bitmask SPECIALIZE,
  const bool has_temperature SPECIALIZE,
  const bool has_top_k SPECIALIZE,
  const bool has_top_p SPECIALIZE,
  threadgroup Logit shared[THREADGROUP_SIZE_IN_SIMDS],
  const ThreadContext thread_context,
  uint batch_idx GROUPS(batch_size),
  uint slice_idx GROUPS(num_slices),
  uint thread_idx THREADS(THREADGROUP_SIZE)
) {
  logits += vocab_size * batch_idx;
  uint64_t rng_seed = 0;
  if (is_stochastic) {
    rng_seed = seeds[batch_idx];
  }
  if (has_bitmask) {
    bitmask += div_ceil(vocab_size, BITS_IN_U32) * batch_idx;
  }
  float recip_temperature = 0.0;
  if (has_temperature) {
    recip_temperature = 1.0 / temperature;
  }

  const float4 batch_state = state[batch_idx];
  Logit candidate_logit_pre_filter = Logit::load(logits, index_from_lane(batch_state.y));
  if (has_temperature) {
    candidate_logit_pre_filter.value *= recip_temperature;
  }
  const float pre_filter_logit_max = batch_state.z;
  const float pre_filter_logit_norm = batch_state.w;

  const AboveCandidateStats stats = scan_above_candidate(
      logits,
      bitmask,
      rng_seed,
      recip_temperature,
      candidate_logit_pre_filter,
      pre_filter_logit_max,
      pre_filter_logit_norm,
      slice_noise_blocks(slice_idx, num_slices, vocab_size),
      vocab_size,
      is_stochastic,
      has_bitmask,
      has_temperature,
      has_top_k,
      has_top_p,
      shared,
      thread_context,
      thread_idx
  );

  if (thread_idx == 0) {
    partials[batch_idx * num_slices + slice_idx] = float4(
        float(stats.num_above),
        stats.mass_above,
        stats.next_candidate_post_gumbel.value,
        lane_from_index(stats.next_candidate_post_gumbel.index)
    );
  }
}

// Combine the loop partials, decide iteration 0, and — in the rare case the
// first candidate is rejected — run the remaining rejection-loop iterations
// serially over the vocab, exactly as the pre-split kernel did.
template <typename T>
VARIANTS(T, float, bfloat)
KERNEL(SamplingFinalize)(
  const device T* logits,
  device uint32_t* output,
  const device float4* loop_partials,
  const device float4* state,
  const device uint64_t* seeds OPTIONAL(is_stochastic),
  const device uint32_t* bitmask OPTIONAL(has_bitmask),
  const constant float& temperature OPTIONAL(has_temperature),
  const constant uint32_t& top_k OPTIONAL(has_top_k),
  const constant float& top_p OPTIONAL(has_top_p),
  const constant float& min_p OPTIONAL(has_min_p),
  const constant uint32_t& vocab_size,
  const constant uint32_t& batch_size,
  const constant uint32_t& num_slices,
  const bool is_stochastic SPECIALIZE,
  const bool has_bitmask SPECIALIZE,
  const bool has_temperature SPECIALIZE,
  const bool has_top_k SPECIALIZE,
  const bool has_top_p SPECIALIZE,
  const bool has_min_p SPECIALIZE,
  threadgroup Logit shared[THREADGROUP_SIZE_IN_SIMDS],
  const ThreadContext thread_context,
  uint batch_idx GROUPS(batch_size),
  uint thread_idx THREADS(THREADGROUP_SIZE)
) {
  logits += vocab_size * batch_idx;
  output += batch_idx;
  uint64_t rng_seed = 0;
  if (is_stochastic) {
    rng_seed = seeds[batch_idx];
  }
  if (has_bitmask) {
    bitmask += div_ceil(vocab_size, BITS_IN_U32) * batch_idx;
  }
  float recip_temperature = 0.0;
  if (has_temperature) {
    recip_temperature = 1.0 / temperature;
  }
  float log_min_p;
  if (has_min_p) {
    log_min_p = log(min_p);
  }

  const float4 batch_state = state[batch_idx];
  Logit candidate_logit_post_gumbel = {batch_state.x, index_from_lane(batch_state.y)};
  const float pre_filter_logit_max = batch_state.z;
  const float pre_filter_logit_norm = batch_state.w;

  // Combine the slice partials on simdgroup 0 and broadcast.
  threadgroup float* shared_scalars = (threadgroup float*)shared;
  if (thread_context.simdgroup_index == 0) {
    const bool active = thread_context.simd_lane_id < num_slices;
    const float4 partial = active
        ? loop_partials[batch_idx * num_slices + thread_context.simd_lane_id]
        : float4(0.0f, 0.0, -INFINITY, INACTIVE_LANE);
    const uint32_t num_above = uint32_t(simd_sum(partial.x));
    const float mass_above = simd_sum(partial.y);
    const Logit next = SimdReduceMaxLogit::simd_reduce(Logit{partial.z, index_from_lane(partial.w)});
    if (thread_context.simd_lane_id == 0) {
      shared_scalars[0] = float(num_above);
      shared_scalars[1] = mass_above;
      shared_scalars[2] = next.value;
      shared_scalars[3] = lane_from_index(next.index);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint32_t num_above_candidate = uint32_t(shared_scalars[0]);
  float mass_above_candidate = shared_scalars[1];
  Logit next_candidate_logit_post_gumbel = {shared_scalars[2], index_from_lane(shared_scalars[3])};
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint32_t iteration = 0; iteration < MAX_ITERS; iteration++) {
    Logit candidate_logit_pre_filter = Logit::load(logits, candidate_logit_post_gumbel.index);
    if (has_temperature) {
      candidate_logit_pre_filter.value *= recip_temperature;
    }

    if (iteration > 0) {
      // Serial tail: recompute the stats against the current candidate.
      const AboveCandidateStats stats = scan_above_candidate(
          logits,
          bitmask,
          rng_seed,
          recip_temperature,
          candidate_logit_pre_filter,
          pre_filter_logit_max,
          pre_filter_logit_norm,
          uint2(0, div_ceil(vocab_size, ELEMS_PER_NOISE_BLOCK)),
          vocab_size,
          is_stochastic,
          has_bitmask,
          has_temperature,
          has_top_k,
          has_top_p,
          shared,
          thread_context,
          thread_idx
      );
      num_above_candidate = stats.num_above;
      mass_above_candidate = stats.mass_above;
      next_candidate_logit_post_gumbel = stats.next_candidate_post_gumbel;
    }

    bool filters_passed = true;

    if (has_top_k && num_above_candidate >= top_k) {
      filters_passed = false;
    }
    if (has_top_p && mass_above_candidate >= top_p) {
      filters_passed = false;
    }
    if (has_min_p && candidate_logit_pre_filter.value < pre_filter_logit_max + log_min_p) {
      filters_passed = false;
    }

    if (filters_passed || iteration == MAX_ITERS - 1) {
      if (thread_idx == 0) {
        *output = candidate_logit_post_gumbel.index;
      }
      return;
    }

    candidate_logit_post_gumbel = next_candidate_logit_post_gumbel;
  }
}
