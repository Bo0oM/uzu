use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::{
    array::ArrayElement,
    backends::{
        common::gpu_types::trie::TrieNode,
        cpu::kernel::attention::{
            kv_row::{KvRowSource, read_kv_row},
            mask::should_use_key,
        },
    },
};

#[kernel(AttentionSinglePass)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_DIM, 64, 128, 256, 512)]
pub fn attention_single_pass<T: ArrayElement + Float, const HEAD_DIM: u32>(
    queries: *const T,
    #[optional(!kv_int8)] keys: Option<*const T>,
    #[optional(!kv_int8)] values: Option<*const T>,
    #[optional(kv_int8)] keys_q8: Option<*const i8>,
    #[optional(kv_int8)] values_q8: Option<*const i8>,
    #[optional(kv_int8)] key_scales: Option<*const f32>,
    #[optional(kv_int8)] value_scales: Option<*const f32>,
    #[optional(kv_int8)] num_kv_heads: Option<u32>,
    out: *mut T,
    gqa_factor: u32,
    sequence_length: u32,
    k_head_stride: u32,
    k_seq_stride: u32,
    v_head_stride: u32,
    v_seq_stride: u32,
    #[optional(is_kv_cache_ring)] ring_params: Option<crate::backends::common::gpu_types::ring::RingParams>,
    scale: f32,
    #[optional(is_trie)] trie: Option<*const TrieNode>,
    #[optional(is_sliding_window)] sliding_window_size: Option<u32>,
    #[optional(has_sinks)] sinks: Option<*const T>,
    num_heads: u32,
    suffix_length: u32,
    #[specialize] has_sinks: bool,
    #[specialize] kv_int8: bool,
    #[specialize] is_kv_cache_ring: bool,
    #[specialize] is_causal: bool,
    #[specialize] is_trie: bool,
    #[specialize] is_sliding_window: bool,
) {
    assert_eq!(ring_params.is_some(), is_kv_cache_ring);
    assert_eq!(sliding_window_size.is_some(), is_sliding_window);

    let value_dim = HEAD_DIM;

    let prefix_length = sequence_length - suffix_length;
    let suffix_position = if let Some(ring_params) = ring_params {
        ring_params.ring_length
    } else {
        prefix_length
    };

    for head_idx in 0..num_heads {
        for q_seq_idx in 0..suffix_length {
            let kv_head_idx = head_idx / gqa_factor;
            let o_offset = q_seq_idx * num_heads + head_idx;
            let q_offset = head_idx * suffix_length + q_seq_idx;

            let query_position = if is_trie {
                let trie_node = unsafe { &*trie.unwrap().add(q_seq_idx as usize) };
                suffix_position + trie_node.height
            } else {
                suffix_position + q_seq_idx
            };

            let queries: *const T = unsafe { queries.add((q_offset * HEAD_DIM) as usize) };
            let out: *mut T = unsafe { out.add((o_offset * value_dim) as usize) };
            let key_source = KvRowSource {
                int8_base: keys_q8,
                float_base: keys,
                scales: key_scales,
                kv_int8,
                num_kv_heads,
                kv_head_idx,
                head_stride: k_head_stride,
                seq_stride: k_seq_stride,
            };
            let value_source = KvRowSource {
                int8_base: values_q8,
                float_base: values,
                scales: value_scales,
                kv_int8,
                num_kv_heads,
                kv_head_idx,
                head_stride: v_head_stride,
                seq_stride: v_seq_stride,
            };

            // Read the query and 0 the output accumulator
            let mut q = vec![0.0f32; HEAD_DIM as usize];
            let mut o = vec![0.0f32; HEAD_DIM as usize];
            let mut key_row = vec![0.0f32; HEAD_DIM as usize];
            let mut value_row = vec![0.0f32; HEAD_DIM as usize];
            for j in 0..HEAD_DIM as usize {
                q[j] = scale * unsafe { *queries.add(j) }.to_f32().unwrap();
            }

            let mut max_score = f32::NEG_INFINITY;
            let mut sum_exp_score = 0.0f32;
            if has_sinks {
                let q_head_idx = head_idx % num_heads;
                max_score = unsafe { *sinks.unwrap().add(q_head_idx as usize) }.to_f32().unwrap();
                sum_exp_score = 1.0;
            }

            // For each key
            for i in 0..sequence_length {
                if should_use_key(
                    ring_params,
                    trie,
                    sliding_window_size,
                    q_seq_idx,
                    prefix_length,
                    suffix_position,
                    query_position,
                    i,
                    is_causal,
                ) {
                    read_kv_row(&key_source, i, &mut key_row);

                    // Compute the i-th score
                    let mut score = 0.0f32;
                    for j in 0..HEAD_DIM as usize {
                        score += q[j] * key_row[j];
                    }

                    // Update the accumulators
                    let new_max = f32::max(max_score, score);
                    let factor = (max_score - new_max).exp();
                    let exp_score = (score - new_max).exp();

                    max_score = new_max;
                    sum_exp_score = sum_exp_score * factor + exp_score;

                    // Update the output accumulator
                    read_kv_row(&value_source, i, &mut value_row);
                    for j in 0..HEAD_DIM as usize {
                        o[j] = o[j] * factor + exp_score * value_row[j];
                    }
                }
            }

            // Write the output
            for j in 0..HEAD_DIM as usize {
                unsafe {
                    *out.add(j) = T::from(o[j] / sum_exp_score).unwrap();
                }
            }
        }
    }
}
