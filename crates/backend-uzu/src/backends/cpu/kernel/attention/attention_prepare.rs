use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

fn apply_rope<ElementT: ArrayElement + Float, RopeT: ArrayElement + Float>(
    head: *const ElementT,
    cosines: *const RopeT,
    sines: *const RopeT,
    batch_idx: usize,
    head_dim_idx: usize,
    rope_dim: usize,
) -> ElementT {
    let half_rope_dim = rope_dim / 2;
    let paired_idx = if head_dim_idx < half_rope_dim {
        head_dim_idx + half_rope_dim
    } else {
        head_dim_idx - half_rope_dim
    };
    let input = unsafe { (*head.add(head_dim_idx)).to_f32().unwrap() };
    let paired = unsafe { (*head.add(paired_idx)).to_f32().unwrap() };
    let signed_paired = if head_dim_idx < half_rope_dim {
        -paired
    } else {
        paired
    };
    let cos_val = unsafe { (*cosines.add(batch_idx * rope_dim + head_dim_idx)).to_f32().unwrap() };
    let sin_val = unsafe { (*sines.add(batch_idx * rope_dim + head_dim_idx)).to_f32().unwrap() };

    ElementT::from(input * cos_val + signed_paired * sin_val).unwrap()
}

#[kernel(AttentionPrepare)]
#[variants(ElementT, bf16)]
#[variants(RopeT, f32)]
pub fn attention_prepare<ElementT: ArrayElement + Float, RopeT: ArrayElement + Float>(
    qkv: *const ElementT,
    queries: *mut ElementT,
    #[optional(has_kv && !kv_int8)] keys: Option<*mut ElementT>,
    #[optional(has_kv && !kv_int8)] values: Option<*mut ElementT>,
    #[optional(has_kv && kv_int8)] keys_q8: Option<*mut i8>,
    #[optional(has_kv && kv_int8)] values_q8: Option<*mut i8>,
    #[optional(has_kv && kv_int8)] key_scales: Option<*mut f32>,
    #[optional(has_kv && kv_int8)] value_scales: Option<*mut f32>,
    #[optional(has_rope)] cosines: Option<*const RopeT>,
    #[optional(has_rope)] sines: Option<*const RopeT>,
    num_q_heads: u32,
    #[optional(has_kv)] num_kv_heads: Option<u32>,
    head_dim: u32,
    #[optional(has_rope)] rope_dim: Option<u32>,
    #[optional(has_kv)] kv_token_offset: Option<u32>,
    batch_dim: u32,
    #[specialize] has_kv: bool,
    #[specialize] has_rope: bool,
    #[specialize] kv_int8: bool,
) {
    assert!(num_q_heads > 0 || has_kv, "attention prepare without KV requires at least one query head");
    assert!(head_dim > 0, "attention prepare requires nonzero head_dim");
    assert_eq!(keys.is_some(), has_kv && !kv_int8, "attention prepare keys presence mismatch");
    assert_eq!(values.is_some(), has_kv && !kv_int8, "attention prepare values presence mismatch");
    assert_eq!(keys_q8.is_some(), has_kv && kv_int8, "attention prepare keys_q8 presence mismatch");
    assert_eq!(values_q8.is_some(), has_kv && kv_int8, "attention prepare values_q8 presence mismatch");
    assert_eq!(key_scales.is_some(), has_kv && kv_int8, "attention prepare key_scales presence mismatch");
    assert_eq!(value_scales.is_some(), has_kv && kv_int8, "attention prepare value_scales presence mismatch");
    assert_eq!(num_kv_heads.is_some(), has_kv, "attention prepare num_kv_heads presence mismatch");
    assert_eq!(kv_token_offset.is_some(), has_kv, "attention prepare kv_token_offset presence mismatch");
    assert_eq!(cosines.is_some(), has_rope, "attention prepare cosines presence mismatch");
    assert_eq!(sines.is_some(), has_rope, "attention prepare sines presence mismatch");
    assert_eq!(rope_dim.is_some(), has_rope, "attention prepare rope_dim presence mismatch");

    let (num_kv_heads, kv_token_offset) =
        if let (Some(num_kv_heads), Some(kv_token_offset)) = (num_kv_heads, kv_token_offset) {
            assert!(num_kv_heads > 0, "attention prepare has_kv requires nonzero num_kv_heads");
            (num_kv_heads as usize, kv_token_offset as usize)
        } else {
            (0, 0)
        };
    let keys = keys.unwrap_or(std::ptr::null_mut());
    let values = values.unwrap_or(std::ptr::null_mut());
    let keys_q8 = keys_q8.unwrap_or(std::ptr::null_mut());
    let values_q8 = values_q8.unwrap_or(std::ptr::null_mut());
    let key_scales = key_scales.unwrap_or(std::ptr::null_mut());
    let value_scales = value_scales.unwrap_or(std::ptr::null_mut());

    let (cosines, sines, rope_dim) = if let (Some(cosines), Some(sines), Some(rope_dim)) = (cosines, sines, rope_dim) {
        assert!(rope_dim > 0, "attention prepare has_rope requires nonzero rope_dim");
        assert!(rope_dim <= head_dim, "attention prepare rope_dim exceeds head_dim");
        assert!(rope_dim.is_multiple_of(2), "attention prepare rope_dim must be even");
        (cosines, sines, rope_dim as usize)
    } else {
        (std::ptr::null(), std::ptr::null(), 0)
    };

    let num_q_heads = num_q_heads as usize;
    let head_dim = head_dim as usize;
    let batch_dim = batch_dim as usize;
    let total_heads = if has_kv {
        num_q_heads + num_kv_heads * 2
    } else {
        num_q_heads
    };

    let mut row = vec![ElementT::zero(); head_dim];
    for batch_idx in 0..batch_dim {
        for head_idx in 0..total_heads {
            let qkv_head = unsafe { qkv.add(batch_idx * total_heads * head_dim + head_idx * head_dim) };
            let is_query = !has_kv || head_idx < num_q_heads;
            let is_key = has_kv && head_idx >= num_q_heads && head_idx < num_q_heads + num_kv_heads;

            for (head_dim_idx, slot) in row.iter_mut().enumerate() {
                let mut element = unsafe { *qkv_head.add(head_dim_idx) };
                if has_rope && head_dim_idx < rope_dim && (is_query || is_key) {
                    element = apply_rope(qkv_head, cosines, sines, batch_idx, head_dim_idx, rope_dim);
                }
                *slot = element;
            }

            if is_query {
                for (head_dim_idx, &element) in row.iter().enumerate() {
                    let query_offset = head_idx * batch_dim * head_dim + batch_idx * head_dim + head_dim_idx;
                    unsafe {
                        *queries.add(query_offset) = element;
                    }
                }
                continue;
            }

            let kv_head = if is_key {
                head_idx - num_q_heads
            } else {
                head_idx - num_q_heads - num_kv_heads
            };
            let kv_row = (kv_token_offset + batch_idx) * num_kv_heads + kv_head;
            let kv_offset = kv_row * head_dim;

            if !kv_int8 {
                let target = if is_key {
                    keys
                } else {
                    values
                };
                for (head_dim_idx, &element) in row.iter().enumerate() {
                    unsafe {
                        *target.add(kv_offset + head_dim_idx) = element;
                    }
                }
                continue;
            }

            // Mirrors the Metal path: symmetric int8 with one absmax scale
            // per (token, kv head).
            let absmax = row.iter().fold(0.0f32, |acc, e| acc.max(e.to_f32().unwrap().abs()));
            let scale = absmax.max(1e-8) / 127.0;
            let (target, scales) = if is_key {
                (keys_q8, key_scales)
            } else {
                (values_q8, value_scales)
            };
            unsafe {
                *scales.add(kv_row) = scale;
            }
            for (head_dim_idx, &element) in row.iter().enumerate() {
                let quantized = (element.to_f32().unwrap() / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
                unsafe {
                    *target.add(kv_offset + head_dim_idx) = quantized;
                }
            }
        }
    }
}
