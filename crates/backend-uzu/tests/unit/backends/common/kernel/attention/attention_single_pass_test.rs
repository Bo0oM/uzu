use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use proc_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Allocation, Backend, Context, Encoder, Kernels, kernel::AttentionSinglePassKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
    },
};

struct Input<T: ArrayElement + Float> {
    queries: Box<[T]>,
    keys: Box<[T]>,
    values: Box<[T]>,
    num_heads: u32,
    gqa_factor: u32,
    sequence_length: u32,
    suffix_length: u32,
    head_dim: u32,
    scale: f32,
    do_causal: bool,
}

fn get_input<T: ArrayElement + Float>(
    num_heads: u32,
    num_kv_heads: u32,
    sequence_length: u32,
    suffix_length: u32,
    head_dim: u32,
    do_causal: bool,
) -> Input<T> {
    let gqa_factor = num_heads / num_kv_heads;

    // queries: [num_heads * suffix_length, head_dim]
    let q_size = (num_heads * suffix_length * head_dim) as usize;
    let mut queries = vec![T::zero(); q_size];
    for i in 0..q_size {
        queries[i] = T::from((i as f32 * 0.13 + 0.5).sin() * 0.5).unwrap();
    }

    // keys: [num_kv_heads, sequence_length, head_dim]
    let k_size = (num_kv_heads * sequence_length * head_dim) as usize;
    let mut keys = vec![T::zero(); k_size];
    for i in 0..k_size {
        keys[i] = T::from((i as f32 * 0.07 + 1.0).cos() * 0.5).unwrap();
    }

    // values: [num_kv_heads, sequence_length, head_dim]
    let v_size = (num_kv_heads * sequence_length * head_dim) as usize;
    let mut values = vec![T::zero(); v_size];
    for i in 0..v_size {
        values[i] = T::from((i as f32 * 0.11 + 2.0).sin() * 0.5).unwrap();
    }

    let scale = 1.0 / (head_dim as f32).sqrt();

    Input {
        queries: queries.into_boxed_slice(),
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        num_heads,
        gqa_factor,
        sequence_length,
        suffix_length,
        head_dim,
        scale,
        do_causal,
    }
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
        &context,
        T::data_type(),
        input.head_dim,
        false,
        false,
        false,
        input.do_causal,
        false,
        false,
    )
    .expect("Failed to create AttentionSinglePassKernel");

    let queries_allocation = alloc_allocation_with_data::<B, T>(&context, &input.queries);
    let keys_allocation = alloc_allocation_with_data::<B, T>(&context, &input.keys);
    let values_allocation = alloc_allocation_with_data::<B, T>(&context, &input.values);

    let output_size = (input.suffix_length * input.num_heads * input.head_dim) as usize;
    let mut output_allocation = alloc_allocation::<B, T>(&context, output_size);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        &queries_allocation,
        Some(&keys_allocation),
        Some(&values_allocation),
        None::<&Allocation<B>>,
        None::<&Allocation<B>>,
        None::<&Allocation<B>>,
        None::<&Allocation<B>>,
        None,
        &mut output_allocation,
        input.gqa_factor,
        input.sequence_length,
        input.sequence_length * input.head_dim,
        input.head_dim,
        input.sequence_length * input.head_dim,
        input.head_dim,
        None,
        input.scale,
        None::<&Allocation<B>>,
        None,
        None::<&Allocation<B>>,
        input.num_heads,
        input.suffix_length,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<B, T>(&output_allocation)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(input);
        let msg = format!(
            "AttentionSinglePass failed (backend={}, heads={}, seq={}, suffix={}, head_dim={}, causal={})",
            std::any::type_name::<B>(),
            input.num_heads,
            input.sequence_length,
            input.suffix_length,
            input.head_dim,
            input.do_causal,
        );
        assert_eq_float::<T>(expected, &output, eps, &msg);
    });
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    // Non-causal, single token
    let input = get_input::<T>(4, 4, 8, 1, 64, false);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);

    // Non-causal, multiple tokens
    let input = get_input::<T>(4, 4, 8, 4, 64, false);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);
}

fn test_causal<T: ArrayElement + Float + Debug + Display>() {
    // Causal, single token decode
    let input = get_input::<T>(4, 4, 16, 1, 64, true);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);

    // Causal, multi-token prefill
    let input = get_input::<T>(4, 4, 8, 4, 64, true);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);
}

fn test_gqa<T: ArrayElement + Float + Debug + Display>() {
    // GQA: 8 query heads, 2 kv heads
    let input = get_input::<T>(8, 2, 8, 1, 64, false);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);

    // GQA causal
    let input = get_input::<T>(8, 2, 8, 4, 64, true);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);
}

fn test_head_dim<T: ArrayElement + Float + Debug + Display>(head_dim: u32) {
    let input = get_input::<T>(4, 4, 8, 2, head_dim, true);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);
}

/// Symmetric per-(position, kv head) absmax quantization mirroring
/// AttentionPrepare; scales are laid out `[position, kv_head]`.
fn quantize_kv<T: ArrayElement + Float>(
    data: &[T],
    num_kv_heads: usize,
    sequence_length: usize,
    head_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut quantized = vec![0i8; data.len()];
    let mut scales = vec![0f32; sequence_length * num_kv_heads];
    for kv_head in 0..num_kv_heads {
        for position in 0..sequence_length {
            let row = &data[(kv_head * sequence_length + position) * head_dim..][..head_dim];
            let absmax = row.iter().fold(0f32, |acc, v| acc.max(v.to_f32().unwrap().abs()));
            let scale = absmax.max(1e-8) / 127.0;
            scales[position * num_kv_heads + kv_head] = scale;
            for (j, v) in row.iter().enumerate() {
                let q = (v.to_f32().unwrap() / scale).round_ties_even().clamp(-127.0, 127.0);
                quantized[(kv_head * sequence_length + position) * head_dim + j] = q as i8;
            }
        }
    }
    (quantized, scales)
}

fn get_output_q8<T: ArrayElement + Float, B: Backend>(
    input: &Input<T>,
    keys_q8: &[i8],
    values_q8: &[i8],
    key_scales: &[f32],
    value_scales: &[f32],
    num_kv_heads: u32,
) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
        &context,
        T::data_type(),
        input.head_dim,
        false,
        true,
        false,
        input.do_causal,
        false,
        false,
    )
    .expect("Failed to create AttentionSinglePassKernel");

    let queries_allocation = alloc_allocation_with_data::<B, T>(&context, &input.queries);
    let keys_allocation = alloc_allocation_with_data::<B, i8>(&context, keys_q8);
    let values_allocation = alloc_allocation_with_data::<B, i8>(&context, values_q8);
    let key_scales_allocation = alloc_allocation_with_data::<B, f32>(&context, key_scales);
    let value_scales_allocation = alloc_allocation_with_data::<B, f32>(&context, value_scales);

    let output_size = (input.suffix_length * input.num_heads * input.head_dim) as usize;
    let mut output_allocation = alloc_allocation::<B, T>(&context, output_size);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        &queries_allocation,
        None::<&Allocation<B>>,
        None::<&Allocation<B>>,
        Some(&keys_allocation),
        Some(&values_allocation),
        Some(&key_scales_allocation),
        Some(&value_scales_allocation),
        Some(num_kv_heads),
        &mut output_allocation,
        input.gqa_factor,
        input.sequence_length,
        input.sequence_length * input.head_dim,
        input.head_dim,
        input.sequence_length * input.head_dim,
        input.head_dim,
        None,
        input.scale,
        None::<&Allocation<B>>,
        None,
        None::<&Allocation<B>>,
        input.num_heads,
        input.suffix_length,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<B, T>(&output_allocation)
}

fn test_kv_int8<T: ArrayElement + Float + Debug + Display>() {
    let (num_heads, num_kv_heads, sequence_length, suffix_length, head_dim) = (8u32, 2u32, 16u32, 2u32, 64u32);
    let mut input = get_input::<T>(num_heads, num_kv_heads, sequence_length, suffix_length, head_dim, true);

    let (keys_q8, key_scales) =
        quantize_kv(&input.keys, num_kv_heads as usize, sequence_length as usize, head_dim as usize);
    let (values_q8, value_scales) =
        quantize_kv(&input.values, num_kv_heads as usize, sequence_length as usize, head_dim as usize);

    // Reference: the plain kernel run over host-dequantized K/V — the int8
    // kernel must reproduce it exactly up to accumulation order.
    for i in 0..input.keys.len() {
        let scale_idx = kv_scale_index(i, num_kv_heads, sequence_length, head_dim);
        input.keys[i] = T::from(keys_q8[i] as f32 * key_scales[scale_idx]).unwrap();
        input.values[i] = T::from(values_q8[i] as f32 * value_scales[scale_idx]).unwrap();
    }
    let expected = get_output::<T, Cpu>(&input);

    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    };
    let cpu_output = get_output_q8::<T, Cpu>(&input, &keys_q8, &values_q8, &key_scales, &value_scales, num_kv_heads);
    assert_eq_float::<T>(&expected, &cpu_output, eps, "AttentionSinglePass kv_int8 CPU mismatch");

    for_each_non_cpu_backend!(|B| {
        let output = get_output_q8::<T, B>(&input, &keys_q8, &values_q8, &key_scales, &value_scales, num_kv_heads);
        let msg = format!("AttentionSinglePass kv_int8 failed (backend={})", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &output, eps, &msg);
    });
}

/// Maps a flat `[kv_head, position, head_dim]` element index to its scale slot.
fn kv_scale_index(
    flat: usize,
    num_kv_heads: u32,
    sequence_length: u32,
    head_dim: u32,
) -> usize {
    let row = flat / head_dim as usize;
    let kv_head = row / sequence_length as usize;
    let position = row % sequence_length as usize;
    position * num_kv_heads as usize + kv_head
}

#[uzu_test]
fn test_kv_int8_f32() {
    test_kv_int8::<f32>();
}

#[uzu_test]
fn test_kv_int8_bf16() {
    test_kv_int8::<bf16>();
}

// Basic tests
#[uzu_test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[uzu_test]
fn test_basic_f16() {
    test_basic::<f16>();
}

#[uzu_test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

// Causal tests
#[uzu_test]
fn test_causal_f32() {
    test_causal::<f32>();
}

#[uzu_test]
fn test_causal_f16() {
    test_causal::<f16>();
}

#[uzu_test]
fn test_causal_bf16() {
    test_causal::<bf16>();
}

// GQA tests
#[uzu_test]
fn test_gqa_f32() {
    test_gqa::<f32>();
}

#[uzu_test]
fn test_gqa_f16() {
    test_gqa::<f16>();
}

#[uzu_test]
fn test_gqa_bf16() {
    test_gqa::<bf16>();
}

// Head dim 128
#[uzu_test]
fn test_head_dim_128_f32() {
    test_head_dim::<f32>(128);
}

#[uzu_test]
fn test_head_dim_128_f16() {
    test_head_dim::<f16>(128);
}

#[uzu_test]
fn test_head_dim_128_bf16() {
    test_head_dim::<bf16>(128);
}

#[uzu_test]
fn test_head_dim_512_f32() {
    test_head_dim::<f32>(512);
}

#[uzu_test]
fn test_head_dim_512_f16() {
    test_head_dim::<f16>(512);
}

#[uzu_test]
fn test_head_dim_512_bf16() {
    test_head_dim::<bf16>(512);
}
