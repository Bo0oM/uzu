#![cfg(backend = "metal")]

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use num_traits::Float;
use proc_macros::uzu_bench;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend,
            gpu_types::QuantizationMethod,
            kernel::{Kernels, matmul::MatmulKernel},
        },
        metal::{Metal, MetalContext},
    },
    tests::{
        cold_pool::ColdPool,
        matmul::{
            QuantBuffers, QuantInput, gemma3_batched_layer_shapes, gemma3_layer_shapes, iter_encode_loop,
            quant_arguments, qwen3_layer_shapes,
        },
        util::type_short_name,
    },
};

fn bench_layer_shapes_typed<T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &MetalContext,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
    shapes: impl Iterator<Item = (&'static str, crate::tests::matmul::Shape)>,
) {
    let mut group = c.benchmark_group(format!("{}/Kernel/Qwen3Layers/{}", type_short_name::<Metal>(), label));
    for (layer, shape) in shapes {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 42);
        let mut buffers =
            ColdPool::new(input.weight_buffer_bytes(), || QuantBuffers::<Metal, T>::allocate(context, &input));
        let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
            context,
            T::data_type(),
            T::data_type(),
            T::data_type(),
        )
        .unwrap();

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(format!("{layer}_{shape}")), |b| {
            iter_encode_loop::<Metal, _>(context, b, |encoder| {
                matmul.encode(quant_arguments(buffers.next_mut(), &input), encoder).expect("encode qwen3 layer");
            });
        });
    }
}

#[uzu_bench]
fn bench_qwen3_layers(c: &mut Criterion) {
    let context = crate::tests::util::shared_metal_context();
    bench_layer_shapes_typed::<bf16>(
        c,
        &context,
        "ScaleBias_BF16_gs128_4b",
        128,
        4,
        QuantizationMethod::ScaleBias,
        qwen3_layer_shapes(4),
    );
    bench_layer_shapes_typed::<bf16>(
        c,
        &context,
        "ZP_BF16_gs128_4b",
        128,
        4,
        QuantizationMethod::ScaleZeroPoint,
        qwen3_layer_shapes(4),
    );
    // The production gemma-3-1b-4bit configuration (decode, m = 1).
    bench_layer_shapes_typed::<bf16>(
        c,
        &context,
        "Gemma3_ScaleBias_BF16_gs64_4b",
        64,
        4,
        QuantizationMethod::ScaleBias,
        gemma3_layer_shapes(),
    );
    // Draft-verification batch sizes on the same forms; run with
    // UZU_GEMV_M_TILE=4|8 to dispatch the batched quantized tile.
    bench_layer_shapes_typed::<bf16>(
        c,
        &context,
        "Gemma3Batched_ScaleBias_BF16_gs64_4b",
        64,
        4,
        QuantizationMethod::ScaleBias,
        gemma3_batched_layer_shapes(),
    );
}
