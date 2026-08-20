#![cfg(backend = "metal")]

//! Microbenchmarks for the attention cores.
//!
//! These shapes are chosen to land on each of the three paths
//! `AttentionCores::encode` dispatches to, because two of them were never
//! being measured. A suffix longer than 8 tokens goes to the GEMM core; a
//! single-token suffix goes to two-pass once the context passes 1024 and to
//! single-pass below it. The engine benchmark only ever ran short prompts, so
//! the two-pass path -- the one every long chat ends up on -- had no number at
//! all.
//!
//! Worth measuring at all because attention is not a rounding error at length:
//! on Qwen3.5-0.8B-M the prefill curve gives up 37% of its peak by 27k tokens,
//! which works out to attention being 43% of prefill cost there.

use std::mem::size_of;

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context},
        metal::Metal,
    },
    encodable_block::mixer::attention::{
        core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments, AttentionCores, AttentionKvQuant},
        state::AttentionStateType,
    },
    tests::{helpers::alloc_allocation, matmul::iter_encode_loop_named, util::type_short_name},
};

/// Qwen3-0.6B's attention shape: 16 query heads over 8 KV groups, 128 wide.
const HEAD_DIM: u32 = 128;
const NUM_Q_HEADS: u32 = 16;
const NUM_GROUPS: u32 = 8;

struct Case {
    label: &'static str,
    prefix: u32,
    suffix: u32,
    /// Whether the cache is the quantized one. It is a separate kernel path,
    /// not a variation of the same one, and the engine turns it on by itself
    /// at exactly the long contexts these cases cover.
    kv_int8: bool,
}

/// `prefix` is what the cache already holds, `suffix` what this pass adds --
/// the pair that decides the path, so each case names the one it exercises.
const CASES: &[Case] = &[
    Case {
        label: "decode_single_pass_ctx512",
        prefix: 512,
        suffix: 1,
        kv_int8: false,
    },
    Case {
        label: "decode_two_pass_ctx4k",
        prefix: 4096,
        suffix: 1,
        kv_int8: false,
    },
    Case {
        label: "decode_two_pass_ctx16k",
        prefix: 16384,
        suffix: 1,
        kv_int8: false,
    },
    Case {
        label: "prefill_gemm_suffix512",
        prefix: 0,
        suffix: 512,
        kv_int8: false,
    },
    Case {
        label: "prefill_gemm_ctx8k_suffix512",
        prefix: 8192,
        suffix: 512,
        kv_int8: false,
    },
    Case {
        label: "decode_two_pass_ctx4k_int8",
        prefix: 4096,
        suffix: 1,
        kv_int8: true,
    },
    Case {
        label: "decode_two_pass_ctx16k_int8",
        prefix: 16384,
        suffix: 1,
        kv_int8: true,
    },
];

#[uzu_bench]
fn bench_attention_cores(c: &mut Criterion) {
    let context = <Metal as Backend>::Context::new().expect("metal context");
    let group_path = format!("{}/Kernel/Attention/Cores_BF16", type_short_name::<Metal>());
    let mut group = c.benchmark_group(group_path.clone());

    for case in CASES {
        let cores = AttentionCores::<Metal>::new(
            AttentionCoreNewArguments {
                head_dim: HEAD_DIM,
                num_groups: NUM_GROUPS,
                num_q_heads: NUM_Q_HEADS,
                has_sinks: false,
                is_kv_cache_ring: false,
                is_causal: true,
                is_trie: false,
                sliding_window_size: None,
                scale: None,
                data_type: bf16::data_type(),
                kv_int8: case.kv_int8,
            },
            &context,
        )
        .expect("attention cores");

        let rows = (case.prefix + case.suffix) as usize;
        let queries = alloc_allocation::<Metal, bf16>(
            &context,
            NUM_Q_HEADS as usize * case.suffix as usize * HEAD_DIM as usize,
        );
        let plane = rows * NUM_GROUPS as usize * HEAD_DIM as usize;
        let (keys, values) = if case.kv_int8 {
            (alloc_allocation::<Metal, u8>(&context, plane), alloc_allocation::<Metal, u8>(&context, plane))
        } else {
            (alloc_allocation::<Metal, bf16>(&context, plane), alloc_allocation::<Metal, bf16>(&context, plane))
        };
        // Scales live in plain buffers, one per (token, kv head), the way the
        // attention state holds them.
        let scale_bytes = rows * NUM_GROUPS as usize * size_of::<f32>();
        let scales = case.kv_int8.then(|| {
            (
                context.create_buffer(scale_bytes).expect("key scales"),
                context.create_buffer(scale_bytes).expect("value scales"),
            )
        });
        let state_type = AttentionStateType::Full {
            length: case.prefix,
        };

        // Scores and the value-weighted sum, one multiply-add each per (query
        // head, suffix token, context token, channel).
        group.throughput(Throughput::Elements(
            2 * u64::from(NUM_Q_HEADS) * u64::from(case.suffix) * rows as u64 * u64::from(HEAD_DIM),
        ));
        group.bench_function(BenchmarkId::from_parameter(case.label), |b| {
            let benchmark_path = format!("{group_path}/{}", case.label);
            iter_encode_loop_named::<Metal, _>(&context, b, &benchmark_path, |encoder| {
                let output = cores
                    .encode(
                        AttentionCoreEncodeArguments {
                            queries: &queries,
                            keys: &keys,
                            values: &values,
                            suffix_length: case.suffix,
                            trie: None,
                            sinks: None,
                            state_type: &state_type,
                            kv_quant: scales.as_ref().map(|(key_scales, value_scales)| AttentionKvQuant {
                                key_scales,
                                value_scales,
                            }),
                        },
                        encoder,
                    )
                    .expect("attention encode");
                drop(output);
            });
        });
    }
}
