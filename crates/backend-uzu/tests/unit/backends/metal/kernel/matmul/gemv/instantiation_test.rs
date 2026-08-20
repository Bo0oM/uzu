#![cfg(backend = "metal")]

//! Every tile the policy can select must be a tile the shaders compiled.
//!
//! `GemvSpecialization::new` restates, in Rust, which template instantiations
//! `gemv.metal` actually produces — three predicates whose comments say
//! "Mirrors the ... constraint in gemv.metal". The `CONSTRAINT` lines there are
//! the authority, and nothing links the two: editing one leaves the other
//! stale with no compile error, and the failure surfaces at runtime either as
//! a pipeline-creation error or, worse, as a silently downgraded tile.
//!
//! Deriving the host predicate from the same constraint list is the proper
//! fix and wants build-system work. Until then this sweeps the shapes the
//! policy can be asked about and encodes each one, which creates the pipeline
//! — so a desync fails here rather than in someone's chat session.

use half::bf16;
use proc_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder,
            gpu_types::{
                QuantizationMethod,
                gemm::{GemmBPrologueKind, GemmDTransform},
            },
            kernel::{
                Kernels,
                matmul::{MatmulKernel, MatmulShape},
            },
        },
        metal::{
            DeviceTier, Metal,
            kernel::matmul::gemv::{GemvDispatch, GemvSpecialization},
        },
    },
    tests::matmul::{QuantBuffers, QuantInput, quant_arguments},
};

/// Decode and small-batch shapes drawn from the models on the stand: gemma-3
/// (k=1152, 6912), Qwen3-0.6B (k=1024), Qwen3.5 (k=2048), and the readouts,
/// which are the widest n the policy ever sees.
const SHAPES: &[(u32, u32)] = &[
    (1536, 1152),
    (1152, 1024),
    (13824, 1152),
    (1152, 6912),
    (2048, 2048),
    (4096, 2048),
    (6144, 2048),
    (262_144, 1152),
    (151_936, 1024),
];

#[uzu_test]
fn every_selectable_tile_is_instantiated() {
    let context = <Metal as Backend>::Context::new().expect("metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("matmul kernel");

    let mut checked = 0usize;
    for &(n, k) in SHAPES {
        for &(bits, group_size) in &[(4u32, 32u32), (4, 64), (8, 32), (8, 64)] {
            if !k.is_multiple_of(group_size) {
                continue;
            }
            // m = 1 is decode; m = 4 is a speculative verification pass, which
            // is where the batched-tile slice comes in.
            for m in [1u32, 4] {
                let input = QuantInput::<bf16>::new(m, k, n, group_size, bits, QuantizationMethod::ScaleBias, 7);
                let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
                let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
                kernel
                    .encode(quant_arguments(&mut buffers, &input), &mut encoder)
                    .unwrap_or_else(|error| panic!("n={n} k={k} bits={bits} gs={group_size} m={m}: {error:?}"));
                encoder.end_encoding().submit().wait_until_completed().expect("submit");
                checked += 1;
            }
        }
    }

    assert!(checked >= 40, "expected a real sweep, got {checked} shapes");
}

/// The tile policy branches on the device tier, so the sweep above only proves
/// the tiles *this* machine asks for exist. A phone reaches different branches
/// (`DeviceTier::SmallApple9` on A17+, `SmallApple8` on A15/A16), and nobody
/// runs the suite there before shipping.
///
/// Tier is a parameter of `select_shape`, and creating a pipeline only needs
/// the function to be in the metallib — so every tier can be checked from one
/// desktop. That covers iOS too: no `CONSTRAINT` in `gemv.metal` mentions the
/// platform, so the iPhone SDK compiles the same instantiation set as the macOS
/// one, and a tile legal here is legal there.
const TIERS: &[DeviceTier] =
    &[DeviceTier::SmallApple9, DeviceTier::SmallApple8, DeviceTier::SmallLegacy, DeviceTier::Large];

#[uzu_test]
fn every_tier_selects_an_instantiated_tile() {
    let context = <Metal as Backend>::Context::new().expect("metal context");
    let mut dispatch = GemvDispatch::new(bf16::data_type(), bf16::data_type(), bf16::data_type());

    // `None` is the full-precision path, which has its own tier branch in
    // `policy::fp_values_per_thread`.
    let quantizations: &[Option<(u32, u32)>] =
        &[None, Some((4, 32)), Some((4, 64)), Some((8, 32)), Some((8, 64))];

    let mut checked = 0usize;
    for &tier in TIERS {
        for &(n, k) in SHAPES {
            for quantization in quantizations {
                if let Some((_, group_size)) = quantization
                    && !k.is_multiple_of(*group_size)
                {
                    continue;
                }
                for m in [1u32, 4] {
                    // The RHT output transform forces a 32-row threadgroup, which
                    // sends tile selection down `rht_rows_guard`.
                    for d_transform in [GemmDTransform::empty(), GemmDTransform::RHT] {
                        if d_transform.contains(GemmDTransform::RHT) && !n.is_multiple_of(32) {
                            continue;
                        }
                        let shape = MatmulShape {
                            m,
                            n,
                            k,
                            b_transpose: true,
                            b_leading_dimension: None,
                            b_prologue: match quantization {
                                Some(_) => GemmBPrologueKind::ScaleBiasDequant,
                                None => GemmBPrologueKind::FullPrecision,
                            },
                            b_bits: quantization.map(|(bits, _)| bits),
                            b_group_size: quantization.map(|(_, group_size)| group_size),
                            signed_codes: false,
                            a_full_precision: true,
                            gathered: false,
                            d_transform,
                        };
                        let Some(specialization) = GemvSpecialization::select_shape(
                            &shape,
                            bf16::data_type(),
                            bf16::data_type(),
                            bf16::data_type(),
                            tier,
                        ) else {
                            continue;
                        };
                        dispatch.get_or_create(&context, specialization).unwrap_or_else(|error| {
                            panic!("{tier:?} n={n} k={k} q={quantization:?} m={m} d={d_transform:?}: {error:?}")
                        });
                        checked += 1;
                    }
                }
            }
        }
    }

    assert!(checked >= 4 * 40, "expected every tier to be swept, got {checked} specializations");
}
