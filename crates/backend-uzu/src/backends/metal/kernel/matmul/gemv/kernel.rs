use std::{
    collections::{HashMap, hash_map::Entry},
    sync::OnceLock,
};

use super::{
    autotune,
    policy::{self, DEFAULT_FP_VALUES_PER_THREAD, DEFAULT_RESULTS_PER_SIMDGROUP},
};
use crate::{
    backends::{
        common::{
            Allocation, BufferArg, Encoder,
            gpu_types::{
                HADAMARD_TRANSFORM_BLOCK_SIZE,
                gemm::{GemmBPrologueKind, GemmDTransform},
            },
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulShape},
        },
        metal::{Metal, context::MetalContext, device_tier::DeviceTier, kernel::GemvMetalKernel},
    },
    data_type::DataType,
};

const DEFAULT_GEMV_MAX_BATCH: u32 = 8;
static GEMV_MAX_BATCH: OnceLock<u32> = OnceLock::new();

fn max_gemv_batch_threshold() -> u32 {
    *GEMV_MAX_BATCH.get_or_init(|| {
        // TODO: remove magic env var
        std::env::var("UZU_GEMV_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_GEMV_MAX_BATCH)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemvSpecialization {
    b_prologue: GemmBPrologueKind,
    /// Staging/multiply type of the quantized source (F16 on 2xFP16 tiers).
    math: DataType,
    group_size: u32,
    bits: u32,
    output_transform: GemmDTransform,
    input_aligned: bool,
    values_per_thread: u32,
    k_split: u32,
    results_per_simdgroup: u32,
    num_simdgroups: u32,
    packs: u32,
    /// Batch elements sharing one weight pass per threadgroup (1 = classic).
    m_tile: u32,
    gathered: bool,
    signed_codes: bool,
}

impl GemvSpecialization {
    pub(crate) fn select_shape(
        shape: &MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        device_tier: DeviceTier,
    ) -> Option<GemvSpecialization> {
        if !shape.b_transpose || !shape.a_full_precision {
            return None;
        }
        let is_quant = shape.is_quant();
        let bad_leading_dimension = if is_quant {
            shape.b_leading_dimension.is_some()
        } else {
            shape.b_leading_dimension.is_some_and(|ld| ld != shape.k)
        };
        if bad_leading_dimension {
            return None;
        }
        if shape.d_transform.contains(GemmDTransform::ACCUMULATE) && !shape.n.is_multiple_of(32) {
            return None;
        }
        if shape.d_transform.contains(GemmDTransform::RHT) && !shape.n.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE) {
            return None;
        }
        if shape.n < DEFAULT_RESULTS_PER_SIMDGROUP || shape.m > max_gemv_batch_threshold() {
            return None;
        }
        if !is_quant {
            let mixed_precision = weights_data_type == DataType::F32
                && (input_data_type != DataType::F32 || output_data_type != DataType::F32);
            if mixed_precision {
                return None;
            }
        }

        let bits = shape.b_bits.unwrap_or(0);
        let values_per_thread = if is_quant {
            DEFAULT_FP_VALUES_PER_THREAD
        } else {
            policy::fp_values_per_thread(shape.k, device_tier)
        };
        let group_size = shape.b_group_size.unwrap_or(0);
        let tile_precheck_packs = if is_quant && bits == 4 && group_size == 64 {
            None // resolved from the tile below
        } else {
            Some(2)
        };
        let input_aligned_for = |packs: u32| {
            let block_size = if !is_quant {
                policy::fp_k_block(values_per_thread)
            } else {
                // 32 lanes x packs words x (32 / bits) values per word.
                32 * packs * (32 / bits)
            };
            shape.k.is_multiple_of(block_size)
        };
        let has_rht = shape.d_transform.contains(GemmDTransform::RHT);
        let bf16_io = input_data_type == DataType::BF16 && output_data_type == DataType::BF16;
        let tile = if is_quant && bf16_io {
            policy::quant_tile(shape.m, shape.n, shape.k, bits, has_rht, device_tier)
        } else if is_quant || has_rht {
            // Non-bf16 quant IO and fp+RHT keep the default tile (the only
            // one instantiated for those modes).
            policy::DEFAULT_TILE
        } else {
            // packs is irrelevant on the fp path; the closure ignores it there.
            policy::fp_tile(shape.m, shape.n, shape.k, input_aligned_for(2), values_per_thread, device_tier)
        };
        // Mirrors the M_TILE constraint in gemv.metal: batched tiles exist
        // for the dense-lane 4-bit gs64 and fp bf16 slices only, and the
        // kernel's batch grid assumes m divides exactly into tiles.
        let batched_slice_exists = if is_quant {
            bits == 4 && group_size == 64
        } else {
            weights_data_type == DataType::BF16
        };
        let m_tile = if batched_slice_exists && bf16_io && !shape.gathered && !has_rht {
            let selected = policy::gemv_m_tile(device_tier, shape.m, shape.n, shape.k, is_quant);
            if selected == shape.m {
                selected
            } else {
                1
            }
        } else {
            1
        };
        let packs = if m_tile > 1 && is_quant {
            1
        } else {
            tile_precheck_packs.unwrap_or(if tile.packs == 1 {
                1
            } else {
                2
            })
        };
        let input_aligned = input_aligned_for(packs);
        // Mirrors the MT constraint in gemv.metal: only the 4-bit bf16
        // gs32/gs64 slice has half pipelines instantiated.
        let half_eligible = is_quant && bits == 4 && (group_size == 32 || group_size == 64) && bf16_io;
        let math = if half_eligible
            && policy::quant_half_math(device_tier, shape.m, shape.n, shape.k, group_size, input_aligned)
        {
            DataType::F16
        } else {
            DataType::F32
        };
        Some(Self {
            b_prologue: shape.b_prologue,
            math,
            group_size,
            bits,
            output_transform: shape.d_transform,
            input_aligned,
            values_per_thread,
            // Batched quant tiles have no split variants; batched fp keeps
            // split-K only on deep k (down k=6912 m=4: -19% at KS4), while
            // shallow-k batched forms lose to it (upgate: 223 vs 102 us).
            k_split: if m_tile > 1 && (is_quant || shape.k < 2048) {
                1
            } else {
                tile.k_split
            },
            results_per_simdgroup: tile.results_per_simdgroup,
            num_simdgroups: tile.num_simdgroups,
            packs,
            m_tile,
            gathered: shape.gathered,
            signed_codes: shape.signed_codes,
        })
    }
}

fn rows_per_threadgroup(
    k_split: u32,
    results_per_simdgroup: u32,
    num_simdgroups: u32,
) -> u32 {
    (num_simdgroups / k_split) * results_per_simdgroup
}

pub(crate) struct GemvDispatch {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    pipelines: HashMap<GemvSpecialization, GemvMetalKernel>,
}

impl GemvDispatch {
    pub(crate) fn new(
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Self {
        Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            pipelines: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemvSpecialization,
    ) -> Result<&GemvMetalKernel, MatmulError<Metal>> {
        match self.pipelines.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemvMetalKernel::new(
                    context,
                    self.input_data_type,
                    self.weights_data_type,
                    self.output_data_type,
                    specialization.math,
                    specialization.b_prologue,
                    specialization.group_size,
                    specialization.bits,
                    specialization.values_per_thread,
                    specialization.packs,
                    specialization.k_split,
                    specialization.input_aligned,
                    specialization.results_per_simdgroup,
                    specialization.num_simdgroups,
                    specialization.m_tile,
                    specialization.output_transform,
                    specialization.gathered,
                    specialization.signed_codes,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub(crate) fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        specialization: GemvSpecialization,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let rht_factors = arguments.d_transform.rht_factors;
        let soft_cap = arguments.d_transform.soft_cap;

        let MatmulArguments {
            a,
            b,
            d,
            m,
            n,
            k,
            gather_indices,
            ..
        } = arguments;
        let MatmulA::FullPrecision {
            values: a,
            offset: a_offset,
        } = a
        else {
            return Err(MatmulError::IncompatibleA {
                path: "Gemv",
                reason: "prepared int8 activations require GEMM",
            });
        };

        let group_count_x = n.div_ceil(rows_per_threadgroup(
            specialization.k_split,
            specialization.results_per_simdgroup,
            specialization.num_simdgroups,
        ));

        let context = encoder.context();
        let pipeline = self.get_or_create(context, specialization)?;

        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => {
                pipeline.encode(
                    weights,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    (a, a_offset),
                    &mut *d,
                    output_bias,
                    rht_factors,
                    gather_indices,
                    k,
                    n,
                    m / specialization.m_tile,
                    ab_scale,
                    group_count_x,
                    soft_cap,
                    encoder,
                );
            },
            quant_b @ (MatmulB::ScaleBiasDequant {
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                ..
            }
            | MatmulB::ScaleSymmetricDequant {
                ..
            }) => {
                let (weights, scales, zero_points, biases) = match quant_b {
                    MatmulB::ScaleBiasDequant {
                        b: w,
                        scales,
                        biases,
                        ..
                    } => (w, scales, None, Some(biases)),
                    MatmulB::ScaleZeroPointDequant {
                        b: w,
                        scales,
                        zero_points,
                        ..
                    } => (w, scales, Some(zero_points), None),
                    MatmulB::ScaleSymmetricDequant {
                        b: w,
                        scales,
                        ..
                    } => (w, scales, None, None),
                    MatmulB::FullPrecision {
                        ..
                    } => unreachable!(),
                };
                pipeline.encode(
                    weights,
                    Some(scales),
                    zero_points,
                    biases,
                    (a, a_offset),
                    &mut *d,
                    output_bias,
                    rht_factors,
                    gather_indices,
                    k,
                    n,
                    m / specialization.m_tile,
                    ab_scale,
                    group_count_x,
                    soft_cap,
                    encoder,
                );

                // First-launch tile calibration: on the first sight of a
                // quantized decode shape, time the candidate tiles on these
                // same buffers and install the winner for all later
                // dispatches (and launches, via the on-disk cache). The
                // dispatch above already ran with the shipped default, so
                // decoding is never blocked on calibration.
                if m == 1
                    && !specialization.gathered
                    && specialization.bits != 0
                    && autotune::needs_calibration(context, n, k, specialization.group_size, specialization.bits)
                {
                    self.calibrate(
                        context,
                        &specialization,
                        weights,
                        scales,
                        zero_points,
                        biases,
                        (a, a_offset),
                        &mut *d,
                        output_bias,
                        rht_factors,
                        k,
                        n,
                        ab_scale,
                        soft_cap,
                    );
                }
            },
        }

        Ok(())
    }

    /// Times the candidate tiles for one quantized decode shape and records
    /// the winner (policy override + on-disk cache). Runs once per shape per
    /// device on the caller's live buffers, so the measured dispatches match
    /// production exactly; the output buffer is scratch here — the caller's
    /// real dispatch already produced its result.
    #[allow(clippy::too_many_arguments)]
    fn calibrate<'b, TB: BufferArg<'b, Metal>>(
        &mut self,
        context: &MetalContext,
        base: &GemvSpecialization,
        weights: TB,
        scales: &Allocation<Metal>,
        zero_points: Option<&Allocation<Metal>>,
        biases: Option<&Allocation<Metal>>,
        a: (&Allocation<Metal>, usize),
        d: &mut Allocation<Metal>,
        output_bias: Option<&Allocation<Metal>>,
        rht_factors: Option<&Allocation<Metal>>,
        k: u32,
        n: u32,
        ab_scale: f32,
        soft_cap: Option<f32>,
    ) {
        let has_rht = base.output_transform.contains(GemmDTransform::RHT);
        let fitted = policy::GemvTile {
            num_simdgroups: base.num_simdgroups,
            k_split: base.k_split,
            results_per_simdgroup: base.results_per_simdgroup,
            packs: base.packs,
        };
        let mut best: Option<(policy::GemvTile, std::time::Duration)> = None;
        for tile in autotune::candidates(base.group_size, base.bits, fitted) {
            let rows = (tile.num_simdgroups / tile.k_split) * tile.results_per_simdgroup;
            if n < tile.results_per_simdgroup || (has_rht && rows != 32) {
                continue;
            }
            let block = 32 * tile.packs * (32 / base.bits);
            let candidate = GemvSpecialization {
                input_aligned: k.is_multiple_of(block),
                k_split: tile.k_split,
                results_per_simdgroup: tile.results_per_simdgroup,
                num_simdgroups: tile.num_simdgroups,
                packs: tile.packs,
                ..*base
            };
            let group_count_x = n.div_ceil(rows_per_threadgroup(
                candidate.k_split,
                candidate.results_per_simdgroup,
                candidate.num_simdgroups,
            ));
            let Ok(mut encoder) = Encoder::<Metal>::new(context) else {
                break;
            };
            if self.get_or_create(context, candidate).is_err() {
                continue;
            }
            let pipeline = self.pipelines.get(&candidate).expect("pipeline just inserted");
            for _ in 0..autotune::ITERATIONS_PER_CANDIDATE {
                pipeline.encode(
                    weights,
                    Some(scales),
                    zero_points,
                    biases,
                    a,
                    &mut *d,
                    output_bias,
                    rht_factors,
                    None::<&Allocation<Metal>>,
                    k,
                    n,
                    1,
                    ab_scale,
                    group_count_x,
                    soft_cap,
                    &mut encoder,
                );
            }
            let Ok(completed) = encoder.end_encoding().submit().wait_until_completed() else {
                continue;
            };
            let elapsed = completed.gpu_execution_time();
            if best.as_ref().is_none_or(|(_, best_time)| elapsed < *best_time) {
                best = Some((tile, elapsed));
            }
        }
        match best {
            Some((winner, _)) => autotune::record_winner(context, n, k, base.group_size, base.bits, winner),
            None => autotune::mark_resolved(context, n, k, base.group_size, base.bits),
        }
    }
}
