use std::cmp::Ordering;

use half::bf16;
use num_traits::Float;

use crate::{
    array::ArrayElement,
    backends::{
        common::{BufferArg, BufferArgMut, Encoder, kernel::UnifiedSamplingKernel},
        cpu::{Cpu, context::CpuContext, error::CpuError},
    },
    data_type::DataType,
    encodable_block::sampling::{gumbel_float, revidx},
    utils::pointers::{SendPtr, SendPtrMut},
};

const CANDIDATE_SEED: usize = 64;
const CANDIDATE_GROWTH: usize = 8;

// NOTE: top_k + top_p combination is not exactly matching lalamo ("parallel" here, should be top-k then top-p)
#[allow(clippy::too_many_arguments)]
pub fn unified_sampling<T: ArrayElement + Float>(
    logits: *const T,
    output: *mut u32,
    seeds: Option<*const u64>,
    bitmask: Option<*const u32>,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    vocab_size: u32,
    batch_size: u32,
) {
    let vocab = vocab_size as usize;
    debug_assert!(vocab > 0, "vocab_size must be positive");
    let filtered = top_k.is_some() || top_p.is_some() || min_p.is_some();

    let mut scores = vec![0.0f32; vocab];
    let mut candidates: Vec<u32> = if filtered {
        (0..vocab_size).collect()
    } else {
        Vec::new()
    };
    let mut kept: Vec<(u32, f32)> = Vec::new();

    for batch_idx in 0..batch_size {
        let row = unsafe { std::slice::from_raw_parts(logits.wrapping_add((vocab_size * batch_idx) as usize), vocab) };
        for (score, logit) in scores.iter_mut().zip(row) {
            // NaN logits would break the comparator's strict weak ordering and
            // could win argmax; treat them as masked-out.
            let value = logit.to_f32().unwrap();
            *score = if value.is_nan() {
                f32::NEG_INFINITY
            } else {
                value
            };
        }

        if let Some(bitmask) = bitmask {
            let bitmask = unsafe {
                std::slice::from_raw_parts(
                    bitmask.wrapping_add((vocab_size.div_ceil(u32::BITS) * batch_idx) as usize),
                    vocab_size.div_ceil(u32::BITS) as usize,
                )
            };
            for (logit_index, logit) in scores.iter_mut().enumerate() {
                if bitmask[logit_index / (u32::BITS as usize)] & (1 << (logit_index % (u32::BITS as usize))) == 0 {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }

        if let Some(temperature) = temperature {
            let recip_temperature = 1.0 / temperature;
            for logit in scores.iter_mut() {
                *logit *= recip_temperature;
            }
        }

        if filtered {
            filter_candidates(&mut scores, &mut candidates, &mut kept, top_k, top_p, min_p);
        }

        if let Some(seeds) = seeds {
            let seed = unsafe { *seeds.wrapping_add(batch_idx as usize) };
            for (logit_index, logit) in scores.iter_mut().enumerate() {
                if *logit != f32::NEG_INFINITY {
                    *logit += gumbel_float(seed, revidx(logit_index as u32));
                }
            }
        }

        let mut argmax = 0usize;
        let mut best = scores[0];
        for (index, &value) in scores.iter().enumerate().skip(1) {
            if value > best {
                best = value;
                argmax = index;
            }
        }

        unsafe { *output.wrapping_add(batch_idx as usize) = argmax as u32 }
    }
}

#[inline(always)]
fn candidate_cmp(
    scores: &[f32],
    left: u32,
    right: u32,
) -> Ordering {
    scores[right as usize].total_cmp(&scores[left as usize]).then(left.cmp(&right))
}

fn filter_candidates(
    scores: &mut [f32],
    candidates: &mut [u32],
    kept: &mut Vec<(u32, f32)>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
) {
    let vocab = scores.len();
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut limit = vocab;
    if let Some(top_k) = top_k {
        limit = limit.min(top_k as usize);
    }
    let min_p_threshold = min_p.map(|min_p| max + min_p.ln());
    if let Some(threshold) = min_p_threshold
        && threshold.is_finite()
    {
        limit = limit.min(scores.iter().filter(|score| **score >= threshold).count().max(1));
    }
    limit = limit.max(1);

    let norm = top_p.map(|_| {
        let mut sum = 0.0f32;
        let mut compensation = 0.0f32;
        for &score in scores.iter() {
            let term = (score - max).exp() - compensation;
            let next = sum + term;
            compensation = (next - sum) - term;
            sum = next;
        }
        sum
    });

    let mut cap = if top_p.is_some() {
        CANDIDATE_SEED.min(limit)
    } else {
        limit
    };

    loop {
        if cap < vocab {
            candidates.select_nth_unstable_by(cap - 1, |left, right| candidate_cmp(scores, *left, *right));
        }
        candidates[..cap].sort_unstable_by(|left, right| candidate_cmp(scores, *left, *right));

        kept.clear();
        let mut mass = 0.0f32;
        let mut stopped = false;
        for (rank, &index) in candidates[..cap].iter().enumerate() {
            let value = scores[index as usize];
            if top_k.is_some_and(|top_k| rank as u32 >= top_k)
                || top_p.is_some_and(|top_p| mass >= top_p)
                || min_p_threshold.is_some_and(|threshold| value < threshold)
            {
                stopped = true;
                break;
            }
            kept.push((index, value));
            if let Some(norm) = norm {
                mass += (value - max).exp() / norm;
            }
        }

        if stopped || cap == limit {
            break;
        }
        cap = (cap * CANDIDATE_GROWTH).min(limit);
    }

    scores.fill(f32::NEG_INFINITY);
    for &(index, value) in kept.iter() {
        scores[index as usize] = value;
    }
}

#[allow(non_snake_case)]
pub struct UnifiedSamplingCpuKernel {
    T: DataType,
}

impl UnifiedSamplingKernel for UnifiedSamplingCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &CpuContext,
        #[allow(non_snake_case)] T: DataType,
        _is_stochastic: bool,
        _has_bitmask: bool,
        _has_temperature: bool,
        _has_top_k: bool,
        _has_top_p: bool,
        _has_min_p: bool,
    ) -> Result<Self, CpuError> {
        Ok(Self {
            T,
        })
    }

    fn encode<'logits, 'output, 'seeds, 'bitmask, 'encoder>(
        &self,
        logits: impl BufferArg<'logits, Cpu>,
        output: impl BufferArgMut<'output, Cpu>,
        seeds: Option<impl BufferArg<'seeds, Cpu>>,
        bitmask: Option<impl BufferArg<'bitmask, Cpu>>,
        temperature: Option<f32>,
        top_k: Option<u32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        vocab_size: u32,
        batch_size: u32,
        encoder: &'encoder mut Encoder<Cpu>,
    ) -> Result<(), CpuError> {
        let logits = unsafe {
            let (buffer, offset, _) = logits.into_parts();
            SendPtr((&*buffer.downcast().get()).as_ptr().byte_add(offset))
        };
        let output = unsafe {
            let (buffer, offset, _) = output.into_parts();
            SendPtrMut((&mut *buffer.downcast().get()).as_mut_ptr().byte_add(offset))
        };
        let seeds = seeds.map(|arg| unsafe {
            let (buffer, offset, _) = arg.into_parts();
            SendPtr((&*buffer.downcast().get()).as_ptr().byte_add(offset))
        });
        let bitmask = bitmask.map(|arg| unsafe {
            let (buffer, offset, _) = arg.into_parts();
            SendPtr((&*buffer.downcast().get()).as_ptr().byte_add(offset))
        });
        macro_rules! push {
            ($t:ty) => {
                encoder.as_command_buffer_mut().push_command(move || {
                    unified_sampling::<$t>(
                        logits.as_ptr() as _,
                        output.as_ptr() as _,
                        seeds.map(|p| p.as_ptr() as _),
                        bitmask.map(|p| p.as_ptr() as _),
                        temperature,
                        top_k,
                        top_p,
                        min_p,
                        vocab_size,
                        batch_size,
                    )
                })
            };
        }
        match self.T {
            DataType::F32 => push!(f32),
            DataType::BF16 => push!(bf16),
            variant => unimplemented!("variant doesn't exist: {variant:?}"),
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/backends/cpu/kernel/sampling/unified_sampling_test.rs"]
mod tests;
