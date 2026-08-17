use std::mem::size_of;

use super::{
    SamplingCombineMetalKernel, SamplingFinalizeMetalKernel, SamplingLoopPartialMetalKernel,
    SamplingPartialScanMetalKernel,
};
use crate::backends::{
    common::{Buffer, BufferArg, BufferArgMut, Encoder, kernel::UnifiedSamplingKernel},
    metal::{Metal, context::MetalContext, error::MetalError},
};

// One partial-scan threadgroup per slice; the combine kernels reduce the
// slices within a single simdgroup, so this must stay at most 32.
const NUM_SLICES: u32 = 32;

// Raw (buffer, offset, length) view so one BufferArgMut can feed several
// generated kernel encodes.
#[derive(Clone, Copy)]
struct RawArg<'a>(&'a dyn Buffer<Backend = Metal>, usize, usize);

impl<'a> BufferArg<'a, Metal> for RawArg<'a> {
    fn into_parts(self) -> (&'a dyn Buffer<Backend = Metal>, usize, usize) {
        (self.0, self.1, self.2)
    }
}

impl<'a> BufferArgMut<'a, Metal> for RawArg<'a> {
    fn into_parts(self) -> (&'a dyn Buffer<Backend = Metal>, usize, usize) {
        (self.0, self.1, self.2)
    }
}

/// The sampling pipeline: a partial scan over vocab slices feeds a one-simd
/// combine; with top-k/top-p/min-p filters the first rejection-loop iteration
/// runs sliced as well, and only its (rare) continuation stays serial. The
/// gumbel noise of a logit is a pure function of (seed, index), so every
/// stage and the CPU reference draw identical noise.
pub struct SamplingMetalKernel {
    partial_scan: SamplingPartialScanMetalKernel,
    combine: SamplingCombineMetalKernel,
    loop_partial: Option<SamplingLoopPartialMetalKernel>,
    finalize: Option<SamplingFinalizeMetalKernel>,
}

impl UnifiedSamplingKernel for SamplingMetalKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        #[allow(non_snake_case)] T: crate::data_type::DataType,
        is_stochastic: bool,
        has_bitmask: bool,
        has_temperature: bool,
        has_top_k: bool,
        has_top_p: bool,
        has_min_p: bool,
    ) -> Result<Self, MetalError> {
        let has_filters = has_top_k || has_top_p || has_min_p;
        Ok(Self {
            partial_scan: SamplingPartialScanMetalKernel::new(
                context,
                T,
                is_stochastic,
                has_bitmask,
                has_temperature,
                has_top_p,
                has_min_p,
            )?,
            combine: SamplingCombineMetalKernel::new(context, has_filters, has_top_p)?,
            loop_partial: has_filters
                .then(|| {
                    SamplingLoopPartialMetalKernel::new(
                        context,
                        T,
                        is_stochastic,
                        has_bitmask,
                        has_temperature,
                        has_top_k,
                        has_top_p,
                    )
                })
                .transpose()?,
            finalize: has_filters
                .then(|| {
                    SamplingFinalizeMetalKernel::new(
                        context,
                        T,
                        is_stochastic,
                        has_bitmask,
                        has_temperature,
                        has_top_k,
                        has_top_p,
                        has_min_p,
                    )
                })
                .transpose()?,
        })
    }

    fn encode<'logits, 'output, 'seeds, 'bitmask, 'encoder>(
        &self,
        logits: impl BufferArg<'logits, Metal>,
        output: impl BufferArgMut<'output, Metal>,
        seeds: Option<impl BufferArg<'seeds, Metal>>,
        bitmask: Option<impl BufferArg<'bitmask, Metal>>,
        temperature: Option<f32>,
        top_k: Option<u32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        vocab_size: u32,
        batch_size: u32,
        encoder: &'encoder mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        if batch_size == 0 {
            return Ok(());
        }
        let output = {
            let (buffer, offset, length) = output.into_parts();
            RawArg(buffer, offset, length)
        };

        let partial_bytes = batch_size as usize * NUM_SLICES as usize * 4 * size_of::<f32>();
        let mut scan_partials = encoder.allocate_scratch(partial_bytes)?;
        let mut state = encoder.allocate_scratch(batch_size as usize * 4 * size_of::<f32>())?;

        self.partial_scan.encode(
            logits,
            &mut scan_partials,
            seeds,
            bitmask,
            temperature,
            vocab_size,
            batch_size,
            NUM_SLICES,
            encoder,
        );
        self.combine.encode(&scan_partials, output, &mut state, batch_size, NUM_SLICES, encoder);

        let (Some(loop_partial), Some(finalize)) = (&self.loop_partial, &self.finalize) else {
            return Ok(());
        };
        let mut loop_partials = encoder.allocate_scratch(partial_bytes)?;
        loop_partial.encode(
            logits,
            &state,
            &mut loop_partials,
            seeds,
            bitmask,
            temperature,
            vocab_size,
            batch_size,
            NUM_SLICES,
            encoder,
        );
        finalize.encode(
            logits,
            output,
            &loop_partials,
            &state,
            seeds,
            bitmask,
            temperature,
            top_k,
            top_p,
            min_p,
            vocab_size,
            batch_size,
            NUM_SLICES,
            encoder,
        );
        Ok(())
    }
}
