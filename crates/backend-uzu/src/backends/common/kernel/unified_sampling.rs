use crate::{
    backends::common::{Backend, BufferArg, BufferArgMut, Encoder},
    data_type::DataType,
};

/// One sampled token id per batch row: gumbel-max over masked/tempered
/// logits, constrained by the optional top-k/top-p/min-p filters. The gumbel
/// noise of a logit is a pure function of (seed, logit index), so every
/// backend and grid layout draws identical samples for the same seed.
pub trait UnifiedSamplingKernel: Sized + Send + Sync {
    type Backend: Backend;

    #[allow(non_snake_case)]
    fn new(
        context: &<Self::Backend as Backend>::Context,
        T: DataType,
        is_stochastic: bool,
        has_bitmask: bool,
        has_temperature: bool,
        has_top_k: bool,
        has_top_p: bool,
        has_min_p: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    #[allow(clippy::too_many_arguments)]
    fn encode<'logits, 'output, 'seeds, 'bitmask, 'encoder>(
        &self,
        logits: impl BufferArg<'logits, Self::Backend>,
        output: impl BufferArgMut<'output, Self::Backend>,
        seeds: Option<impl BufferArg<'seeds, Self::Backend>>,
        bitmask: Option<impl BufferArg<'bitmask, Self::Backend>>,
        temperature: Option<f32>,
        top_k: Option<u32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        vocab_size: u32,
        batch_size: u32,
        encoder: &'encoder mut Encoder<Self::Backend>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}
