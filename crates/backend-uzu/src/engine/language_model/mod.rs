use std::{fs::File, io, io::BufReader, path::Path, sync::Arc};

use thiserror::Error;

use crate::{
    backends::common::{Backend, Context, DeviceCapabilities, Kernels, kernel::ContextRingUpdateKernel},
    config::{model::{generation::GenerationConfig, language_model::LanguageModelConfig}, token_mixer::AnyTokenMixerConfig},
    data_type::DataType,
    encodable_block::{
        decoder::{Decoder, DecoderError},
        mixer::attention::state::kv_cache_int8_override,
        sampling::{Sampling, SamplingMethod},
    },
    engine::Engine,
    parameters::{HeaderLoadingError, ParameterLoader, ParameterLoaderError},
    speculators::{
        Speculator,
        dflash_tfm::{DFlashSpeculatorLoadError, DFlashTfmSpeculator},
        prompt_lookup::PromptLookupSpeculator,
    },
};

pub mod state;
pub mod stream;

#[cfg(grammar)]
pub mod grammar;

/// Fallback window for a model-declared repetition penalty when the context
/// length is unknown; long enough to cover the repetition loops the penalty
/// exists to break, short enough to stay a rounding error in memory.
const DEFAULT_SUFFIX_REPETITION_LENGTH: u32 = 512;

pub struct LanguageModel<B: Backend> {
    engine: Arc<Engine<B>>,
    decoder: Decoder<B>,
    speculator: Option<Speculator<B>>,
    sampling: Sampling<B>,
    context_ring_update: <B::Kernels as Kernels>::ContextRingUpdateKernel,
    generation_config: GenerationConfig,
    #[cfg(grammar)]
    vocab_size: usize,
}

#[derive(Debug, Error)]
pub enum EngineLoadLanguageModelError<B: Backend> {
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("HeaderLoading error: {0}")]
    HeaderLoading(#[from] HeaderLoadingError),
    #[error("ParameterLoader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
    #[error("Speculator error: {0}")]
    Speculator(#[from] DFlashSpeculatorLoadError<B>),
}

// Default-on since the ADR-11 gates passed (copy-heavy +83..+136%, chat
// within -2%, long/sampled/qual checks bitwise-clean); UZU_PROMPT_LOOKUP=0
// switches CPU prompt-lookup drafting off. Only reaches bf16 models with
// tree-verify support and no shipped DFlash speculator.
fn prompt_lookup_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("UZU_PROMPT_LOOKUP").map_or(true, |v| v != "0"))
}

/// Whether compressing the KV cache to int8 pays for this model.
///
/// Decode reads every weight once per token, so the KV cache only matters in
/// proportion to it. Measured on M1 Max: Qwen3-0.6B at a 16k context carries
/// 1.9 GB of cache against 1.1 GB of weights — a 63% share — and int8 buys 31
/// to 39% throughput; Foundation-Sec-8B at the same context carries 2.2 GB
/// against 16.1 GB — 12% — and int8 costs 4%, because quantising work grows
/// with the cache while the saving stays a fraction of a small term. The
/// threshold sits between the two measured points, nearer the losing one so a
/// model has to clearly benefit before its cache stops being exact.
const KV_TRAFFIC_SHARE_FOR_INT8: f64 = 0.35;

/// The context a session is expected to run at.
///
/// Deliberately not `recommended_context_length`, which answers a different
/// question — how much context the model *permits*. Where sparse buffers make
/// an unused cache free, that method returns the architectural maximum: 128k
/// for LFM2-350M, 256k for Qwen3.5-0.8B-M. Sizing this decision by that number
/// says the cache dominates the per-token traffic of almost every small model,
/// and it does not, because the cache never fills. Reusing it was tried and
/// measured: LFM2-350M flips to a quantized cache and loses, 389 t/s against
/// 394 exact.
///
/// So the clamp stays, and it is an estimate of what a session reaches rather
/// than of what the model allows.
fn intended_context_length(config: &LanguageModelConfig) -> Option<u32> {
    let platform_ceiling: u32 = if cfg!(target_os = "ios") {
        8192
    } else {
        16384
    };
    config
        .decoder_config
        .transformer_config
        .layer_configs
        .iter()
        .filter_map(|layer| layer.rope_config.as_ref().map(|rope| *rope.max_sequence_length()))
        .max()
        .map(|max_length| max_length.min(platform_ceiling))
}

fn kv_int8_worthwhile<B: Backend>(
    config: &LanguageModelConfig,
    weight_loader: &ParameterLoader<B>,
    data_type: DataType,
) -> bool {
    if let Some(forced) = kv_cache_int8_override() {
        return forced;
    }
    let weight_bytes = weight_loader.total_weight_bytes();
    if weight_bytes == 0 {
        return false;
    }
    let element_bytes = data_type.size_in_bits() as u64 / 8;
    // The context a session will actually run at, not the architectural
    // maximum: a cache sized for 128k that never fills would justify itself on
    // paper and lose in practice.
    let Some(context_length) = intended_context_length(config) else {
        return false;
    };
    let kv_bytes: u64 = config
        .decoder_config
        .transformer_config
        .layer_configs
        .iter()
        .filter_map(|layer| match &layer.mixer_config {
            AnyTokenMixerConfig::AttentionConfig(attention) => (!attention.is_kv_sharing).then(|| {
                // Keys and values, one row each per token and KV head.
                let per_token = 2 * attention.num_groups as u64 * attention.head_dim as u64 * element_bytes;
                // A sliding-window layer never holds more than its window, however
                // long the session runs. Charging it the full context is how
                // gemma-3-1b, whose 22 of 26 layers see only 512 tokens, looked
                // like a 44% cache when it is really 12%, and got a quantized
                // cache that cost it 4% of decode.
                let held = attention.sliding_window_size.map_or(context_length, |window| window.min(context_length));
                per_token.saturating_mul(held as u64)
            }),
            _ => None,
        })
        .sum();
    if kv_bytes == 0 {
        return false;
    }
    let share = kv_bytes as f64 / (kv_bytes + weight_bytes) as f64;
    share >= KV_TRAFFIC_SHARE_FOR_INT8
}

impl<B: Backend> Engine<B> {
    pub fn load_language_model(
        self: &Arc<Self>,
        model_path: &Path,
    ) -> Result<LanguageModel<B>, EngineLoadLanguageModelError<B>> {
        let config: LanguageModelConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;

        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weight_loader = ParameterLoader::new(&weights_file, &*self.context)?;

        // TODO
        let speculator_path = model_path.join("speculator");
        let speculator_path = speculator_path.exists().then_some(speculator_path);

        self.build_language_model(config, &weight_loader, speculator_path.as_deref())
    }

    // TODO: better design
    pub fn load_language_model_random(
        self: &Arc<Self>,
        config_path: &Path,
        header_path: &Path,
        seed: u64,
    ) -> Result<LanguageModel<B>, EngineLoadLanguageModelError<B>> {
        let config: LanguageModelConfig = serde_json::from_reader(BufReader::new(File::open(config_path)?))?;

        let header_file = File::open(header_path)?;
        let weight_loader = ParameterLoader::new_random(&header_file, &*self.context, seed)?;

        self.build_language_model(config, &weight_loader, None)
    }

    fn build_language_model(
        self: &Arc<Self>,
        config: LanguageModelConfig,
        weight_loader: &ParameterLoader<B>,
        speculator_path: Option<&Path>,
    ) -> Result<LanguageModel<B>, EngineLoadLanguageModelError<B>> {
        let data_type = DataType::BF16;
        let kv_int8 = kv_int8_worthwhile(&config, weight_loader, data_type);

        let decoder = Decoder::new(
            self.context.as_ref(),
            &config.decoder_config,
            &weight_loader.tree().subtree("decoder"),
            data_type,
            kv_int8,
        )?;

        assert!(
            speculator_path.is_none() || decoder.speculation_supported(),
            "attempted to load speculator for a model that doesn't support one"
        );

        let speculator = if let Some(speculator_path) = speculator_path {
            Some(Speculator::DFlash(DFlashTfmSpeculator::new(speculator_path, self.context.clone())?))
        } else if decoder.speculation_supported() && !weight_loader.has_integer_tensors() && prompt_lookup_enabled() {
            // Quantized models stay out: their batched verification pass
            // costs ~4x the single-token pass (dequant is ALU-bound, see
            // ADR-11), which no realistic acceptance rate pays back.
            Some(Speculator::PromptLookup(PromptLookupSpeculator::default()))
        } else {
            None
        };

        let sampling = Sampling::new(data_type, config.decoder_config.vocab_size);

        let context_ring_update = <B::Kernels as Kernels>::ContextRingUpdateKernel::new(&self.context)
            .map_err(EngineLoadLanguageModelError::Backend)?;

        weight_loader.tree().assert_all_tensors_validated()?;

        let generation_config = config.generation_config;

        #[cfg(grammar)]
        let vocab_size = config.decoder_config.vocab_size as usize;

        Ok(LanguageModel {
            engine: self.clone(),
            decoder,
            speculator,
            sampling,
            context_ring_update,
            generation_config,
            #[cfg(grammar)]
            vocab_size,
        })
    }
}

impl<B: Backend> LanguageModel<B> {
    pub fn max_context_length(&self) -> Option<u32> {
        self.decoder.max_context_length()
    }

    pub fn recommended_context_length(&self) -> Option<u32> {
        let max_context_length = self.max_context_length();

        // TODO: This is not the correct way to do it, there should be a real memory model
        if self.engine.context.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS) {
            // We just assume that all mixers use sparse if it's available to make max context free until it's actually used
            // Currenlty true for all mixers in uzu:
            // - full attention uses sparse if it's available to make max context free until it's actually used
            // - sliding window attention is bound, usually well below the recommended max context size on non-sparse (but can be made to use sparse if we care about it enough)
            // - short conv/mamba2/delta net are constant state size
            max_context_length
        } else if let Some(max_context_length) = max_context_length {
            // If sparse buffers aren't supported and model has finite maximum context length we assume that kv cache is expensive enough that we should probably clamp it to
            // something reasonable-ish for the platform. This is very primitive but works I guess...
            let platform_recommended_context_length = if cfg!(target_os = "ios") {
                8192
            } else {
                16384
            };

            Some(u32::min(max_context_length, platform_recommended_context_length))
        } else {
            // We just assume that unlimited context means constant state size on all mixers and is thus free
            None
        }
    }

    pub fn speculation_supported(&self) -> bool {
        self.decoder.speculation_supported()
    }

    pub fn default_sampling_method(&self) -> SamplingMethod {
        // HuggingFace applies `repetition_penalty` over the whole context and has
        // no field for a window, so checkpoints ship the penalty alone (Llama-3.1
        // fine-tunes commonly set 1.1). The sampler penalises a trailing window
        // and cannot run without one, so pair the two here: without a window the
        // penalty would panic the first decode.
        let mut method = SamplingMethod::Stochastic {
            temperature: self.generation_config.temperature,
            top_k: self.generation_config.top_k,
            top_p: self.generation_config.top_p,
            min_p: self.generation_config.min_p,
            repetition_penalty: self.generation_config.repetition_penalty,
            suffix_repetition_length: self.generation_config.suffix_repetition_length,
        };
        self.pair_repetition_penalty_with_a_window(&mut method);
        method
    }

    /// The sampler penalises a trailing window and cannot run without one, so a
    /// penalty that arrives without a window gets the default.
    ///
    /// HuggingFace applies `repetition_penalty` over the whole context and has
    /// no field for a window, so checkpoints ship the penalty alone (Llama-3.1
    /// fine-tunes commonly set 1.1) — and a caller building the method itself
    /// has the same gap. Filling it in at each construction site was the first
    /// fix and it missed one: a request arriving through the bridge with a
    /// penalty and no window still reached the panic. Both paths go through
    /// `LanguageModelStream::new`, so the pairing belongs there, where the
    /// model that knows the default is also in hand.
    pub(crate) fn pair_repetition_penalty_with_a_window(
        &self,
        method: &mut SamplingMethod,
    ) {
        let SamplingMethod::Stochastic {
            repetition_penalty: Some(_),
            suffix_repetition_length: window @ None,
            ..
        } = method
        else {
            return;
        };
        *window = Some(self.recommended_context_length().unwrap_or(DEFAULT_SUFFIX_REPETITION_LENGTH));
    }

    pub fn generation_config(&self) -> &GenerationConfig {
        &self.generation_config
    }
}

#[cfg(test)]
mod kv_traffic_tests {
    use proc_macros::uzu_test;

    /// The rule the engine applies, restated on the shapes it was fitted to.
    /// Qwen3-0.6B at a 16k context is cache-dominated and int8 buys 31-39%;
    /// Foundation-Sec-8B at the same context is weight-dominated and int8 costs
    /// 4%; gemma-3-1b-4bit looks cache-dominated until its sliding windows are
    /// counted, and int8 costs it 4%. All measured on M1 Max, 2026-08-18.
    ///
    /// `windows` gives each layer's sliding window, `None` for a layer that
    /// keeps the whole context.
    fn share(
        windows: &[Option<u64>],
        kv_heads: u64,
        head_dim: u64,
        context: u64,
        weight_bytes: u64,
    ) -> f64 {
        let per_token = kv_heads * head_dim * 2 * 2;
        let kv_bytes: u64 =
            windows.iter().map(|window| per_token * window.map_or(context, |window| window.min(context))).sum();
        kv_bytes as f64 / (kv_bytes + weight_bytes) as f64
    }

    fn full_attention(layers: usize) -> Vec<Option<u64>> {
        vec![None; layers]
    }

    #[uzu_test]
    fn small_model_at_long_context_is_cache_bound() {
        let qwen = share(&full_attention(28), 8, 128, 16384, 1_100_000_000);
        assert!(qwen > super::KV_TRAFFIC_SHARE_FOR_INT8, "share {qwen}");
    }

    #[uzu_test]
    fn large_model_stays_weight_bound() {
        let foundation_sec = share(&full_attention(32), 8, 128, 16384, 16_100_000_000);
        assert!(foundation_sec < super::KV_TRAFFIC_SHARE_FOR_INT8, "share {foundation_sec}");
    }

    #[uzu_test]
    fn small_model_at_short_context_stays_exact() {
        let qwen_short = share(&full_attention(28), 8, 128, 2048, 1_100_000_000);
        assert!(qwen_short < super::KV_TRAFFIC_SHARE_FOR_INT8, "share {qwen_short}");
    }

    /// gemma-3-1b-4bit: 22 of its 26 layers see only 512 tokens. Charging every
    /// layer the full context puts it at 44% and hands it a quantized cache
    /// that costs 4% of decode; counting the windows puts it at 12%, where it
    /// belongs.
    #[uzu_test]
    fn sliding_windows_are_not_charged_the_whole_context() {
        let mut windows = vec![Some(512); 22];
        windows.extend(full_attention(4));
        let gemma = share(&windows, 1, 256, 16384, 806_000_000);
        assert!(gemma < super::KV_TRAFFIC_SHARE_FOR_INT8, "share {gemma}");

        let ignoring_windows = share(&full_attention(26), 1, 256, 16384, 806_000_000);
        assert!(
            ignoring_windows > super::KV_TRAFFIC_SHARE_FOR_INT8,
            "the window-blind reading has to land on the other side of the threshold, \
             otherwise this test would pass without the fix: {ignoring_windows}"
        );
    }
}
