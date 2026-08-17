use crate::{
    backends::common::{
        Allocation, Backend, BufferArg, Encoder, Kernels,
        kernel::{KVCacheDequantKernel, attention_gemm::AttentionGemmCore as AttentionGemmCoreTrait},
    },
    data_type::DataType,
    encodable_block::mixer::attention::{
        core::{fallback::AttentionFallbackCore, single_pass::AttentionSinglePassCore, two_pass::AttentionTwoPassCore},
        state::AttentionStateType,
    },
};

mod fallback;
mod single_pass;
mod two_pass;

pub struct AttentionCoreNewArguments {
    pub head_dim: u32,
    pub num_groups: u32,
    pub num_q_heads: u32,
    pub has_sinks: bool,
    pub is_kv_cache_ring: bool,
    pub is_causal: bool,
    pub is_trie: bool,
    pub sliding_window_size: Option<u32>,
    pub scale: Option<f32>,
    pub data_type: DataType,
    pub kv_int8: bool,
}

/// Scale side-buffers of an int8 KV cache (ADR-8); present iff `keys`/`values`
/// in the encode arguments hold quantized planes.
pub struct AttentionKvQuant<'a, B: Backend> {
    pub key_scales: &'a B::DenseBuffer,
    pub value_scales: &'a B::DenseBuffer,
}

impl<B: Backend> Clone for AttentionKvQuant<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: Backend> Copy for AttentionKvQuant<'_, B> {}

pub struct AttentionCoreEncodeArguments<'a, B: Backend, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>> {
    pub queries: &'a Allocation<B>,
    pub keys: KT,
    pub values: VT,
    pub suffix_length: u32,
    pub trie: Option<&'a Allocation<B>>,
    pub sinks: Option<&'a Allocation<B>>,
    pub state_type: &'a AttentionStateType,
    pub kv_quant: Option<AttentionKvQuant<'a, B>>,
}

pub struct AttentionCores<B: Backend> {
    head_dim: u32,
    num_groups: u32,
    data_type: DataType,
    gemm: Option<<B::Kernels as Kernels>::AttentionGemmCore>,
    fallback: Option<AttentionFallbackCore<B>>,
    two_pass: AttentionTwoPassCore<B>,
    single_pass: AttentionSinglePassCore<B>,
    dequant: Option<<B::Kernels as Kernels>::KVCacheDequantKernel>,
}

impl<B: Backend> AttentionCores<B> {
    pub fn new(
        arguments: AttentionCoreNewArguments,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let gemm = if <<B::Kernels as Kernels>::AttentionGemmCore as AttentionGemmCoreTrait<B>>::is_supported(
            &arguments, context,
        )? {
            Some(<<B::Kernels as Kernels>::AttentionGemmCore as AttentionGemmCoreTrait<B>>::new(context, &arguments)?)
        } else {
            None
        };
        let fallback = if arguments.head_dim == 512 && !arguments.is_trie {
            Some(AttentionFallbackCore::new(&arguments, context)?)
        } else {
            None
        };
        let two_pass = AttentionTwoPassCore::new(&arguments, context)?;
        let single_pass = AttentionSinglePassCore::new(&arguments, context)?;
        let dequant = arguments
            .kv_int8
            .then(|| <B::Kernels as Kernels>::KVCacheDequantKernel::new(context, arguments.data_type))
            .transpose()?;

        Ok(Self {
            head_dim: arguments.head_dim,
            num_groups: arguments.num_groups,
            data_type: arguments.data_type,
            gemm,
            fallback,
            two_pass,
            single_pass,
            dequant,
        })
    }
    pub fn encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, <B as Backend>::Error> {
        // Prefill cores keep their full-precision layouts: expand a quantized
        // cache into scratch first and re-enter with plain buffers (ADR-8).
        let use_matrix_core = arguments.suffix_length > 8 && (self.gemm.is_some() || self.fallback.is_some());
        if let Some(kv_quant) = arguments.kv_quant
            && use_matrix_core
        {
            let rows = arguments.state_type.physical_prefix_length() + arguments.suffix_length;
            let mut keys =
                encoder.allocate_scratch_for_shape(&[rows, self.num_groups, self.head_dim], self.data_type)?;
            let mut values =
                encoder.allocate_scratch_for_shape(&[rows, self.num_groups, self.head_dim], self.data_type)?;
            let element_count = rows * self.num_groups * self.head_dim;
            let dequant = self.dequant.as_ref().expect("quantized KV cache requires the dequant bridge kernel");
            encoder.push_debug_group("kv cache dequant");
            dequant.encode(arguments.keys, kv_quant.key_scales, &mut keys, self.head_dim, element_count, encoder);
            dequant.encode(arguments.values, kv_quant.value_scales, &mut values, self.head_dim, element_count, encoder);
            encoder.pop_debug_group();
            return self.encode(
                AttentionCoreEncodeArguments {
                    queries: arguments.queries,
                    keys: &keys,
                    values: &values,
                    suffix_length: arguments.suffix_length,
                    trie: arguments.trie,
                    sinks: arguments.sinks,
                    state_type: arguments.state_type,
                    kv_quant: None,
                },
                encoder,
            );
        }

        encoder.push_debug_group("attention core");

        let output = if use_matrix_core {
            if let Some(gemm) = &self.gemm {
                gemm.encode(arguments, encoder)
            } else {
                self.fallback.as_ref().unwrap().encode(arguments, encoder)
            }
        } else if arguments.state_type.physical_prefix_length() + arguments.suffix_length > 1024 {
            self.two_pass.encode(arguments, encoder)
        } else {
            self.single_pass.encode(arguments, encoder)
        }?;

        encoder.pop_debug_group();

        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/encodable_block/attention_test.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/encodable_block/attention_gemm_test.rs"]
mod gemm_tests;
