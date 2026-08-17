use crate::{
    backends::common::{Allocation, Backend, BufferArg, Encoder, Kernels, kernel::AttentionSinglePassKernel},
    data_type::DataType,
    encodable_block::mixer::attention::core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
};

/// Mirrors LOAD_WIDTH in attention_single_pass.metal.
const KV_LOAD_WIDTH: u32 = 4;

pub struct AttentionSinglePassCore<B: Backend> {
    head_dim: u32,
    num_groups: u32,
    num_q_heads: u32,
    sliding_window_size: Option<u32>,
    scale: Option<f32>,
    data_type: DataType,
    kernel: <B::Kernels as Kernels>::AttentionSinglePassKernel,
    kernel_q8: Option<<B::Kernels as Kernels>::AttentionSinglePassKernel>,
}

impl<B: Backend> AttentionSinglePassCore<B> {
    pub fn new(
        arguments: &AttentionCoreNewArguments,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let make_kernel = |kv_int8: bool| {
            <B::Kernels as Kernels>::AttentionSinglePassKernel::new(
                context,
                arguments.data_type,
                arguments.head_dim,
                arguments.has_sinks,
                kv_int8,
                arguments.is_kv_cache_ring,
                arguments.is_causal,
                arguments.is_trie,
                arguments.sliding_window_size.is_some(),
            )
        };
        let kernel = make_kernel(false)?;
        let kernel_q8 = arguments.kv_int8.then(|| make_kernel(true)).transpose()?;

        Ok(Self {
            head_dim: arguments.head_dim,
            num_groups: arguments.num_groups,
            num_q_heads: arguments.num_q_heads,
            sliding_window_size: arguments.sliding_window_size,
            scale: arguments.scale,
            data_type: arguments.data_type,
            kernel,
            kernel_q8,
        })
    }

    pub fn encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output = encoder
            .allocate_constant_for_shape(&[arguments.suffix_length, self.num_q_heads, self.head_dim], self.data_type)?;
        let head_stride = self.head_dim;
        let seq_stride = self.num_groups * self.head_dim;
        // The kernel loads K/V as vec<T, KV_LOAD_WIDTH>; both strides must stay multiples of that width.
        debug_assert_eq!(head_stride % KV_LOAD_WIDTH, 0, "K/V head stride must stay a KV_LOAD_WIDTH multiple");
        debug_assert_eq!(seq_stride % KV_LOAD_WIDTH, 0, "K/V sequence stride must stay a KV_LOAD_WIDTH multiple");

        macro_rules! encode_with {
            ($kernel:expr, $keys:expr, $values:expr, $keys_q8:expr, $values_q8:expr, $key_scales:expr, $value_scales:expr, $num_kv_heads:expr) => {
                $kernel.encode(
                    arguments.queries,
                    $keys,
                    $values,
                    $keys_q8,
                    $values_q8,
                    $key_scales,
                    $value_scales,
                    $num_kv_heads,
                    &mut output,
                    self.num_q_heads / self.num_groups,
                    arguments.state_type.physical_prefix_length() + arguments.suffix_length,
                    head_stride,
                    seq_stride,
                    head_stride,
                    seq_stride,
                    arguments.state_type.ring_params(),
                    self.scale.unwrap_or(1.0f32 / (self.head_dim as f32).sqrt()),
                    arguments.trie,
                    self.sliding_window_size,
                    arguments.sinks,
                    self.num_q_heads,
                    arguments.suffix_length,
                    encoder,
                )
            };
        }
        if let Some(kv_quant) = arguments.kv_quant {
            let kernel = self.kernel_q8.as_ref().expect("quantized KV cache requires the int8 kernel variant");
            encode_with!(
                kernel,
                None::<&Allocation<B>>,
                None::<&Allocation<B>>,
                Some(arguments.keys),
                Some(arguments.values),
                Some(kv_quant.key_scales),
                Some(kv_quant.value_scales),
                Some(self.num_groups)
            );
        } else {
            encode_with!(
                self.kernel,
                Some(arguments.keys),
                Some(arguments.values),
                None::<&Allocation<B>>,
                None::<&Allocation<B>>,
                None::<&Allocation<B>>,
                None::<&Allocation<B>>,
                None
            );
        }

        Ok(output)
    }
}
