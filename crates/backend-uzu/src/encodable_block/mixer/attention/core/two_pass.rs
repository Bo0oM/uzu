use crate::{
    backends::common::{
        Allocation, Backend, BufferArg, Encoder, Kernels,
        kernel::{AttentionTwoPass1Kernel, AttentionTwoPass2Kernel},
    },
    data_type::DataType,
    encodable_block::mixer::attention::core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
};

const INNER_DATA_TYPE: DataType = DataType::F32;
const TWO_PASS_BLOCKS: u32 = 32;

pub struct AttentionTwoPassCore<B: Backend> {
    head_dim: u32,
    num_groups: u32,
    num_q_heads: u32,
    sliding_window_size: Option<u32>,
    scale: Option<f32>,
    data_type: DataType,
    pass_1: <B::Kernels as Kernels>::AttentionTwoPass1Kernel,
    pass_1_q8: Option<<B::Kernels as Kernels>::AttentionTwoPass1Kernel>,
    pass_2: <B::Kernels as Kernels>::AttentionTwoPass2Kernel,
}

impl<B: Backend> AttentionTwoPassCore<B> {
    pub fn new(
        arguments: &AttentionCoreNewArguments,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let make_pass_1 = |kv_int8: bool| {
            <B::Kernels as Kernels>::AttentionTwoPass1Kernel::new(
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
        let pass_1 = make_pass_1(false)?;
        let pass_1_q8 = arguments.kv_int8.then(|| make_pass_1(true)).transpose()?;

        let pass_2 =
            <B::Kernels as Kernels>::AttentionTwoPass2Kernel::new(context, arguments.data_type, arguments.head_dim)?;

        Ok(Self {
            head_dim: arguments.head_dim,
            num_groups: arguments.num_groups,
            num_q_heads: arguments.num_q_heads,
            sliding_window_size: arguments.sliding_window_size,
            scale: arguments.scale,
            data_type: arguments.data_type,
            pass_1,
            pass_1_q8,
            pass_2,
        })
    }

    pub fn encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut partials = encoder.allocate_scratch_for_shape(
            &[arguments.suffix_length, self.num_q_heads, TWO_PASS_BLOCKS, self.head_dim],
            INNER_DATA_TYPE,
        )?;
        let mut sums = encoder.allocate_scratch_for_shape(
            &[arguments.suffix_length, self.num_q_heads, TWO_PASS_BLOCKS],
            INNER_DATA_TYPE,
        )?;
        let mut maxs = encoder.allocate_scratch_for_shape(
            &[arguments.suffix_length, self.num_q_heads, TWO_PASS_BLOCKS],
            INNER_DATA_TYPE,
        )?;

        macro_rules! encode_pass_1 {
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
                    &mut partials,
                    &mut sums,
                    &mut maxs,
                    self.num_q_heads / self.num_groups,
                    arguments.state_type.physical_prefix_length() + arguments.suffix_length,
                    self.head_dim,
                    self.num_groups * self.head_dim,
                    self.head_dim,
                    self.num_groups * self.head_dim,
                    arguments.state_type.ring_params(),
                    self.scale.unwrap_or(1.0f32 / (self.head_dim as f32).sqrt()),
                    self.num_q_heads,
                    arguments.suffix_length,
                    arguments.trie,
                    self.sliding_window_size,
                    arguments.sinks,
                    encoder,
                )
            };
        }
        if let Some(kv_quant) = arguments.kv_quant {
            let kernel = self.pass_1_q8.as_ref().expect("quantized KV cache requires the int8 kernel variant");
            encode_pass_1!(
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
            encode_pass_1!(
                self.pass_1,
                Some(arguments.keys),
                Some(arguments.values),
                None::<&Allocation<B>>,
                None::<&Allocation<B>>,
                None::<&Allocation<B>>,
                None::<&Allocation<B>>,
                None
            );
        }

        let mut output = encoder
            .allocate_constant_for_shape(&[arguments.suffix_length, self.num_q_heads, self.head_dim], self.data_type)?;
        self.pass_2.encode(&partials, &sums, &maxs, &mut output, self.num_q_heads, arguments.suffix_length, encoder);

        Ok(output)
    }
}
