use std::any::Any;

use crate::{
    array::size_for_shape,
    backends::common::{
        Backend, Buffer, Context, DeviceCapabilities, Encoder, Kernels, SparseBuffer,
        gpu_types::{Copy, ring::RingParams},
        kernel::KVCacheUpdateKernel,
    },
    data_type::DataType,
    encodable_block::mixer::{MixerState, attention::Attention},
};

pub(crate) const ATTENTION_SUFFIX_CAPACITY: u32 = 1024; // TODO: remove hardcoded suffix capacity

/// Explicit `UZU_KV_INT8` setting, when there is one: `0` forces the exact
/// cache, anything else forces int8. Without it the engine decides per model
/// from how much of the per-token traffic the cache accounts for.
pub(crate) fn kv_cache_int8_override() -> Option<bool> {
    static SETTING: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *SETTING.get_or_init(|| std::env::var("UZU_KV_INT8").ok().map(|value| value != "0"))
}

pub enum AttentionStateType {
    Full {
        length: u32,
    },
    Ring {
        offset: u32,
        length: u32,
        max_length: u32,
    },
}

impl AttentionStateType {
    pub fn physical_prefix_length(&self) -> u32 {
        match self {
            Self::Full {
                length,
            } => *length,
            Self::Ring {
                max_length,
                ..
            } => *max_length,
        }
    }

    pub fn ring_params(&self) -> Option<RingParams> {
        let Self::Ring {
            offset,
            length,
            max_length: _,
        } = self
        else {
            return None;
        };

        Some(RingParams {
            ring_offset: *offset,
            ring_length: *length,
        })
    }
}

pub struct AttentionState<B: Backend> {
    pub cur_context_length: u32,
    pub elements_prepared: u32,
    pub element_dim: u32,
    pub data_type: DataType,
    pub state_type: AttentionStateType,
    pub is_sparse: bool,
    pub kv_int8: bool,
    pub num_kv_heads: u32,
    pub keys: Box<dyn Buffer<Backend = B>>,
    pub values: Box<dyn Buffer<Backend = B>>,
    pub key_scales: Option<B::DenseBuffer>,
    pub value_scales: Option<B::DenseBuffer>,
    pub kv_cache_update: <B::Kernels as Kernels>::KVCacheUpdateKernel,
}

impl<B: Backend> AttentionState<B> {
    pub fn create_empty(
        attention: &Attention<B>,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        if let Some(max_context_length) = max_context_length {
            assert!(
                attention.max_rope_length.is_none_or(|max_rope_length| max_context_length <= max_rope_length),
                "Attention state max_prefix_elements overflows RoPE"
            );
        }

        let data_type = attention.data_type;

        let max_prefix_elements = if attention.is_causal
            && let Some(sliding_window_size) = attention.sliding_window_size
        {
            sliding_window_size
        } else if let Some(max_context_length) = max_context_length {
            max_context_length
        } else {
            attention
                .max_rope_length
                .expect("Cannot create full attention state with unlimited length for with no RoPE")
        };

        let state_type = if attention.is_causal && attention.sliding_window_size.is_some() {
            AttentionStateType::Ring {
                offset: 0,
                length: 0,
                max_length: max_prefix_elements,
            }
        } else {
            AttentionStateType::Full {
                length: 0,
            }
        };

        let max_elements = max_prefix_elements + ATTENTION_SUFFIX_CAPACITY;
        let num_kv_heads = attention.num_kv_heads.unwrap();
        let element_size = num_kv_heads * attention.head_dim;
        // The layer already decided this and built its kernels around the
        // answer; deriving it a second time here is how a cache the prepare
        // kernel cannot write ended up being allocated.
        let kv_int8 = attention.kv_int8;
        let cache_data_type = if kv_int8 {
            DataType::U8
        } else {
            data_type
        };
        let kv_buffer_bytes = size_for_shape(&[max_elements, element_size], cache_data_type);

        let is_ring = matches!(state_type, AttentionStateType::Ring { .. });
        let is_sparse = !is_ring && context.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS);

        let (keys, values): (Box<dyn Buffer<Backend = B>>, Box<dyn Buffer<Backend = B>>) = if is_sparse {
            (
                Box::new(context.create_sparse_buffer(kv_buffer_bytes)?),
                Box::new(context.create_sparse_buffer(kv_buffer_bytes)?),
            )
        } else {
            (Box::new(context.create_buffer(kv_buffer_bytes)?), Box::new(context.create_buffer(kv_buffer_bytes)?))
        };

        let (key_scales, value_scales) = if kv_int8 {
            let scale_bytes = size_for_shape(&[max_elements, num_kv_heads], DataType::F32);
            (Some(context.create_buffer(scale_bytes)?), Some(context.create_buffer(scale_bytes)?))
        } else {
            (None, None)
        };

        let kv_cache_update = <B::Kernels as Kernels>::KVCacheUpdateKernel::new(context, data_type)?;

        Ok(Self {
            cur_context_length: 0,
            elements_prepared: 0,
            element_dim: element_size,
            data_type: cache_data_type,
            state_type,
            is_sparse,
            kv_int8,
            num_kv_heads,
            keys,
            values,
            key_scales,
            value_scales,
            kv_cache_update,
        })
    }
}

impl<B: Backend> MixerState<B> for AttentionState<B> {
    fn prepare(
        &mut self,
        context_length: u32,
        suffix_length: u32,
        context: &B::Context,
    ) -> Result<(), B::Error> {
        if !self.is_sparse {
            return Ok(());
        }

        assert!(suffix_length <= ATTENTION_SUFFIX_CAPACITY, "attention suffix length exceeds hardcoded capacity");
        let elements_required = context_length + suffix_length;
        let bytes_required = size_for_shape(&[elements_required, self.element_dim], self.data_type);
        let bytes_prepared = size_for_shape(&[self.elements_prepared, self.element_dim], self.data_type);

        let keys = (self.keys.as_mut() as &mut dyn Any).downcast_mut::<B::SparseBuffer>().unwrap();
        let values = (self.values.as_mut() as &mut dyn Any).downcast_mut::<B::SparseBuffer>().unwrap();

        for buffer in [keys, values] {
            let buffer_page_size = buffer.page_size_bytes();
            let buffer_start_page = bytes_prepared.div_ceil(buffer_page_size);
            let buffer_end_page = bytes_required.div_ceil(buffer_page_size);

            if buffer_end_page > buffer_start_page {
                buffer.map(context, &(buffer_start_page..buffer_end_page))?;
            }
        }

        self.elements_prepared = elements_required;

        Ok(())
    }

    fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        assert!(accepted_indices.is_sorted_by(|a, b| a < b), "invalid accepted indicies");

        let copies = match &mut self.state_type {
            AttentionStateType::Full {
                length,
            } => {
                let copies = accepted_indices
                    .iter()
                    .copied()
                    .zip(0u32..)
                    .filter(|&(accepted_index, index)| index != accepted_index)
                    .map(|(accepted_index, index)| Copy {
                        source: *length + accepted_index,
                        destination: *length + index,
                    })
                    .collect::<Vec<Copy>>();

                *length += accepted_indices.len() as u32;

                copies
            },
            AttentionStateType::Ring {
                offset,
                length,
                max_length,
            } => {
                let mut copies = Vec::new();
                for accepted_index in accepted_indices {
                    copies.push(Copy {
                        source: *max_length + *accepted_index,
                        destination: (*offset + *length) % *max_length,
                    });

                    if length < max_length {
                        *length += 1;
                    } else {
                        *offset = (*offset + 1) % *max_length;
                    }
                }
                copies
            },
        };

        // The copy kernel is typed by the model dtype; the int8 cache rows are
        // the same bytes viewed as half as many of those elements, and the f32
        // scale rows move in a second pass over the scale buffers.
        let copy_element_dim = if self.kv_int8 {
            self.element_dim / 2
        } else {
            self.element_dim
        };
        for copies_chunk in copies.chunks(B::MAX_INLINE_BYTES / size_of::<Copy>()) {
            self.kv_cache_update.encode(
                self.keys.as_mut(),
                self.values.as_mut(),
                copies_chunk,
                copies_chunk.len() as u32,
                copy_element_dim,
                encoder,
            );
            if let (Some(key_scales), Some(value_scales)) = (&mut self.key_scales, &mut self.value_scales) {
                self.kv_cache_update.encode(
                    &mut *key_scales,
                    &mut *value_scales,
                    copies_chunk,
                    copies_chunk.len() as u32,
                    self.num_kv_heads * 2,
                    encoder,
                );
            }
        }

        self.cur_context_length += accepted_indices.len() as u32;

        Ok(())
    }
}
