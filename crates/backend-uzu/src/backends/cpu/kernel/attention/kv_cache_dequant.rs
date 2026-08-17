use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

#[kernel(KVCacheDequant)]
#[variants(T, f32, f16, bf16)]
pub fn kv_cache_dequant<T: ArrayElement + Float>(
    quantized: *const i8,
    scales: *const f32,
    dequantized: *mut T,
    head_dim: u32,
    element_count: u32,
) {
    let head_dim = head_dim as usize;
    for element_idx in 0..element_count as usize {
        let scale = unsafe { *scales.add(element_idx / head_dim) };
        let value = unsafe { *quantized.add(element_idx) } as f32 * scale;
        unsafe {
            *dequantized.add(element_idx) = T::from(value).unwrap();
        }
    }
}
