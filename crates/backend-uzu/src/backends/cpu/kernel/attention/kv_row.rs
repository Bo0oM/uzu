use num_traits::Float;

use crate::array::ArrayElement;

/// Arguments shared by the CPU attention kernels for reading one KV row.
pub(super) struct KvRowSource<T> {
    pub int8_base: Option<*const i8>,
    pub float_base: Option<*const T>,
    pub scales: Option<*const f32>,
    pub kv_int8: bool,
    pub num_kv_heads: Option<u32>,
    pub kv_head_idx: u32,
    pub head_stride: u32,
    pub seq_stride: u32,
}

/// Reads one KV row into f32, transparently dequantizing int8 rows; mirrors
/// the Metal fast paths element for element.
pub(super) fn read_kv_row<T: ArrayElement + Float>(
    source: &KvRowSource<T>,
    position: u32,
    row: &mut [f32],
) {
    let offset = (source.kv_head_idx * source.head_stride + position * source.seq_stride) as usize;
    if source.kv_int8 {
        let base = unsafe { source.int8_base.unwrap().add(offset) };
        let scale = unsafe {
            *source.scales.unwrap().add((position * source.num_kv_heads.unwrap() + source.kv_head_idx) as usize)
        };
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = unsafe { *base.add(j) } as f32 * scale;
        }
    } else {
        let base = unsafe { source.float_base.unwrap().add(offset) };
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = unsafe { *base.add(j) }.to_f32().unwrap();
        }
    }
}
