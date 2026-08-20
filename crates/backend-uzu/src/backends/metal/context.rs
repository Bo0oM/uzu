use std::{
    collections::HashMap,
    path::Path,
    sync::{OnceLock, 
        Arc, Weak,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};

use metal::{
    MTL4CommandQueue, MTL4CommandQueueExt, MTLBuffer, MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager,
    MTLCommandBufferExt, MTLCommandQueue, MTLCommandQueueExt, MTLComputePipelineState, MTLDevice, MTLDeviceExt,
    MTLEvent, MTLFunctionConstantValues, MTLLibrary, MTLResourceOptions, MTLSharedEvent, MTLSparsePageSize,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::{Mutex, MutexGuard};

use super::{
    Metal,
    device_tier::{DeviceTier, device_tier_for_device},
    error::MetalError,
    metal_extensions::{DeviceExt, LibraryPipelineExtensions},
};
use crate::backends::{
    common::{Allocation, AllocationPool, AllocationType, Allocator, Backend, Context, DeviceCapabilities},
    metal::{
        command_buffer::MetalCommandBufferInitial,
        sparse::{MetalSparseBuffer, MetalSparseHeapPool, MetalSparseMappingOpsBatch},
    },
};

pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub command_queue4: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    timeline_event: Retained<ProtocolObject<dyn MTLEvent>>,
    timeline_value: AtomicU64,
    allocator: Arc<Allocator<Metal>>,
    peak_memory_usage: AtomicUsize,
    library_cache: Mutex<HashMap<usize, Retained<ProtocolObject<dyn MTLLibrary>>>>,
    pipeline_cache: Mutex<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
    sparse_heap_pool: Mutex<MetalSparseHeapPool>,
    device_tier: DeviceTier,
    /// Formatted once: this is read on every quantized decode dispatch to key
    /// the autotune cache, and building it costs a String plus two device
    /// queries.
    device_label: OnceLock<String>,
    supports_mxu: bool,
    weak_self: Weak<MetalContext>,
    // Separate from `timeline_event` on purpose: making the hot-path timeline shared costs ~6%
    // on decode. Signalled only when the sparse path is about to wait on it.
    sparse_mapping_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    sparse_mapping_lock: Mutex<()>,
    // True when the last timeline participant was the sparse mapping queue, so
    // the next work submit must encode a cross-queue wait. Same-queue work
    // submits order themselves by commit order + hazard tracking and skip it.
    timeline_cross_queue_dirty: AtomicBool,
}

/// Budget must cover a full drain of the queue, prefill included.
const SPARSE_MAPPING_WAIT_MS: u64 = 5_000;
const SPARSE_MAPPING_WAIT_ATTEMPTS: u32 = 6;

impl MetalContext {
    pub fn supports_mxu(&self) -> bool {
        self.supports_mxu
    }

    pub(crate) fn device_tier(&self) -> DeviceTier {
        self.device_tier
    }

    /// Stable device identity for on-disk calibration caches.
    pub(crate) fn device_label(&self) -> &str {
        self.device_label.get_or_init(|| format!("{} ({} cores)", self.device.name(), self.device.gpu_core_count()))
    }

    pub(super) fn update_peak_memory_usage(&self) {
        self.peak_memory_usage.fetch_max(self.device.current_allocated_size(), Ordering::Relaxed);
    }

    fn library(
        &self,
        data: &'static [u8],
        compressed: bool,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError> {
        // `data` always comes from an `include_bytes!` constant, so its address is a stable, unique key.
        let key = data.as_ptr() as usize;
        if let Some(library) = self.library_cache.lock().get(&key) {
            return Ok(library.clone());
        }

        let maybe_uncompressed_data_owned;
        let data = if compressed {
            maybe_uncompressed_data_owned = zstd::decode_all(data).map_err(MetalError::CannotDecompressLibrary)?;

            &maybe_uncompressed_data_owned
        } else {
            data
        };

        let library = self
            .device
            .new_library_with_data(data)
            .map_err(|nserror| MetalError::CannotCreateLibrary(nserror.to_string()))?;
        self.library_cache.lock().insert(key, library.clone());

        Ok(library)
    }

    pub fn compute_pipeline_state(
        &self,
        library_data: &'static [u8],
        library_compressed: bool,
        cache_key: &str,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        if let Some(pipeline) = self.pipeline_cache.lock().get(cache_key) {
            return Ok(pipeline.clone());
        }

        let pipeline =
            self.library(library_data, library_compressed)?.compute_pipeline_state(function_name, constants)?;
        self.pipeline_cache.lock().insert(cache_key.to_string(), pipeline.clone());

        Ok(pipeline)
    }

    pub(super) fn sparse_heap_pool(&self) -> MutexGuard<'_, MetalSparseHeapPool> {
        self.sparse_heap_pool.lock()
    }

    /// Queues mapping updates and waits: releasing their buffers or heaps any earlier wedges the GPU.
    pub(super) fn sparse_update_mappings_blocking(
        &self,
        mappings: &[MetalSparseMappingOpsBatch],
    ) {
        let Some(completion_value) = self.enqueue_sparse_mappings(mappings, true) else {
            return;
        };

        for _ in 0..SPARSE_MAPPING_WAIT_ATTEMPTS {
            if self.sparse_mapping_event.wait_until_signaled_value_timeout_ms(completion_value, SPARSE_MAPPING_WAIT_MS)
            {
                return;
            }
            eprintln!("Still waiting for sparse mapping update {completion_value} to complete on the GPU");
        }

        // The wait exists to keep these resources alive; returning without it would free them under a
        // live command — the very fault this path guards. Leak them instead: bounded memory beats a
        // wedged GPU, and only a queue that already stopped making progress can get here.
        for op in mappings {
            std::mem::forget(op.buffer.clone());
            std::mem::forget(op.heap.clone());
        }
        eprintln!(
            "Sparse mapping update {completion_value} never completed; leaking its buffers and heaps, the GPU is stuck"
        );
    }

    /// Queues mapping updates without waiting; safe only while their buffers and heaps stay alive.
    pub(super) fn sparse_update_mappings(
        &self,
        mappings: &[MetalSparseMappingOpsBatch],
    ) {
        self.enqueue_sparse_mappings(mappings, false);
    }

    /// Returns the timeline value the batch signals on completion, or `None` if there was nothing to do.
    fn enqueue_sparse_mappings(
        &self,
        mappings: &[MetalSparseMappingOpsBatch],
        signal_cpu_visible: bool,
    ) -> Option<u64> {
        if mappings.is_empty() {
            return None;
        }

        // Ticket and encoding must stay atomic — the queue runs commands in submission order.
        let _guard = self.sparse_mapping_lock.lock();

        let wait_value = self.timeline_get_and_increment();
        self.command_queue4.wait_for_event_value(&self.timeline_event, wait_value);
        for op in mappings {
            self.command_queue4.update_buffer_mappings(&op.buffer, Some(op.heap.lock().heap()), &op.mtl_operations);
        }
        let completion_value = wait_value + 1;
        self.command_queue4.signal_event_value(&self.timeline_event, completion_value);
        self.timeline_cross_queue_dirty.store(true, Ordering::Release);
        if signal_cpu_visible {
            self.command_queue4
                .signal_event_value(ProtocolObject::from_ref(&*self.sparse_mapping_event), completion_value);
        }

        Some(completion_value)
    }

    pub(super) fn timeline_get_and_increment(&self) -> u64 {
        self.timeline_value.fetch_add(1, Ordering::Release)
    }

    /// Takes a work-queue ticket and reports whether the sparse mapping queue
    /// touched the timeline since the last submit.
    ///
    /// The two must be read together under the same lock the mapping path
    /// holds. A submit that took its ticket after a mapping took one, but
    /// before the mapping set the flag, would see it clear and skip the wait
    /// its work needs — its commands could then run against buffers whose
    /// remap is still in flight. Single-stream decode never interleaves the
    /// two, so this only bites with two streams on one context.
    pub(super) fn take_work_ticket(&self) -> (u64, bool) {
        let _guard = self.sparse_mapping_lock.lock();
        let ticket = self.timeline_get_and_increment();
        let needs_cross_queue_wait = self.timeline_cross_queue_dirty.swap(false, Ordering::AcqRel);
        (ticket, needs_cross_queue_wait)
    }

    pub(super) fn timeline_event(&self) -> &ProtocolObject<dyn MTLEvent> {
        &self.timeline_event
    }
}

impl Context for MetalContext {
    type Backend = Metal;

    fn new() -> Result<Arc<Self>, MetalError> {
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            <dyn MTLDevice>::system_default().ok_or(MetalError::CannotOpenDevice)?;

        let command_queue =
            device.new_command_queue_with_max_command_buffer_count(1024).ok_or(MetalError::CannotCreateCommandQueue)?;

        let command_queue4 = device.new_mtl4_command_queue().ok_or(MetalError::CannotCreateCommandQueueMtl4)?;

        let gpu_core_count = device.gpu_core_count();
        let device_tier = device_tier_for_device(gpu_core_count, device.as_ref());
        let supports_mxu = device.supports_mxu();
        let page_size = MTLSparsePageSize::KB256;
        let heap_capacity = Metal::ALLOCATION_GRANULARITY;
        let sparse_pool = MetalSparseHeapPool::new(page_size, heap_capacity);
        let timeline_event = device.new_event().ok_or(MetalError::CannotCreateEvent)?;
        let sparse_mapping_event = device.new_shared_event().ok_or(MetalError::CannotCreateEvent)?;

        Ok(Arc::new_cyclic(|weak_self| Self {
            device,
            command_queue,
            command_queue4,
            timeline_event,
            timeline_value: AtomicU64::new(0),
            allocator: Allocator::new(weak_self.clone()),
            peak_memory_usage: AtomicUsize::new(0),
            library_cache: Mutex::new(HashMap::new()),
            pipeline_cache: Mutex::new(HashMap::new()),
            sparse_heap_pool: Mutex::new(sparse_pool),
            device_tier,
            device_label: OnceLock::new(),
            supports_mxu,
            weak_self: weak_self.clone(),
            sparse_mapping_event,
            sparse_mapping_lock: Mutex::new(()),
            timeline_cross_queue_dirty: AtomicBool::new(true),
        }))
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or(MetalError::CannotCreateBuffer)?;

        self.update_peak_memory_usage();

        Ok(buffer)
    }

    fn create_allocation(
        &self,
        size: usize,
        allocation_type: AllocationType<Metal>,
    ) -> Result<Allocation<Metal>, MetalError> {
        self.allocator.allocate(size, allocation_type)
    }

    fn create_allocation_pool(
        &self,
        reusable: bool,
    ) -> AllocationPool<Metal> {
        self.allocator.create_pool(reusable)
    }

    fn create_command_buffer(
        &self,
        name: Option<&str>,
    ) -> Result<MetalCommandBufferInitial, MetalError> {
        let command_buffer = self.command_queue.command_buffer().ok_or(MetalError::CannotCreateCommandBuffer)?;
        command_buffer.set_label(name);
        let context = self.weak_self.upgrade().unwrap(); // never fails
        Ok(MetalCommandBufferInitial::new(command_buffer, context))
    }

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error> {
        let sparse_page_size = self.sparse_heap_pool.lock().page_size();
        let context = self.weak_self.upgrade().ok_or(MetalError::CannotCreateBuffer)?;
        MetalSparseBuffer::new(context, capacity, sparse_page_size)
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        Some(self.peak_memory_usage.load(Ordering::Relaxed))
    }

    fn enable_capture() {
        unsafe {
            std::env::set_var("METAL_CAPTURE_ENABLED", "1");
        }
    }

    fn start_capture(
        &self,
        trace_path: &Path,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        let capture_manager = MTLCaptureManager::shared_capture_manager();
        let capture_descriptor = MTLCaptureDescriptor::new();
        capture_descriptor.set_destination(MTLCaptureDestination::GPUTraceDocument);
        capture_descriptor.set_output_path(Some(&trace_path.with_added_extension("gputrace")));

        self.command_queue.set_label(Some("uzu_command_queue"));
        capture_descriptor.set_capture_object(Some(self.command_queue.as_ref()));

        capture_manager
            .start_capture_with_descriptor_error(&capture_descriptor)
            .map_err(|nserror| MetalError::CannotStartGpuCapture(nserror.to_string()))?;

        Ok(())
    }

    fn stop_capture(&self) -> Result<(), <Self::Backend as Backend>::Error> {
        MTLCaptureManager::shared_capture_manager().stop_capture();

        Ok(())
    }

    fn device_capabilities(&self) -> DeviceCapabilities {
        let mut capabilities = DeviceCapabilities::empty();
        // A/B knob for the sparse KV path (ADR-10.V): the second MTL4 queue
        // and its cross-queue syncs are a suspected per-token cost on iOS.
        let sparse_disabled = std::env::var_os("UZU_DISABLE_SPARSE").is_some();
        if !sparse_disabled && self.device.supports_placement_sparse_resources() {
            capabilities |= DeviceCapabilities::SPARSE_BUFFERS;
        }
        if self.supports_mxu {
            capabilities |= DeviceCapabilities::NATIVE_INT8_MATMUL;
        }
        capabilities
    }
}
