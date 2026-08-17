//! Engine-level decode probe for on-device runs (ADR-10.A): reports wall-clock
//! prefill and decode throughput without the cli, so dinghy can run it on an
//! iPhone. Gated by UZU_DEVICE_DECODE_PROBE so the regular suite skips it.
use std::time::Instant;

use proc_macros::uzu_test;
use test_runner::path::get_test_model_path;

/// iOS/macOS peak process footprint (the number the OS shows as app memory),
/// mirroring the trymirai metrics-page "Resident memory" definition.
#[cfg(any(target_os = "ios", target_os = "macos"))]
fn phys_footprint_bytes() -> Option<u64> {
    // task_vm_info.phys_footprint via task_info; layout per <mach/task_info.h>.
    #[repr(C)]
    #[derive(Default)]
    struct TaskVmInfo {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        device: u64,
        device_peak: u64,
        internal: u64,
        internal_peak: u64,
        external: u64,
        external_peak: u64,
        reusable: u64,
        reusable_peak: u64,
        purgeable_volatile_pmap: u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual: u64,
        compressed: u64,
        compressed_peak: u64,
        compressed_lifetime: u64,
        phys_footprint: u64,
    }
    unsafe extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(task: u32, flavor: u32, info: *mut TaskVmInfo, count: *mut u32) -> i32;
    }
    const TASK_VM_INFO: u32 = 22;
    let mut info = TaskVmInfo::default();
    let mut count = (size_of::<TaskVmInfo>() / size_of::<u32>()) as u32;
    let result = unsafe { task_info(mach_task_self(), TASK_VM_INFO, &mut info, &mut count) };
    (result == 0).then_some(info.phys_footprint)
}

use crate::{
    encodable_block::sampling::SamplingMethod,
    engine::{Engine, language_model::stream::{LanguageModelStream, LanguageModelStreamOptions}},
    tests::helpers::for_each_non_cpu_backend,
};

const PREFILL_TOKENS: u64 = 64;
const DECODE_TOKENS: usize = 64;

// Real-prompt mode: UZU_PROBE_PROMPT_IDS is a comma-separated token-id list
// (tokenized host-side); the generated ids are printed for host-side
// detokenization and a coherence check.
fn prompt_ids_from_env() -> Option<Vec<u64>> {
    let raw = std::env::var("UZU_PROBE_PROMPT_IDS").ok()?;
    let ids: Vec<u64> = raw.split(',').filter_map(|id| id.trim().parse().ok()).collect();
    (!ids.is_empty()).then_some(ids)
}

#[uzu_test]
fn device_decode_probe() {
    if std::env::var_os("UZU_DEVICE_DECODE_PROBE").is_none() {
        return;
    }
    for_each_non_cpu_backend!(|B| {
        let model_path = get_test_model_path();
        let engine = Engine::<B>::new().unwrap();
        let load_start = Instant::now();
        let model = engine.load_language_model(&model_path).unwrap();
        let load = load_start.elapsed();
        let mut state = model.create_empty_state(Some(4096)).unwrap();
        // Real prompt ids when provided; arbitrary in-vocabulary ids otherwise.
        let input: Vec<u64> =
            prompt_ids_from_env().unwrap_or_else(|| (0..PREFILL_TOKENS).map(|token| 1000 + token).collect());
        let decode_tokens: usize = std::env::var("UZU_PROBE_DECODE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DECODE_TOKENS);
        let options = LanguageModelStreamOptions {
            sampling_method: SamplingMethod::Greedy,
            #[cfg(grammar)]
            grammar: None,
        };

        let start = Instant::now();
        let mut stream = LanguageModelStream::new(&model, &input, &mut state, options).unwrap();
        let first = stream.next().expect("no first token").expect("first token failed");
        let prefill = start.elapsed();

        let (gpu_nanos_before, submits_before) = crate::backends::common::gpu_trace_snapshot();
        let decode_start = Instant::now();
        let mut output_ids = vec![first];
        for _ in 1..decode_tokens {
            match stream.next() {
                Some(Ok(token)) => output_ids.push(token),
                Some(Err(error)) => panic!("decode failed: {error:?}"),
                None => break,
            }
        }
        let decode = decode_start.elapsed();
        let (gpu_nanos_after, submits_after) = crate::backends::common::gpu_trace_snapshot();
        if submits_after > submits_before {
            // UZU_GPU_TRACE=1: wall vs GPU-busy over the decode loop decides
            // whether GPU-resident decode has anything to reclaim (uzu-ktq).
            let busy_ms = (gpu_nanos_after - gpu_nanos_before) as f64 / 1e6;
            println!(
                "DEVICE_DECODE_GPU_TRACE wall_ms={:.1} gpu_busy_ms={busy_ms:.1} submits={} idle_frac={:.3}",
                decode.as_secs_f64() * 1e3,
                submits_after - submits_before,
                1.0 - busy_ms / (decode.as_secs_f64() * 1e3),
            );
        }
        let generated = output_ids.len();
        let decode_tps = (generated.saturating_sub(1)) as f64 / decode.as_secs_f64();
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        let footprint_mib = phys_footprint_bytes().map(|bytes| bytes as f64 / (1024.0 * 1024.0)).unwrap_or(-1.0);
        #[cfg(not(any(target_os = "ios", target_os = "macos")))]
        let footprint_mib = -1.0f64;
        println!(
            "DEVICE_DECODE_PROBE model={} load_ms={:.0} prefill_tokens={} ttft_ms={:.1} decode_tokens={generated} decode_tps={decode_tps:.2} footprint_mib={footprint_mib:.0}",
            model_path.file_name().unwrap().to_string_lossy(),
            load.as_secs_f64() * 1e3,
            input.len(),
            prefill.as_secs_f64() * 1e3,
        );
        println!(
            "DEVICE_DECODE_IDS {}",
            output_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",")
        );
    });
}
