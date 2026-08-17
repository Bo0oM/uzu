use std::{
    env,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use criterion::Bencher;
use test_runner::env_vars;

use crate::backends::common::{Backend, Context, Encoder};

static CAPTURE_TAKEN: AtomicBool = AtomicBool::new(false);

fn should_capture_benchmark(benchmark_path: &str) -> bool {
    env_vars::enabled(env_vars::UZU_CAPTURE_BENCH)
        && benchmark_path.starts_with("Metal/")
        && env::var(env_vars::UZU_CAPTURE_BENCH_FILTER).map_or(true, |filter| benchmark_path.contains(&filter))
        && !CAPTURE_TAKEN.swap(true, Ordering::AcqRel)
}

fn benchmark_capture_path() -> PathBuf {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock before Unix epoch").as_secs();
    env::var(env_vars::UZU_CAPTURE_BENCH_DIR)
        .map(PathBuf::from)
        .unwrap_or(env::current_dir().unwrap())
        .join(format!("uzu_bench-{timestamp}.gputrace"))
}

fn start_benchmark_capture<B: Backend>(
    context: &B::Context,
    benchmark_path: &str,
) -> bool {
    if !should_capture_benchmark(benchmark_path) {
        return false;
    }

    let path = benchmark_capture_path();
    context.start_capture(&path).expect("failed to start benchmark GPU capture");
    println!("GPU benchmark capture started for {benchmark_path}: {path:?}");
    true
}

#[cfg(all(feature = "metal", target_os = "ios"))]
fn drain_autoreleased<T>(f: impl FnOnce() -> T) -> T {
    objc2::rc::autoreleasepool(|_| f())
}

#[cfg(not(all(feature = "metal", target_os = "ios")))]
fn drain_autoreleased<T>(f: impl FnOnce() -> T) -> T {
    f()
}

pub fn iter_encode_loop<B: Backend, F>(
    context: &B::Context,
    bencher: &mut Bencher,
    mut encode: F,
) where
    F: FnMut(&mut Encoder<B>),
{
    iter_encode_loop_named(context, bencher, "unnamed_benchmark", |encoder| encode(encoder));
}

pub fn iter_encode_loop_named<B: Backend, F>(
    context: &B::Context,
    bencher: &mut Bencher,
    benchmark_path: &str,
    mut encode: F,
) where
    F: FnMut(&mut Encoder<B>),
{
    // Encoding every iteration into one command buffer makes its host-side
    // footprint proportional to n_iters (criterion ramps it into the hundreds
    // of thousands), which trips the jetsam limit on iPhone. Chunking bounds
    // the footprint; the GPU execution times of the chunks simply add up.
    // On iOS a single command buffer must also finish inside the GPU watchdog
    // budget (seconds), so slow kernels need far smaller chunks: override
    // with UZU_BENCH_ITERS_PER_CB.
    let iters_per_command_buffer: u64 =
        std::env::var("UZU_BENCH_ITERS_PER_CB").ok().and_then(|value| value.parse().ok()).unwrap_or(8192);
    bencher.iter_custom(move |n_iters| {
        let capture = start_benchmark_capture::<B>(context, benchmark_path);
        let mut total = std::time::Duration::ZERO;
        let mut remaining = n_iters;
        while remaining > 0 {
            let chunk = remaining.min(iters_per_command_buffer);
            // On iOS the autoreleased command-buffer objects of a long bench
            // loop accumulate to a jetsam kill without a pool drain per chunk.
            total += drain_autoreleased(|| {
                let mut encoder = Encoder::<B>::new(context).unwrap();
                for _ in 0..chunk {
                    encode(&mut encoder);
                }
                let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
                completed.gpu_execution_time()
            });
            remaining -= chunk;
        }
        if capture {
            context.stop_capture().expect("failed to stop benchmark GPU capture");
        }
        total
    });
}
