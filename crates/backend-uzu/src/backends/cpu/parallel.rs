use std::{
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread::JoinHandle,
};

const MIN_WORK_PER_THREAD: usize = 1 << 16;
// More than one chunk per thread: P and E cores differ several-fold in throughput.
const CHUNKS_PER_THREAD: usize = 8;

pub(crate) fn available_threads() -> usize {
    if let Ok(raw) = std::env::var("UZU_CPU_THREADS") {
        match raw.parse::<usize>() {
            Ok(threads) if threads > 0 => return threads,
            _ => {
                // A typo in the variable must not silently change what a
                // benchmark measures.
                eprintln!("ignoring invalid UZU_CPU_THREADS={raw:?}, using available parallelism");
            },
        }
    }
    std::thread::available_parallelism().map(|threads| threads.get()).unwrap_or(1)
}

struct Job {
    run: unsafe fn(*const (), Range<usize>),
    data: *const (),
    chunk: usize,
    chunks: usize,
    units: usize,
    next: AtomicUsize,
}

// SAFETY: `data` points at the submitter's `F: Fn(Range<usize>) + Sync`
// closure (the `Sync` bound is enforced by `for_each_chunk`), so shared
// calls from worker threads are sound, and `run` is the matching monomorphic
// trampoline. The pointer never outlives the closure: workers only reach the
// job through `State::job`, which `Retract` clears — waiting for every
// active worker — before `for_each_chunk` returns or unwinds.
unsafe impl Send for Job {}
// SAFETY: see `Send` above; all shared access goes through `&F` of a `Sync`
// closure and the atomic `next` counter.
unsafe impl Sync for Job {}

impl Job {
    /// # Safety
    ///
    /// `self.data` must still point at the live closure `self.run` was
    /// instantiated for; guaranteed by the publication protocol above.
    unsafe fn run_claimed(&self) {
        loop {
            let index = self.next.fetch_add(1, Ordering::Relaxed);
            if index >= self.chunks {
                return;
            }
            let start = index * self.chunk;
            unsafe { (self.run)(self.data, start..(start + self.chunk).min(self.units)) };
        }
    }
}

#[derive(Default)]
struct State {
    job: Option<&'static Job>,
    generation: u64,
    active: usize,
    panicked: bool,
    stop: bool,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    idle: Condvar,
}

pub(crate) struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

/// Retracts the job and drains workers even if the submitter unwinds, so the job never
/// outlives the closure it points at.
struct Retract<'a>(&'a Shared);

impl Drop for Retract<'_> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(|error| error.into_inner());
        state.job = None;
        while state.active > 0 {
            state = self.0.idle.wait(state).unwrap_or_else(|error| error.into_inner());
        }
    }
}

impl Pool {
    pub(crate) fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared::default());
        let workers = (1..threads)
            .map(|_| {
                let shared = shared.clone();
                std::thread::spawn(move || worker(&shared))
            })
            .collect();

        Self {
            shared,
            workers,
        }
    }

    /// Runs `compute` over `0..units` split into chunks across the pool.
    ///
    /// NOT reentrant and not callable concurrently on one `Pool`: there is a
    /// single `State::job` slot, so a second simultaneous submitter would
    /// overwrite the first job, `Retract` would wait on the wrong workers,
    /// and `panicked` could be attributed to the wrong caller. The engine
    /// upholds this by submitting only from the single `CpuContext` command
    /// thread; the debug assert below catches violations.
    pub(crate) fn for_each_chunk<F>(
        &self,
        units: usize,
        work: usize,
        compute: F,
    ) where
        F: Fn(Range<usize>) + Sync,
    {
        let threads = (self.workers.len() + 1).min(units).min((work / MIN_WORK_PER_THREAD).max(1));
        if threads <= 1 {
            compute(0..units);
            return;
        }

        /// # Safety
        ///
        /// `data` must be the `&F` this trampoline was monomorphized for;
        /// `for_each_chunk` passes `&raw const compute` and keeps `compute`
        /// alive until `Retract` has drained every worker.
        unsafe fn call<F: Fn(Range<usize>)>(
            data: *const (),
            range: Range<usize>,
        ) {
            unsafe { (*(data as *const F))(range) }
        }

        let chunk = units.div_ceil(threads * CHUNKS_PER_THREAD).max(1);
        let job = Job {
            run: call::<F>,
            data: (&raw const compute) as *const (),
            chunk,
            chunks: units.div_ceil(chunk),
            units,
            next: AtomicUsize::new(0),
        };

        // SAFETY: `Retract` below clears the job and waits for every worker that took it,
        // on the normal path and while unwinding, so workers never see it after this call.
        let published: &'static Job = unsafe { std::mem::transmute::<&Job, &'static Job>(&job) };

        {
            let mut state = self.shared.state.lock().unwrap_or_else(|error| error.into_inner());
            debug_assert!(state.job.is_none(), "for_each_chunk is not reentrant: a job is already published");
            state.job = Some(published);
            state.generation = state.generation.wrapping_add(1);
        }
        for _ in 1..threads {
            self.shared.wake.notify_one();
        }

        {
            let _retract = Retract(&self.shared);
            unsafe { job.run_claimed() };
        }

        let mut state = self.shared.state.lock().unwrap_or_else(|error| error.into_inner());
        if std::mem::take(&mut state.panicked) {
            drop(state);
            panic!("cpu kernel panicked on a pool worker");
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap_or_else(|error| error.into_inner());
            state.stop = true;
        }
        self.shared.wake.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker(shared: &Shared) {
    let mut seen = 0u64;
    loop {
        let job = {
            let mut state = shared.state.lock().unwrap_or_else(|error| error.into_inner());
            loop {
                if state.stop {
                    return;
                }
                match state.job {
                    Some(job) if state.generation != seen => {
                        seen = state.generation;
                        state.active += 1;
                        break job;
                    },
                    _ => {
                        state = shared.wake.wait(state).unwrap_or_else(|error| error.into_inner());
                    },
                }
            }
        };

        let outcome = catch_unwind(AssertUnwindSafe(|| unsafe { job.run_claimed() }));

        let mut state = shared.state.lock().unwrap_or_else(|error| error.into_inner());
        state.active -= 1;
        state.panicked |= outcome.is_err();
        if state.active == 0 {
            shared.idle.notify_all();
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/backends/cpu/parallel_test.rs"]
mod tests;
