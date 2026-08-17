use crate::util::enable_benchmark_gpu_capture_if_requested;

pub enum UzuTest {
    Bench(&'static dyn Fn()),
    Test(&'static test::TestDescAndFn),
}

// Keeps the main thread inside UIApplicationMain so FrontBoard sees a
// completed launch; without it the 20-second launch watchdog (0x8BADF00D)
// kills any bench run. The work runs on a background thread and exits the
// process when done.
#[cfg(target_os = "ios")]
fn run_with_app_shim(run: impl FnOnce() + Send + 'static) -> ! {
    use std::ffi::{c_char, c_int, c_void};
    #[link(name = "UIKit", kind = "framework")]
    unsafe extern "C" {
        fn UIApplicationMain(
            argc: c_int,
            argv: *mut *mut c_char,
            principal_class_name: *const c_void,
            delegate_class_name: *const c_void,
        ) -> c_int;
    }
    std::thread::spawn(move || {
        run();
        std::process::exit(0);
    });
    unsafe {
        UIApplicationMain(0, std::ptr::null_mut(), std::ptr::null(), std::ptr::null());
    }
    unreachable!("UIApplicationMain returned")
}

pub fn uzu_harness(tests: &[&UzuTest]) {
    // Tests and benches must not self-calibrate tiles: calibration skews
    // criterion numbers and slows parity runs. Explicit env still wins.
    if std::env::var_os("UZU_TILE_AUTOTUNE").is_none() {
        unsafe { std::env::set_var("UZU_TILE_AUTOTUNE", "0") };
    }
    let args = std::env::args().collect::<Vec<String>>();
    let benchmarks = args.contains(&"--bench".to_string());
    if benchmarks {
        #[cfg(target_os = "ios")]
        crate::path::ios_set_current_dir();
        enable_benchmark_gpu_capture_if_requested();
        let bench_tests: Vec<&'static dyn Fn()> = tests
            .iter()
            .filter_map(|test| match test {
                UzuTest::Bench(test) => Some(*test),
                UzuTest::Test(_) => None,
            })
            .collect::<Vec<_>>();
        #[cfg(target_os = "ios")]
        if std::env::var_os("UZU_IOS_APP_SHIM").is_some() {
            // The bench closures are 'static top-level functions; the wrapper
            // only carries the references across the thread boundary.
            struct ForceSend<T>(T);
            unsafe impl<T> Send for ForceSend<T> {}
            let payload = ForceSend(bench_tests);
            run_with_app_shim(move || {
                let payload = payload;
                criterion::runner(payload.0.as_slice());
            });
        }
        criterion::runner(bench_tests.as_slice());
    } else {
        let default_tests: Vec<&'static test::TestDescAndFn> = tests
            .iter()
            .filter_map(|test| match test {
                UzuTest::Bench(_) => None,
                UzuTest::Test(test) => Some(*test),
            })
            .collect::<Vec<_>>();
        #[cfg(target_os = "ios")]
        if std::env::var_os("UZU_IOS_APP_SHIM").is_some() {
            struct ForceSend<T>(T);
            unsafe impl<T> Send for ForceSend<T> {}
            let payload = ForceSend(default_tests);
            run_with_app_shim(move || {
                let payload = payload;
                test::test_main_static(payload.0.as_slice());
            });
        }
        test::test_main_static(&default_tests)
    }
}
