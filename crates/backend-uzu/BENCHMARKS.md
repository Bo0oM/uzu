# Kernel Benchmarks

Criterion-based microbenchmarks for Metal kernels. Runs on macOS (native)
and on iPhone (via `cargo-dinghy`). Results from both are consolidated
under a single `target/criterion/<label>/` tree so you can compare
baselines side by side.

## Prerequisites

- Rust nightly: `rustup toolchain install nightly`
- For iOS runs:
    - Apple target: `rustup target add aarch64-apple-ios`
    - `cargo-dinghy` built from the [trymirai/dinghy](https://github.com/trymirai/dinghy/tree/mirai):

      ```bash
      cargo install \
        --git https://github.com/trymirai/dinghy \
        --branch mirai \
        cargo-dinghy
      ```

    - A physical device connected via USB with a device ID from
      `xcrun devicectl list devices`.

## Available benchmark groups

Benchmarks live inside the `backend-uzu` library test target (registered
with `#[uzu_bench]` under `crates/backend-uzu/tests/unit/`), so the Cargo
target is `--lib`.

| Group id                                    | Filter                              | Declared in |
|---------------------------------------------|-------------------------------------|---------------|
| `Metal/Kernel/Matmul/GEMM`, `.../GEMM_MXU`   | `Metal/Kernel/Matmul`               | `matmul/gemm_bench.rs` |
| `Metal/Kernel/A8W/w4`, `.../w8`              | `Metal/Kernel/A8W`                  | `matmul/a8w_bench.rs` |
| `Metal/Kernel/UnifiedQuantizedGemm/…`        | `Metal/Kernel/UnifiedQuantizedGemm` | `matmul/quant_gemm_bench.rs` |
| `Metal/Kernel/Gemv/…`                        | `Metal/Kernel/Gemv`                 | `matmul/quant_gemv_bench.rs` |
| `Metal/Kernel/Qwen3Layers/…`                 | `Metal/Kernel/Qwen3Layers`          | `matmul/qwen3_bench.rs` |
| `Metal/Kernel/GDNTreeVerify/…`               | `Metal/Kernel/GDNTreeVerify`        | `gdn/tree_verify/*_bench.rs` |
| `Model loading`                              | `Model loading`                     | `session/model_loading_bench.rs` |

The prefix is substituted per backend (`Metal` or `Cpu`), so the same groups also exist
under `Cpu/Kernel/…`.

The prefix `Metal/Kernel/Matmul` runs both `GEMM` and `GEMM_MXU` in one pass.
`Model loading` requires the test model path configured by the test helpers
(`TEST_MODEL`, see `crates/test-runner/src/path.rs`).

**What is not here yet.** There are no benchmark groups for `RMSNorm`, the sampling
kernels, `ChatSession run`, or a whole forward pass; adding them is a separate task.

Until they exist, end-to-end numbers come from `cli bench <model_path> <task_path>
<output_path>` (`crates/cli/src/bench/`): TTFT, prompt t/s, generate t/s, memory, average
power, joules per token, N runs with mean and deviation. `greedy: true` in the task gives a
deterministic output suitable for byte-exact comparison.

## Output layout

Every run writes into `target/criterion/<label>/…`, where `<label>` is a
free-form name you choose (e.g. `m2_max`, `a19`). The Criterion baseline
you saved lives at `target/criterion/<label>/<benchmark-path>/<baseline-name>/`.

## Running on macOS

From the repo root. Use an **absolute** `CRITERION_HOME` so it doesn't
resolve relative to the package dir:

```bash
CRITERION_HOME="$PWD/target/criterion/m2_max" cargo bench \
  -p backend-uzu \
  --lib -- "Metal/Kernel/Matmul" \
  --save-baseline matmul_baseline_m2_max
```

Set `UZU_CAPTURE_BENCH=1` to capture the first matching benchmark command
buffer as a Metal `.gputrace`. `UZU_CAPTURE_BENCH_FILTER` is an optional
benchmark path substring; `UZU_CAPTURE_BENCH_DIR` defaults to the current
directory.

## Running on iPhone (via `cargo-dinghy`)

Run one benchmark group at a time to avoid the iOS watchdog killing the
app.

Set `IPHONEOS_DEPLOYMENT_TARGET` (value from `platforms.toml` `[envs]`)
for all iOS builds; without it the link step fails with undefined
symbols (e.g. `___chkstk_darwin`) because objects are built for a newer
SDK than the default deployment target.

Key flags:

- `DINGHY_SKIP_APPLE_HOST=1` — **required for this crate**, on both `test`
  and `bench`. Without it dinghy regenerates a host crate that compiles
  our sources into its own, so `include!(concat!(env!("OUT_DIR"),
  "/traits.rs"))` in `backends/common/kernel/mod.rs` resolves to the
  runner's `OUT_DIR`, where our build script never ran. The build fails
  with `couldn't find file .../dinghy-generated-apple-runner/.../traits.rs`
  after a full iOS compile, so the cost of forgetting it is about fifteen
  minutes. With the flag set, dinghy packages the already-built binary
  instead.
- `-e UZU_IOS_APP_SHIM=1` — on-device env var that runs the harness
  inside a UIApplication. Without it the iOS watchdog kills anything
  long-running (criterion in particular) with SIGKILL.

- `-e CRITERION_HOME=criterion/a19` — on-device env var. Path is
  relative to the app's cwd (`Documents/`), so this becomes
  `Documents/criterion/a19/` on device. Keep it directly under
  `Documents/` — nested parents (e.g. `Documents/target/`) do not exist
  on a fresh install and the pre-run sync cannot create them.
- `--sync-dirs "$(pwd)/target/criterion=Documents/criterion"` —
  syncs the criterion tree between host and device before and after the
  run, so results written on device land back in the repo's
  `target/criterion/`. `$(pwd)` is required (absolute path) because the
  cargo runner is launched with cwd set to the package dir, not the
  workspace root.

Running the test suite on device follows the same shape:

```bash
DEVICE=$(xcrun devicectl list devices | awk '/iPhone/ {print $3; exit}')

IPHONEOS_DEPLOYMENT_TARGET=26.4 DINGHY_SKIP_APPLE_HOST=1 cargo dinghy \
  -d "$DEVICE" -e UZU_IOS_APP_SHIM=1 \
  test -p backend-uzu --lib -- --nocapture
```

One test fails there and is expected to: `test_metadata_loading` opens
the test model's weights, which are not synced to the device. Everything
else passes, including the two paths that cannot run on M1 Max at all --
`MetalDeltaNetChunkedPrefill` and the `NATIVE_INT8_MATMUL` activation
path -- both verified on A19 on 2026-08-19.

### End-to-end decode on device

The per-model numbers in the README come from `device_decode_probe`, not
from criterion. It needs three things, and the third is easy to miss:

- `[test_data]` in `.dinghy.toml` names the model to ship. Only one model
  fits per run, so a sweep rewrites the file between models. The value is
  a host path; dinghy copies the directory into the bundle.
- `-e UZU_DEVICE_DECODE_PROBE=1` — the probe returns immediately without
  it, so the regular suite does not pay for a model load.
- `-e UZU_TEST_MODEL_DIR=test_data/model` — **required**. `copy_test_data`
  puts the directory at `<bundle>/test_data/<key>`, and the probe resolves
  a relative `UZU_TEST_MODEL_DIR` against the executable's directory.
  Without it `get_test_model_path` falls through to the host-side
  `workspace/models/<version>/` branch, which does not exist on the phone,
  and the run fails in `test-runner/src/path.rs` after the whole model has
  already been copied over USB.

```bash
DEVICE=$(xcrun devicectl list devices | awk '/iPhone/ {print $3; exit}')

IPHONEOS_DEPLOYMENT_TARGET=26.4 DINGHY_SKIP_APPLE_HOST=1 cargo dinghy \
  -d "$DEVICE" \
  -e UZU_IOS_APP_SHIM=1 \
  -e UZU_DEVICE_DECODE_PROBE=1 \
  -e UZU_TEST_MODEL_DIR=test_data/model \
  test -p backend-uzu --lib --release device_decode_probe -- --nocapture
```

Discard the first run of a model: the tile autotune calibrates on first
launch and the file cache is cold, which shows up in `ttft_ms` (365 ms
against 171 ms on the second gemma-3-1b-it-4bit run) more than in
`decode_tps`.

`UZU_PROBE_DECODE` sets the decode length and **defaults to 64, while the
published tables are measured at 256**. The two are not comparable, so set
it explicitly on every run that is going into a table and check
`decode_tokens=` in the output line, which is the only place the length is
recorded once the number has been copied into a table. Speculation and the
tile autotune stay on: they are what ships, and upstream has neither, so
leaving them on is what makes the comparison the one a reader cares
about.

```bash
DEVICE=<DEVICE_ID>

IPHONEOS_DEPLOYMENT_TARGET=26.4 DINGHY_SKIP_APPLE_HOST=1 cargo dinghy \
  -d "$DEVICE" \
  -e CRITERION_HOME=criterion/a19 \
  --sync-dirs "$(pwd)/target/criterion=Documents/criterion" \
  bench -p backend-uzu --lib -- \
    "Metal/Kernel/Matmul" \
    --save-baseline matmul_baseline_a19
```

On-device criterion sanitizes group path separators, so results land in
`target/criterion/a19/Metal_Kernel_Matmul_<GEMM|GEMM_MXU>/…/`
on the host, next to any `m2_max/` baselines.

## Viewing reports

Open the Criterion HTML report:

```bash
open target/criterion/report/index.html
```

To inspect a specific label only:

```bash
open target/criterion/m2_max/report/index.html
open target/criterion/a19/report/index.html
```

## Cold GEMV

GEMV-class benches cycle through enough quant-buffer copies to cover a 256 MiB
weight working set before reusing one. This avoids ranking kernels on
cache-warm weights; pools allocate lazily, so criterion filters skip their cost.
