# uzu — performance fork

This repository is a performance-focused fork of [trymirai/uzu](https://github.com/trymirai/uzu), a high-performance inference engine for AI models on Apple silicon. For general product documentation, API usage, bindings, and the model catalog, refer to the [original README](https://github.com/trymirai/uzu/blob/main/README.md). This document describes what this fork changes relative to the upstream baseline (branch point `1bbb59eb`).

## Results

Measured on identical hardware, identical models (weights SHA-256-verified against the original repositories), and an identical protocol. Decode throughput, tokens/s.

### macOS, M1 Max (24-core GPU)

“Kernels” isolates the kernel and pipeline work (speculation and autotune disabled on both sides); “defaults” is the shipping configuration of this fork — prompt-lookup speculation enabled and the first-launch tile autotune already warmed. Δ compares the best fork configuration (bold) against upstream; on a free-form prompt, where speculation finds nothing to accept, the two fork configurations measure within ~2% of each other.

| Model | Upstream | Fork, kernels | Fork, defaults | Δ (best) |
|---|---|---|---|---|
| LFM2-350M (bf16) | 347.5 | **372.0** | 363.8 | +7.1% |
| Qwen3-0.6B (bf16) | 172.6 | **186.0** | 185.7 | +7.8% |
| gemma-3-1b-it (bf16) | 125.5¹ | **132.2** | 132.0 | +5.3% |
| gemma-3-1b-it-4bit | 73.4¹ | 141.6 | **146.7** | ×2.00 |
| Qwen3.5-0.8B-M (4-bit, RHT) | 238.4¹ | **276.0** | 272.6 | +15.8% |
| Qwen3.5-4B-M (4-bit, RHT) | 81.1¹ | 85.4 | **90.0** | +11.0% |

### iOS, iPhone 17 Pro (A19 Pro)

Each measurement is preceded by a two-minute cooldown pause (A-series chips throttle under sustained load). Speculation and autotune disabled on both sides.

| Model | Upstream | This fork | Δ |
|---|---|---|---|
| gemma-3-1b-it-4bit | 75.0 | 102.6 | +37% |
| Qwen3-0.6B (bf16) | 45.7 | 58.4 | +28% |
| Qwen3.5-0.8B-M | 125.7 | 134.3 | +6.9% |
| LFM2-350M (bf16) | 95.5 | 99.8 | +4.5% |

Time to first token improves across the board (e.g. gemma-3-1b-it-4bit on A19: 850 ms → ~200 ms). Relative to the published upstream release metrics (uzu 0.5.14, A-series), gemma-3-1b-it-4bit decode improves ×11.5 and outperforms MLX on generation speed, TTFT, and memory on the same device class.

### Copy-heavy workloads (speculative decoding)

Verbatim-quoting task (extract six sentences word-for-word from a supplied text), M1 Max, greedy, 3 runs. This exercises the fork's draft-free speculative decoding, which upstream does not have; the generated text is bitwise identical across all three configurations.

| Model | Upstream | Fork, no speculation | Fork, defaults | Δ |
|---|---|---|---|---|
| gemma-3-1b-it (bf16) | 125.9 | 131.7 | **233.9** | ×1.86 |
| Qwen3-0.6B (bf16) | 171.8 | 186.1 | **335.7** | ×1.95 |

¹ Upstream cannot run these four models on macOS 26.6 at all: its attention GEMM kernel at `head_dim = 256` crashes the system shader compiler (SIGSEGV in the LLVM register allocator inside MTLCompilerService). Upstream numbers were collected with a minimal 6-line fix from this fork applied (see “Bug fixes”).

## Changes

### GPU kernels

- **Quantized GEMV rewrite.** In-place nibble dequantization with full-rate integer-to-float conversion (masked words feed the multipliers directly; activation lanes are pre-scaled by exact powers of two), replacing the shift-and-convert unpack path.
- **Per-device-tier tile tables.** GEMV tile selection (simdgroups, rows per simdgroup, split-K, lane pack depth) is fitted per device tier (M1-class, A15–A16, A17/M3+, Max/Ultra) and per layer shape, with environment-variable probes for sweeps.
- **Single-pack lane mode (`PACKS = 1`).** Dense per-lane loads that hide dequantization latency on shallow-reduction shapes; opt-in per shape via the tile tables (cuts gemma qkv kernel time by 39% and fused up/gate by 49% on A19).
- **FP16 dual-issue path (A19).** The quantized GEMV reduction can run on the double-rate FP16 pipes where the hardware provides them; enabled per shape on measured wins only (+1.8% end-to-end on A19; other tiers retain FP32).
- **Batched GEMV (`M_TILE`).** One threadgroup runs 4 or 8 batch elements through a single weight pass, amortizing weight reads and dequantization. Fitted policy per tier; the enabler for speculative verification (a 4-token verification pass costs ~1.5× a single-token pass on bf16 instead of ~5×).
- **Attention register-pressure cap.** The attention GEMM caps the register-resident PV operand; at `head_dim = 256` the uncapped variant requires 128 float registers per lane.
- **GEMM row-unaligned tile fix.** Tiles partial in rows now use clamped vector loads instead of per-element predicated loads; quantized GEMM at `m = 4` improves 2.6–2.9× to parity with `m = 8`, and any prefill with a partial trailing row tile benefits.
- **int8 KV cache** with per-(token, head) scales and a dequantization bridge for prefill.
- **Top-k candidate selection** in sampling instead of sorting the full vocabulary.

### Submission pipeline

- Collapsed the three-command-buffer sandwich into encodeWait/SignalEvent on the working command buffer.
- Matmul encoding on a worker pool; per-group work hoisted out of the inner loop; per-dispatch work removed from the encoder; allocator pool numbers recycled.
- Host-side: regexes compiled once; release builds use `codegen-units = 1`.

### Speculative decoding without a draft model

Prompt-lookup speculation on the existing tree-verification infrastructure: drafts are proposed from trigram matches against the token history on the CPU (no weights, no GPU cost), verified exactly by a batched pass. Drafting starts paused and engages only when consecutive full-length lookup hits indicate a verbatim span; an acceptance window pauses it again when drafts stop landing. Enabled by default (`UZU_PROMPT_LOOKUP=0` disables); quantized models are excluded automatically (their batched pass economics do not pay back).

- Copy-heavy generation (verbatim quoting, extraction): gemma bf16 ×1.78, Qwen3-0.6B ×1.80 over the fork's own non-speculative decode on M1 Max (×1.86 / ×1.95 over upstream — see Results); ×2.36 on iPhone 17 Pro.
- Free-form chat: on par with baseline (within run-to-run noise).
- Greedy output on copy workloads is bitwise identical to non-speculative decoding; long-generation and 10-question factual smoke checks are bitwise identical as well.

### Model tooling

- **encoding.json inference.** The engine's local model registries infer the chat-template configuration from the recorded repository id or the model directory name when `encoding.json` is missing (side-loaded and freshly converted models previously failed to start chat), and persist it next to the weights.

### Bug fixes

- **macOS 26.6 shader-compiler crash** (upstream-affecting): attention GEMM at `head_dim = 256` (gemma-3, Qwen3.5) crashes MTLCompilerService with the current Metal toolchain. Fixed by the register-pressure cap above; the minimal six-line form of the fix is an upstream candidate.
- **Quantized GEMM `m = 4` pathology** (2.7× slower than `m = 8`) — the row-unaligned tile fix above.

### Diagnostics and benchmarking

- `device_decode_probe`: an engine-level decode probe runnable on iOS devices via cargo-dinghy (model shipped in the app bundle), reporting TTFT, decode throughput, and peak footprint.
- Environment probes for sweeps and A/B runs: `UZU_QUANT_TILE`, `UZU_QUANT_TILE_MAP`, `UZU_GEMV_HALF`, `UZU_GEMV_M_TILE`, `UZU_GEMV_MAX_BATCH`, `UZU_PROMPT_LOOKUP`, `UZU_SPEC_DEBUG`, `UZU_PSO_TRACE`, `UZU_ENC_TRACE`.
- On-device criterion benchmarking recipes for iPhone (see `crates/backend-uzu/BENCHMARKS.md`), including thermal-validity protocol notes.

## Validation

- Model weights used in all comparisons are bitwise identical to the original Hugging Face repositories (SHA-256).
- Short-prompt greedy outputs are bitwise identical to upstream on all six benchmark models; long-prompt outputs are coherent and semantically parallel (expected under changed FMA ordering).
- Test suite: 594 passing; clippy clean.

## License

Same as upstream: MIT. See [LICENSE](LICENSE).
