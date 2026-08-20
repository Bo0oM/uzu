# uzu — performance fork

This repository is a performance-focused fork of [trymirai/uzu](https://github.com/trymirai/uzu), a high-performance inference engine for AI models on Apple silicon. For general product documentation, API usage, bindings, and the model catalog, refer to the [original README](https://github.com/trymirai/uzu/blob/main/README.md). This document describes what this fork changes relative to the upstream baseline (branch point `1bbb59eb`).

## Results

Measured on identical hardware, identical models (weights SHA-256-verified against the original repositories), and an identical protocol: the two binaries run back to back on a cooled machine, alternating which side goes first so neither arm gets the warm-up. Decode throughput, tokens/s.

Both sides run as a user would run them; on this fork that means prompt-lookup speculation and the first-launch tile autotune enabled. Turning both off measures within ~1% of the defaults on free-form prompts, so there is no separate ablation column.

### macOS, M1 Max (24-core GPU)

| Model | Upstream | This fork | Δ |
|---|---|---|---|
| LFM2-350M (bf16) | 347.6 | **372.8** | +7.3% |
| Qwen3-0.6B (bf16) | 173.3 | **183.2** | +5.7% |
| gemma-3-1b-it (bf16)¹ | 126.2 | **132.1** | +4.7% |
| gemma-3-1b-it-4bit¹ | 73.9 | **156.2** | ×2.11 |
| Qwen3.5-0.8B-M (4-bit, RHT)¹ | 240.8 | **279.6** | +16.1% |
| Qwen3.5-4B-M (4-bit, RHT)¹ | 80.4 | **89.4** | +11.2% |

Both columns come from one session: for each model the two binaries run back to back, twice each in the order fork, upstream, upstream, fork, so a drifting machine cannot favour either arm. Before each model a canary — a fixed short decode — has to hold still and stay near what this machine is known to do; the canary that admitted each row is recorded, and across the six rows they agree within 1.5%. That is the point of the whole apparatus: on this stand the machine's own state is worth about 5%, which is the same size as most of the numbers in the Δ column, so a table whose two halves were measured on different afternoons says more about the afternoons than about the code.

Two ways that has gone wrong here, both now fixed. The first version of this table read 12% low on gemma-3-1b-it-4bit because the tile cache was keyed on the engine version, which does not move between builds of a development branch, so it kept serving winners measured against kernels that had since been rewritten — silently, since a stale tile is still a valid one; the key now carries a hash of the compiled shaders. And the canary's stability check was itself broken for its whole existence, admitting any three readings however far apart, which is how a 2.5% swing in the machine came to look like a regression in gemma-3-1b-it.

### iOS, iPhone 17 Pro (A19 Pro)

| Model | Upstream | This fork | Δ |
|---|---|---|---|
| gemma-3-1b-it-4bit | 75.0 | **103.4** | +38% |
| Qwen3-0.6B (bf16) | 45.7 | **59.0** | +29% |
| Qwen3.5-0.8B-M | 125.7 | **136.6** | +8.7% |
| LFM2-350M (bf16) | 95.5 | **100.3** | +5.0% |

Re-measured on the current build. 256 decode tokens after a 64-token prompt — the probe's own default is 64, and the two lengths are not comparable. The phone has no equivalent of the desktop's thermal gate, and it matters: LFM2-350M reads 100.3 on a rested device and 91 after an hour of back-to-back runs, so a model measured last in a sweep is measured on the hottest phone. Every row here was taken with the device rested.

Time to first token improves across the board (e.g. gemma-3-1b-it-4bit on A19: 850 ms → ~200 ms). Relative to the published upstream release metrics (uzu 0.5.14, A-series), gemma-3-1b-it-4bit decode improves ×11.5 and outperforms MLX on generation speed, TTFT, and memory on the same device class.

### Copy-heavy workloads (speculative decoding)

Verbatim-quoting task — copy the first six sentences of a supplied ~4900-character text word for word, M1 Max, greedy, 3 runs. This exercises the fork's draft-free speculative decoding, which upstream does not have; the generated text is bitwise identical across all three configurations.

| Model | Upstream | Fork, no speculation | Fork, defaults | Δ |
|---|---|---|---|---|
| gemma-3-1b-it (bf16) | 125.6 | 131.4 | **185.7** | ×1.48 |
| Qwen3-0.6B (bf16) | 172.5 | 188.2 | **238.3** | ×1.38 |
| Qwen3-0.6B-q8emb | — | 206.6 | **253.1** | +22% |

The last row is a model this fork's own `--quantize-embeddings` produces; upstream cannot run it. How much speculation buys depends on how much of the answer is quoted rather than composed — a task that copies longer spans pays back more.

¹ Upstream cannot run these four models on macOS 26.6 at all: its attention GEMM kernel at `head_dim = 256` crashes the system shader compiler (SIGSEGV in the LLVM register allocator inside MTLCompilerService). Upstream numbers were collected with a minimal 6-line fix from this fork applied (see "Bug fixes").

## Changes

### GPU kernels

- **Quantized GEMV rewrite.** In-place nibble dequantization with full-rate integer-to-float conversion (masked words feed the multipliers directly; activation lanes are pre-scaled by exact powers of two), replacing the shift-and-convert unpack path.
- **Per-device-tier tile tables.** GEMV tile selection (simdgroups, rows per simdgroup, split-K, lane pack depth) is fitted per device tier (M1-class, A15–A16, A17/M3+, Max/Ultra) and per layer shape, with environment-variable probes for sweeps.
- **Single-pack lane mode (`PACKS = 1`).** Dense per-lane loads that hide dequantization latency on shallow-reduction shapes; opt-in per shape via the tile tables (cuts gemma qkv kernel time by 39% and fused up/gate by 49% on A19).
- **FP16 dual-issue path (A19).** The quantized GEMV reduction can run on the double-rate FP16 pipes where the hardware provides them; enabled per shape on measured wins only (+1.8% end-to-end on A19; other tiers retain FP32).
- **Batched GEMV (`M_TILE`).** One threadgroup runs 4 or 8 batch elements through a single weight pass, amortizing weight reads and dequantization. Fitted policy per tier; the enabler for speculative verification (a 4-token verification pass costs ~1.5× a single-token pass on bf16 instead of ~5×).
- **Quantized GEMM tile policy.** The wide tile needs whole 64-column blocks, not a large `n`; the old policy also demanded `n ≥ 6144`, which sent every projection narrower than that — qkv, output, down — to a tile 12% slower. Prefill improves 9.8% on gemma-3-1b-4bit and 6–7% on 8-bit models; bf16 is indifferent to the tile and is left alone.
- **Attention: skipping blocks the mask would discard.** The GEMM path drops ring blocks that are entirely dead — before the live window (+4.5% on ~1200-token prompts, where the ring is still empty on the first chunk) and below the sliding-window floor (+3.45% on ~4800-token prompts). Both are numerically neutral by construction.
- **Attention register-pressure cap.** The attention GEMM caps the register-resident PV operand; at `head_dim = 256` the uncapped variant requires 128 float registers per lane.
- **GEMM row-unaligned tile fix.** Tiles partial in rows now use clamped vector loads instead of per-element predicated loads; quantized GEMM at `m = 4` improves 2.6–2.9× to parity with `m = 8`, and any prefill with a partial trailing row tile benefits.
- **int8 KV cache, enabled per model rather than by hand.** Per-(token, head) scales with a dequantization bridge for prefill. The engine decides at load time by comparing the cache traffic a session will actually generate against the model's weight bytes: below 35% the cache is not what decode waits on and quantizing it only costs accuracy. Sliding-window layers are charged only their window, not the whole context — the difference between a correct 12% reading and a wrong 44% one on gemma-3-1b. Measured: Qwen3-0.6B at 16k gains 31%, an 8B at the same context loses 4%, and each now gets the right answer without a flag.
- **Top-k candidate selection** in sampling instead of sorting the full vocabulary.

### Submission pipeline

- Collapsed the three-command-buffer sandwich into encodeWait/SignalEvent on the working command buffer.
- Matmul encoding on a worker pool; per-group work hoisted out of the inner loop; per-dispatch work removed from the encoder; allocator pool numbers recycled.
- CPU matmul dispatch: code, mask, midpoint and sign flip are properties of a dispatch, not of a column, and are computed once per dispatch instead of once per column (−10% dispatch time).
- Host-side: regexes compiled once; release builds use `codegen-units = 1`.

### Speculative decoding without a draft model

Prompt-lookup speculation on the existing tree-verification infrastructure: drafts are proposed from trigram matches against the token history on the CPU (no weights, no GPU cost), verified exactly by a batched pass. Drafting starts paused. It resumes on hindsight rather than on hope: over a window of already-generated tokens the engine counts the drafts the lookup *would* have got right, and only resumes when that predicts a pass worth more than it costs. An acceptance window pauses it again when drafts stop landing. Enabled by default (`UZU_PROMPT_LOOKUP=0` disables); quantized models are excluded automatically, since their batched pass costs about 4.2× a single-token pass and cannot pay that back.

- Copy-heavy generation (verbatim quoting, extraction): gemma bf16 ×1.78, Qwen3-0.6B ×1.80 over the fork's own non-speculative decode on M1 Max (see Results); ×2.36 on iPhone 17 Pro.
- Free-form chat: on par with baseline (within run-to-run noise).
- Greedy output on copy workloads is bitwise identical to non-speculative decoding; long-generation and 10-question factual smoke checks are bitwise identical as well.

### Model tooling

- **Vendored lalamo fork** (`lalamo/`, pinned to the 0.5.16 model-format schema) with a new `--quantize-embeddings {4,8}` conversion flag: quantizes the tied embedding table only (it doubles as the vocabulary readout and accounts for 20–32% of per-token weight traffic on bf16 exports). Measured: Qwen3-0.6B +8.6%, LFM2-350M +7.0% decode at a 0.68% RMS weight error (1.7·10⁻³ maximum absolute, Qwen3-0.6B at 8 bits and group 64), matching the table an external quantizer produces for the same settings.
- **A quantizer that refuses to ship a degraded model.** Group scales were computed in the weight's own dtype, so for bfloat16 every scale below 2⁻⁷ clamped to that floor — which every embedding group hit, leaving roughly four effective bits out of eight and 7.62% RMS reconstruction error where 0.68% was correct. Decompression now goes through float32. Since a silent precision loss is exactly what hid there, conversion also measures it: the observed error is compared against what the bit width and group size predict, and a ratio above three aborts the conversion instead of writing the file.
- **Downloader auto-conversion.** `tools/downloader` falls back to the vendored lalamo for Hugging Face repositories that are not in the cloud registry: download, convert, and register in one command.
- **Full weight quantization** (`--quantize 8`) alongside the embedding-only flag, and a Foundation-Sec-8B-Reasoning model spec with its chat template — rendering, parsing and tokens — so the model's reasoning output is parsed out of the visible answer rather than handed to the caller verbatim.
- **encoding.json inference.** The engine's local model registries infer the chat-template configuration from the recorded repository id or the model directory name when `encoding.json` is missing (side-loaded and freshly converted models previously failed to start chat), and persist it next to the weights.

### Bug fixes

- **macOS 26.6 shader-compiler crash** (upstream-affecting): attention GEMM at `head_dim = 256` (gemma-3, Qwen3.5) crashes MTLCompilerService with the current Metal toolchain. Fixed by the register-pressure cap above; the minimal six-line form of the fix is an upstream candidate.
- **Quantized GEMM `m = 4` pathology** (2.7× slower than `m = 8`) — the row-unaligned tile fix above.
- **cargo-dinghy device listing** tolerates paired devices without a `cpuType` entry (e.g. a Watch surfaced through its paired iPhone).
- **Repetition penalty panicked** when the context ring wrapped.
- **NaN logits sorted incoherently** in sampling; they now compare as negative infinity through a total ordering.
- **The GDN update dropped the tail of its dv split** when the head dimension was not a multiple of the block count.
- **Two components decided the int8 KV cache independently** and disagreed, so a model below the traffic threshold allocated a quantized cache its layer had built no kernel to write, and panicked on the first prefill. The layer owns the decision; the predicate is private to it.
- **The sliding-window floor skipped live key blocks on the trie path**, where a query's position comes from its node height rather than its row index.
- **Autotune calibration timed its candidates into the caller's live output buffer**, which an accumulating epilogue would have corrupted.
- **Prompt-lookup drafting never turned itself off** once the lookup stopped proposing: the acceptance window only advanced on passes that lost a draft, while the requested batch kept forcing a blocking resolve on every token.
- **The tile cache outlived the kernels it was measured against.** Keyed on the engine version, which does not move on a development branch, it kept serving stale winners — 7.5% of gemma-3-1b-it-4bit decode. The key now carries a hash of the compiled Metal libraries, so a shader change invalidates it and a comment does not.
- **The sampling pipeline carried token indices as float bit patterns**, making every index below 2²³ a denormal and the inactive-lane marker a NaN payload — correct only for as long as nothing on the path flushed a denormal. Indices ride the lane as values now, which a float holds exactly below 2²⁴.
- **A model was called quantized because its embedding table was**, so `--quantize-embeddings` exports lost speculative decoding for no reason (+22% on a verbatim-copy task once given it back).
- **An inferred chat template was written to disk even when the model's name named two families**, baking in whichever the fixed match order reached first.

### Diagnostics and benchmarking

- `device_decode_probe`: an engine-level decode probe runnable on iOS devices via cargo-dinghy (model shipped in the app bundle), reporting TTFT, decode throughput, and peak footprint.
- Environment probes for sweeps and A/B runs: `UZU_QUANT_TILE`, `UZU_QUANT_TILE_MAP`, `UZU_GEMV_HALF`, `UZU_GEMV_M_TILE`, `UZU_GEMV_MAX_BATCH`, `UZU_PROMPT_LOOKUP`, `UZU_SPEC_DEBUG`, `UZU_PSO_TRACE`, `UZU_ENC_TRACE`.
- On-device criterion benchmarking recipes for iPhone (see `crates/backend-uzu/BENCHMARKS.md`), including the two flags without which no iOS run starts.
- **Attention microbenchmarks** covering each path `AttentionCores::encode` dispatches to. Two of the three had no number at all before: the engine benchmark only ran short prompts, so the two-pass path every long chat ends up on was never executed. It is worth measuring — twenty-eight layers of the 16k figure is 11.4 ms per token against a 14.6 ms step, so attention is around 78% of decode at that context.
- **A benchmark gate that knows when a number is worth taking**: battery temperature, CPU idle, and a canary decode that waits for consecutive readings to agree rather than for a recorded best to be matched — a best taken under conditions that no longer hold otherwise keeps the gate shut forever.
- **A regression check against a reference binary**, not against recorded numbers. This stand's baseline wanders by 10-20% over a day, so a numbers-based reference called five regressions on an unchanged tree; running both binaries minutes apart and alternating their order cancels that, and is how an int8 KV regression that had gone unnoticed was found.

## Validation

- Model weights used in all comparisons are bitwise identical to the original Hugging Face repositories (SHA-256).
- Short-prompt greedy outputs are bitwise identical to upstream on all six benchmark models; long-prompt outputs are coherent and semantically parallel (expected under changed FMA ordering).
- Test suite: 608 passing on both Qwen3-0.6B and LFM2-350M; clippy clean. On A19, all but one pass, the exception being a test that opens weights not synced to the device.

## License

Same as upstream: MIT. See [LICENSE](LICENSE).
