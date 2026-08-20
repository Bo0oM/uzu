use crate::backends::metal::device_tier::DeviceTier;

// One full vectorized fp K block is values_per_thread * 32 lanes.
pub(crate) const DEFAULT_FP_VALUES_PER_THREAD: u32 = 4;

/// Lane depth of the fp GEMV source. The deep-lane variant reads 16 B per
/// lane and wins ~6-7% on SmallLegacy when k tiles evenly (Qwen3-0.6B +7.1%,
/// LFM2 +6.2%); on k % 256 != 0 rows (gemma's 1152) the coarser blocks cost
/// split-K slices and lose ~10%, so those keep the 4-value lanes. Other tiers
/// keep 4 until their fleet tables are re-swept with this axis.
pub(crate) fn fp_values_per_thread(
    k: u32,
    tier: DeviceTier,
) -> u32 {
    if tier == DeviceTier::SmallLegacy && k.is_multiple_of(fp_k_block(8)) {
        8
    } else {
        DEFAULT_FP_VALUES_PER_THREAD
    }
}

pub(crate) fn fp_k_block(values_per_thread: u32) -> u32 {
    values_per_thread * 32
}
pub(crate) const DEFAULT_RESULTS_PER_SIMDGROUP: u32 = 4;
pub(crate) const DEFAULT_NUM_SIMDGROUPS: u32 = 8;

/// Tile specialization selected for one GEMV dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemvTile {
    /// SIMD groups launched in one threadgroup.
    pub num_simdgroups: u32,
    /// Number of split-K slices reduced by one threadgroup.
    pub k_split: u32,
    /// Output rows computed by each SIMD group.
    pub results_per_simdgroup: u32,
    /// Per-lane pack depth of the quantized source (1 or 2; see gemv.metal).
    pub packs: u32,
}

/// Whether the quantized source stages activations in half for this dispatch.
/// Half runs the qdot reduction on the double-rate FP16 pipes, which only
/// exist on Apple9+ phones (A19); A15-A18 and the M-series run FP16 at FP32
/// rate and keep float. The win is form-dependent (A19 gemma gs64 sweep, Aug
/// 2026: qkv -6%, down -3.4%, vocab readout -2.2%, but o +6.5% and upgate
/// +3.6%), so only the measured winning cells opt in; cells sharing gemma's
/// qkv/o bucket split on input alignment (unaligned qkv wins, aligned o
/// loses). Other group sizes stay float until swept. UZU_GEMV_HALF=0/1
/// forces it off/on globally for A/B runs and Mac-side parity checks.
pub(crate) fn quant_half_math(
    tier: DeviceTier,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
    input_aligned: bool,
) -> bool {
    static FORCED: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let forced = FORCED
        .get_or_init(|| std::env::var("UZU_GEMV_HALF").ok().and_then(|s| s.parse::<u32>().ok()).map(|v| v != 0));
    if let Some(forced) = *forced {
        return forced;
    }
    // Fitted on m = 1 decode forms only, matching the quant tile tables.
    if tier != DeviceTier::SmallApple9 || group_size != 64 || m != 1 {
        return false;
    }
    let k_bucket = table_bucket_index(k, &QUANT_K_BUCKET_MAXES);
    let n_bucket = table_bucket_index(n, &QUANT_N_BUCKET_MAXES);
    match (k_bucket, n_bucket) {
        (1, 1) => !input_aligned,
        (2, 1) | (1, 6) => true,
        _ => false,
    }
}

/// PROBE (sweep tooling): UZU_GEMV_M_TILE=1|4|8 overrides the fitted batched
/// tile choice (1 forces the classic per-element grid for A/B).
fn gemv_m_tile_probe() -> Option<u32> {
    static PROBE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *PROBE.get_or_init(|| {
        std::env::var("UZU_GEMV_M_TILE").ok().and_then(|s| s.parse().ok()).filter(|m| matches!(m, 1 | 4 | 8))
    })
}

/// Batch elements sharing one weight pass per threadgroup. Fitted on the
/// 24-core M1 Max stand (SmallLegacy tier; criterion GPU-time vs the classic
/// per-element grid at the same m, gemma forms, Aug 2026): quantized gs64
/// m=4 wins on every form (down -61%, readout -49%,
/// upgate -41%, qkv -14%, o -4%); fp bf16 m=4 wins except on small-n
/// shallow-k forms like o (n=1152, k=1024) where the classic grid's extra
/// threadgroups hide latency better; fp m=8 only pays on wide-n forms
/// (upgate -51%, readout -57%) and loses badly elsewhere to register
/// pressure (o +216%). Other tiers keep the classic grid until swept.
pub(crate) fn gemv_m_tile(
    tier: DeviceTier,
    m: u32,
    n: u32,
    k: u32,
    is_quant: bool,
) -> u32 {
    if let Some(forced) = gemv_m_tile_probe() {
        return if forced == m {
            forced
        } else {
            1
        };
    }
    match tier {
        DeviceTier::SmallLegacy => {
            if is_quant {
                if m == 4 {
                    4
                } else {
                    1
                }
            } else {
                match m {
                    4 if n >= 1536 || k >= 2048 => 4,
                    8 if n >= 8192 => 8,
                    _ => 1,
                }
            }
        },
        // A19 (Aug 2026 sweep): the classic quant grid is already near-free
        // on deep-k/huge-n forms (SLC shares weights across batch elements:
        // down m4 = 1.03x m1), so quant batching pays only on shallow-k
        // moderate-n forms (qkv -38/-51%, o -41/-49%, upgate -13/-25% at
        // m4/m8); fp batching wins everywhere except small-n shallow-k o,
        // and Dynamic Caching keeps even M_TILE=8 out of register trouble.
        DeviceTier::SmallApple9 => {
            if is_quant {
                if matches!(m, 4 | 8) && k < 2048 && n <= 16384 {
                    m
                } else {
                    1
                }
            } else if matches!(m, 4 | 8) && (n >= 1536 || k >= 2048) {
                m
            } else {
                1
            }
        },
        _ => 1,
    }
}

/// Per-shape tile overrides produced by the first-launch calibration
/// (gemv::autotune). Read on every selection; writes happen once per shape
/// per process.
/// Keyed by the same quadruple the calibration resolves and the on-disk cache
/// stores. Shape alone is not enough: two models can share `(n, k)` and differ
/// in scale group, and the winning tile's `packs` is fitted per group.
type AutotuneKey = (u32, u32, u32, u32);

static AUTOTUNE_OVERRIDES: std::sync::RwLock<Vec<(AutotuneKey, GemvTile)>> = std::sync::RwLock::new(Vec::new());

fn autotune_override(key: AutotuneKey) -> Option<GemvTile> {
    AUTOTUNE_OVERRIDES.read().ok()?.iter().find(|(entry, _)| *entry == key).map(|(_, tile)| *tile)
}

pub(crate) fn set_autotune_override(
    n: u32,
    k: u32,
    group_size: u32,
    bits: u32,
    tile: GemvTile,
) {
    let key = (n, k, group_size, bits);
    if let Ok(mut overrides) = AUTOTUNE_OVERRIDES.write() {
        overrides.retain(|(entry, _)| *entry != key);
        overrides.push((key, tile));
    }
}

const SMALL_G13_HUGE_N: u32 = 32768;
const SMALL_G13_WIDE_ROW_N: u32 = 6144;
const DEEP_K: u32 = 8192;
const FP_LARGE_SPLIT_K_MIN_DEPTH: u32 = 512;
const FP_K_DEPTH_N_MAX: u32 = 4095;
const FP_K_DEPTH_DEEP_MIN: u32 = 3072;
const FP_K_DEPTH_VERY_DEEP_RATIO: u32 = 16;

const fn tile(
    num_simdgroups: u32,
    k_split: u32,
    results_per_simdgroup: u32,
) -> GemvTile {
    GemvTile {
        num_simdgroups,
        k_split,
        results_per_simdgroup,
        packs: 2,
    }
}

const fn qtile(
    num_simdgroups: u32,
    results_per_simdgroup: u32,
) -> GemvTile {
    tile(num_simdgroups, 1, results_per_simdgroup)
}

/// Opts a fitted cell into the dense single-pack quantized source; the kernel
/// grid only instantiates it for 4-bit gs64, so other dispatches clamp back
/// to two packs in kernel.rs.
const fn with_packs1(mut selected: GemvTile) -> GemvTile {
    selected.packs = 1;
    selected
}

// One quantized 4-bit K block at the default PACKS = 2: 16 values per lane *
// 32 lanes. Fitted tables and split caps keep this granularity; the real
// per-pipeline block (which depends on the packs knob) lives in kernel.rs.
const QUANT_K_BLOCK: u32 = 512;

// PROBE (sweep tooling, parsed once per process): UZU_QUANT_TILE=sg,r[,ks[,packs]]
// forces one tile globally; UZU_QUANT_TILE_MAP="n:k:sg:r[:ks[:packs]];..." overrides
// per shape; UZU_QUANT_TILE_LOG=1 prints selections. Forced tiles still pass
// the split cap and RHT guard. To sweep quantized split-K, also widen the
// K_SPLIT constraint in gemv.metal locally.
struct QuantTileProbe {
    forced: Option<GemvTile>,
    map: Vec<(u32, u32, GemvTile)>,
    log: bool,
}

fn quant_tile_probe() -> &'static QuantTileProbe {
    static PROBE: std::sync::OnceLock<QuantTileProbe> = std::sync::OnceLock::new();
    PROBE.get_or_init(|| {
        let parse_tile = |sg: &str, r: &str, ks: Option<&str>, packs: Option<&str>| -> Option<GemvTile> {
            let ks = ks.and_then(|ks| ks.parse().ok()).unwrap_or(1);
            let mut forced = tile(sg.parse().ok()?, ks, r.parse().ok()?);
            forced.packs = packs.and_then(|packs| packs.parse().ok()).unwrap_or(2);
            Some(forced)
        };
        let mut map = Vec::new();
        if let Ok(spec) = std::env::var("UZU_QUANT_TILE_MAP") {
            for entry in spec.split(';') {
                let mut it = entry.split(':');
                if let (Some(n), Some(k), Some(sg), Some(r)) = (it.next(), it.next(), it.next(), it.next())
                    && let (Ok(n), Ok(k)) = (n.parse(), k.parse())
                    && let Some(forced) = parse_tile(sg, r, it.next(), it.next())
                {
                    map.push((n, k, forced));
                }
            }
        }
        let forced = std::env::var("UZU_QUANT_TILE").ok().and_then(|spec| {
            let mut it = spec.split(',');
            parse_tile(it.next()?, it.next()?, it.next(), it.next())
        });
        QuantTileProbe {
            forced,
            map,
            log: std::env::var_os("UZU_QUANT_TILE_LOG").is_some(),
        }
    })
}

// The kernel grid instantiates quantized split-K at 8 simdgroups only, and a
// slice with no complete K block would only ever see the unaligned tail.
fn cap_quant_k_split(
    k: u32,
    selected: GemvTile,
) -> GemvTile {
    let k_split = cap_k_split_to_complete_fp_k_blocks(k, selected.k_split, QUANT_K_BLOCK);
    GemvTile {
        num_simdgroups: if k_split > 1 {
            DEFAULT_NUM_SIMDGROUPS
        } else {
            selected.num_simdgroups
        },
        k_split,
        packs: selected.packs,
        results_per_simdgroup: selected.results_per_simdgroup,
    }
}

// The RHT epilogue transforms exactly one 32-row hadamard block per
// threadgroup; a tile with any other row count writes zeros past row 32.
fn rht_rows_guard(
    has_rht: bool,
    selected: GemvTile,
) -> GemvTile {
    let rows = (selected.num_simdgroups / selected.k_split) * selected.results_per_simdgroup;
    if has_rht && rows != 32 {
        DEFAULT_TILE
    } else {
        selected
    }
}

pub(crate) const DEFAULT_TILE: GemvTile = qtile(DEFAULT_NUM_SIMDGROUPS, DEFAULT_RESULTS_PER_SIMDGROUP);
// Qxy = qtile(num_simdgroups=x, results_per_simdgroup=y), with KS1.
const Q21: GemvTile = qtile(2, 1);
const Q22: GemvTile = qtile(2, 2);
const Q24: GemvTile = qtile(2, 4);
const Q42: GemvTile = qtile(4, 2);
const Q44: GemvTile = qtile(4, 4);
const Q48: GemvTile = qtile(4, 8);
const Q82: GemvTile = qtile(8, 2);
const Q28: GemvTile = qtile(2, 8);
const QUANT_N_BUCKET_MAXES: [u32; 6] = [512, 2048, 4096, 8192, 16384, 32768];
const QUANT_K_BUCKET_MAXES: [u32; 3] = [512, 2048, 8192];
const QUANT_RHT_TUNED_N_MIN_EXCLUSIVE: u32 = 2048;
const QUANT_RHT_TUNED_N_MAX: u32 = 4096;
const QUANT_RHT_TUNED_K_MIN: u32 = 2048;

fn table_bucket_index(
    value: u32,
    bucket_maxes: &[u32],
) -> usize {
    bucket_maxes.partition_point(|&max| value > max)
}

fn cap_k_split_to_complete_fp_k_blocks(
    k: u32,
    preferred: u32,
    k_block: u32,
) -> u32 {
    // K_SPLIT variants are powers of two. Do not split beyond the number of
    // complete vectorized K blocks each slice can own.
    let complete_blocks = k / k_block;
    if complete_blocks == 0 {
        return 1;
    }
    preferred.min((1 << complete_blocks.ilog2()).min(DEFAULT_NUM_SIMDGROUPS))
}

fn preferred_fp_k_split(
    m: u32,
    n: u32,
    k: u32,
) -> u32 {
    if m <= 2 {
        return 8;
    }
    if m <= 4 {
        return if n <= 16384 {
            8
        } else {
            1
        };
    }
    if n <= 512 {
        return 8;
    }
    if n <= 1024 {
        return if n != 0 && k / n >= FP_K_DEPTH_VERY_DEEP_RATIO {
            8
        } else {
            4
        };
    }
    if n <= FP_K_DEPTH_N_MAX {
        return if n != 0 && k / n >= FP_K_DEPTH_VERY_DEEP_RATIO {
            8
        } else if k >= FP_K_DEPTH_DEEP_MIN {
            4
        } else {
            2
        };
    }
    1
}

/// Selects the full-precision GEMV tile. `m` is the input-vector count,
/// `n` is the output row count, and `k` is the reduction depth.
pub(crate) fn fp_tile(
    m: u32,
    n: u32,
    k: u32,
    input_aligned: bool,
    values_per_thread: u32,
    tier: DeviceTier,
) -> GemvTile {
    // FP sweeps covered SG2/SG4/SG8; SG changes did not produce portable
    // confirmed wins, so shipped FP policy keeps SG8 and tunes KS/R only.
    let should_disable_k_split = !input_aligned
        || (m == 1 && tier == DeviceTier::Large && k < FP_LARGE_SPLIT_K_MIN_DEPTH)
        || (m == 1 && tier == DeviceTier::SmallLegacy && n >= SMALL_G13_HUGE_N);

    let k_split = if should_disable_k_split {
        1
    } else {
        cap_k_split_to_complete_fp_k_blocks(k, preferred_fp_k_split(m, n, k), fp_k_block(values_per_thread))
    };

    // R1 won most single-row FP sweeps; Large devices only switch back to R4
    // for deep-K rows, while legacy wide rows keep R4.
    let results_per_simdgroup = if tier == DeviceTier::SmallLegacy && m == 1 && n >= SMALL_G13_WIDE_ROW_N {
        DEFAULT_RESULTS_PER_SIMDGROUP
    } else if m == 1 && (k <= DEEP_K || tier != DeviceTier::Large) {
        1
    } else {
        DEFAULT_RESULTS_PER_SIMDGROUP
    };

    tile(DEFAULT_NUM_SIMDGROUPS, k_split, results_per_simdgroup)
}

/// Selects the quantized GEMV tile. `m` is the input-vector count, `n` is the
/// output row count, `k` is the reduction depth, and `bits` is the quant width.
pub(crate) fn quant_tile(
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
    bits: u32,
    has_rht: bool,
    tier: DeviceTier,
) -> GemvTile {
    // These tables are fitted for batch-1 Q4 only; Q8/future widths keep the
    // deterministic default until they have their own cold sweep.
    if m != 1 || bits != 4 {
        return DEFAULT_TILE;
    }
    let probe = quant_tile_probe();
    let selected = quant_tile_uncapped(n, k, group_size, bits, has_rht, tier, probe);
    // Every path passes the split cap and the RHT row guard, so probe-forced
    // tiles and future table rows cannot select an invalid split or break the
    // 32-row hadamard invariant.
    let selected = rht_rows_guard(has_rht, cap_quant_k_split(k, selected));
    if probe.log {
        eprintln!(
            "QUANTSHAPE n={n} k={k} rht={has_rht} -> SG{},KS{},R{}",
            selected.num_simdgroups, selected.k_split, selected.results_per_simdgroup
        );
    }
    let mut selected = if n < selected.results_per_simdgroup {
        // Coarse N buckets can include n < R; keep the default R4 tile for tiny rows.
        DEFAULT_TILE
    } else {
        selected
    };
    // Single-pack lanes are instantiated for the gs64 slice alone. The table's
    // `with_packs1` cells are fitted there, and a non-gs64 shape reaching one
    // used to have its `packs` put back by the dispatcher — which meant the
    // table said one thing and the dispatch did another. The table answers for
    // itself now.
    if selected.packs == 1 && group_size != 64 {
        selected.packs = 2;
    }
    selected
}

fn quant_tile_uncapped(
    n: u32,
    k: u32,
    group_size: u32,
    bits: u32,
    has_rht: bool,
    tier: DeviceTier,
    probe: &QuantTileProbe,
) -> GemvTile {
    for &(pn, pk, forced) in &probe.map {
        if pn == n && pk == k {
            return forced;
        }
    }
    if let Some(forced) = probe.forced {
        return forced;
    }
    // First-launch autotune winners sit between the sweep probes and the
    // fitted tables: an explicit env probe always wins, a calibrated shape
    // beats the shipped default.
    if let Some(tuned) = autotune_override((n, k, group_size, bits)) {
        return tuned;
    }
    if has_rht {
        // This special case mirrors quant bucket edges: n in (2048, 4096]
        // and k at or above the 2048 boundary.
        return if tier == DeviceTier::Large
            && n > QUANT_RHT_TUNED_N_MIN_EXCLUSIVE
            && n <= QUANT_RHT_TUNED_N_MAX
            && k >= QUANT_RHT_TUNED_K_MIN
        {
            qtile(4, 8)
        } else {
            DEFAULT_TILE
        };
    }

    let k_bucket = table_bucket_index(k, &QUANT_K_BUCKET_MAXES);
    let n_bucket = table_bucket_index(n, &QUANT_N_BUCKET_MAXES);
    // Q4 BF16 decode choices from June 2026 gemv_fine_tune sweeps; omitted
    // cells keep SG8_KS1_R4. Other quant widths keep DEFAULT_TILE until swept.
    match (tier, k_bucket, n_bucket) {
        (DeviceTier::Large, 0, 1) => Q42,
        (DeviceTier::Large, 1, 0) => Q21,
        (DeviceTier::Large, 1, 1..=3) => Q22,
        (DeviceTier::Large, 1, 4) => Q21,
        (DeviceTier::Large, 1, 5) => Q24,
        (DeviceTier::Large, 2, 1) => Q42,
        (DeviceTier::Large, 3, 1) => Q22,

        (DeviceTier::SmallApple9, 0, 1) => Q44,
        (DeviceTier::SmallApple9, 1, 0) => Q42,
        // packs=1 on shallow-k gs64 (A19 sweep, Aug 2026): qkv -39%,
        // upgate -49% at Q48, vocab readout -4%; o/down keep packs=2.
        (DeviceTier::SmallApple9, 1, 1) => with_packs1(Q22),
        (DeviceTier::SmallApple9, 1, 2) => Q42,
        (DeviceTier::SmallApple9, 1, 4) => with_packs1(Q48),
        (DeviceTier::SmallApple9, 1, 5) => Q42,
        (DeviceTier::SmallApple9, 1, 6) => with_packs1(DEFAULT_TILE),
        (DeviceTier::SmallApple9, 2, 1) => Q42,
        (DeviceTier::SmallApple9, 3, 1) => Q22,

        (DeviceTier::SmallApple8, 0, 1) => Q44,
        (DeviceTier::SmallApple8, 1, _) | (DeviceTier::SmallApple8, 2, 1) => Q82,

        (DeviceTier::SmallLegacy, 0, 1) => Q48,
        // Shallow-k non-RHT rows amortize the A-vector load and dequant over more rows per
        // simdgroup: measured on gemma-3-1b-it-4bit (k=1152), qkv n=1536 gains 4.4% at SG8/R4,
        // fused up/gate n=13824 gains 7.3% and the vocab readout 7.1% at R8; o-proj is neutral.
        // The 0.8B readout (n=248320, k=1024) gains 0.8% on the same R8 cell.
        (DeviceTier::SmallLegacy, 1, 0) => Q82,
        // Under the direct-convert nibble unpack (Aug 2026) block-unaligned k
        // prefers few wide simdgroups (gemma qkv k=1152: -10% at SG2/R8),
        // while the aligned rows of the same bucket keep the default.
        // packs=1 (Aug 2026): shallow-k gs64 forms hide dequant latency with
        // dense per-lane loads (gemma engine +55%); non-gs64 dispatches on
        // these cells clamp back to packs=2 in kernel.rs.
        (DeviceTier::SmallLegacy, 1, 1) if !k.is_multiple_of(QUANT_K_BLOCK) => with_packs1(Q28),
        (DeviceTier::SmallLegacy, 1, 1) => with_packs1(DEFAULT_TILE),
        (DeviceTier::SmallLegacy, 1, 2..=3) => Q82,
        (DeviceTier::SmallLegacy, 1, 4) => with_packs1(Q28),
        (DeviceTier::SmallLegacy, 1, 6) => with_packs1(Q48),
        // gemma down (k=6912): least-bad row under the direct-convert unpack.
        (DeviceTier::SmallLegacy, 2, 1) => with_packs1(Q48),

        _ => DEFAULT_TILE,
    }
}

#[cfg(test)]
#[path = "../../../../../../tests/unit/backends/metal/kernel/matmul/gemv/policy_test.rs"]
mod tests;
