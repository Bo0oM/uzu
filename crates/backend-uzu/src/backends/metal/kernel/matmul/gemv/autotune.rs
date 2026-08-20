//! First-launch tile calibration for the quantized GEMV.
//!
//! The shipped tile tables are fitted per device tier, but tiers span many
//! chips. On the first dispatch of each quantized decode shape this module
//! times a handful of candidate tiles on the real buffers (~10 dispatches
//! per candidate, a few hundred microseconds each), installs the winner as
//! a policy override, and persists it to a per-device cache so later
//! launches skip the measurement entirely. UZU_TILE_AUTOTUNE=0 disables;
//! explicit sweep probes (UZU_QUANT_TILE / UZU_QUANT_TILE_MAP) take
//! precedence and suppress calibration.

use std::{
    collections::HashSet,
    fs,
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

use serde::{Deserialize, Serialize};

use super::policy::{self, GemvTile};
use crate::backends::metal::context::MetalContext;

pub(crate) const ITERATIONS_PER_CANDIDATE: u32 = 10;

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("UZU_TILE_AUTOTUNE").map_or(true, |v| v != "0")
            && std::env::var_os("UZU_QUANT_TILE").is_none()
            && std::env::var_os("UZU_QUANT_TILE_MAP").is_none()
    })
}

/// One calibrated shape in the on-disk cache.
#[derive(Serialize, Deserialize, Clone)]
struct CacheEntry {
    n: u32,
    k: u32,
    group_size: u32,
    bits: u32,
    num_simdgroups: u32,
    results_per_simdgroup: u32,
    k_split: u32,
    packs: u32,
}


#[derive(Serialize, Deserialize, Default)]
struct CacheFile {
    device: String,
    engine: String,
    /// Hash of the compiled shaders the winners were measured against.
    ///
    /// The rest of the key — device and package version — does not move
    /// between builds of a development branch, so without this the cache kept
    /// serving tiles measured against kernels that had since been rewritten.
    /// Silently: a stale tile is still a valid one, just no longer the
    /// fastest. It cost 7.5% of gemma-3-1b-4bit decode before this field
    /// existed, and the branch's own benchmark table was written from it.
    #[serde(default)]
    shaders: String,
    entries: Vec<CacheEntry>,
}

fn cache_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join("Library/Caches/uzu/tile_cache.json"))
}

struct AutotuneState {
    /// Shapes already resolved this process (calibrated, cached, or failed).
    resolved: HashSet<(u32, u32, u32, u32)>,
    cache: CacheFile,
    cache_loaded_for: Option<String>,
}

static STATE: Mutex<Option<AutotuneState>> = Mutex::new(None);

fn with_state<R>(
    device: &str,
    f: impl FnOnce(&mut AutotuneState) -> R,
) -> R {
    let mut guard = STATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let state = guard.get_or_insert_with(|| AutotuneState {
        resolved: HashSet::new(),
        cache: CacheFile::default(),
        cache_loaded_for: None,
    });
    if state.cache_loaded_for.as_deref() != Some(device) {
        state.cache = load_cache(device);
        state.cache_loaded_for = Some(device.to_string());
        for entry in state.cache.entries.clone() {
            policy::set_autotune_override(
                entry.n,
                entry.k,
                entry.group_size,
                entry.bits,
                GemvTile {
                    num_simdgroups: entry.num_simdgroups,
                    k_split: entry.k_split,
                    results_per_simdgroup: entry.results_per_simdgroup,
                    packs: entry.packs,
                },
            );
            state.resolved.insert((entry.n, entry.k, entry.group_size, entry.bits));
        }
    }
    f(state)
}

fn load_cache(device: &str) -> CacheFile {
    let Some(path) = cache_path() else {
        return CacheFile::default();
    };
    let Ok(text) = fs::read_to_string(&path) else {
        return CacheFile::default();
    };
    match serde_json::from_str::<CacheFile>(&text) {
        Ok(cache)
            if cache.device == device && cache.engine == crate::VERSION && cache.shaders == crate::backends::metal::kernel::METAL_SHADER_FINGERPRINT =>
        {
            cache
        },
        _ => CacheFile::default(),
    }
}

fn persist_cache(cache: &CacheFile) {
    let Some(path) = cache_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(serialized) = serde_json::to_string_pretty(cache) {
        // Best-effort: a read-only sandbox keeps the session override only.
        let _ = fs::write(&path, serialized);
    }
}

/// Whether this (shape, quant) still needs calibration.
pub(crate) fn needs_calibration(
    context: &MetalContext,
    n: u32,
    k: u32,
    group_size: u32,
    bits: u32,
) -> bool {
    if !enabled() {
        return false;
    }
    let device = context.device_label();
    with_state(device, |state| !state.resolved.contains(&(n, k, group_size, bits)))
}

/// Candidate tiles for a quantized decode shape. The fitted default is
/// always measured too, so a well-fitted tier keeps its table.
pub(crate) fn candidates(
    group_size: u32,
    bits: u32,
    fitted: GemvTile,
) -> Vec<GemvTile> {
    let tile = |sg: u32, r: u32, packs: u32| GemvTile {
        num_simdgroups: sg,
        k_split: 1,
        results_per_simdgroup: r,
        packs,
    };
    let mut list = vec![fitted];
    let packs1_exists = bits == 4 && group_size == 64;
    for (sg, r) in [(8, 4), (4, 8), (2, 8), (8, 2), (4, 4), (2, 2)] {
        list.push(tile(sg, r, 2));
        if packs1_exists {
            list.push(tile(sg, r, 1));
        }
    }
    list.dedup_by_key(|t| (t.num_simdgroups, t.results_per_simdgroup, t.k_split, t.packs));
    list
}

/// Records the measured winner: installs the policy override, marks the
/// shape resolved, and persists the cache.
pub(crate) fn record_winner(
    context: &MetalContext,
    n: u32,
    k: u32,
    group_size: u32,
    bits: u32,
    winner: GemvTile,
) {
    let device = context.device_label();
    with_state(device, |state| {
        policy::set_autotune_override(n, k, group_size, bits, winner);
        state.resolved.insert((n, k, group_size, bits));
        state.cache.device = device.to_string();
        state.cache.engine = crate::VERSION.to_string();
        state.cache.shaders = crate::backends::metal::kernel::METAL_SHADER_FINGERPRINT.to_string();
        state.cache.entries.retain(|e| !(e.n == n && e.k == k && e.group_size == group_size && e.bits == bits));
        state.cache.entries.push(CacheEntry {
            n,
            k,
            group_size,
            bits,
            num_simdgroups: winner.num_simdgroups,
            results_per_simdgroup: winner.results_per_simdgroup,
            k_split: winner.k_split,
            packs: winner.packs,
        });
        persist_cache(&state.cache);
    });
}

/// Marks a shape resolved without an override (calibration failed or was
/// not applicable) so it is not retried every dispatch.
pub(crate) fn mark_resolved(
    context: &MetalContext,
    n: u32,
    k: u32,
    group_size: u32,
    bits: u32,
) {
    let device = context.device_label();
    with_state(device, |state| {
        state.resolved.insert((n, k, group_size, bits));
    });
}
