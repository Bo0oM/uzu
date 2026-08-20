use proc_macros::uzu_test;

use super::*;

#[uzu_test]
fn fp_policy_cases() {
    #[rustfmt::skip]
    let cases = [
        (DeviceTier::Large,    1, 12288, 1536, true,  4, tile(8, 8, 1)),
        (DeviceTier::Large,    1, 12288, 1536, false, 4, tile(8, 1, 1)),
        (DeviceTier::Large,    1,  1536,  256, true,  4, tile(8, 1, 1)),
        (DeviceTier::SmallApple9, 1, 1536,  256, true,  4, tile(8, 2, 1)),
        (DeviceTier::Large,    8, 12288, 1536, true,  4, tile(8, 1, 4)),
        (DeviceTier::Large,    1,  1536, 12288, true, 4, tile(8, 8, 4)),
        (DeviceTier::SmallLegacy, 1, 262144, 1536, true, 4, tile(8, 1, 4)),
        // Deep 8-value lanes halve the complete-block count, so the split cap
        // tightens accordingly.
        (DeviceTier::SmallLegacy, 1, 1536, 1024, true, 8, tile(8, 4, 1)),
    ];

    for (tier, m, n, k, aligned, vpt, expected) in cases {
        assert_eq!(fp_tile(m, n, k, aligned, vpt, tier), expected, "tier={tier:?} m={m} n={n} k={k}");
    }
}

#[uzu_test]
fn fp_values_per_thread_cases() {
    // Deep lanes only on the swept tier and only when k tiles into 256-value
    // blocks; gemma's 1152 keeps the 4-value lanes (split-K matters more).
    assert_eq!(fp_values_per_thread(1024, DeviceTier::SmallLegacy), 8);
    assert_eq!(fp_values_per_thread(1152, DeviceTier::SmallLegacy), 4);
    assert_eq!(fp_values_per_thread(1024, DeviceTier::Large), 4);
    assert_eq!(fp_values_per_thread(1024, DeviceTier::SmallApple9), 4);
}

#[uzu_test]
fn quant_policy_cases() {
    #[rustfmt::skip]
    let cases = [
        (DeviceTier::Large,    1,    256, 1536, 4, false, qtile(2, 1)),
        (DeviceTier::Large,    1, 262144, 1536, 4, false, DEFAULT_TILE),
        (DeviceTier::SmallApple9, 1, 1536,  256, 4, false, qtile(4, 4)),
        (DeviceTier::SmallApple8, 1, 2048, 1536, 4, false, qtile(8, 2)),
        (DeviceTier::SmallLegacy, 1,  256, 1536, 4, false, qtile(8, 2)),
        (DeviceTier::SmallLegacy, 1, 1536,  256, 4, false, qtile(4, 8)),
        (DeviceTier::Large,    2,   2048, 1536, 4, false, DEFAULT_TILE),
        (DeviceTier::Large,    1,   2048, 1536, 8, false, DEFAULT_TILE),
        (DeviceTier::Large,    1,   2560, 9216, 4, true,  qtile(4, 8)),
    ];

    for (tier, m, n, k, bits, has_rht, expected) in cases {
        // Group 64 throughout: these rows exercise the fitted tables, which
        // are keyed on shape, not on the scale group.
        assert_eq!(
            quant_tile(m, n, k, 64, bits, has_rht, tier),
            expected,
            "tier={tier:?} m={m} n={n} k={k} bits={bits}"
        );
    }
}

#[uzu_test]
fn quant_k_split_cap_cases() {
    // Under two complete 512-value K blocks the split collapses to 1; above,
    // it is bounded by complete blocks and forced onto 8 simdgroups (the only
    // grid the quantized split kernels are instantiated for).
    assert_eq!(cap_quant_k_split(256, tile(4, 8, 2)), tile(4, 1, 2));
    assert_eq!(cap_quant_k_split(512, tile(8, 2, 4)), tile(8, 1, 4));
    assert_eq!(cap_quant_k_split(1024, tile(4, 8, 2)), tile(8, 2, 2));
    assert_eq!(cap_quant_k_split(1152, tile(8, 8, 4)), tile(8, 2, 4));
    assert_eq!(cap_quant_k_split(6912, tile(8, 8, 4)), tile(8, 8, 4));
    assert_eq!(cap_quant_k_split(4096, tile(8, 4, 1)), tile(8, 4, 1));
    assert_eq!(cap_quant_k_split(4096, tile(8, 1, 4)), tile(8, 1, 4));
}

#[uzu_test]
fn rht_rows_guard_cases() {
    // RHT keeps only tiles covering exactly one 32-row hadamard block.
    assert_eq!(rht_rows_guard(true, tile(8, 1, 4)), tile(8, 1, 4));
    assert_eq!(rht_rows_guard(true, tile(8, 2, 8)), tile(8, 2, 8));
    assert_eq!(rht_rows_guard(true, tile(4, 1, 8)), tile(4, 1, 8));
    assert_eq!(rht_rows_guard(true, tile(8, 1, 8)), DEFAULT_TILE);
    assert_eq!(rht_rows_guard(true, tile(8, 4, 4)), DEFAULT_TILE);
    assert_eq!(rht_rows_guard(false, tile(8, 1, 8)), tile(8, 1, 8));
}
