#include "../../common/dsl.h"
#include "../../generated/gemm.h"
#include "common/b_source.h"
#include "common/epilogue.h"
#include "common/output_tile.h"
#include "common/reduce.h"

using namespace metal;
using namespace uzu::gemm;

template <
    typename AT,
    typename BT,
    typename DT,
    typename MT,
    GemmBPrologueKind B_PROLOGUE,
    uint GROUP_SIZE,
    uint BITS,
    uint VALUES_PER_THREAD,
    uint PACKS,
    uint K_SPLIT,
    bool INPUT_ALIGNED,
    uint RESULTS_PER_SIMDGROUP,
    uint NUM_SIMDGROUPS,
    uint M_TILE>
VARIANTS(AT, bfloat, float)
VARIANTS(BT, bfloat, float)
VARIANTS(DT, bfloat, float)
// Staging/multiply type of the quantized source's activations; half wins only
// on tiers with double-rate FP16 ALUs (A19: down k=6912 -34% at kernel level),
// so only the 4-bit bf16 slice real models dispatch is instantiated.
VARIANTS(MT, float, half)
CONSTRAINT(
    MT == "float" ||
    (BITS == 4 && AT == "bfloat" && DT == "bfloat" && (GROUP_SIZE == 32 || GROUP_SIZE == 64)))
CONSTRAINT(BT != "float" || (AT == "float" && DT == "float"))
VARIANTS(
    B_PROLOGUE,
    GemmBPrologueKind::FullPrecision,
    GemmBPrologueKind::ScaleBiasDequant,
    GemmBPrologueKind::ScaleZeroPointDequant,
    GemmBPrologueKind::ScaleSymmetricDequant)
VARIANTS(GROUP_SIZE, 0, 16, 32, 64, 128)
VARIANTS(BITS, 0, 4, 8)
// Lane depth of the fp source; the quantized source derives its own from the
// pack factor, so the axis is pinned there.
VARIANTS(VALUES_PER_THREAD, 4, 8)
CONSTRAINT(BITS == 0 || VALUES_PER_THREAD == 4)
// Per-lane pack depth of the quantized source; 1 wins only on shallow-k
// gs64 forms (see quantized_b_source.h), so only that slice is instantiated.
VARIANTS(PACKS, 1, 2)
CONSTRAINT(PACKS == 2 || (BITS == 4 && GROUP_SIZE == 64))
VARIANTS(K_SPLIT, 1, 2, 4, 8)
VARIANTS(INPUT_ALIGNED, false, true)
VARIANTS(RESULTS_PER_SIMDGROUP, 1, 2, 4, 8)
VARIANTS(NUM_SIMDGROUPS, 2, 4, 8)
CONSTRAINT((B_PROLOGUE == GemmBPrologueKind::FullPrecision) == (BITS == 0))
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
CONSTRAINT(B_PROLOGUE == GemmBPrologueKind::FullPrecision || BT != "float")
// The quantized source supports split-K, but no tuned table selects it, so
// no split variants are instantiated. M1 Max (Aug 2026): kernel-level wins
// on deep aligned k at R4 (-5..-14%) do not survive at engine level - RHT
// rows force R8 (which loses), and gemma's k is block-unaligned. To sweep
// split-K on other tiers, widen this constraint locally and drive tiles via
// UZU_QUANT_TILE / UZU_QUANT_TILE_MAP (policy.rs caps the split per shape).
CONSTRAINT(B_PROLOGUE == GemmBPrologueKind::FullPrecision || K_SPLIT == 1)
CONSTRAINT(K_SPLIT <= NUM_SIMDGROUPS)
// Only selector-reachable tiles are instantiated (fleet-tuned tables): fp
// always runs 8 simdgroups with 1 or 4 rows each; non-default quantized
// tiles exist for bf16 IO only. Widen locally when sweeping new configs.
CONSTRAINT(BITS != 0 || NUM_SIMDGROUPS == 8)
CONSTRAINT(BITS != 0 || RESULTS_PER_SIMDGROUP == 1 || RESULTS_PER_SIMDGROUP == 4)
CONSTRAINT(
    BITS == 0 || (NUM_SIMDGROUPS == 8 && RESULTS_PER_SIMDGROUP == 4) ||
    (AT == "bfloat" && DT == "bfloat"))
// Batched decode tile: one threadgroup runs M_TILE consecutive batch
// elements through a single weight pass, amortizing weight reads and the
// dequant ALU across the batch (the enabler for draft-verification decode,
// where the m=8 pass must not cost m times the m=1 pass). Instantiated only
// for the dense-lane 4-bit gs64 bf16 slice; gather and RHT dispatches stay
// on M_TILE = 1 (the host mirrors this).
VARIANTS(M_TILE, 1, 4, 8)
// fp batched tiles keep the split-K axis (deep-k forms live on it at m = 1);
// the quantized grid still pins batched tiles to K_SPLIT = 1.
CONSTRAINT(
    M_TILE == 1 ||
    (AT == "bfloat" && DT == "bfloat" &&
     (BITS == 0 || (BITS == 4 && GROUP_SIZE == 64 && PACKS == 1 && K_SPLIT == 1))))
KERNEL(Gemv)(
    const device uint32_t* b,
    const device BT* scales
        OPTIONAL(B_PROLOGUE != GemmBPrologueKind::FullPrecision),
    const device uint8_t* zero_points
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant),
    const device BT* biases
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant),
    const device AT* a,
    device DT* d,
    const device BT* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* hadamard_factors
        OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const device uint* gather_indices OPTIONAL(gathered),
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    // Batch GROUPS, not elements: the host passes m / M_TILE here, and each
    // threadgroup covers M_TILE consecutive batch elements.
    const constant uint& batch_size,
    const constant float& ab_scale,
    const constant uint& group_count_x,
    const constant float& soft_cap
        OPTIONAL(output_transform.contains(GemmDTransform::SOFT_CAP)),
    const GemmDTransform output_transform SPECIALIZE,
    const bool gathered SPECIALIZE,
    const bool signed_codes SPECIALIZE,
    threadgroup float shared_results[NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP],
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(group_count_x),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(NUM_SIMDGROUPS)
) {
  typedef float U;
  thread U result[M_TILE * RESULTS_PER_SIMDGROUP] = {0};

  OutputTile<K_SPLIT, NUM_SIMDGROUPS, RESULTS_PER_SIMDGROUP> tile =
      OutputTile<K_SPLIT, NUM_SIMDGROUPS, RESULTS_PER_SIMDGROUP>::make(out_block_idx, simd_group, out_vec_size);
  const uint batch_base = batch_idx * M_TILE;
  d += batch_base * out_vec_size + tile.out_row;

  BSource<
      BT,
      AT,
      U,
      MT,
      M_TILE,
      B_PROLOGUE,
      GROUP_SIZE,
      BITS,
      VALUES_PER_THREAD,
      PACKS,
      K_SPLIT,
      RESULTS_PER_SIMDGROUP,
      INPUT_ALIGNED>::
      accumulate(
      result,
      b,
      scales,
      zero_points,
      biases,
      a,
      gather_indices,
      gathered,
      in_vec_size,
      out_vec_size,
      tile.out_row,
      batch_base,
      simd_lane,
      tile.k_slice,
      signed_codes
  );

  METAL_PRAGMA_UNROLL
  for (uint bt = 0; bt < M_TILE; bt++) {
    thread U(&slice)[RESULTS_PER_SIMDGROUP] =
        *reinterpret_cast<thread U(*)[RESULTS_PER_SIMDGROUP]>(result + bt * RESULTS_PER_SIMDGROUP);

    Reduce<U, K_SPLIT, NUM_SIMDGROUPS, RESULTS_PER_SIMDGROUP>::run(
        slice,
        shared_results,
        simd_group,
        simd_lane,
        tile.row_group,
        tile.k_slice
    );

    Epilogue<BT, DT, U, RESULTS_PER_SIMDGROUP>::store(
        slice,
        d + bt * out_vec_size,
        output_bias,
        hadamard_factors,
        shared_results,
        ab_scale,
        soft_cap,
        output_transform,
        tile.out_row,
        out_vec_size,
        out_block_idx,
        simd_group,
        simd_lane,
        tile.row_group,
        tile.writer
    );

    // Split-K reduces through shared_results; the next batch element must
    // not overwrite it while slice 0 is still reading this one.
    if constexpr (K_SPLIT > 1 && M_TILE > 1) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}
