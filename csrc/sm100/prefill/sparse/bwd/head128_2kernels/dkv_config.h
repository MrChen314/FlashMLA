#pragma once

#include <cuda.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

#include "params.h"
#include "defines.h"

namespace sm100::bwd::head128_2kernels::dkv {

using namespace cute;

static constexpr int D_QK = 576;
static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr int D_ROPE = D_Q - D_V;

static constexpr int D_tQ = 384;
static constexpr int NUM_tQ_TILES = D_tQ / 64;
static constexpr int D_sQ = D_QK - D_tQ;
static constexpr int NUM_sQ_TILES = D_sQ / 64;
static_assert(D_sQ % 64 == 0 && D_tQ % 64 == 0 && D_sQ + D_tQ == D_Q);

static constexpr int B_H = 128;
static constexpr int B_TOPK = 128;

// dV = S^T @ dO : [M, N] = [128, 512], reduction K = 128
// 2CTA output split: each CTA owns dV_local[64, 512]
static constexpr int D_dV_CHUNK = 256;

// dK = dS^T @ Q : [M, N] = [128, 576], reduction K = 128
// 2CTA output split: each CTA owns dK_local[64, 576]
static constexpr int D_dK_tQ0 = 256;
static constexpr int D_dK_tQ1 = 128;
static constexpr int D_dK_sQ = D_sQ;

static_assert(D_dV_CHUNK % 16 == 0);
static_assert(D_dK_tQ0 % 16 == 0);
static_assert(D_dK_tQ1 % 16 == 0);
static_assert(D_dK_sQ % 16 == 0);
static_assert(D_dK_tQ0 + D_dK_tQ1 == D_tQ);
static_assert(D_dK_tQ0 + D_dK_tQ1 + D_dK_sQ == D_Q);

// For 2CTA dKV MMA:
// - global M = 128, each CTA owns M/2 = 64 output rows
// - global K = 128 and must stay complete for every CTA-local operand
// - global N is split across CTA-local B operands and reassembled by cta_group::2
template<int N_GLOBAL>
using SmemLayoutOperandBTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<N_GLOBAL / 2>, Int<B_H>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

template<int M_GLOBAL>
using SmemLayoutOperandATiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_INTER_Atom<bf16>{},
    Shape<Int<M_GLOBAL / 2>, Int<B_H>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

// Q is staged for dK as B[K, N] = Q[128, 576], so each CTA keeps [128, 288] in GEMM-B view.
using SmemLayoutQ = SmemLayoutOperandBTiles<D_Q>;
using SmemLayoutQ_tQ0 = SmemLayoutOperandBTiles<D_dK_tQ0>;
using SmemLayoutQ_tQ1 = SmemLayoutOperandBTiles<D_dK_tQ1>;
using SmemLayoutQ_sQ = SmemLayoutOperandBTiles<D_dK_sQ>;

// dO is staged for dV as B[K, N] = dO[128, 512], so each CTA keeps [128, 256] in GEMM-B view.
using SmemLayoutdO = SmemLayoutOperandBTiles<D_V>;
using SmemLayoutdOChunk = SmemLayoutOperandBTiles<D_dV_CHUNK>;

// S / dS are staged directly in the A[M, K] view for 2CTA dKV MMA.
using SmemLayoutS = SmemLayoutOperandATiles<B_TOPK>;
using SmemLayoutdS = SmemLayoutS;
using SmemLayoutdSTransposed = SmemLayoutS;

using TiledMMA_dV = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_TOPK, D_dV_CHUNK, UMMA::Major::MN, UMMA::Major::MN>{}
));

using TiledMMA_dK_tQ0 = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_TOPK, D_dK_tQ0, UMMA::Major::MN, UMMA::Major::MN>{}
));

using TiledMMA_dK_tQ1 = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_TOPK, D_dK_tQ1, UMMA::Major::MN, UMMA::Major::MN>{}
));

using TiledMMA_dK_sQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_TOPK, D_dK_sQ, UMMA::Major::MN, UMMA::Major::MN>{}
));

struct tmem_cols {
    // Single 2CTA dKV TMEM buffer reused across dV / dK phases.
    //
    // In cta_group::2 mode, a float output tile with global width N occupies N/2 TMEM columns.
    // Therefore:
    // - dV[128, 512] needs 256 cols
    // - dK[128, 576] needs 288 cols
    //
    // Each CTA owns output rows [64, N], i.e. the output is split on M rather than K.
    static constexpr int dKV = 0;

    // dV = S^T @ dO, split as [256 | 256]
    static constexpr int dV_part0 = dKV;
    static constexpr int dV_part1 = dV_part0 + D_dV_CHUNK / 2;

    // dK = dS^T @ Q, split as [256 | 128 | 192]
    static constexpr int dK_tQ0 = dKV;
    static constexpr int dK_tQ1 = dK_tQ0 + D_dK_tQ0 / 2;
    static constexpr int dK_sQ = dK_tQ1 + D_dK_tQ1 / 2;

    static constexpr int kNumUsedCols = dK_sQ + D_dK_sQ / 2;
};

static_assert(tmem_cols::dV_part1 == tmem_cols::dV_part0 + D_dV_CHUNK / 2);
static_assert(tmem_cols::dK_tQ1 == tmem_cols::dK_tQ0 + D_dK_tQ0 / 2);
static_assert(tmem_cols::dK_sQ == tmem_cols::dK_tQ1 + D_dK_tQ1 / 2);
static_assert(tmem_cols::kNumUsedCols == D_Q / 2);

struct alignas(128) SharedMemoryPlan {
    // dKV-only kernel keeps only the operands needed by
    // dV = S^T @ dO and dK = dS^T @ Q.
    array_aligned<bf16, cosize_v<SmemLayoutQ>> q;
    array_aligned<bf16, cosize_v<SmemLayoutdO>> dO;
    struct {
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> s;
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> ds;
    } s_ds;
    char is_k_valid[B_TOPK / 8];

    transac_bar_t bar_prologue_q;
    transac_bar_t bar_prologue_dO;
    transac_bar_t bar_s_ready;
    transac_bar_t bar_ds_ready;
    transac_bar_t bar_k_valid_free;
    transac_bar_t bar_k_valid_ready;
    transac_bar_t bar_dv_ready;
    transac_bar_t bar_dv_done;
    transac_bar_t bar_dk_tQ_ready;
    transac_bar_t bar_dk_tQ_done;
    transac_bar_t bar_dk_sQ_ready;
    transac_bar_t bar_dk_sQ_done;

    array_aligned<uint32_t, 1> tmem_start_addr;
};

static constexpr size_t SMEM_SIZE = sizeof(SharedMemoryPlan);

}  // namespace sm100::bwd::head128_2kernels::dkv
