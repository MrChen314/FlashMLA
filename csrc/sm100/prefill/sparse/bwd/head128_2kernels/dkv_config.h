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

template<
    typename Shape_Q, typename TMA_Q,
    typename Shape_QRoPE, typename TMA_QRoPE,
    typename Shape_dO, typename TMA_dO,
    typename Shape_S, typename TMA_S,
    typename Shape_dS, typename TMA_dS
>
struct TmaParams {
    Shape_Q shape_Q;
    TMA_Q tma_Q;
    Shape_QRoPE shape_Q_rope;
    TMA_QRoPE tma_Q_rope;
    Shape_dO shape_dO;
    TMA_dO tma_dO;
    Shape_S shape_S;
    TMA_S tma_S;
    Shape_dS shape_dS;
    TMA_dS tma_dS;
};

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
static constexpr int NUM_THREADS = 16 * 32;

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
// We keep the 2CTA MMA path, but stage operands in baseline-style MN-major
// layouts first to prioritize correctness over layout tuning.
template<int N_GLOBAL>
using SmemLayoutOperandBTiles = decltype(ku::make_umma_canonical_mn_major_layout<N_GLOBAL / 2, B_H, 128>());

template<int M_GLOBAL>
using SmemLayoutOperandATiles = decltype(ku::make_umma_canonical_mn_major_layout<M_GLOBAL / 2, B_H, 0>());

// Q / dO are consumed by dKV as CTA-local B operands in MN-major.
// The NoPE part keeps one CTA-local half [256, 128], and RoPE keeps [32, 128].
using SmemLayoutQ = SmemLayoutOperandBTiles<D_V>;
using SmemLayoutQRoPE = SmemLayoutOperandBTiles<D_ROPE>;
using SmemLayoutQ_tQ0 = SmemLayoutOperandBTiles<D_dK_tQ0>;
using SmemLayoutQ_tQ1 = SmemLayoutOperandBTiles<D_dK_tQ1>;
using SmemLayoutQ_sQ = SmemLayoutOperandBTiles<D_dK_sQ>;

// dO is staged in the same CTA-local MN-major view as Q.
using SmemLayoutdO = SmemLayoutOperandBTiles<D_V>;
using SmemLayoutdOChunk = SmemLayoutOperandBTiles<D_dV_CHUNK>;

// S / dS are staged directly in the CTA-local MN-major A[M, K] view.
using SmemLayoutS = SmemLayoutOperandATiles<B_TOPK>;
using SmemLayoutdS = SmemLayoutS;
using SmemLayoutdSTransposed = SmemLayoutS;

using TiledMMA_dKV = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_TOPK, 256, UMMA::Major::MN, UMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}
));

using TiledMMA_dKV_RoPE = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_TOPK, D_ROPE, UMMA::Major::MN, UMMA::Major::MN>{}
));

struct tmem_cols {
    // fp32 [64, 512] -> 128kb
    static constexpr int dKV = 0;
    // fp32 [64, 64] -> 16kb
    static constexpr int dKV_RoPE = 256;
    static constexpr int kNumUsedCols = dKV_RoPE + D_ROPE / 2;
};

struct alignas(128) SharedMemoryPlan {
    // dKV-only kernel keeps only the operands needed by
    // dV = S^T @ dO and dK = dS^T @ Q.
    array_aligned<bf16, cosize_v<SmemLayoutQ>> q;
    array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;
    array_aligned<bf16, cosize_v<SmemLayoutdO>> dO;
    struct {
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> s;
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> ds;
    } s_ds;

    transac_bar_t bar_q_nope_ready;
    transac_bar_t bar_q_rope_ready;
    transac_bar_t bar_dO_ready;
    transac_bar_t bar_s_tile_ready;
    transac_bar_t bar_ds_tile_ready;
    transac_bar_t bar_dkv_nope_ready;
    transac_bar_t bar_dkv_rope_ready;

    array_aligned<uint32_t, 1> tmem_start_addr;
};

static constexpr size_t SMEM_SIZE = sizeof(SharedMemoryPlan);

}  // namespace sm100::bwd::head128_2kernels::dkv
